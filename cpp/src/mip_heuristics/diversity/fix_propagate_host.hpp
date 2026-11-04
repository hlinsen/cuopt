/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// Host-side fix-and-propagate. Integer variables are fixed one at a time in a caller-given order
// and value preference; after each fixing the branch-and-bound bound propagation (domain_t: all
// rows, incremental activities, clique implications) propagates the change. A conflict triggers a
// one-level backtrack (try the other value); if both values conflict the pattern is reported as
// conflicting.
//
// References
//   [KK]  N.-C. Kempke, T. Koch. Fix-and-Propagate Heuristics Using Low-Precision First-Order LP
//         Solutions for Large-Scale Mixed-Integer Linear Optimization. arXiv:2503.10344.
//   [SRF] D. Salvagnin, R. Roberti, M. Fischetti. A fix-propagate-repair heuristic for mixed
//         integer programming. Mathematical Programming Computation (2024),
//         doi:10.1007/s12532-024-00269-5. ([KK] builds on its depth-first modes.)
//   [GBHW] G. Gamrath, T. Berthold, S. Heinz, M. Winkler. Structure-driven fix-and-propagate
//         heuristics for mixed integer programming. Mathematical Programming Computation 11
//         (2019), doi:10.1007/s12532-019-00159-1. (LP-free fix-and-propagate before the LP, as in
//         their locks/clique/vbounds heuristics; our LP-free start is a simpler bound-based one.)
// Terminology follows [KK]: a variable strategy orders the integers, a value strategy picks the
// value each integer is fixed to. Both are computed from a reference point: the root LP solution,
// or zero clamped to the bounds when no LP is available yet.
// Not implemented here: the Dual variable strategy of [KK], its integer-aware three-way branching,
// the clique-cover orders, propagators and in-pass repair of [SRF], and the dynamic order of the
// [GBHW] locks heuristic (its lock updates only change the value here, as in [SRF] loosedyn).
// Continuous bounds are tightened inside the domain only (conflict detection, never handed to an
// LP), subject to a minimum relative improvement.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <dual_simplex/domain.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/user_problem.hpp>
#include <mip_heuristics/presolve/conflict_graph/clique_table.cuh>

#include <memory>

namespace cuopt::mathematical_optimization::mip {

// Variable strategy: integers with a larger key are fixed first.
// The integer values are the settings values (mip_hyper_heuristic_fix_propagate_*_strategy).
enum class fix_propagate_variable_strategy_t {
  FRACTIONALITY = 0,  // closest to integral first; same measure as [KK] Frac
  LP_VALUE_DESC = 1,  // largest reference value first; not in [KK]
  REDUCED_COST  = 2,  // most negative reduced cost first; [KK] RedCost orders by |red. cost|
  UPLOCKS       = 3,  // fewest up-locks first, ties by larger reference value; cf. [SRF] locks
  LOCKS         = 4,  // most locks (up + down) first, ties by larger reference value: the order of
                      // the [GBHW] locks heuristic
  REDUCED_COST_ABS = 5,  // largest |reduced cost| first: [KK] RedCost
  TYPE             = 6,  // binaries, then general integers, in formulation order: [KK] Type,
                         // [SRF] type
};

// Value strategy: the value an integer is fixed to (fix_and_propagate clamps it to the current
// domain).
enum class fix_propagate_value_strategy_t {
  UPPER_BOUND              = 0,  // LP-free
  LOWER_BOUND              = 1,  // LP-free
  ROUND_UP                 = 2,  // round any fractional reference value up
  ROUND_UP_OR_NEG_REDCOST  = 3,  // round up if fractional or reduced cost < -param
  ROUND_UP_ABOVE_THRESHOLD = 4,  // round up if the fractional part is > param
  LP_RANDOMIZED            = 5,  // round up with probability equal to the fractional part:
                                 // the LP-based fixing of [KK]
  FEWER_LOCKS = 6,               // round toward the side with fewer locks (the locks heuristic
                                 // fixes to that bound); ties round a fractional value up
  GOOD_OBJ = 7,                  // the bound that improves the objective, the lower one without an
                                 // objective coefficient: [SRF] goodobj (LP-free)
  BAD_OBJ      = 8,  // the bound that worsens the objective, same tie: [SRF] badobj (LP-free)
  RANDOM_BOUND = 9,  // lower or upper bound with probability 1/2: [SRF] random (LP-free)
  FEWER_LOCKS_DYNAMIC = 10,  // FEWER_LOCKS with the locks recounted at fixing time over the rows
                             // that can still be violated: [SRF] loosedyn
};

inline bool fix_propagate_dynamic_locks(fix_propagate_value_strategy_t strategy)
{
  return strategy == fix_propagate_value_strategy_t::FEWER_LOCKS_DYNAMIC;
}

// value strategies that draw one uniform number per integer
inline bool fix_propagate_randomized(fix_propagate_value_strategy_t strategy)
{
  return strategy == fix_propagate_value_strategy_t::LP_RANDOMIZED ||
         strategy == fix_propagate_value_strategy_t::RANDOM_BOUND;
}

// Type grouping ([SRF]: the first step of every order but LR): binaries before general integers,
// the order key deciding within each group. Only TYPE groups.
inline bool fix_propagate_groups_by_type(fix_propagate_variable_strategy_t strategy)
{
  return strategy == fix_propagate_variable_strategy_t::TYPE;
}

template <typename i_t, typename f_t>
f_t fix_propagate_order_key(fix_propagate_variable_strategy_t strategy,
                            f_t reference_value,
                            f_t reduced_cost,
                            i_t uplocks,
                            i_t downlocks = 0)
{
  switch (strategy) {
    case fix_propagate_variable_strategy_t::FRACTIONALITY:
      return std::abs(reference_value - std::floor(reference_value) - f_t(0.5));
    case fix_propagate_variable_strategy_t::LP_VALUE_DESC: return reference_value;
    case fix_propagate_variable_strategy_t::REDUCED_COST: return -reduced_cost;
    case fix_propagate_variable_strategy_t::UPLOCKS:
      return -(f_t)uplocks + f_t(1e-3) * reference_value;
    case fix_propagate_variable_strategy_t::LOCKS:
      return (f_t)(uplocks + downlocks) + f_t(1e-3) * reference_value;
    case fix_propagate_variable_strategy_t::REDUCED_COST_ABS: return std::abs(reduced_cost);
    case fix_propagate_variable_strategy_t::TYPE:
      return 0;  // formulation order within each type group
  }
  return reference_value;
}

// Preferred value of one integer. For the dynamic-locks strategies this is the value taken on a
// lock tie; fix_and_propagate decides the direction at fixing time (see
// fix_propagate_dynamic_lock_values). The LP-free bound strategies fall back to rounding the
// reference value up when the chosen bound is infinite ([SRF] assumes finite bounds).
template <typename f_t>
f_t fix_propagate_preferred_value(fix_propagate_value_strategy_t strategy,
                                  f_t reference_value,
                                  f_t lb,
                                  f_t ub,
                                  f_t reduced_cost,
                                  f_t value_param,
                                  f_t integrality_tolerance,
                                  f_t uniform_draw          = 0,
                                  int uplocks               = 0,
                                  int downlocks             = 0,
                                  f_t objective_coefficient = 0)
{
  const f_t floor_value     = std::floor(reference_value + integrality_tolerance);
  const f_t fractional_part = reference_value - floor_value;
  const f_t round_up = fractional_part > integrality_tolerance ? floor_value + 1 : floor_value;
  auto bound = [&](f_t bound_value) { return std::isfinite(bound_value) ? bound_value : round_up; };
  switch (strategy) {
    case fix_propagate_value_strategy_t::UPPER_BOUND: return bound(ub);
    case fix_propagate_value_strategy_t::LOWER_BOUND: return bound(lb);
    case fix_propagate_value_strategy_t::ROUND_UP: return round_up;
    case fix_propagate_value_strategy_t::ROUND_UP_OR_NEG_REDCOST:
      return (fractional_part > integrality_tolerance || reduced_cost < -value_param)
               ? std::min(floor_value + 1, ub)
               : floor_value;
    case fix_propagate_value_strategy_t::ROUND_UP_ABOVE_THRESHOLD:
      return fractional_part > value_param ? floor_value + 1 : floor_value;
    case fix_propagate_value_strategy_t::LP_RANDOMIZED:
      return uniform_draw < fractional_part ? floor_value + 1
                                            : floor_value;  // uniform_draw ~ U(0,1)
    case fix_propagate_value_strategy_t::FEWER_LOCKS:
      if (uplocks != downlocks) {
        return uplocks < downlocks ? std::min(floor_value + 1, ub) : floor_value;
      }
      return round_up;
    case fix_propagate_value_strategy_t::GOOD_OBJ:
      return bound(objective_coefficient < 0 ? ub : lb);
    case fix_propagate_value_strategy_t::BAD_OBJ: return bound(objective_coefficient > 0 ? ub : lb);
    case fix_propagate_value_strategy_t::RANDOM_BOUND:
      return bound(uniform_draw < f_t(0.5) ? ub : lb);
    case fix_propagate_value_strategy_t::FEWER_LOCKS_DYNAMIC: return round_up;
  }
  return floor_value;
}

// Values of a dynamic-locks variable when its up-locks (first) or down-locks (second) are fewer at
// fixing time; false when the strategy does not use dynamic locks.
template <typename f_t>
bool fix_propagate_dynamic_lock_values(fix_propagate_value_strategy_t strategy,
                                       f_t reference_value,
                                       f_t ub,
                                       f_t integrality_tolerance,
                                       f_t& up_value,
                                       f_t& down_value)
{
  if (strategy == fix_propagate_value_strategy_t::FEWER_LOCKS_DYNAMIC) {
    const f_t floor_value = std::floor(reference_value + integrality_tolerance);
    up_value              = std::min(floor_value + 1, ub);
    down_value            = floor_value;
    return true;
  }
  return false;
}

// A row side still locks its variables when it can be violated given the current activity range:
// the <= side while the maximum activity exceeds the right-hand side, the >= side while the minimum
// activity is below the left-hand side. Sides already redundant stop counting ([SRF] loosedyn).
template <typename f_t>
void fix_propagate_active_sides(f_t min_activity,
                                f_t max_activity,
                                f_t row_lb,
                                f_t row_ub,
                                f_t tolerance,
                                bool& le_active,
                                bool& ge_active)
{
  le_active = std::isfinite(row_ub) && !(max_activity <= row_ub + tolerance);
  ge_active = std::isfinite(row_lb) && !(min_activity >= row_lb - tolerance);
}

template <typename i_t, typename f_t>
struct h_fix_propagate_t {
  // the constraint matrix on the host, row-wise and column-wise (problem_t's offsets / variables /
  // coefficients and reverse_* views), and the constraint bounds: used for reduced costs, locks and
  // dynamic locks, not for propagation
  std::vector<i_t> row_offsets;  // size n_constraints + 1
  std::vector<i_t> row_columns;
  std::vector<f_t> row_values;
  std::vector<f_t> row_lb, row_ub;
  std::vector<i_t> column_offsets;  // size n_variables + 1
  std::vector<i_t> column_rows;
  std::vector<f_t> column_values;
  // values of the last pass (integer entries are the fixings)
  std::vector<f_t> pass_values;
  f_t absolute_tolerance    = 1e-6;
  f_t integrality_tolerance = 1e-5;
  int64_t work              = 0;
  // propagation work budget of one pass (nonzeros processed); exceeding it ends the pass like a
  // conflict, so one pass cannot hold local search for long on models with very dense rows
  int64_t work_limit  = std::numeric_limits<int64_t>::max();
  bool work_limit_hit = false;
  // a continuous tightening is kept if it shrinks the domain by at least continuous_min_improvement
  // of its width (0 keeps every tightening, above 1 none; infinite bounds becoming finite are kept)
  f_t continuous_min_improvement   = 0.3;
  int64_t n_continuous_tightenings = 0;

  // propagation engine: the branch-and-bound bound propagation (simplex::domain_t) on the problem
  // in its LP form (equality rows with slack columns appended after the original columns)
  simplex::simplex_solver_settings_t<i_t, f_t> settings;
  std::unique_ptr<simplex::lp_problem_t<i_t, f_t>> lp;
  csr_matrix_t<i_t, f_t> Arow{1, 1, 0};
  std::vector<simplex::variable_type_t> var_types;
  std::vector<f_t> initial_lower, initial_upper;
  simplex::domain_t<i_t, f_t> domain;
  i_t n_original_columns = 0;

  // Builds the propagation domain from the problem (same conversion as branch and bound).
  void build_domain(const simplex::user_problem_t<i_t, f_t>& user_problem,
                    const clique_table_t<i_t, f_t>* clique_table)
  {
    settings.log.log     = false;
    settings.barrier     = false;
    settings.dualize     = 0;
    settings.primal_tol  = absolute_tolerance;
    settings.integer_tol = integrality_tolerance;
    lp = std::make_unique<simplex::lp_problem_t<i_t, f_t>>(user_problem.handle_ptr, 1, 1, 1);
    std::vector<i_t> new_slacks;
    simplex::dualize_info_t<i_t, f_t> dualize_info;
    simplex::convert_user_problem(user_problem, settings, *lp, new_slacks, dualize_info);
    n_original_columns = user_problem.num_cols;
    var_types          = user_problem.var_types;
    var_types.resize(lp->num_cols, simplex::variable_type_t::CONTINUOUS);
    lp->A.to_compressed_row(Arow);
    initial_lower = lp->lower;
    initial_upper = lp->upper;
    simplex::domain_params domain_parameters;
    domain_parameters.min_relative_improvement = continuous_min_improvement;
    domain.set_params(domain_parameters);
    domain.implied_bounds = nullptr;
    domain.clique_table   = clique_table;
  }

  // fix variable to value and propagate; false (state restored) on conflict
  bool try_fix(i_t variable, f_t value)
  {
    simplex::bound_change_t<i_t, f_t> change{.var       = variable,
                                             .new_upper = value,
                                             .new_lower = value,
                                             .origin    = simplex::bound_change_origin_t::BRANCH};
    const bool propagated = domain.apply_and_propagate(Arow, var_types, settings, change, *lp);
    work += (int64_t)domain.last_nnz_processed;
    if (propagated) {
      for (size_t k = 0; k < domain.size(); ++k) {
        const i_t changed_variable = domain[k].var;
        if (domain[k].origin == simplex::bound_change_origin_t::BOUND_PROPAGATION &&
            changed_variable < n_original_columns &&
            var_types[changed_variable] == simplex::variable_type_t::CONTINUOUS) {
          n_continuous_tightenings++;
        }
      }
      domain.clear();  // never backtracked past an accepted fixing
      return true;
    }
    domain.backtrack_to_parent(*lp);
    return false;
  }

  // up- and down-locks of the integer variable over the rows that can still be violated under the
  // current bounds (fix_propagate_active_sides); the scan counts toward the pass's work
  void dynamic_locks(i_t variable, i_t& uplocks, i_t& downlocks)
  {
    uplocks = downlocks = 0;
    for (i_t entry = column_offsets[variable]; entry < column_offsets[variable + 1]; ++entry) {
      const i_t row    = column_rows[entry];
      f_t min_activity = 0, max_activity = 0;
      for (i_t k = row_offsets[row]; k < row_offsets[row + 1]; ++k) {
        const i_t column      = row_columns[k];
        const f_t coefficient = row_values[k];
        min_activity +=
          coefficient > 0 ? coefficient * lp->lower[column] : coefficient * lp->upper[column];
        max_activity +=
          coefficient > 0 ? coefficient * lp->upper[column] : coefficient * lp->lower[column];
      }
      work += row_offsets[row + 1] - row_offsets[row];
      bool le_active, ge_active;
      fix_propagate_active_sides<f_t>(min_activity,
                                      max_activity,
                                      row_lb[row],
                                      row_ub[row],
                                      absolute_tolerance,
                                      le_active,
                                      ge_active);
      const f_t coefficient = column_values[entry];
      if ((coefficient > 0 && le_active) || (coefficient < 0 && ge_active)) { uplocks++; }
      if ((coefficient > 0 && ge_active) || (coefficient < 0 && le_active)) { downlocks++; }
    }
  }

  // fix-and-propagate pass: order = integer vars in processing order, preferred_values = preferred
  // value per var (indexed by var id). One-level backtrack: on a conflict the other end of the
  // domain is tried; if both conflict, the remaining integers get their preferred values without
  // propagation (the pattern is reported as conflicting), as when the pass exceeds work_limit.
  // Returns the number of conflicts; values in pass_values. dynamic_variables (optional, by var
  // id): variables whose value is decided at fixing time by their dynamic locks: dynamic_up_values
  // when they have fewer up-locks, dynamic_down_values when fewer down-locks, preferred_values on a
  // tie.
  i_t fix_and_propagate(const std::vector<i_t>& order,
                        const std::vector<f_t>& preferred_values,
                        i_t& n_flipped,
                        i_t& n_implied,
                        const std::vector<char>* dynamic_variables  = nullptr,
                        const std::vector<f_t>* dynamic_up_values   = nullptr,
                        const std::vector<f_t>* dynamic_down_values = nullptr)
  {
    lp->lower = initial_lower;
    lp->upper = initial_upper;
    domain.clear();
    work                     = 0;
    n_continuous_tightenings = 0;
    n_flipped                = 0;
    n_implied                = 0;
    work_limit_hit           = false;
    i_t n_conflicts          = 0;
    if (!domain.propagate_full(Arow, var_types, settings, *lp)) { n_conflicts++; }
    work += (int64_t)domain.last_nnz_processed;
    domain.clear();
    bool propagate = n_conflicts == 0;
    for (auto variable : order) {
      const f_t lb = lp->lower[variable], ub = lp->upper[variable];
      if (lb == ub) {
        if (lb != preferred_values[variable]) n_implied++;
        continue;
      }
      f_t preferred = preferred_values[variable];
      if (propagate && dynamic_variables != nullptr && (*dynamic_variables)[variable]) {
        i_t up_locks = 0, down_locks = 0;
        dynamic_locks(variable, up_locks, down_locks);
        if (up_locks < down_locks) {
          preferred = (*dynamic_up_values)[variable];
        } else if (down_locks < up_locks) {
          preferred = (*dynamic_down_values)[variable];
        }
      }
      const f_t value = std::min(std::max(preferred, lb), ub);
      if (propagate && work > work_limit) {
        work_limit_hit = true;
        n_conflicts++;
        propagate = false;
      }
      if (!propagate) {
        lp->lower[variable] = lp->upper[variable] = value;
        continue;
      }
      if (try_fix(variable, value)) { continue; }
      const f_t other_value = (value == lb) ? ub : lb;
      if (try_fix(variable, other_value)) {
        n_flipped++;
        continue;
      }
      n_conflicts++;
      propagate           = false;
      lp->lower[variable] = lp->upper[variable] = value;
    }
    pass_values.assign(lp->lower.begin(), lp->lower.begin() + n_original_columns);
    return n_conflicts;
  }
};

}  // namespace cuopt::mathematical_optimization::mip
