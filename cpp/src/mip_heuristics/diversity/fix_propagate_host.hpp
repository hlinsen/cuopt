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
//         SCIP's locks/clique/vbounds heuristics; our LP-free start is a simpler bound-based one.)
// Terminology follows [KK]: a variable strategy orders the integers, a value strategy picks the
// value each integer is fixed to. Both are computed from a reference point: the root LP solution,
// or zero clamped to the bounds when no LP is available yet.
// Not implemented here: the Dual variable strategy of [KK], its integer-aware three-way branching,
// the clique-cover orders and the in-pass (WalkSAT-style) repair of [SRF], and the dynamic order of
// SCIP's locks heuristic (its lock updates only change the value here, as in [SRF] loosedyn).
// Continuous bounds are tightened inside the domain only (conflict detection, never handed to an
// LP), subject to a minimum relative improvement.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <dual_simplex/domain.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/user_problem.hpp>
#include <mip_heuristics/presolve/conflict_graph/clique_table.cuh>

#include <memory>

namespace cuopt::mathematical_optimization::mip {

// Variable strategy: integers with a larger key are fixed first.
// The integer values are the settings values (mip_hyper_heuristic_fix_propagate_*_strategy).
enum class fp_variable_strategy_t {
  FRACTIONALITY = 0,  // closest to integral first; same measure as [KK] Frac
  LP_VALUE_DESC = 1,  // largest reference value first; not in [KK]
  REDUCED_COST  = 2,  // most negative reduced cost first; [KK] RedCost orders by |red. cost|
  UPLOCKS       = 3,  // fewest up-locks first, ties by larger reference value; cf. [SRF] locks
  LOCKS         = 4,  // most locks (up + down) first, ties by larger reference value: the order of
                      // SCIP's locks heuristic
  REDUCED_COST_ABS = 5,  // largest |reduced cost| first: [KK] RedCost
  TYPE             = 6,  // binaries, then general integers, in formulation order: [KK] Type,
                         // [SRF] type
};

// Value strategy: the value an integer is fixed to (fix_and_propagate clamps it to the current
// domain).
enum class fp_value_strategy_t {
  UPPER_BOUND              = 0,  // LP-free
  LOWER_BOUND              = 1,  // LP-free
  ROUND_UP                 = 2,  // round any fractional reference value up
  ROUND_UP_OR_NEG_REDCOST  = 3,  // round up if fractional or reduced cost < -param
  ROUND_UP_ABOVE_THRESHOLD = 4,  // round up if the fractional part is > param
  LP_RANDOMIZED            = 5,  // round up with probability equal to the fractional part:
                                 // the LP-based fixing of [KK]
  FEWER_LOCKS = 6,               // round toward the side with fewer locks (SCIP's locks heuristic
                                 // fixes to that bound); ties round a fractional value up
  GOOD_OBJ = 7,                  // the bound that improves the objective, the lower one without an
                                 // objective coefficient: [SRF] goodobj (LP-free)
  BAD_OBJ      = 8,  // the bound that worsens the objective, same tie: [SRF] badobj (LP-free)
  RANDOM_BOUND = 9,  // lower or upper bound with probability 1/2: [SRF] random (LP-free)
  FEWER_LOCKS_DYNAMIC = 10,  // FEWER_LOCKS with the locks recounted at fixing time over the rows
                             // that can still be violated: [SRF] loosedyn
  SCIP_LOCKS = 11,  // SCIP's locks heuristic: dynamic locks on binaries only (general integers
                    // round up as ROUND_UP, after all binaries), ties up when the objective
                    // coefficient is <= 0, else up with probability 0.67
};

inline bool fp_dynamic_locks(fp_value_strategy_t s)
{
  return s == fp_value_strategy_t::FEWER_LOCKS_DYNAMIC || s == fp_value_strategy_t::SCIP_LOCKS;
}

// Type grouping ([SRF]: the first step of every order but LR): binaries before general integers,
// the order key deciding within each group. TYPE and SCIP_LOCKS always group.
inline bool fp_groups_by_type(fp_variable_strategy_t s, fp_value_strategy_t v, bool grouping)
{
  return grouping || s == fp_variable_strategy_t::TYPE || v == fp_value_strategy_t::SCIP_LOCKS;
}

// SCIP's locks heuristic: probability of rounding up on a lock tie with a positive objective
// coefficient (heuristics/locks/roundupprobability)
constexpr double fp_scip_locks_roundup_probability = 0.67;

template <typename i_t, typename f_t>
f_t fp_order_key(fp_variable_strategy_t s, f_t x, f_t red, i_t uplocks, i_t downlocks = 0)
{
  switch (s) {
    case fp_variable_strategy_t::FRACTIONALITY: return std::abs(x - std::floor(x) - f_t(0.5));
    case fp_variable_strategy_t::LP_VALUE_DESC: return x;
    case fp_variable_strategy_t::REDUCED_COST: return -red;
    case fp_variable_strategy_t::UPLOCKS: return -(f_t)uplocks + f_t(1e-3) * x;
    case fp_variable_strategy_t::LOCKS: return (f_t)(uplocks + downlocks) + f_t(1e-3) * x;
    case fp_variable_strategy_t::REDUCED_COST_ABS: return std::abs(red);
    case fp_variable_strategy_t::TYPE: return 0;  // formulation order within each type group
  }
  return x;
}

// Preferred value of one integer. For the dynamic-locks strategies this is the value taken on a
// lock tie; fix_and_propagate decides the direction at fixing time (see fp_dynamic_lock_values).
// The LP-free bound strategies fall back to rounding the reference value up when the chosen bound
// is infinite ([SRF] assumes finite bounds).
template <typename f_t>
f_t fp_preferred_value(fp_value_strategy_t s,
                       f_t x,
                       f_t lb,
                       f_t ub,
                       f_t red,
                       f_t param,
                       f_t int_tol,
                       f_t u         = 0,
                       int uplocks   = 0,
                       int downlocks = 0,
                       f_t obj       = 0)
{
  const f_t fl       = std::floor(x + int_tol);
  const f_t fr       = x - fl;
  const f_t round_up = fr > int_tol ? fl + 1 : fl;
  auto bound         = [&](f_t b) { return std::isfinite(b) ? b : round_up; };
  switch (s) {
    case fp_value_strategy_t::UPPER_BOUND: return bound(ub);
    case fp_value_strategy_t::LOWER_BOUND: return bound(lb);
    case fp_value_strategy_t::ROUND_UP: return round_up;
    case fp_value_strategy_t::GOOD_OBJ: return bound(obj < 0 ? ub : lb);
    case fp_value_strategy_t::BAD_OBJ: return bound(obj > 0 ? ub : lb);
    case fp_value_strategy_t::RANDOM_BOUND: return bound(u < f_t(0.5) ? ub : lb);
    case fp_value_strategy_t::FEWER_LOCKS_DYNAMIC: return round_up;
    case fp_value_strategy_t::SCIP_LOCKS:
      if (lb != 0 || ub != 1) { return round_up; }  // general integers: not fixed by locks
      return (obj <= 0 || u < f_t(fp_scip_locks_roundup_probability)) ? ub : lb;
    case fp_value_strategy_t::ROUND_UP_OR_NEG_REDCOST:
      return (fr > int_tol || red < -param) ? std::min(fl + 1, ub) : fl;
    case fp_value_strategy_t::ROUND_UP_ABOVE_THRESHOLD: return fr > param ? fl + 1 : fl;
    case fp_value_strategy_t::LP_RANDOMIZED: return u < fr ? fl + 1 : fl;  // u ~ U(0,1)
    case fp_value_strategy_t::FEWER_LOCKS:
      if (uplocks != downlocks) { return uplocks < downlocks ? std::min(fl + 1, ub) : fl; }
      return fr > int_tol ? fl + 1 : fl;
  }
  return fl;
}

// Values of a dynamic-locks variable when its up-locks (first) or down-locks (second) are fewer at
// fixing time; nothing when the strategy does not use dynamic locks for this variable (general
// integers under SCIP_LOCKS).
template <typename f_t>
bool fp_dynamic_lock_values(
  fp_value_strategy_t s, f_t x, f_t lb, f_t ub, f_t int_tol, f_t& up_value, f_t& down_value)
{
  if (s == fp_value_strategy_t::FEWER_LOCKS_DYNAMIC) {
    const f_t fl = std::floor(x + int_tol);
    up_value     = std::min(fl + 1, ub);
    down_value   = fl;
    return true;
  }
  if (s == fp_value_strategy_t::SCIP_LOCKS && lb == 0 && ub == 1) {
    up_value   = 1;
    down_value = 0;
    return true;
  }
  return false;
}

// [SRF] Sect. 5: the shift of variable j (coefficient a, domain [lo, hi] within its original
// [lo0, hi0]) that repairs a violated row, where gap = U - min activity (row too high, negative) or
// L - max activity (row too low, positive). Only variables whose domain was reduced can move: a
// fixed binary flips when the flip goes the right way; other variables shift their interval by
// gap / a, rounded toward zero for integers and clipped to the original bounds. 0 = no move.
template <typename f_t>
f_t fp_repair_shift(
  f_t a, bool too_high, f_t gap, f_t lo, f_t hi, f_t lo0, f_t hi0, bool is_integer, f_t int_tol)
{
  if (a == 0 || !(lo > lo0 || hi < hi0)) { return 0; }
  f_t s = gap / a;
  if (is_integer && lo0 == 0 && hi0 == 1) {
    if (lo != hi) { return 0; }
    const f_t flip = lo > f_t(0.5) ? -1 : 1;
    return (flip > 0) == (s > 0) ? flip : 0;
  }
  if (is_integer) { s = s > 0 ? std::floor(s + int_tol) : std::ceil(s - int_tol); }
  s = std::min(s, hi0 - hi);
  s = std::max(s, lo0 - lo);
  (void)too_high;
  return std::isfinite(s) ? s : 0;
}

// A row side still locks its variables when it can be violated given the current activity range:
// the <= side while the maximum activity exceeds the right-hand side, the >= side while the minimum
// activity is below the left-hand side. Sides already redundant stop counting ([SRF] loosedyn,
// SCIP heuristics/locks/updatelocks).
template <typename f_t>
void fp_active_sides(
  f_t min_act, f_t max_act, f_t row_lb, f_t row_ub, f_t tol, bool& le_active, bool& ge_active)
{
  le_active = std::isfinite(row_ub) && !(max_act <= row_ub + tol);
  ge_active = std::isfinite(row_lb) && !(min_act >= row_lb - tol);
}

template <typename i_t, typename f_t>
struct host_fix_propagate_t {
  // integer rows (rows with at least one integer var) in the original numbering: used by the
  // callers for reduced costs, up-locks and logging, not for propagation
  std::vector<i_t> row_offsets;  // size n_rows+1
  std::vector<i_t> row_cols;
  std::vector<f_t> row_vals;
  std::vector<f_t> row_lb, row_ub;
  std::vector<i_t> row_orig;
  // per integer var: list of (row, coef) in the integer rows (indexed by var id, n+1)
  std::vector<i_t> var_offsets;
  std::vector<i_t> var_rows;
  std::vector<f_t> var_vals;
  std::vector<char> is_int;
  // values of the last pass (integer entries are the fixings)
  std::vector<f_t> col_lb;
  std::vector<char> row_dead;
  f_t abs_tol  = 1e-6;
  f_t int_tol  = 1e-5;
  int64_t work = 0;
  // propagation work budget of one pass (nonzeros processed); exceeding it ends the pass like a
  // conflict, so one pass cannot hold local search for long on models with very dense rows
  int64_t work_limit     = std::numeric_limits<int64_t>::max();
  bool work_limit_hit    = false;
  i_t n_binary_only_rows = 0;
  i_t n_mixed_rows       = 0;
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
  std::vector<f_t> lp_lower0, lp_upper0;
  simplex::domain_t<i_t, f_t> domain;
  i_t n_orig_cols = 0;

  void build(i_t n_vars,
             i_t n_cstrs,
             const std::vector<i_t>& offsets,
             const std::vector<i_t>& cols,
             const std::vector<f_t>& vals,
             const std::vector<f_t>& clb,
             const std::vector<f_t>& cub,
             const std::vector<f_t>&,
             const std::vector<f_t>&,
             const std::vector<i_t>& integer_indices)
  {
    is_int.assign(n_vars, 0);
    for (auto v : integer_indices) {
      is_int[v] = 1;
    }
    row_offsets.assign(1, 0);
    row_cols.clear();
    row_vals.clear();
    row_lb.clear();
    row_ub.clear();
    row_orig.clear();
    n_binary_only_rows = 0;
    n_mixed_rows       = 0;
    for (i_t r = 0; r < n_cstrs; ++r) {
      bool has_int = false, has_cont = false;
      for (i_t k = offsets[r]; k < offsets[r + 1]; ++k) {
        if (is_int[cols[k]]) {
          has_int = true;
        } else {
          has_cont = true;
        }
      }
      if (!has_int) continue;
      if (has_cont) {
        n_mixed_rows++;
      } else {
        n_binary_only_rows++;
      }
      for (i_t k = offsets[r]; k < offsets[r + 1]; ++k) {
        row_cols.push_back(cols[k]);
        row_vals.push_back(vals[k]);
      }
      row_offsets.push_back((i_t)row_cols.size());
      row_lb.push_back(clb[r]);
      row_ub.push_back(cub[r]);
      row_orig.push_back(r);
    }
    const i_t n_rows = (i_t)row_lb.size();
    std::vector<i_t> cnt(n_vars + 1, 0);
    for (i_t r = 0; r < n_rows; ++r) {
      for (i_t k = row_offsets[r]; k < row_offsets[r + 1]; ++k) {
        if (is_int[row_cols[k]]) cnt[row_cols[k] + 1]++;
      }
    }
    for (i_t v = 0; v < n_vars; ++v) {
      cnt[v + 1] += cnt[v];
    }
    var_offsets = cnt;
    var_rows.assign(var_offsets[n_vars], 0);
    var_vals.assign(var_offsets[n_vars], 0);
    std::vector<i_t> pos(var_offsets.begin(), var_offsets.end() - 1);
    for (i_t r = 0; r < n_rows; ++r) {
      for (i_t k = row_offsets[r]; k < row_offsets[r + 1]; ++k) {
        i_t c = row_cols[k];
        if (is_int[c]) {
          var_rows[pos[c]] = r;
          var_vals[pos[c]] = row_vals[k];
          pos[c]++;
        }
      }
    }
    // every column of the integer rows (the repair also shifts continuous domains)
    std::vector<i_t> ccnt(n_vars + 1, 0);
    for (i_t k = 0; k < (i_t)row_cols.size(); ++k) {
      ccnt[row_cols[k] + 1]++;
    }
    for (i_t v = 0; v < n_vars; ++v) {
      ccnt[v + 1] += ccnt[v];
    }
    csc_offsets = ccnt;
    csc_rows.assign(csc_offsets[n_vars], 0);
    csc_vals.assign(csc_offsets[n_vars], 0);
    std::vector<i_t> cpos(csc_offsets.begin(), csc_offsets.end() - 1);
    for (i_t r = 0; r < n_rows; ++r) {
      for (i_t k = row_offsets[r]; k < row_offsets[r + 1]; ++k) {
        const i_t c       = row_cols[k];
        csc_rows[cpos[c]] = r;
        csc_vals[cpos[c]] = row_vals[k];
        cpos[c]++;
      }
    }
  }

  // ---- in-pass repair: [SRF] RepairWalk (Sect. 5, Fig. 4) ----
  // On a conflict the pass propagates the conflicting fixing to the infeasible state and walks on
  // it: pick a violated integer row at random, shift the domain of one of its variables whose
  // domain was reduced (a flip for a fixed binary) so that the row's violation decreases, choosing
  // a zero-damage shift if one exists, else a random candidate with probability repair_noise, else
  // a least-damage one (damage = total increase of violation over the other rows). The best state
  // of the walk is kept; at zero violation the slack bounds are reset and the state propagated
  // again, and the pass goes on. The walk uses the integer rows in their original form.
  bool repair_enabled       = false;
  int repair_max_iterations = 200;   // [SRF]: 200 per call
  double repair_noise       = 0.75;  // [SRF]: p = 0.75
  int repair_max_candidates = 64;    // candidates evaluated per iteration (sampled beyond this)
  unsigned repair_seed      = 1;
  int n_repair_calls        = 0;  // of the last pass
  int n_repair_success      = 0;
  int n_repair_invisible    = 0;  // failures: no integer row violated at the infeasible state,
  int n_repair_unsolved     = 0;  // the walk did not reach zero violation,
  int n_repair_reprop_fail  = 0;  // the repaired state did not propagate
  // walk statistics of the last pass (summed over its calls): violated rows at the start and at the
  // best state, iterations, iterations without a candidate move
  int64_t walk_rows_start = 0, walk_rows_best = 0, walk_iters = 0, walk_no_cand = 0;
  std::vector<i_t> csc_offsets, csc_rows;
  std::vector<f_t> csc_vals;
  struct row_act_t {
    f_t min_fin{0}, max_fin{0};
    i_t min_inf{0}, max_inf{0};
  };
  std::vector<row_act_t> act;
  std::vector<f_t> viol;
  std::vector<i_t> viol_rows, viol_pos;  // violated rows and their positions (-1: not violated)
  f_t total_viol = 0;

  static void add_contribution(row_act_t& a, f_t coef, f_t lo, f_t hi, f_t sign)
  {
    const f_t mn = coef > 0 ? lo : hi;
    const f_t mx = coef > 0 ? hi : lo;
    if (std::isfinite(mn)) {
      a.min_fin += sign * coef * mn;
    } else {
      a.min_inf += sign > 0 ? 1 : -1;
    }
    if (std::isfinite(mx)) {
      a.max_fin += sign * coef * mx;
    } else {
      a.max_inf += sign > 0 ? 1 : -1;
    }
  }
  f_t act_min(i_t r) const
  {
    return act[r].min_inf > 0 ? -std::numeric_limits<f_t>::infinity() : act[r].min_fin;
  }
  f_t act_max(i_t r) const
  {
    return act[r].max_inf > 0 ? std::numeric_limits<f_t>::infinity() : act[r].max_fin;
  }
  f_t row_tol(i_t r) const
  {
    const f_t b = std::isfinite(row_ub[r]) ? row_ub[r] : (std::isfinite(row_lb[r]) ? row_lb[r] : 0);
    return abs_tol * std::max<f_t>(1, std::abs(b));
  }
  f_t violation_of(i_t r, f_t mn, f_t mx) const
  {
    f_t v = 0;
    if (std::isfinite(row_lb[r]) && mx < row_lb[r]) { v = row_lb[r] - mx; }
    if (std::isfinite(row_ub[r]) && mn > row_ub[r]) { v = std::max(v, mn - row_ub[r]); }
    return v > row_tol(r) ? v : 0;
  }
  void set_violation(i_t r, f_t v)
  {
    total_viol += v - viol[r];
    viol[r] = v;
    if (v > 0 && viol_pos[r] < 0) {
      viol_pos[r] = (i_t)viol_rows.size();
      viol_rows.push_back(r);
    } else if (v == 0 && viol_pos[r] >= 0) {
      const i_t last         = viol_rows.back();
      viol_rows[viol_pos[r]] = last;
      viol_pos[last]         = viol_pos[r];
      viol_rows.pop_back();
      viol_pos[r] = -1;
    }
  }
  void compute_all_activities()
  {
    const i_t m = n_rows();
    act.assign(m, row_act_t{});
    viol.assign(m, 0);
    viol_pos.assign(m, -1);
    viol_rows.clear();
    total_viol = 0;
    for (i_t r = 0; r < m; ++r) {
      for (i_t k = row_offsets[r]; k < row_offsets[r + 1]; ++k) {
        const i_t c = row_cols[k];
        add_contribution(act[r], row_vals[k], lp->lower[c], lp->upper[c], 1);
      }
      set_violation(r, violation_of(r, act_min(r), act_max(r)));
    }
    work += (int64_t)row_cols.size();
  }
  // moves column c to [lo, hi], updating the activities and violations of its rows
  void move_column(i_t c, f_t lo, f_t hi)
  {
    for (i_t k = csc_offsets[c]; k < csc_offsets[c + 1]; ++k) {
      const i_t r = csc_rows[k];
      add_contribution(act[r], csc_vals[k], lp->lower[c], lp->upper[c], -1);
      add_contribution(act[r], csc_vals[k], lo, hi, 1);
      set_violation(r, violation_of(r, act_min(r), act_max(r)));
    }
    work += csc_offsets[c + 1] - csc_offsets[c];
    lp->lower[c] = lo;
    lp->upper[c] = hi;
  }
  // total violation increase over the rows of c (except row i) if c moves by shift (both bounds)
  f_t shift_damage(i_t c, f_t shift, i_t i)
  {
    f_t damage = 0;
    for (i_t k = csc_offsets[c]; k < csc_offsets[c + 1]; ++k) {
      const i_t r = csc_rows[k];
      if (r == i) { continue; }
      const f_t d = csc_vals[k] * shift;
      damage += std::max<f_t>(0, violation_of(r, act_min(r) + d, act_max(r) + d) - viol[r]);
    }
    work += csc_offsets[c + 1] - csc_offsets[c];
    return damage;
  }

  // RepairWalk on the current (infeasible) bounds; true if it reaches zero violation, the bounds
  // then holding the repaired state (the best state of the walk is restored otherwise)
  bool repair_walk(std::mt19937& g)
  {
    compute_all_activities();
    if (viol_rows.empty()) {  // infeasibility not visible in the integer rows
      n_repair_invisible++;
      return false;
    }
    std::uniform_real_distribution<double> U(0., 1.);
    walk_rows_start += (int64_t)viol_rows.size();
    size_t best_rows = viol_rows.size();
    struct move_t {
      i_t c;
      f_t lo, hi;  // bounds before the move
    };
    std::vector<move_t> moves;
    f_t best_total  = total_viol;
    size_t best_len = 0;
    struct cand_t {
      i_t c;
      f_t shift, damage;
    };
    std::vector<cand_t> cands;
    std::vector<i_t> ks;
    for (int it = 0; it < repair_max_iterations && !viol_rows.empty() && work <= work_limit; ++it) {
      walk_iters++;
      const i_t i  = viol_rows[std::uniform_int_distribution<size_t>(0, viol_rows.size() - 1)(g)];
      const f_t mn = act_min(i), mx = act_max(i);
      const bool too_high = std::isfinite(row_ub[i]) && mn > row_ub[i];
      ks.clear();
      for (i_t k = row_offsets[i]; k < row_offsets[i + 1]; ++k) {
        ks.push_back(k);
      }
      if ((int)ks.size() > repair_max_candidates) {
        std::shuffle(ks.begin(), ks.end(), g);
        ks.resize(repair_max_candidates);
      }
      cands.clear();
      f_t best_damage = std::numeric_limits<f_t>::infinity();
      for (i_t k : ks) {
        const i_t c = row_cols[k];
        const f_t s = fp_repair_shift<f_t>(row_vals[k],
                                           too_high,
                                           too_high ? row_ub[i] - mn : row_lb[i] - mx,
                                           lp->lower[c],
                                           lp->upper[c],
                                           lp_lower0[c],
                                           lp_upper0[c],
                                           is_int[c] != 0,
                                           int_tol);
        if (s == 0) { continue; }
        const f_t d = row_vals[k] * s;
        if (!(violation_of(i, mn + d, mx + d) < viol[i])) { continue; }
        const f_t damage = shift_damage(c, s, i);
        cands.push_back({c, s, damage});
        best_damage = std::min(best_damage, damage);
      }
      if (cands.empty()) {
        walk_no_cand++;
        continue;
      }
      // [SRF] Fig. 4: a zero-damage move at random, else noise: random candidate, else least damage
      std::vector<size_t> pool;
      if (best_damage <= 0) {
        for (size_t q = 0; q < cands.size(); ++q) {
          if (cands[q].damage <= 0) { pool.push_back(q); }
        }
      } else if (U(g) < repair_noise) {
        for (size_t q = 0; q < cands.size(); ++q) {
          pool.push_back(q);
        }
      } else {
        for (size_t q = 0; q < cands.size(); ++q) {
          if (cands[q].damage <= best_damage) { pool.push_back(q); }
        }
      }
      const auto& m = cands[pool[std::uniform_int_distribution<size_t>(0, pool.size() - 1)(g)]];
      moves.push_back({m.c, lp->lower[m.c], lp->upper[m.c]});
      move_column(m.c, lp->lower[m.c] + m.shift, lp->upper[m.c] + m.shift);
      if (total_viol < best_total) {
        best_total = total_viol;
        best_len   = moves.size();
        best_rows  = viol_rows.size();
      }
    }
    while (moves.size() > best_len) {
      const auto mv = moves.back();
      moves.pop_back();
      move_column(mv.c, mv.lo, mv.hi);
    }
    walk_rows_best += (int64_t)best_rows;
    if (!viol_rows.empty()) { n_repair_unsolved++; }
    return viol_rows.empty();
  }

  // Builds the propagation domain from the problem (same conversion as branch and bound).
  void build_domain(const simplex::user_problem_t<i_t, f_t>& user_problem,
                    const clique_table_t<i_t, f_t>* clique_table)
  {
    settings.log.log     = false;
    settings.barrier     = false;
    settings.dualize     = 0;
    settings.primal_tol  = abs_tol;
    settings.integer_tol = int_tol;
    lp = std::make_unique<simplex::lp_problem_t<i_t, f_t>>(user_problem.handle_ptr, 1, 1, 1);
    std::vector<i_t> new_slacks;
    simplex::dualize_info_t<i_t, f_t> dualize_info;
    simplex::convert_user_problem(user_problem, settings, *lp, new_slacks, dualize_info);
    n_orig_cols = user_problem.num_cols;
    var_types   = user_problem.var_types;
    var_types.resize(lp->num_cols, simplex::variable_type_t::CONTINUOUS);
    lp->A.to_compressed_row(Arow);
    lp_lower0 = lp->lower;
    lp_upper0 = lp->upper;
    simplex::domain_params params;
    params.min_relative_improvement = continuous_min_improvement;
    domain.set_params(params);
    domain.implied_bounds = nullptr;
    domain.clique_table   = clique_table;
  }

  i_t n_rows() const { return (i_t)row_lb.size(); }

  // fix var v to val and propagate; false (state restored) on conflict
  bool try_fix(i_t v, f_t val)
  {
    simplex::bound_change_t<i_t, f_t> change{.var       = v,
                                             .new_upper = val,
                                             .new_lower = val,
                                             .origin    = simplex::bound_change_origin_t::BRANCH};
    const bool ok = domain.apply_and_propagate(Arow, var_types, settings, change, *lp);
    work += (int64_t)domain.last_nnz_processed;
    if (ok) {
      for (size_t k = 0; k < domain.size(); ++k) {
        const i_t j = domain[k].var;
        if (domain[k].origin == simplex::bound_change_origin_t::BOUND_PROPAGATION &&
            j < n_orig_cols && var_types[j] == simplex::variable_type_t::CONTINUOUS) {
          n_continuous_tightenings++;
        }
      }
      domain.clear();  // never backtracked past an accepted fixing
      return true;
    }
    domain.backtrack_to_parent(*lp);
    return false;
  }

  // up- and down-locks of integer var v over the rows that can still be violated under the current
  // bounds (fp_active_sides); the scan counts toward the pass's work
  void dynamic_locks(i_t v, i_t& uplocks, i_t& downlocks)
  {
    uplocks = downlocks = 0;
    const f_t tol       = abs_tol;
    for (i_t kk = var_offsets[v]; kk < var_offsets[v + 1]; ++kk) {
      const i_t r = var_rows[kk];
      f_t min_act = 0, max_act = 0;
      for (i_t k = row_offsets[r]; k < row_offsets[r + 1]; ++k) {
        const i_t c = row_cols[k];
        const f_t a = row_vals[k];
        min_act += a > 0 ? a * lp->lower[c] : a * lp->upper[c];
        max_act += a > 0 ? a * lp->upper[c] : a * lp->lower[c];
      }
      work += row_offsets[r + 1] - row_offsets[r];
      bool le_active, ge_active;
      fp_active_sides<f_t>(min_act, max_act, row_lb[r], row_ub[r], tol, le_active, ge_active);
      const f_t a = var_vals[kk];
      if ((a > 0 && le_active) || (a < 0 && ge_active)) { uplocks++; }
      if ((a > 0 && ge_active) || (a < 0 && le_active)) { downlocks++; }
    }
  }

  // fix-and-propagate pass: order = integer vars in processing order, pref = preferred value per
  // var (indexed by var id). One-level backtrack: on a conflict the other end of the domain is
  // tried; if both conflict, the remaining integers get their preferred values without propagation
  // (the pattern is reported as conflicting), as when the pass exceeds work_limit. Returns the
  // number of conflicts; values in col_lb.
  // dyn (optional, by var id): variables whose value is decided at fixing time by their dynamic
  // locks: dyn_up when they have fewer up-locks, dyn_down when fewer down-locks, pref on a tie.
  i_t fix_and_propagate(const std::vector<i_t>& order,
                        const std::vector<f_t>& pref,
                        i_t& n_flipped,
                        i_t& n_implied,
                        const std::vector<char>* dyn     = nullptr,
                        const std::vector<f_t>* dyn_up   = nullptr,
                        const std::vector<f_t>* dyn_down = nullptr)
  {
    lp->lower = lp_lower0;
    lp->upper = lp_upper0;
    domain.clear();
    work                     = 0;
    n_continuous_tightenings = 0;
    row_dead.assign(n_rows(), 0);
    n_flipped            = 0;
    n_implied            = 0;
    work_limit_hit       = false;
    n_repair_calls       = 0;
    n_repair_success     = 0;
    n_repair_invisible   = 0;
    n_repair_unsolved    = 0;
    n_repair_reprop_fail = 0;
    walk_rows_start = walk_rows_best = walk_iters = walk_no_cand = 0;
    i_t n_conflicts                                              = 0;
    if (!domain.propagate_full(Arow, var_types, settings, *lp)) { n_conflicts++; }
    work += (int64_t)domain.last_nnz_processed;
    domain.clear();
    bool propagate = n_conflicts == 0;
    for (auto v : order) {
      const f_t lb = lp->lower[v], ub = lp->upper[v];
      if (lb == ub) {
        if (lb != pref[v]) n_implied++;
        continue;
      }
      f_t pv = pref[v];
      if (propagate && dyn != nullptr && (*dyn)[v]) {
        i_t up = 0, down = 0;
        dynamic_locks(v, up, down);
        if (up < down) {
          pv = (*dyn_up)[v];
        } else if (down < up) {
          pv = (*dyn_down)[v];
        }
      }
      const f_t p = std::min(std::max(pv, lb), ub);
      if (propagate && work > work_limit) {
        work_limit_hit = true;
        n_conflicts++;
        propagate = false;
      }
      if (!propagate) {
        lp->lower[v] = lp->upper[v] = p;
        continue;
      }
      if (try_fix(v, p)) { continue; }
      const f_t q = (p == lb) ? ub : lb;
      if (try_fix(v, q)) {
        n_flipped++;
        continue;
      }
      if (repair_enabled && repair_conflict(v, p)) { continue; }
      n_conflicts++;
      propagate    = false;
      lp->lower[v] = lp->upper[v] = p;
    }
    col_lb.assign(lp->lower.begin(), lp->lower.begin() + n_orig_cols);
    return n_conflicts;
  }

  // fixes v to p, propagates to the infeasible state and repairs it (repair_walk); true when the
  // repaired state propagates again, the pass then continuing from it
  bool repair_conflict(i_t v, f_t p)
  {
    n_repair_calls++;
    lp->lower[v] = lp->upper[v] = p;
    domain.clear();
    bool ok = domain.propagate_full(Arow, var_types, settings, *lp);
    work += (int64_t)domain.last_nnz_processed;
    domain.clear();
    if (ok) {
      n_repair_success++;
      return true;
    }
    std::mt19937 g(repair_seed + (unsigned)n_repair_calls);
    if (!repair_walk(g)) { return false; }
    // slack bounds were tightened from the old activities: let propagation derive them again
    for (size_t c = n_orig_cols; c < lp->lower.size(); ++c) {
      lp->lower[c] = lp_lower0[c];
      lp->upper[c] = lp_upper0[c];
    }
    ok = domain.propagate_full(Arow, var_types, settings, *lp);
    work += (int64_t)domain.last_nnz_processed;
    domain.clear();
    n_repair_success += ok;
    n_repair_reprop_fail += !ok;
    return ok;
  }
};

}  // namespace cuopt::mathematical_optimization::mip
