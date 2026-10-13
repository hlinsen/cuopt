/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// Host-side fix-and-propagate over the rows that touch integer variables.
// Integer (binary) variables are fixed one at a time in a caller-given order and value
// preference; after each fixing, activity-based bound propagation runs over the integer rows
// only (continuous variables keep their static bounds and only contribute activity ranges).
// A conflict triggers a one-level backtrack (try the other value); if both values conflict the
// preferred value is kept and the violated rows are marked dead (reported as conflicts).
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
// Not implemented here: the LP-based fixing of [KK] (round up with probability equal to the
// fractional part), its Dual and Type variable strategies, its integer-aware three-way branching,
// and the clique/implication propagators and repair of [SRF].
// By default continuous bounds are tightened too (for conflict detection only, never handed to an
// LP), with a minimum relative improvement as in HiGHS; with that switched off, propagation is
// restricted to rows with an integer and chains through continuous variables are only caught by
// the completion LP.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

// Variable strategy: integers with a larger key are fixed first.
// The integer values are the settings values (mip_hyper_heuristic_fix_propagate_*_strategy).
enum class fp_variable_strategy_t {
  FRACTIONALITY       = 0,  // closest to integral first; same measure as [KK] Frac
  LP_VALUE_DESC       = 1,  // largest reference value first; not in [KK]
  LP_VALUE_DESC_NOISE = 2,  // reference value + noise * U(0,1), largest first; not in [KK]
  REDUCED_COST        = 3,  // most negative reduced cost first; [KK] RedCost orders by |red. cost|
  UPLOCKS = 4,  // fewest up-locks first, ties by larger reference value; cf. [SRF] locks
};

// Value strategy: the value an integer is fixed to (fix_and_propagate clamps it to the current
// domain).
enum class fp_value_strategy_t {
  UPPER_BOUND              = 0,  // LP-free
  LOWER_BOUND              = 1,  // LP-free
  ROUND_UP                 = 2,  // round any fractional reference value up
  ROUND_UP_OR_NEG_REDCOST  = 3,  // round up if fractional or reduced cost < -param
  ROUND_UP_ABOVE_THRESHOLD = 4,  // round up if the fractional part is > param
};

template <typename i_t, typename f_t>
f_t fp_order_key(fp_variable_strategy_t s, f_t x, f_t red, i_t uplocks, f_t noise)
{
  switch (s) {
    case fp_variable_strategy_t::FRACTIONALITY: return std::abs(x - std::floor(x) - f_t(0.5));
    case fp_variable_strategy_t::LP_VALUE_DESC: return x;
    case fp_variable_strategy_t::LP_VALUE_DESC_NOISE: return x + noise;
    case fp_variable_strategy_t::REDUCED_COST: return -red;
    case fp_variable_strategy_t::UPLOCKS: return -(f_t)uplocks + f_t(1e-3) * x;
  }
  return x;
}

template <typename f_t>
f_t fp_preferred_value(
  fp_value_strategy_t s, f_t x, f_t lb, f_t ub, f_t red, f_t param, f_t int_tol)
{
  const f_t fl = std::floor(x + int_tol);
  const f_t fr = x - fl;
  switch (s) {
    case fp_value_strategy_t::UPPER_BOUND: return ub;
    case fp_value_strategy_t::LOWER_BOUND: return lb;
    case fp_value_strategy_t::ROUND_UP: return fr > int_tol ? fl + 1 : fl;
    case fp_value_strategy_t::ROUND_UP_OR_NEG_REDCOST:
      return (fr > int_tol || red < -param) ? std::min(fl + 1, ub) : fl;
    case fp_value_strategy_t::ROUND_UP_ABOVE_THRESHOLD: return fr > param ? fl + 1 : fl;
  }
  return fl;
}

template <typename i_t, typename f_t>
struct host_fix_propagate_t {
  // selected rows: rows with at least one integer var, or all rows when continuous bounds are
  // propagated too (chains through continuous-only rows)
  std::vector<i_t> row_offsets;  // size n_rows+1
  std::vector<i_t> row_cols;
  std::vector<f_t> row_vals;
  std::vector<f_t> row_lb, row_ub;
  std::vector<i_t> row_orig;
  // per var: list of (row, coef) in selected rows; integer vars only unless continuous bounds are
  // propagated (indexed by var id in the original numbering, n+1)
  std::vector<i_t> var_offsets;
  std::vector<i_t> var_rows;
  std::vector<f_t> var_vals;
  std::vector<char> is_int;
  // working domains: integer bounds are the fixings; continuous bounds are only tightened for
  // conflict detection and never handed to an LP
  std::vector<f_t> col_lb, col_ub;
  std::vector<f_t> orig_lb, orig_ub;
  // activities
  std::vector<f_t> min_act, max_act;
  std::vector<i_t> min_inf, max_inf;
  std::vector<char> row_dead;
  // trail of bound changes: (var, old_lb, old_ub)
  struct trail_entry_t {
    i_t var;
    f_t lb, ub;
  };
  std::vector<trail_entry_t> trail;
  std::vector<i_t> queue;
  std::vector<char> in_queue;
  f_t abs_tol            = 1e-6;
  f_t rel_tol            = 1e-9;
  int64_t work           = 0;
  i_t n_binary_only_rows = 0;
  i_t n_mixed_rows       = 0;
  // continuous bound propagation (set before build): a tightened continuous bound is kept only if
  // it is tighter by more than 1000 * abs_tol and shrinks the domain by at least
  // continuous_min_improvement of its width (of the bound magnitude if one side is infinite);
  // infinite bounds becoming finite are always kept. Same acceptance rule as HiGHS (0.3).
  bool propagate_continuous        = false;
  f_t continuous_min_improvement   = 0.3;
  int64_t n_continuous_tightenings = 0;

  void build(i_t n_vars,
             i_t n_cstrs,
             const std::vector<i_t>& offsets,
             const std::vector<i_t>& cols,
             const std::vector<f_t>& vals,
             const std::vector<f_t>& clb,
             const std::vector<f_t>& cub,
             const std::vector<f_t>& vlb,
             const std::vector<f_t>& vub,
             const std::vector<i_t>& integer_indices)
  {
    is_int.assign(n_vars, 0);
    for (auto v : integer_indices) {
      is_int[v] = 1;
    }
    orig_lb = vlb;
    orig_ub = vub;
    row_offsets.clear();
    row_offsets.push_back(0);
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
      if (!has_int && !propagate_continuous) continue;
      if (has_int && has_cont) {
        n_mixed_rows++;
      } else if (has_int) {
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
    auto tracked     = [&](i_t c) { return propagate_continuous || is_int[c]; };
    std::vector<i_t> cnt(n_vars + 1, 0);
    for (i_t r = 0; r < n_rows; ++r) {
      for (i_t k = row_offsets[r]; k < row_offsets[r + 1]; ++k) {
        if (tracked(row_cols[k])) cnt[row_cols[k] + 1]++;
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
        if (tracked(c)) {
          var_rows[pos[c]] = r;
          var_vals[pos[c]] = row_vals[k];
          pos[c]++;
        }
      }
    }
  }

  i_t n_rows() const { return (i_t)row_lb.size(); }

  void contrib(f_t a, f_t lb, f_t ub, f_t& mn, f_t& mx, i_t& mn_inf, i_t& mx_inf) const
  {
    if (a > 0) {
      if (std::isfinite(lb)) {
        mn += a * lb;
      } else {
        mn_inf++;
      }
      if (std::isfinite(ub)) {
        mx += a * ub;
      } else {
        mx_inf++;
      }
    } else {
      if (std::isfinite(ub)) {
        mn += a * ub;
      } else {
        mn_inf++;
      }
      if (std::isfinite(lb)) {
        mx += a * lb;
      } else {
        mx_inf++;
      }
    }
  }

  // move the contribution of a coefficient a in row r from bounds [olb, oub] to [nlb, nub]
  void update_activity(i_t r, f_t a, f_t olb, f_t oub, f_t nlb, f_t nub)
  {
    if (std::isfinite(olb) && std::isfinite(oub) && std::isfinite(nlb) && std::isfinite(nub)) {
      if (a > 0) {
        min_act[r] += a * (nlb - olb);
        max_act[r] += a * (nub - oub);
      } else {
        min_act[r] += a * (nub - oub);
        max_act[r] += a * (nlb - olb);
      }
      return;
    }
    f_t omn = 0, omx = 0, nmn = 0, nmx = 0;
    i_t omni = 0, omxi = 0, nmni = 0, nmxi = 0;
    contrib(a, olb, oub, omn, omx, omni, omxi);
    contrib(a, nlb, nub, nmn, nmx, nmni, nmxi);
    min_act[r] += nmn - omn;
    max_act[r] += nmx - omx;
    min_inf[r] += nmni - omni;
    max_inf[r] += nmxi - omxi;
  }

  void reset_domains()
  {
    col_lb      = orig_lb;
    col_ub      = orig_ub;
    const i_t m = n_rows();
    min_act.assign(m, 0);
    max_act.assign(m, 0);
    min_inf.assign(m, 0);
    max_inf.assign(m, 0);
    row_dead.assign(m, 0);
    in_queue.assign(m, 0);
    trail.clear();
    queue.clear();
    n_continuous_tightenings = 0;
    for (i_t r = 0; r < m; ++r) {
      recompute_row(r);
    }
  }

  void recompute_row(i_t r)
  {
    f_t mn = 0, mx = 0;
    i_t mni = 0, mxi = 0;
    for (i_t k = row_offsets[r]; k < row_offsets[r + 1]; ++k) {
      i_t c = row_cols[k];
      contrib(row_vals[k], col_lb[c], col_ub[c], mn, mx, mni, mxi);
    }
    min_act[r] = mn;
    max_act[r] = mx;
    min_inf[r] = mni;
    max_inf[r] = mxi;
  }

  f_t tol(f_t b) const { return abs_tol + rel_tol * std::max<f_t>(1., std::abs(b)); }

  bool row_infeasible(i_t r) const
  {
    if (min_inf[r] == 0 && min_act[r] > row_ub[r] + tol(row_ub[r])) return true;
    if (max_inf[r] == 0 && max_act[r] < row_lb[r] - tol(row_lb[r])) return true;
    return false;
  }

  // change the domain of var v to [lb, ub] (subset of current); update activities and queue rows
  void set_domain(i_t v, f_t lb, f_t ub)
  {
    trail.push_back({v, col_lb[v], col_ub[v]});
    f_t olb = col_lb[v], oub = col_ub[v];
    col_lb[v] = lb;
    col_ub[v] = ub;
    for (i_t k = var_offsets[v]; k < var_offsets[v + 1]; ++k) {
      i_t r = var_rows[k];
      update_activity(r, var_vals[k], olb, oub, lb, ub);
      if (!in_queue[r]) {
        in_queue[r] = 1;
        queue.push_back(r);
      }
    }
  }

  void undo_to(size_t trail_size)
  {
    while (trail.size() > trail_size) {
      auto e = trail.back();
      trail.pop_back();
      i_t v  = e.var;
      f_t lb = col_lb[v], ub = col_ub[v];
      col_lb[v] = e.lb;
      col_ub[v] = e.ub;
      for (i_t k = var_offsets[v]; k < var_offsets[v + 1]; ++k) {
        update_activity(var_rows[k], var_vals[k], lb, ub, e.lb, e.ub);
      }
    }
    for (auto r : queue) {
      in_queue[r] = 0;
    }
    queue.clear();
  }

  // HiGHS-style acceptance of a tightened continuous bound (new upper bound nub over [lb, ub])
  bool accept_continuous_ub(f_t lb, f_t ub, f_t nub) const
  {
    if (!std::isfinite(nub) || std::abs(nub) > 1e15) return false;
    if (!std::isfinite(ub)) return true;
    if (!(nub + 1000. * abs_tol < ub)) return false;
    const f_t denom = std::isfinite(lb) ? ub - lb : std::max(std::abs(ub), std::abs(nub));
    return denom > 0 && (ub - nub) >= continuous_min_improvement * denom;
  }
  bool accept_continuous_lb(f_t lb, f_t ub, f_t nlb) const
  {
    if (!std::isfinite(nlb) || std::abs(nlb) > 1e15) return false;
    if (!std::isfinite(lb)) return true;
    if (!(nlb - 1000. * abs_tol > lb)) return false;
    const f_t denom = std::isfinite(ub) ? ub - lb : std::max(std::abs(lb), std::abs(nlb));
    return denom > 0 && (nlb - lb) >= continuous_min_improvement * denom;
  }

  // propagate queued rows; returns false on conflict (a live row became infeasible)
  bool propagate()
  {
    size_t head        = 0;
    auto clear_pending = [&]() {
      for (size_t h = head; h < queue.size(); ++h) {
        in_queue[queue[h]] = 0;
      }
      queue.clear();
    };
    while (head < queue.size()) {
      i_t r       = queue[head++];
      in_queue[r] = 0;
      if (row_dead[r]) continue;
      work += row_offsets[r + 1] - row_offsets[r];
      if (row_infeasible(r)) { recompute_row(r); }
      if (row_infeasible(r)) {
        clear_pending();
        return false;
      }
      // implications on the unfixed vars of the row (continuous ones only if enabled)
      const bool ub_finite = std::isfinite(row_ub[r]) && min_inf[r] == 0;
      const bool lb_finite = std::isfinite(row_lb[r]) && max_inf[r] == 0;
      if (!ub_finite && !lb_finite) continue;
      const f_t slack_ub = ub_finite ? row_ub[r] + tol(row_ub[r]) - min_act[r] : 0;
      const f_t slack_lb = lb_finite ? max_act[r] - (row_lb[r] - tol(row_lb[r])) : 0;
      for (i_t k = row_offsets[r]; k < row_offsets[r + 1]; ++k) {
        i_t c  = row_cols[k];
        f_t lb = col_lb[c], ub = col_ub[c];
        if (lb == ub) continue;
        const f_t a = row_vals[k];
        if (is_int[c]) {
          f_t range = std::abs(a) * (ub - lb);
          f_t nlb = lb, nub = ub;
          // moving c from its min-activity value by its range must not exceed the slack
          if (ub_finite && range > slack_ub) {
            // cannot take the "max-contribution" end
            f_t steps = std::floor(slack_ub / std::abs(a) + 1e-9);
            if (a > 0) {
              nub = std::min(nub, lb + steps);
            } else {
              nlb = std::max(nlb, ub - steps);
            }
          }
          if (lb_finite && range > slack_lb) {
            f_t steps = std::floor(slack_lb / std::abs(a) + 1e-9);
            if (a > 0) {
              nlb = std::max(nlb, ub - steps);
            } else {
              nub = std::min(nub, lb + steps);
            }
          }
          if (nlb > nub) {
            clear_pending();
            return false;
          }
          if (nlb != lb || nub != ub) { set_domain(c, nlb, nub); }
          continue;
        }
        if (!propagate_continuous) continue;
        // continuous var: real-valued implied bounds (its own min/max contribution is finite
        // because the row has no infinite contributions on that side)
        f_t nlb = lb, nub = ub;
        if (ub_finite) {
          if (a > 0) {
            const f_t cand = lb + slack_ub / a;
            if (accept_continuous_ub(lb, ub, cand)) nub = cand;
          } else {
            const f_t cand = ub - slack_ub / -a;
            if (accept_continuous_lb(lb, ub, cand)) nlb = cand;
          }
        }
        if (lb_finite) {
          if (a > 0) {
            const f_t cand = ub - slack_lb / a;
            if (accept_continuous_lb(nlb, nub, cand)) nlb = cand;
          } else {
            const f_t cand = lb + slack_lb / -a;
            if (accept_continuous_ub(nlb, nub, cand)) nub = cand;
          }
        }
        if (nlb > nub) {
          if (nlb - nub > tol(std::max(std::abs(nlb), std::abs(nub)))) {
            clear_pending();
            return false;
          }
          nlb = nub = 0.5 * (nlb + nub);
        }
        if (nlb != lb || nub != ub) {
          n_continuous_tightenings++;
          set_domain(c, nlb, nub);
        }
      }
    }
    queue.clear();
    return true;
  }

  // fix var v to val and propagate; on conflict undo and return false
  bool try_fix(i_t v, f_t val)
  {
    if (val < col_lb[v] || val > col_ub[v]) return false;
    size_t mark = trail.size();
    set_domain(v, val, val);
    if (propagate()) return true;
    undo_to(mark);
    return false;
  }

  // force var v to val, marking rows that become infeasible dead
  void force_fix(i_t v, f_t val)
  {
    set_domain(v, val, val);
    // mark directly violated rows dead, then propagate the rest
    for (auto r : queue) {
      if (row_infeasible(r)) row_dead[r] = 1;
    }
    while (!propagate()) {
      // a conflict surfaced later in the propagation; mark all currently infeasible rows dead
      for (i_t k = var_offsets[v]; k < var_offsets[v + 1]; ++k) {
        i_t r = var_rows[k];
        if (row_infeasible(r)) row_dead[r] = 1;
      }
      bool any = false;
      for (i_t r = 0; r < n_rows(); ++r) {
        if (!row_dead[r] && row_infeasible(r)) {
          row_dead[r] = 1;
          any         = true;
        }
      }
      if (!any) break;
    }
  }

  // fix-and-propagate pass: order = integer vars in processing order, pref = preferred value per
  // var (indexed by var id). Returns number of forced conflicts; out values in col_lb.
  i_t fix_and_propagate(const std::vector<i_t>& order,
                        const std::vector<f_t>& pref,
                        i_t& n_flipped,
                        i_t& n_implied)
  {
    reset_domains();
    // initial propagation over all rows
    for (i_t r = 0; r < n_rows(); ++r) {
      in_queue[r] = 1;
      queue.push_back(r);
    }
    i_t n_conflicts = 0;
    if (!propagate()) {
      n_conflicts++;
      for (i_t r = 0; r < n_rows(); ++r) {
        if (row_infeasible(r)) row_dead[r] = 1;
      }
      for (i_t r = 0; r < n_rows(); ++r) {
        in_queue[r] = 1;
        queue.push_back(r);
      }
      propagate();
    }
    trail.clear();
    n_flipped = 0;
    n_implied = 0;
    for (auto v : order) {
      if (col_lb[v] == col_ub[v]) {
        if (col_lb[v] != pref[v]) n_implied++;
        continue;
      }
      f_t p = std::min(std::max(pref[v], col_lb[v]), col_ub[v]);
      if (try_fix(v, p)) {
        trail.clear();
        continue;
      }
      f_t q = (p == col_lb[v]) ? col_ub[v] : col_lb[v];
      if (try_fix(v, q)) {
        n_flipped++;
        trail.clear();
        continue;
      }
      n_conflicts++;
      force_fix(v, p);
      trail.clear();
    }
    return n_conflicts;
  }
};

}  // namespace cuopt::mathematical_optimization::mip
