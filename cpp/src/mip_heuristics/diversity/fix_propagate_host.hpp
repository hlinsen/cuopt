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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
struct host_fix_propagate_t {
  // selected rows (rows with at least one integer var)
  std::vector<i_t> row_offsets;  // size n_rows+1
  std::vector<i_t> row_cols;
  std::vector<f_t> row_vals;
  std::vector<f_t> row_lb, row_ub;
  std::vector<i_t> row_orig;
  // per integer var: list of (row, coef) in selected rows
  std::vector<i_t> int_offsets;  // indexed by var id in the original numbering (n+1)
  std::vector<i_t> int_rows;
  std::vector<f_t> int_vals;
  std::vector<char> is_int;
  std::vector<f_t> col_lb, col_ub;  // current domains (integers change, continuous static)
  std::vector<f_t> orig_lb, orig_ub;
  // activities
  std::vector<f_t> min_act, max_act;
  std::vector<i_t> min_inf, max_inf;
  std::vector<char> row_dead;
  // trail of fixed integer vars: (var, old_lb, old_ub)
  struct trail_entry_t {
    i_t var;
    f_t lb, ub;
  };
  std::vector<trail_entry_t> trail;
  std::vector<i_t> queue;
  std::vector<char> in_queue;
  f_t abs_tol = 1e-6;
  f_t rel_tol = 1e-9;
  int64_t work = 0;
  i_t n_binary_only_rows = 0;
  i_t n_mixed_rows       = 0;

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
    int_offsets = cnt;
    int_rows.assign(int_offsets[n_vars], 0);
    int_vals.assign(int_offsets[n_vars], 0);
    std::vector<i_t> pos(int_offsets.begin(), int_offsets.end() - 1);
    for (i_t r = 0; r < n_rows; ++r) {
      for (i_t k = row_offsets[r]; k < row_offsets[r + 1]; ++k) {
        i_t c = row_cols[k];
        if (is_int[c]) {
          int_rows[pos[c]] = r;
          int_vals[pos[c]] = row_vals[k];
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

  void reset_domains()
  {
    col_lb = orig_lb;
    col_ub = orig_ub;
    const i_t m = n_rows();
    min_act.assign(m, 0);
    max_act.assign(m, 0);
    min_inf.assign(m, 0);
    max_inf.assign(m, 0);
    row_dead.assign(m, 0);
    in_queue.assign(m, 0);
    trail.clear();
    queue.clear();
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

  // change the domain of integer var v to [lb, ub] (subset of current); update activities
  void set_domain(i_t v, f_t lb, f_t ub)
  {
    trail.push_back({v, col_lb[v], col_ub[v]});
    f_t olb = col_lb[v], oub = col_ub[v];
    col_lb[v] = lb;
    col_ub[v] = ub;
    for (i_t k = int_offsets[v]; k < int_offsets[v + 1]; ++k) {
      i_t r = int_rows[k];
      f_t a = int_vals[k];
      // integer domains are always finite
      if (a > 0) {
        min_act[r] += a * (lb - olb);
        max_act[r] += a * (ub - oub);
      } else {
        min_act[r] += a * (ub - oub);
        max_act[r] += a * (lb - olb);
      }
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
      for (i_t k = int_offsets[v]; k < int_offsets[v + 1]; ++k) {
        i_t r = int_rows[k];
        f_t a = int_vals[k];
        if (a > 0) {
          min_act[r] += a * (e.lb - lb);
          max_act[r] += a * (e.ub - ub);
        } else {
          min_act[r] += a * (e.ub - ub);
          max_act[r] += a * (e.lb - lb);
        }
      }
    }
    for (auto r : queue) {
      in_queue[r] = 0;
    }
    queue.clear();
  }

  // propagate queued rows; returns false on conflict (a live row became infeasible)
  bool propagate()
  {
    size_t head = 0;
    while (head < queue.size()) {
      i_t r = queue[head++];
      in_queue[r] = 0;
      if (row_dead[r]) continue;
      work += row_offsets[r + 1] - row_offsets[r];
      if (row_infeasible(r)) { recompute_row(r); }
      if (row_infeasible(r)) {
        for (size_t h = head; h < queue.size(); ++h) {
          in_queue[queue[h]] = 0;
        }
        queue.clear();
        return false;
      }
      // implications on unfixed integer vars of the row
      const bool ub_finite = std::isfinite(row_ub[r]) && min_inf[r] == 0;
      const bool lb_finite = std::isfinite(row_lb[r]) && max_inf[r] == 0;
      if (!ub_finite && !lb_finite) continue;
      const f_t slack_ub = ub_finite ? row_ub[r] + tol(row_ub[r]) - min_act[r] : 0;
      const f_t slack_lb = lb_finite ? max_act[r] - (row_lb[r] - tol(row_lb[r])) : 0;
      for (i_t k = row_offsets[r]; k < row_offsets[r + 1]; ++k) {
        i_t c = row_cols[k];
        if (!is_int[c]) continue;
        f_t lb = col_lb[c], ub = col_ub[c];
        if (lb == ub) continue;
        f_t a     = row_vals[k];
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
          for (size_t h = head; h < queue.size(); ++h) {
            in_queue[queue[h]] = 0;
          }
          queue.clear();
          return false;
        }
        if (nlb != lb || nub != ub) { set_domain(c, nlb, nub); }
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
      for (i_t k = int_offsets[v]; k < int_offsets[v + 1]; ++k) {
        i_t r = int_rows[k];
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

  // run the dive: order = integer vars in processing order, pref = preferred value per var
  // (indexed by var id). Returns number of forced conflicts; out values in col_lb.
  i_t dive(const std::vector<i_t>& order,
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
