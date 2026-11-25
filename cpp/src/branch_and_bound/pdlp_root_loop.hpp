/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// Helpers for the PDLP root loop: a B&B-side task that, starting from the first GPU root LP point,
// repeatedly separates basis-free cuts, re-solves the tightened LP with PDLP, and launches a RENS
// sub-MIP on every point. The goal is earlier incumbents (primal integral), not the root bound.

#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>

#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>

#include <linear_algebra/sparse_matrix.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

// Experimental knobs, read from the environment so MIPLIB sweeps need no rebuild.
struct pdlp_root_loop_params_t {
  bool enabled{true};            // CUOPT_PDLP_RC
  int max_passes{3};             // CUOPT_PDLP_RC_PASSES: PDLP re-solves after the first point
  int iteration_limit{5000};     // CUOPT_PDLP_RC_ITERS: per re-solve
  double pass_time_limit{10.0};  // CUOPT_PDLP_RC_PASS_TIME: seconds per re-solve
  // CUOPT_PDLP_RC_ROOT_SETTINGS: 1 = re-solve like the root LP of the heuristics thread
  // (Stable2, absolute primal/dual tolerance, tiny relative tolerance, per-constraint residuals);
  // 0 = Stable3 with one tolerance for everything
  bool root_settings{true};
  double tolerance{1e-6};       // CUOPT_PDLP_RC_TOL: absolute primal/dual tolerance
  double rel_tolerance{1e-12};  // CUOPT_PDLP_RC_REL_TOL: relative tolerance (root settings)
  // CUOPT_PDLP_RC_FROM_BASIS: ignore the early GPU point and start from the root basis solution
  bool from_basis{false};
  bool warm_start{true};           // CUOPT_PDLP_RC_WARM: primal + dual of the previous pass
  bool global_pool{false};         // CUOPT_PDLP_RC_POOL: 1 = stage cuts for the B&B cut pool
  double separation_time{10.0};    // CUOPT_PDLP_RC_SEP_TIME: seconds per separation round
  double loop_time_limit{120.0};   // CUOPT_PDLP_RC_TIME: wall budget for the PDLP/cut loop
  bool stop_at_basis{false};       // CUOPT_PDLP_RC_STOP_AT_BASIS: stop when the root basis exists
  bool trigger_always{false};      // CUOPT_PDLP_RC_ALWAYS: also start from a dual simplex root
  bool submip{true};               // CUOPT_PDLP_RC_SUBMIP: RENS on every PDLP point
  int submip_threads{1};           // CUOPT_PDLP_RC_SUBMIP_THREADS
  int max_submips{2};              // CUOPT_PDLP_RC_MAX_SUBMIPS: concurrent RENS sub-MIPs
  bool submip_outlive_root{true};  // CUOPT_PDLP_RC_SUBMIP_OUTLIVE: keep RENS running into the tree
  // CUOPT_PDLP_RC_SUBMIP_MODE: 0 = classic RENS (fix integral, floor/ceil the rest);
  // 1 = root policy without LP re-solves: RINS when an incumbent exists, else RENS, both to a
  // target fix rate; 2 = RENS to a target fix rate; 3 = RENS and, with an incumbent, RINS
  int submip_mode{0};
  double submip_time{-1.0};  // CUOPT_PDLP_RC_SUBMIP_TIME: seconds per sub-MIP (<0: no cap)
  // CUOPT_PDLP_RC_ADAPT: adapt the target fix rate from sub-MIP outcomes (looser after an
  // infeasible neighbourhood, tighter after a sub-MIP that stalled at its root)
  bool adapt_fixrate{false};
  bool mutation{false};  // CUOPT_PDLP_RC_MUTATION: also a mutation sub-MIP when an incumbent exists
  double snap_tol{
    -1.0};  // CUOPT_PDLP_RC_SNAP_TOL: |x - round(x)| treated as integral (<0: integer_tol)
  double fixrate{
    -1.0};  // CUOPT_PDLP_RC_FIXRATE: target fix rate in mode 1 (<0: adaptive like the root)
  int submip_iters{10000};  // CUOPT_PDLP_RC_SUBMIP_ITERS: simplex iteration budget of a sub-MIP

  static pdlp_root_loop_params_t from_env()
  {
    pdlp_root_loop_params_t p;
    read_env("CUOPT_PDLP_RC", p.enabled);
    read_env("CUOPT_PDLP_RC_PASSES", p.max_passes);
    read_env("CUOPT_PDLP_RC_ITERS", p.iteration_limit);
    read_env("CUOPT_PDLP_RC_PASS_TIME", p.pass_time_limit);
    read_env("CUOPT_PDLP_RC_ROOT_SETTINGS", p.root_settings);
    read_env("CUOPT_PDLP_RC_TOL", p.tolerance);
    read_env("CUOPT_PDLP_RC_REL_TOL", p.rel_tolerance);
    read_env("CUOPT_PDLP_RC_FROM_BASIS", p.from_basis);
    read_env("CUOPT_PDLP_RC_WARM", p.warm_start);
    read_env("CUOPT_PDLP_RC_POOL", p.global_pool);
    read_env("CUOPT_PDLP_RC_SEP_TIME", p.separation_time);
    read_env("CUOPT_PDLP_RC_TIME", p.loop_time_limit);
    read_env("CUOPT_PDLP_RC_STOP_AT_BASIS", p.stop_at_basis);
    read_env("CUOPT_PDLP_RC_ALWAYS", p.trigger_always);
    read_env("CUOPT_PDLP_RC_SUBMIP", p.submip);
    read_env("CUOPT_PDLP_RC_SUBMIP_THREADS", p.submip_threads);
    read_env("CUOPT_PDLP_RC_MAX_SUBMIPS", p.max_submips);
    read_env("CUOPT_PDLP_RC_SUBMIP_OUTLIVE", p.submip_outlive_root);
    read_env("CUOPT_PDLP_RC_SUBMIP_MODE", p.submip_mode);
    read_env("CUOPT_PDLP_RC_MUTATION", p.mutation);
    read_env("CUOPT_PDLP_RC_SNAP_TOL", p.snap_tol);
    read_env("CUOPT_PDLP_RC_FIXRATE", p.fixrate);
    read_env("CUOPT_PDLP_RC_SUBMIP_ITERS", p.submip_iters);
    read_env("CUOPT_PDLP_RC_SUBMIP_TIME", p.submip_time);
    read_env("CUOPT_PDLP_RC_ADAPT", p.adapt_fixrate);
    p.max_passes     = std::max(p.max_passes, 0);
    p.submip_threads = std::max(p.submip_threads, 1);
    p.max_submips    = std::max(p.max_submips, 0);
    return p;
  }

  std::string to_string() const
  {
    return "passes=" + std::to_string(max_passes) + " iters=" + std::to_string(iteration_limit) +
           " pass_time=" + std::to_string(pass_time_limit) +
           " root_settings=" + std::to_string(root_settings) + " tol=" + std::to_string(tolerance) +
           " from_basis=" + std::to_string(from_basis) + " warm=" + std::to_string(warm_start) +
           " pool=" + (global_pool ? "global" : "private") +
           " sep_time=" + std::to_string(separation_time) +
           " time=" + std::to_string(loop_time_limit) +
           " stop_at_basis=" + std::to_string(stop_at_basis) +
           " always=" + std::to_string(trigger_always) + " submip=" + std::to_string(submip) +
           " submip_threads=" + std::to_string(submip_threads) +
           " max_submips=" + std::to_string(max_submips) +
           " submip_outlive=" + std::to_string(submip_outlive_root) +
           " submip_mode=" + std::to_string(submip_mode) + " mutation=" + std::to_string(mutation) +
           " snap_tol=" + std::to_string(snap_tol) + " fixrate=" + std::to_string(fixrate) +
           " submip_iters=" + std::to_string(submip_iters) +
           " submip_time=" + std::to_string(submip_time) +
           " adapt=" + std::to_string(adapt_fixrate);
  }

 private:
  static void read_env(const char* name, bool& value)
  {
    const char* s = std::getenv(name);
    if (s != nullptr && *s != '\0') { value = std::atoi(s) != 0; }
  }
  static void read_env(const char* name, int& value)
  {
    const char* s = std::getenv(name);
    if (s != nullptr && *s != '\0') { value = std::atoi(s); }
  }
  static void read_env(const char* name, double& value)
  {
    const char* s = std::getenv(name);
    if (s != nullptr && *s != '\0') { value = std::strtod(s, nullptr); }
  }
};

// Convert the equality-form simplex LP (one singleton slack column per row, stored as the trailing
// columns listed in new_slacks) to the bounded-row form PDLP expects. Returns an empty model when
// the layout does not match.
template <typename i_t, typename f_t>
io::mps_data_model_t<i_t, f_t> simplex_problem_to_mps_data_model(
  const simplex::lp_problem_t<i_t, f_t>& lp, const std::vector<i_t>& new_slacks)
{
  io::mps_data_model_t<i_t, f_t> model;
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols - static_cast<i_t>(new_slacks.size());
  if (n <= 0) { return model; }

  std::vector<i_t> cols_to_remove(lp.A.n, 0);
  std::vector<i_t> slack_of_row(m, -1);
  for (i_t j : new_slacks) {
    if (j < n || j >= lp.num_cols || cols_to_remove[j] != 0) { return {}; }
    if (lp.A.col_start[j + 1] - lp.A.col_start[j] != 1) { return {}; }
    cols_to_remove[j]                       = 1;
    slack_of_row[lp.A.i[lp.A.col_start[j]]] = j;
  }

  std::vector<f_t> row_lower(m);
  std::vector<f_t> row_upper(m);
  for (i_t i = 0; i < m; ++i) {
    const i_t s = slack_of_row[i];
    if (s < 0) { return {}; }
    // a'x + sigma * s = rhs, lower[s] <= s <= upper[s]
    const f_t sigma = lp.A.x[lp.A.col_start[s]];
    if (sigma == f_t(-1)) {
      row_lower[i] = lp.rhs[i] + lp.lower[s];
      row_upper[i] = lp.rhs[i] + lp.upper[s];
    } else if (sigma == f_t(1)) {
      row_lower[i] = lp.rhs[i] - lp.upper[s];
      row_upper[i] = lp.rhs[i] - lp.lower[s];
    } else {
      return {};
    }
  }

  csc_matrix_t<i_t, f_t> A_structural = lp.A;
  A_structural.remove_columns(cols_to_remove);
  csr_matrix_t<i_t, f_t> A_row(m, n, 0);
  A_structural.to_compressed_row(A_row);
  const i_t nz = A_row.row_start[m];

  model.set_csr_constraint_matrix(
    std::span<const f_t>{A_row.x.data(), static_cast<size_t>(nz)},
    std::span<const i_t>{A_row.j.data(), static_cast<size_t>(nz)},
    std::span<const i_t>{A_row.row_start.data(), static_cast<size_t>(m + 1)});
  model.set_objective_coefficients(
    std::span<const f_t>{lp.objective.data(), static_cast<size_t>(n)});
  model.set_objective_scaling_factor(f_t(1));
  model.set_objective_offset(f_t(0));
  model.set_variable_lower_bounds(std::span<const f_t>{lp.lower.data(), static_cast<size_t>(n)});
  model.set_variable_upper_bounds(std::span<const f_t>{lp.upper.data(), static_cast<size_t>(n)});
  model.set_constraint_lower_bounds(row_lower);
  model.set_constraint_upper_bounds(row_upper);
  model.set_maximize(false);
  return model;
}

// Expand a bounded-row PDLP point into the equality/slack coordinates used by cut generation. Slack
// values and their reduced costs are reconstructed from the equality rows.
template <typename i_t, typename f_t>
bool expand_pdlp_solution(const simplex::lp_problem_t<i_t, f_t>& lp,
                          const std::vector<i_t>& new_slacks,
                          const std::vector<f_t>& structural_x,
                          const std::vector<f_t>& row_dual,
                          const std::vector<f_t>& structural_z,
                          std::vector<f_t>& x,
                          std::vector<f_t>& y,
                          std::vector<f_t>& z)
{
  const i_t n = lp.num_cols - static_cast<i_t>(new_slacks.size());
  if (static_cast<i_t>(structural_x.size()) != n || static_cast<i_t>(structural_z.size()) != n ||
      static_cast<i_t>(row_dual.size()) != lp.num_rows) {
    return false;
  }
  x.assign(lp.num_cols, f_t(0));
  z.assign(lp.num_cols, f_t(0));
  y = row_dual;
  std::copy(structural_x.begin(), structural_x.end(), x.begin());
  std::copy(structural_z.begin(), structural_z.end(), z.begin());

  std::vector<f_t> activity(lp.num_rows, f_t(0));
  for (i_t j = 0; j < n; ++j) {
    for (i_t p = lp.A.col_start[j]; p < lp.A.col_start[j + 1]; ++p) {
      activity[lp.A.i[p]] += lp.A.x[p] * x[j];
    }
  }
  for (i_t j : new_slacks) {
    const i_t p   = lp.A.col_start[j];
    const i_t i   = lp.A.i[p];
    const f_t aij = lp.A.x[p];
    x[j]          = (lp.rhs[i] - activity[i]) / aij;
    z[j]          = lp.objective[j] - aij * y[i];
  }
  const auto finite = [](f_t v) { return std::isfinite(v); };
  return std::all_of(x.begin(), x.end(), finite) && std::all_of(y.begin(), y.end(), finite) &&
         std::all_of(z.begin(), z.end(), finite);
}

// Extend a structural point to the equality/slack coordinates of `lp` (slacks from the rows).
template <typename i_t, typename f_t>
void expand_structural_point(const simplex::lp_problem_t<i_t, f_t>& lp,
                             const std::vector<i_t>& new_slacks,
                             const std::vector<f_t>& structural_x,
                             std::vector<f_t>& x)
{
  const i_t n = lp.num_cols - static_cast<i_t>(new_slacks.size());
  x.assign(lp.num_cols, f_t(0));
  std::copy(structural_x.begin(), structural_x.begin() + n, x.begin());
  std::vector<f_t> activity(lp.num_rows, f_t(0));
  for (i_t j = 0; j < n; ++j) {
    for (i_t p = lp.A.col_start[j]; p < lp.A.col_start[j + 1]; ++p) {
      activity[lp.A.i[p]] += lp.A.x[p] * x[j];
    }
  }
  for (i_t j : new_slacks) {
    const i_t p = lp.A.col_start[j];
    x[j]        = (lp.rhs[lp.A.i[p]] - activity[lp.A.i[p]]) / lp.A.x[p];
  }
}

// Append cuts (cuts * x <= rhs) to a private equality-form LP with one >= 0 slack per cut. Unlike
// add_cuts() this does not touch a basis: the only consumer is PDLP.
template <typename i_t, typename f_t>
bool append_cuts_to_pdlp_problem(const csr_matrix_t<i_t, f_t>& cuts,
                                 const std::vector<f_t>& cut_rhs,
                                 simplex::lp_problem_t<i_t, f_t>& lp,
                                 std::vector<i_t>& new_slacks)
{
  const i_t num_cuts = cuts.m;
  if (num_cuts == 0) { return true; }
  if (static_cast<i_t>(cut_rhs.size()) != num_cuts || cuts.n > lp.num_cols) { return false; }

  csr_matrix_t<i_t, f_t> padded_cuts = cuts;
  padded_cuts.n                      = lp.num_cols;
  csr_matrix_t<i_t, f_t> A_row(lp.num_rows, lp.num_cols, 1);
  lp.A.to_compressed_row(A_row);
  if (A_row.append_rows(padded_cuts) != 0) { return false; }

  const i_t old_rows = lp.num_rows;
  const i_t old_cols = lp.num_cols;
  csc_matrix_t<i_t, f_t> A_col(old_rows + num_cuts, old_cols, 1);
  A_row.to_compressed_col(A_col);

  lp.lower.resize(old_cols + num_cuts, f_t(0));
  lp.upper.resize(old_cols + num_cuts, std::numeric_limits<f_t>::infinity());
  lp.objective.resize(old_cols + num_cuts, f_t(0));

  i_t nz = A_col.col_start[old_cols];
  A_col.col_start.resize(old_cols + num_cuts + 1);
  A_col.i.resize(nz + num_cuts);
  A_col.x.resize(nz + num_cuts);
  for (i_t k = 0; k < num_cuts; ++k) {
    const i_t slack        = old_cols + k;
    A_col.col_start[slack] = nz;
    A_col.i[nz]            = old_rows + k;
    A_col.x[nz]            = f_t(1);
    new_slacks.push_back(slack);
    ++nz;
  }
  A_col.col_start[old_cols + num_cuts] = nz;
  A_col.n                              = old_cols + num_cuts;
  A_col.nz_max                         = nz;

  lp.A        = std::move(A_col);
  lp.num_rows = old_rows + num_cuts;
  lp.num_cols = old_cols + num_cuts;
  lp.rhs.resize(lp.num_rows);
  std::copy(cut_rhs.begin(), cut_rhs.end(), lp.rhs.begin() + old_rows);
  return true;
}

// Classic RENS neighbourhood: fix integer variables that are integral in `solution`, restrict the
// others to [floor, ceil]. Returns the fraction of unfixed integer variables that became fixed.
template <typename i_t, typename f_t>
f_t apply_classic_rens_fixings(const std::vector<f_t>& solution,
                               const std::vector<simplex::variable_type_t>& var_types,
                               f_t integer_tol,
                               f_t fixed_tol,
                               std::vector<f_t>& lower,
                               std::vector<f_t>& upper,
                               std::vector<bool>& bounds_changed)
{
  i_t num_integer = 0;
  i_t num_fixed   = 0;
  for (i_t j = 0; j < static_cast<i_t>(var_types.size()); ++j) {
    if (var_types[j] == simplex::variable_type_t::CONTINUOUS) { continue; }
    if (upper[j] - lower[j] <= fixed_tol) { continue; }
    ++num_integer;
    const f_t old_lower = lower[j];
    const f_t old_upper = upper[j];
    const f_t value     = std::clamp(solution[j], old_lower, old_upper);
    const f_t rounded   = std::round(value);
    if (std::abs(value - rounded) <= integer_tol) {
      lower[j] = upper[j] = std::clamp(rounded, old_lower, old_upper);
      ++num_fixed;
    } else {
      lower[j] = std::max(std::floor(value), old_lower);
      upper[j] = std::min(std::ceil(value), old_upper);
    }
    bounds_changed[j] = lower[j] != old_lower || upper[j] != old_upper;
  }
  return num_integer > 0 ? static_cast<f_t>(num_fixed) / num_integer : f_t(0);
}

}  // namespace cuopt::mathematical_optimization::mip
