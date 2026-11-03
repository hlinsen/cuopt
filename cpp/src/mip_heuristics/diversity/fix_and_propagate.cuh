/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "population.cuh"

#include <mip_heuristics/local_search/local_search.cuh>
#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/solver_context.cuh>
#include <utilities/timer.hpp>

#include <rmm/device_uvector.hpp>

#include <functional>
#include <limits>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

// LP-guided CPU fix-and-propagate on the root LP point (fix_propagate_host.hpp does the pass):
// fix the integers in the configured order to the configured values with bounds propagation and
// one-level backtracking, evaluate the pattern with a cost-aware fixed-integer LP, push and repair
// it, and hand feasible points to the population. Built on the diversity manager's state for one
// call (diversity_manager_t::run_root_fix_and_propagate).
template <typename i_t, typename f_t>
class fix_and_propagate_t {
 public:
  fix_and_propagate_t(mip_solver_context_t<i_t, f_t>& context,
                      problem_t<i_t, f_t>* problem_ptr,
                      population_t<i_t, f_t>& population,
                      local_search_t<i_t, f_t>& ls,
                      cuopt::timer_t& timer,
                      const rmm::device_uvector<f_t>& lp_primal,
                      const rmm::device_uvector<f_t>& lp_dual,
                      std::function<bool()> preempted);
  // one pass with the configured strategies within the configured budget; true when a pattern was
  // repaired and added to the population
  bool run();

 private:
  // feasibility jump repair of a complete but infeasible pattern (integers from the pass,
  // continuous from the LP point): a short fixed-integer LP, then FJ on every variable
  struct fj_repair_t {
    bool feasible{false};
    bool published{false};  // the population took it as its new best
    f_t obj{std::numeric_limits<f_t>::infinity()};
    std::vector<f_t> assignment;
    double time{0};
  };
  fj_repair_t fj_repair_pattern(const std::vector<f_t>& assign, f_t time_limit);
  // fixed-binary LP evaluation of one binary pattern (full host assignment, integers = pattern):
  // cost-aware PDLP on the fixed problem (cold with PSLP, or warm-started without presolve),
  // full objective of the LP point, optional objective-free repair and publish
  struct pattern_eval_t {
    bool usable{false};
    bool optimal{false};
    f_t lp_obj{std::numeric_limits<f_t>::infinity()};
    f_t l2_primal_res{0};
    f_t repaired_obj{std::numeric_limits<f_t>::infinity()};
    bool published{false};
    int iters{0};
    double time{0};
    std::vector<f_t> primal, dual;  // phase-A (PID) fixed-LP primal/dual
    std::vector<f_t> repaired_assignment;
    f_t pushed_lp_obj{std::numeric_limits<f_t>::infinity()};
    f_t pushed_l2_primal_res{-1};
    double push_time{0};
    bool cannot_beat_incumbent{false};  // stopped after stage 1
  };
  pattern_eval_t eval_fixed_pattern(const std::vector<f_t>& assign,
                                    f_t tol,
                                    f_t tlimit,
                                    const std::vector<f_t>* warm_primal,
                                    const std::vector<f_t>* warm_dual,
                                    bool repair,
                                    f_t repair_tlimit,
                                    f_t repair_below  = std::numeric_limits<f_t>::infinity(),
                                    f_t max_push_time = 11.);
  bool check_b_b_preemption() { return preempted(); }

  mip_solver_context_t<i_t, f_t>& context;
  problem_t<i_t, f_t>* problem_ptr;
  population_t<i_t, f_t>& population;
  local_search_t<i_t, f_t>& ls;
  cuopt::timer_t& timer;
  const rmm::device_uvector<f_t>& lp_optimal_solution;
  const rmm::device_uvector<f_t>& lp_dual_optimal_solution;
  std::function<bool()> preempted;
};

}  // namespace cuopt::mathematical_optimization::mip
