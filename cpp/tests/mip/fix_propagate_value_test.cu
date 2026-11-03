/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mip_heuristics/diversity/fix_propagate_host.hpp>

#include <gtest/gtest.h>

#include <random>

namespace cuopt::mathematical_optimization::test {
namespace {
using mip::fix_propagate_preferred_value;
using mip::fix_propagate_value_strategy_t;

constexpr double integrality_tolerance = 1e-6;
constexpr double infinity              = std::numeric_limits<double>::infinity();

double value(fix_propagate_value_strategy_t strategy,
             double reference_value,
             double uniform_draw = 0.,
             double value_param  = 0.)
{
  return fix_propagate_preferred_value<double>(
    strategy, reference_value, 0., infinity, 0., value_param, integrality_tolerance, uniform_draw);
}

double locks_value(double reference_value, int uplocks, int downlocks, double ub = infinity)
{
  return fix_propagate_preferred_value<double>(fix_propagate_value_strategy_t::FEWER_LOCKS,
                                               reference_value,
                                               0.,
                                               ub,
                                               0.,
                                               0.,
                                               integrality_tolerance,
                                               0.,
                                               uplocks,
                                               downlocks);
}
}  // namespace

TEST(FixPropagateValue, RoundUp)
{
  EXPECT_EQ(value(fix_propagate_value_strategy_t::ROUND_UP, 2.3), 3.);
  EXPECT_EQ(value(fix_propagate_value_strategy_t::ROUND_UP, 2.0), 2.);
  EXPECT_EQ(value(fix_propagate_value_strategy_t::ROUND_UP, 2.0 - 1e-9), 2.);
}

TEST(FixPropagateValue, LpRandomizedUsesTheFractionalPart)
{
  const auto strategy = fix_propagate_value_strategy_t::LP_RANDOMIZED;
  EXPECT_EQ(value(strategy, 2.3, 0.29), 3.);
  EXPECT_EQ(value(strategy, 2.3, 0.31), 2.);
  // integral reference values are never moved
  for (double uniform_draw : {0., 0.5, 0.999}) {
    EXPECT_EQ(value(strategy, 4.0, uniform_draw), 4.);
    EXPECT_EQ(value(strategy, 4.0 - 1e-9, uniform_draw), 4.);
  }
}

TEST(FixPropagateValue, LpRandomizedRoundsUpWithTheFractionalProbability)
{
  std::mt19937 generator(7);
  std::uniform_real_distribution<double> uniform(0., 1.);
  for (double fractional_part : {0.05, 0.25, 0.5, 0.9}) {
    const int n_draws = 20000;
    int n_rounded_up  = 0;
    for (int i = 0; i < n_draws; ++i) {
      n_rounded_up += value(fix_propagate_value_strategy_t::LP_RANDOMIZED,
                            1. + fractional_part,
                            uniform(generator)) == 2.;
    }
    EXPECT_NEAR(double(n_rounded_up) / n_draws, fractional_part, 0.015)
      << "fractional part " << fractional_part;
  }
}

TEST(FixPropagateValue, FewerLocksRoundsTowardTheSideWithFewerLocks)
{
  EXPECT_EQ(locks_value(0.2, 1, 3), 1.);      // fewer up-locks: round up
  EXPECT_EQ(locks_value(0.8, 3, 1), 0.);      // fewer down-locks: round down
  EXPECT_EQ(locks_value(2.0, 0, 5, 2.), 2.);  // rounding up stays within the upper bound
  EXPECT_EQ(locks_value(0.4, 2, 2), 1.);      // ties round a fractional value up
  EXPECT_EQ(locks_value(3.0, 2, 2), 3.);      // ties keep an integral value
}

TEST(FixPropagateValue, LocksOrderFixesTheMostLockedVariablesFirst)
{
  using mip::fix_propagate_order_key;
  using mip::fix_propagate_variable_strategy_t;
  const auto strategy = fix_propagate_variable_strategy_t::LOCKS;
  EXPECT_GT((fix_propagate_order_key<int, double>(strategy, 0.1, 0., 3, 4)),
            (fix_propagate_order_key<int, double>(strategy, 0.9, 0., 2, 4)));
  // equal locks: the larger reference value comes first
  EXPECT_GT((fix_propagate_order_key<int, double>(strategy, 0.9, 0., 2, 2)),
            (fix_propagate_order_key<int, double>(strategy, 0.1, 0., 2, 2)));
}

TEST(FixPropagateValue, AbsoluteReducedCostOrderRanksBothSigns)
{
  using mip::fix_propagate_order_key;
  using mip::fix_propagate_variable_strategy_t;
  const auto strategy = fix_propagate_variable_strategy_t::REDUCED_COST_ABS;
  EXPECT_GT((fix_propagate_order_key<int, double>(strategy, 0.5, 5., 0)),
            (fix_propagate_order_key<int, double>(strategy, 0.5, -1., 0)));
  EXPECT_GT((fix_propagate_order_key<int, double>(strategy, 0.5, -5., 0)),
            (fix_propagate_order_key<int, double>(strategy, 0.5, 1., 0)));
  // TYPE keeps the formulation order (equal keys, stable sort)
  EXPECT_EQ(
    (fix_propagate_order_key<int, double>(fix_propagate_variable_strategy_t::TYPE, 0.2, 3., 1)),
    (fix_propagate_order_key<int, double>(fix_propagate_variable_strategy_t::TYPE, 0.9, -3., 4)));
}

TEST(FixPropagateValue, ObjectiveAndRandomBoundValues)
{
  auto preferred = [](fix_propagate_value_strategy_t strategy,
                      double objective_coefficient,
                      double uniform_draw = 0.,
                      double ub           = 4.) {
    return fix_propagate_preferred_value<double>(strategy,
                                                 1.5,
                                                 0.,
                                                 ub,
                                                 0.,
                                                 0.,
                                                 integrality_tolerance,
                                                 uniform_draw,
                                                 0,
                                                 0,
                                                 objective_coefficient);
  };
  EXPECT_EQ(preferred(fix_propagate_value_strategy_t::GOOD_OBJ, -2.),
            4.);  // a negative cost is better at the top
  EXPECT_EQ(preferred(fix_propagate_value_strategy_t::GOOD_OBJ, 2.), 0.);
  EXPECT_EQ(preferred(fix_propagate_value_strategy_t::GOOD_OBJ, 0.),
            0.);  // no objective: the lower bound
  EXPECT_EQ(preferred(fix_propagate_value_strategy_t::BAD_OBJ, 2.), 4.);
  EXPECT_EQ(preferred(fix_propagate_value_strategy_t::BAD_OBJ, -2.), 0.);
  EXPECT_EQ(preferred(fix_propagate_value_strategy_t::RANDOM_BOUND, 0., 0.2), 4.);
  EXPECT_EQ(preferred(fix_propagate_value_strategy_t::RANDOM_BOUND, 0., 0.7), 0.);
  // an infinite chosen bound falls back to rounding the reference value up
  EXPECT_EQ(preferred(fix_propagate_value_strategy_t::GOOD_OBJ, -2., 0., infinity), 2.);
  EXPECT_EQ(preferred(fix_propagate_value_strategy_t::UPPER_BOUND, 0., 0., infinity), 2.);
}

TEST(FixPropagateValue, DynamicLocksDirections)
{
  double up_value = -1, down_value = -1;
  EXPECT_TRUE(mip::fix_propagate_dynamic_lock_values<double>(
    fix_propagate_value_strategy_t::FEWER_LOCKS_DYNAMIC,
    2.4,
    5.,
    integrality_tolerance,
    up_value,
    down_value));
  EXPECT_EQ(up_value, 3.);
  EXPECT_EQ(down_value, 2.);
  // the upper bound caps the up value
  EXPECT_TRUE(mip::fix_propagate_dynamic_lock_values<double>(
    fix_propagate_value_strategy_t::FEWER_LOCKS_DYNAMIC,
    2.0,
    2.,
    integrality_tolerance,
    up_value,
    down_value));
  EXPECT_EQ(up_value, 2.);
  EXPECT_EQ(down_value, 2.);
  // static strategies do not use dynamic locks
  EXPECT_FALSE(
    mip::fix_propagate_dynamic_lock_values<double>(fix_propagate_value_strategy_t::FEWER_LOCKS,
                                                   2.4,
                                                   5.,
                                                   integrality_tolerance,
                                                   up_value,
                                                   down_value));
}

TEST(FixPropagateValue, RowSidesStopLockingOnceRedundant)
{
  bool le_active = false, ge_active = false;
  // x1 + x2 + x3 <= 1 with x1 fixed to 1 and x2, x3 fixed to 0: activity range [1, 1]
  mip::fix_propagate_active_sides<double>(1., 1., -infinity, 1., 1e-6, le_active, ge_active);
  EXPECT_FALSE(le_active);
  EXPECT_FALSE(ge_active);
  // nothing fixed yet: maximum activity 3 can violate the <= side
  mip::fix_propagate_active_sides<double>(0., 3., -infinity, 1., 1e-6, le_active, ge_active);
  EXPECT_TRUE(le_active);
  EXPECT_FALSE(ge_active);
  // equality row 2 <= a.x <= 2 with activity range [0, 3]: both sides still lock
  mip::fix_propagate_active_sides<double>(0., 3., 2., 2., 1e-6, le_active, ge_active);
  EXPECT_TRUE(le_active);
  EXPECT_TRUE(ge_active);
  // an infinite activity bound keeps the side active
  mip::fix_propagate_active_sides<double>(-infinity, infinity, 0., 1., 1e-6, le_active, ge_active);
  EXPECT_TRUE(le_active);
  EXPECT_TRUE(ge_active);
}

}  // namespace cuopt::mathematical_optimization::test
