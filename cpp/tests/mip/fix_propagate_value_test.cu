/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mip_heuristics/diversity/fix_propagate_host.hpp>

#include <gtest/gtest.h>

#include <random>

namespace cuopt::mathematical_optimization::test {
namespace {
using mip::fp_preferred_value;
using mip::fp_value_strategy_t;

constexpr double int_tol = 1e-6;
constexpr double inf     = std::numeric_limits<double>::infinity();

double value(fp_value_strategy_t s, double x, double u = 0., double param = 0.)
{
  return fp_preferred_value<double>(s, x, 0., inf, 0., param, int_tol, u);
}

double locks_value(double x, int uplocks, int downlocks, double ub = inf)
{
  return fp_preferred_value<double>(
    fp_value_strategy_t::FEWER_LOCKS, x, 0., ub, 0., 0., int_tol, 0., uplocks, downlocks);
}
}  // namespace

TEST(FixPropagateValue, RoundUp)
{
  EXPECT_EQ(value(fp_value_strategy_t::ROUND_UP, 2.3), 3.);
  EXPECT_EQ(value(fp_value_strategy_t::ROUND_UP, 2.0), 2.);
  EXPECT_EQ(value(fp_value_strategy_t::ROUND_UP, 2.0 - 1e-9), 2.);
}

TEST(FixPropagateValue, LpRandomizedUsesTheFractionalPart)
{
  const auto s = fp_value_strategy_t::LP_RANDOMIZED;
  EXPECT_EQ(value(s, 2.3, 0.29), 3.);
  EXPECT_EQ(value(s, 2.3, 0.31), 2.);
  // integral reference values are never moved
  for (double u : {0., 0.5, 0.999}) {
    EXPECT_EQ(value(s, 4.0, u), 4.);
    EXPECT_EQ(value(s, 4.0 - 1e-9, u), 4.);
  }
}

TEST(FixPropagateValue, LpRandomizedRoundsUpWithTheFractionalProbability)
{
  std::mt19937 g(7);
  std::uniform_real_distribution<double> U(0., 1.);
  for (double frac : {0.05, 0.25, 0.5, 0.9}) {
    const int n = 20000;
    int ups     = 0;
    for (int i = 0; i < n; ++i) {
      ups += value(fp_value_strategy_t::LP_RANDOMIZED, 1. + frac, U(g)) == 2.;
    }
    EXPECT_NEAR(double(ups) / n, frac, 0.015) << "fractional part " << frac;
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
  using mip::fp_order_key;
  using mip::fp_variable_strategy_t;
  const auto s = fp_variable_strategy_t::LOCKS;
  EXPECT_GT((fp_order_key<int, double>(s, 0.1, 0., 3, 4)),
            (fp_order_key<int, double>(s, 0.9, 0., 2, 4)));
  // equal locks: the larger reference value comes first
  EXPECT_GT((fp_order_key<int, double>(s, 0.9, 0., 2, 2)),
            (fp_order_key<int, double>(s, 0.1, 0., 2, 2)));
}

TEST(FixPropagateValue, AbsoluteReducedCostOrderRanksBothSigns)
{
  using mip::fp_order_key;
  using mip::fp_variable_strategy_t;
  const auto s = fp_variable_strategy_t::REDUCED_COST_ABS;
  EXPECT_GT((fp_order_key<int, double>(s, 0.5, 5., 0)),
            (fp_order_key<int, double>(s, 0.5, -1., 0)));
  EXPECT_GT((fp_order_key<int, double>(s, 0.5, -5., 0)),
            (fp_order_key<int, double>(s, 0.5, 1., 0)));
  // TYPE keeps the formulation order (equal keys, stable sort)
  EXPECT_EQ((fp_order_key<int, double>(fp_variable_strategy_t::TYPE, 0.2, 3., 1)),
            (fp_order_key<int, double>(fp_variable_strategy_t::TYPE, 0.9, -3., 4)));
}

TEST(FixPropagateValue, ObjectiveAndRandomBoundValues)
{
  auto v = [](fp_value_strategy_t s, double obj, double u = 0., double ub = 4.) {
    return fp_preferred_value<double>(s, 1.5, 0., ub, 0., 0., int_tol, u, 0, 0, obj);
  };
  EXPECT_EQ(v(fp_value_strategy_t::GOOD_OBJ, -2.), 4.);  // a negative cost is better at the top
  EXPECT_EQ(v(fp_value_strategy_t::GOOD_OBJ, 2.), 0.);
  EXPECT_EQ(v(fp_value_strategy_t::GOOD_OBJ, 0.), 0.);  // no objective: the lower bound
  EXPECT_EQ(v(fp_value_strategy_t::BAD_OBJ, 2.), 4.);
  EXPECT_EQ(v(fp_value_strategy_t::BAD_OBJ, -2.), 0.);
  EXPECT_EQ(v(fp_value_strategy_t::RANDOM_BOUND, 0., 0.2), 4.);
  EXPECT_EQ(v(fp_value_strategy_t::RANDOM_BOUND, 0., 0.7), 0.);
  // an infinite chosen bound falls back to rounding the reference value up
  EXPECT_EQ(v(fp_value_strategy_t::GOOD_OBJ, -2., 0., inf), 2.);
  EXPECT_EQ(v(fp_value_strategy_t::UPPER_BOUND, 0., 0., inf), 2.);
}

TEST(FixPropagateValue, ScipLocksTieRuleAndDynamicDirections)
{
  auto tie = [](double obj, double u) {
    return fp_preferred_value<double>(
      fp_value_strategy_t::SCIP_LOCKS, 0.4, 0., 1., 0., 0., int_tol, u, 0, 0, obj);
  };
  EXPECT_EQ(tie(-1., 0.99), 1.);  // objective <= 0: up
  EXPECT_EQ(tie(0., 0.99), 1.);
  EXPECT_EQ(tie(1., 0.5), 1.);  // positive objective: up with probability 0.67
  EXPECT_EQ(tie(1., 0.8), 0.);
  double up = -1, down = -1;
  EXPECT_TRUE(mip::fp_dynamic_lock_values<double>(
    fp_value_strategy_t::SCIP_LOCKS, 0.4, 0., 1., int_tol, up, down));
  EXPECT_EQ(up, 1.);
  EXPECT_EQ(down, 0.);
  // general integers are not fixed by SCIP's locks rule
  EXPECT_FALSE(mip::fp_dynamic_lock_values<double>(
    fp_value_strategy_t::SCIP_LOCKS, 2.4, 0., 5., int_tol, up, down));
  EXPECT_TRUE(mip::fp_dynamic_lock_values<double>(
    fp_value_strategy_t::FEWER_LOCKS_DYNAMIC, 2.4, 0., 5., int_tol, up, down));
  EXPECT_EQ(up, 3.);
  EXPECT_EQ(down, 2.);
}

TEST(FixPropagateValue, RowSidesStopLockingOnceRedundant)
{
  bool le = false, ge = false;
  // x1 + x2 + x3 <= 1 with x1 fixed to 1 and x2, x3 fixed to 0: activity range [1, 1]
  mip::fp_active_sides<double>(1., 1., -inf, 1., 1e-6, le, ge);
  EXPECT_FALSE(le);
  EXPECT_FALSE(ge);
  // nothing fixed yet: maximum activity 3 can violate the <= side
  mip::fp_active_sides<double>(0., 3., -inf, 1., 1e-6, le, ge);
  EXPECT_TRUE(le);
  EXPECT_FALSE(ge);
  // equality row 2 <= a.x <= 2 with activity range [0, 3]: both sides still lock
  mip::fp_active_sides<double>(0., 3., 2., 2., 1e-6, le, ge);
  EXPECT_TRUE(le);
  EXPECT_TRUE(ge);
  // an infinite activity bound keeps the side active
  mip::fp_active_sides<double>(-inf, inf, 0., 1., 1e-6, le, ge);
  EXPECT_TRUE(le);
  EXPECT_TRUE(ge);
}

}  // namespace cuopt::mathematical_optimization::test
