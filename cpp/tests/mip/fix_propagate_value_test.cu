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

}  // namespace cuopt::mathematical_optimization::test
