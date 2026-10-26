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

}  // namespace cuopt::mathematical_optimization::test
