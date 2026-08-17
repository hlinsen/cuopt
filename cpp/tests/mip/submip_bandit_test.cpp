/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <branch_and_bound/submip_bandit.hpp>

#include <gtest/gtest.h>

namespace cuopt::mathematical_optimization::mip::test {

using heuristic_t = submip_heuristic_t;

TEST(SubmipBanditTest, ReservesEachColdArmBeforeWaitingForFeedback)
{
  submip_bandit_t bandit;
  const submip_bandit_t::eligibility_t both_enabled{true, true};

  EXPECT_EQ(bandit.select(both_enabled), heuristic_t::RINS);
  EXPECT_EQ(bandit.select(both_enabled), heuristic_t::DINS);
  EXPECT_EQ(bandit.select(both_enabled), std::nullopt);

  EXPECT_EQ(bandit.stats(heuristic_t::RINS).pending(), 1);
  EXPECT_EQ(bandit.stats(heuristic_t::DINS).pending(), 1);
}

TEST(SubmipBanditTest, HonorsEligibilityAndKeepsSingleArmBusy)
{
  submip_bandit_t bandit;

  EXPECT_EQ(bandit.select({false, false}), std::nullopt);
  EXPECT_EQ(bandit.select({false, true}), heuristic_t::DINS);
  EXPECT_EQ(bandit.select({false, true}), heuristic_t::DINS);
  EXPECT_EQ(bandit.stats(heuristic_t::DINS).pending(), 2);
  EXPECT_EQ(bandit.stats(heuristic_t::RINS).pulls, 0);
}

TEST(SubmipBanditTest, AssociatesOutOfOrderFeedbackWithTheSelectedArm)
{
  submip_bandit_t bandit;
  const submip_bandit_t::eligibility_t both_enabled{true, true};

  ASSERT_EQ(bandit.select(both_enabled), heuristic_t::RINS);
  ASSERT_EQ(bandit.select(both_enabled), heuristic_t::DINS);

  bandit.complete(heuristic_t::DINS, true, 2.0);
  bandit.complete(heuristic_t::RINS, false, 3.0);

  EXPECT_EQ(bandit.stats(heuristic_t::RINS).completed, 1);
  EXPECT_EQ(bandit.stats(heuristic_t::RINS).successes, 0);
  EXPECT_DOUBLE_EQ(bandit.stats(heuristic_t::RINS).mean_reward(), 0.0);
  EXPECT_EQ(bandit.stats(heuristic_t::DINS).completed, 1);
  EXPECT_EQ(bandit.stats(heuristic_t::DINS).successes, 1);
  EXPECT_DOUBLE_EQ(bandit.stats(heuristic_t::DINS).mean_reward(), 0.5);
}

TEST(SubmipBanditTest, CancellationRestoresReservation)
{
  submip_bandit_t bandit;
  const submip_bandit_t::eligibility_t both_enabled{true, true};

  ASSERT_EQ(bandit.select(both_enabled), heuristic_t::RINS);
  bandit.cancel(heuristic_t::RINS);

  EXPECT_EQ(bandit.total_pulls(), 0);
  EXPECT_EQ(bandit.stats(heuristic_t::RINS).pulls, 0);
  EXPECT_EQ(bandit.stats(heuristic_t::RINS).pending(), 0);
  EXPECT_EQ(bandit.select(both_enabled), heuristic_t::RINS);
}

TEST(SubmipBanditTest, PrefersFasterArmForEqualImprovement)
{
  submip_bandit_t bandit;
  const submip_bandit_t::eligibility_t both_enabled{true, true};

  ASSERT_EQ(bandit.select(both_enabled), heuristic_t::RINS);
  bandit.complete(heuristic_t::RINS, true, 10.0);
  ASSERT_EQ(bandit.select(both_enabled), heuristic_t::DINS);
  bandit.complete(heuristic_t::DINS, true, 1.0);

  EXPECT_EQ(bandit.select(both_enabled), heuristic_t::DINS);
}

TEST(SubmipBanditTest, ExplorationRetriesArmWithoutPriorSuccess)
{
  submip_bandit_t bandit;
  const submip_bandit_t::eligibility_t both_enabled{true, true};

  ASSERT_EQ(bandit.select(both_enabled), heuristic_t::RINS);
  bandit.complete(heuristic_t::RINS, true, 1.0);
  ASSERT_EQ(bandit.select(both_enabled), heuristic_t::DINS);
  bandit.complete(heuristic_t::DINS, false, 1.0);

  bool retried_dins = false;
  for (int decision = 0; decision < 20; ++decision) {
    const auto selected = bandit.select(both_enabled);
    ASSERT_TRUE(selected.has_value());
    if (*selected == heuristic_t::DINS) {
      retried_dins = true;
      bandit.complete(*selected, false, 1.0);
      break;
    }
    bandit.complete(*selected, true, 1.0);
  }

  EXPECT_TRUE(retried_dins);
}

}  // namespace cuopt::mathematical_optimization::mip::test
