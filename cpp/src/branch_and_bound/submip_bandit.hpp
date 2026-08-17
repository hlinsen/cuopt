/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>

namespace cuopt::mathematical_optimization::mip {

enum class submip_heuristic_t : std::size_t { RINS = 0, DINS = 1, SIZE = 2 };

struct submip_bandit_arm_stats_t {
  std::size_t pulls{};
  std::size_t completed{};
  std::size_t successes{};
  double reward_sum{};
  double cost_sum{};

  std::size_t pending() const
  {
    assert(pulls >= completed);
    return pulls - completed;
  }

  double mean_reward() const { return completed == 0 ? 0.0 : reward_sum / completed; }
  double mean_cost() const { return completed == 0 ? 0.0 : cost_sum / completed; }
};

/**
 * @brief Cost-aware UCB selector for the recursive neighborhood sub-MIP heuristics.
 *
 * A pull is reserved when an arm is selected, before its asynchronous work starts. This prevents
 * concurrent B&B workers from repeatedly selecting the same untried arm. Completion contributes a
 * bounded reward: an accepted incumbent improvement is worth `1 / max(worker_seconds, 1)`, while a
 * call without an improvement is worth zero. The caller must serialize select() and complete().
 */
class submip_bandit_t {
 public:
  static constexpr std::size_t num_arms = static_cast<std::size_t>(submip_heuristic_t::SIZE);
  using eligibility_t                   = std::array<bool, num_arms>;

  std::optional<submip_heuristic_t> select(const eligibility_t& eligible)
  {
    std::size_t num_eligible = 0;
    for (bool is_eligible : eligible) {
      if (is_eligible) { ++num_eligible; }
    }
    if (num_eligible == 0) { return std::nullopt; }

    // Force one cold-start pull of every eligible arm. Reservations count immediately so two
    // concurrent selectors choose different cold arms.
    for (std::size_t arm = 0; arm < num_arms; ++arm) {
      if (eligible[arm] && stats_[arm].pulls == 0) { return reserve(arm); }
    }

    // With one eligible arm there is no decision to learn, so keep the available workers busy even
    // while an earlier pull is still pending.
    if (num_eligible == 1) {
      for (std::size_t arm = 0; arm < num_arms; ++arm) {
        if (eligible[arm]) { return reserve(arm); }
      }
    }

    // Wait for at least one cold-start result before scheduling additional work. Otherwise a third
    // worker would make a decision without feedback from either arm.
    bool has_feedback = false;
    for (std::size_t arm = 0; arm < num_arms; ++arm) {
      has_feedback = has_feedback || (eligible[arm] && stats_[arm].completed > 0);
    }
    if (!has_feedback) { return std::nullopt; }

    double best_score                = -std::numeric_limits<double>::infinity();
    std::size_t best_arm             = num_arms;
    const double log_term            = std::log(static_cast<double>(total_pulls_ + 1));
    constexpr double score_tolerance = 1e-12;

    for (std::size_t arm = 0; arm < num_arms; ++arm) {
      if (!eligible[arm] || stats_[arm].completed == 0) { continue; }

      // Pending pulls are included in the denominator to avoid oversubscribing an arm while its
      // outcome is unknown. The reward itself is bounded in [0, 1].
      const double confidence = std::sqrt(2.0 * log_term / static_cast<double>(stats_[arm].pulls));
      const double score      = stats_[arm].mean_reward() + confidence;
      if (score > best_score + score_tolerance) {
        best_score = score;
        best_arm   = arm;
      }
    }

    if (best_arm == num_arms) { return std::nullopt; }
    return reserve(best_arm);
  }

  void complete(submip_heuristic_t heuristic, bool improved, double worker_seconds)
  {
    const std::size_t arm = static_cast<std::size_t>(heuristic);
    assert(arm < num_arms);
    assert(stats_[arm].completed < stats_[arm].pulls);

    const double cost   = std::max(worker_seconds, 0.0);
    const double reward = improved ? 1.0 / std::max(cost, 1.0) : 0.0;
    ++stats_[arm].completed;
    if (improved) { ++stats_[arm].successes; }
    stats_[arm].reward_sum += reward;
    stats_[arm].cost_sum += cost;
  }

  void cancel(submip_heuristic_t heuristic)
  {
    const std::size_t arm = static_cast<std::size_t>(heuristic);
    assert(arm < num_arms);
    assert(stats_[arm].pending() > 0);
    assert(total_pulls_ > 0);
    --stats_[arm].pulls;
    --total_pulls_;
  }

  const submip_bandit_arm_stats_t& stats(submip_heuristic_t heuristic) const
  {
    const std::size_t arm = static_cast<std::size_t>(heuristic);
    assert(arm < num_arms);
    return stats_[arm];
  }

  std::size_t total_pulls() const { return total_pulls_; }

 private:
  submip_heuristic_t reserve(std::size_t arm)
  {
    assert(arm < num_arms);
    ++stats_[arm].pulls;
    ++total_pulls_;
    return static_cast<submip_heuristic_t>(arm);
  }

  std::array<submip_bandit_arm_stats_t, num_arms> stats_{};
  std::size_t total_pulls_{};
};

}  // namespace cuopt::mathematical_optimization::mip
