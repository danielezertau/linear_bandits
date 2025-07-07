import numpy as np

class UCB1:
    def __init__(self, n_arms, reward_function):
        self.n_arms = n_arms
        self.reward_function = reward_function
        self.counts = np.zeros(n_arms, dtype=int)     # number of times each arm was played
        self.values = np.zeros(n_arms, dtype=float)    # average reward for each arm
        self.total_counts = 0                          # total number of rounds

    def select_arm(self):
        # If any arm hasn't been played, play it first (ensures each arm is sampled once)
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        # Otherwise compute UCB for each arm and select the one with highest UCB value
        ucb_values = self.values + np.sqrt((8 * np.log(self.total_counts)) / self.counts)
        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        # Update counts and average reward for the chosen arm
        self.counts[chosen_arm] += 1
        self.total_counts += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        # incremental update of the average
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward


def run_ucb(bandit, horizon):
    expected_rewards = []

    for t in range(horizon):
        arm = bandit.select_arm()
        reward, expected_reward = bandit.reward_function(arm)
        bandit.update(arm, reward)

        expected_rewards.append(expected_reward)

    return expected_rewards
