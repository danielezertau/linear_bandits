from distributions.dist import Distribution
from distributions.single_coord_noise import SingleCoordNoise
from ucb import UCB1, run_ucb
from utils import *

class PhasedElimination:
    def __init__(self, actions, delta, theta_star, l2_dist_from_sphere, theta_dist: Distribution, num_steps):
        self.A_l = actions
        self.k = actions.shape[0]
        self.k_l = self.k
        self.d = actions.shape[1]
        self.delta = delta
        self.theta_star = theta_star
        self.theta_dist = theta_dist
        self.cov_matrix = theta_dist.get_cov_matrix()
        self.l2_dist_from_sphere = l2_dist_from_sphere
        self.total_steps = num_steps
        self.remaining_steps = self.total_steps
        self.l = 1
        self.theta_hat = np.zeros((self.d, 1))
        self.pi_l = []
        self.variances_l = np.array([arm.T @ self.cov_matrix @ arm for arm in self.A_l])
        self.optimal_arm = self.A_l[np.argmax(self.A_l @ self.theta_star)]
        self.optimal_expected_reward = self.theta_star @ self.optimal_arm
        self.pseudo_regret = []

    def delta_l(self):
        return self.delta / (self.k * self.l * (self.l+1))

    def epsilon_l(self):
        return math.pow(2, - self.l)

    def update_phase_info(self, actions_l):
        self.A_l = actions_l
        self.k_l = actions_l.shape[0]
        variances_l = np.array([arm.T @ self.cov_matrix @ arm for arm in self.A_l])
        self.variances_l = np.maximum(variances_l, self.epsilon_l())
        self.l += 1

    def get_t_l_a(self, pi_arm, arm_variance):
        delta_l = self.delta_l()
        epsilon_l = self.epsilon_l()
        return math.ceil(((16 * self.d * pi_arm) / (math.pow(epsilon_l, 2))) * (math.log2(1 / delta_l)) * arm_variance)

    def optimal_arm_eliminated(self):
        return self.optimal_arm not in self.A_l

    def play_arm(self, arm):
        theta_s = self.theta_dist.sample_theta_s()
        return (theta_s @ arm).item(), (self.theta_star @ arm).item()
    
    def should_eliminate_arm(self, arm):
        gaps = self.A_l @ self.theta_hat
        return np.max(gaps - arm @ self.theta_hat) > 2 * self.epsilon_l()
    
    def play_arm_t_l_a(self, arm_index):
        arm = self.A_l[arm_index]
        pi_arm = self.pi_l[arm_index]
        arm_variance = self.variances_l[arm_index]

        v_mat_l_a = np.zeros((self.d, self.d))
        theta_hat_a = np.zeros((self.d, 1))
        arm = np.expand_dims(arm, axis=1)
        t_l_a = self.get_t_l_a(pi_arm, arm_variance)
        v_mat_l_a += t_l_a * (arm @ arm.T / arm_variance)
    
        print(f"Running arm {arm_index} for {t_l_a} steps")
        num_steps = min(t_l_a, self.remaining_steps)
        for _ in range(num_steps):
            reward, expected_reward = self.play_arm(arm)
            self.pseudo_regret.append(self.optimal_expected_reward - expected_reward)
            theta_hat_a += (arm / arm_variance) * reward
        self.remaining_steps -= num_steps
        return v_mat_l_a, theta_hat_a
    
    def phase(self):
        phase_start_remaining_steps = self.remaining_steps
        support, self.pi_l = frank_wolfe(self.A_l)
        v_mat_l = np.zeros((self.d, self.d))
        self.theta_hat = np.zeros((self.d, 1))
        for arm_index in support:
            if self.remaining_steps <= 0:
                break
            v_mat_l_a, theta_hat_a = self.play_arm_t_l_a(arm_index)
            v_mat_l += v_mat_l_a
            self.theta_hat += theta_hat_a
    
        print("Calculating theta hat")
        v_mat_l_inv = np.linalg.inv(v_mat_l)
        self.theta_hat = v_mat_l_inv @ self.theta_hat
    
        print("Eliminating arms")
        vec_fn = np.vectorize(
            self.should_eliminate_arm,
            otypes=[bool],
            signature='(d)->()'
        )
        old_k_l = self.k_l
        self.update_phase_info(self.A_l[~vec_fn(self.A_l)])
        print(f"Eliminated {old_k_l - self.k_l} arms")
        return phase_start_remaining_steps - self.remaining_steps

    def phase_elimination_alg(self):
        time_to_one_arm = 0
        time_to_ucb = 0
        phase_lengths = []
        while self.remaining_steps > 0:
            print(f"\n\nRunning phase {self.l}"
                  f"\nNumber of arms: {self.k_l}"
                  f"\nRemaining steps: {self.remaining_steps}")
            print(f"Optimal arm eliminated: {self.optimal_arm_eliminated()}")
    
            if self.k_l < self.d:
                # Play UCB algo
                print("Running UCB")
                reward_function = lambda a: self.play_arm(self.A_l[a])
                ucb = UCB1(n_arms=self.k_l, reward_function=reward_function)
                phase_expected_rewards = run_ucb(ucb, self.remaining_steps)
                time_to_ucb = self.total_steps - self.remaining_steps
                self.remaining_steps = 0
                phase_lengths.append(len(phase_expected_rewards))
                self.pseudo_regret += list(self.optimal_expected_reward - np.array(phase_expected_rewards))
            else:
                phase_length = self.phase()
                phase_lengths.append(phase_length)

            if self.k_l == 1 and time_to_one_arm == 0:
                time_to_one_arm = self.total_steps - self.remaining_steps

        return self.pseudo_regret, phase_lengths, time_to_one_arm, time_to_ucb

def main():
    num_steps = 3000000
    k, d, delta = 1000, 50, 0.00001
    actions = sample_matrix_in_unit_ball(k, d)
    
    theta_star = uniform_vector_unit_ball(d)
    l2_dist_from_sphere = 1 - np.linalg.norm(theta_star)

    theta_dist = SingleCoordNoise(theta_star, l2_dist_from_sphere)
    phase_elimination = PhasedElimination(actions, delta, theta_star, l2_dist_from_sphere, theta_dist ,num_steps)
    regret, phase_lengths, time_to_one_arm, time_to_ucb = phase_elimination.phase_elimination_alg()

    # Plot regret
    worst_case = np.array([np.max(actions @ theta_star) - np.min(actions @ theta_star)] * num_steps)
    mean_case = np.array([np.max(actions @ theta_star) - np.mean(actions @ theta_star).item()] * num_steps)
    
    num_phases = len(phase_lengths)
    delta_m = delta / (k * num_phases * (num_phases + 1))
    plot_regret(regret, worst_case, mean_case, phase_lengths, d, delta_m, theta_dist.get_cov_matrix(), time_to_one_arm, time_to_ucb)

if __name__ == '__main__':
    main()
