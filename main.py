from ucb import UCB1, run_ucb
from utils import *

class PhasedElimination:
    def __init__(self, A, delta, theta_star, cov_matrix, r, num_steps):
        self.A_l = A
        self.k = A.shape[0]
        self.k_l = self.k
        self.d = A.shape[1]
        self.delta = delta
        self.theta_star = theta_star
        self.cov_matrix = cov_matrix
        self.r = r
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

    def update_phase_info(self, A_l):
        self.A_l = A_l
        self.k_l = A_l.shape[0]
        variances_l = np.array([arm.T @ self.cov_matrix @ arm for arm in self.A_l])
        self.variances_l = np.maximum(variances_l, self.epsilon_l())
        self.l += 1

    def get_t_l_a(self, pi_arm, arm_variance):
        delta_l = self.delta_l()
        epsilon_l = self.epsilon_l()
        return math.ceil(((16 * self.d * pi_arm) / (math.pow(epsilon_l, 2))) * (math.log2(1 / delta_l)) * arm_variance)

    def optimal_arm_eliminated(self):
        return self.optimal_arm not in self.A_l
    
    def sample_theta_s(self):
        return np.random.multivariate_normal(self.theta_star, self.cov_matrix)
    
    def sample_shifted_uniform(self):
        theta_star = np.array(self.theta_star)
        if np.linalg.norm(theta_star) + self.r > 1:
            raise ValueError("The ball of radius r around mu must lie inside the unit ball.")
    
        # Sample direction uniformly on the unit sphere
        direction = uniform_vector_unit_sphere(self.d)
        
        return theta_star + self.r * direction
    
    def play_arm(self, arm):
        theta_s = self.sample_shifted_uniform()
        return (theta_s @ arm).item(), (self.theta_star @ arm).item()
    
    def should_eliminate_arm(self, arm):
        gaps = self.A_l @ self.theta_hat
        return np.max(gaps - arm @ self.theta_hat) > 2 * self.epsilon_l()
    
    def play_arm_t_l_a(self, arm_index):
        arm = self.A_l[arm_index]
        pi_arm = self.pi_l[arm_index]
        arm_variance = self.variances_l[arm_index]

        V_l_a = np.zeros((self.d, self.d))
        theta_hat_a = np.zeros((self.d, 1))
        arm = np.expand_dims(arm, axis=1)
        T_l_a = self.get_t_l_a(pi_arm, arm_variance)
        V_l_a += T_l_a * (arm @ arm.T / arm_variance)
    
        print(f"Running arm {arm_index} for {T_l_a} steps")
        num_steps = min(T_l_a, self.remaining_steps)
        for _ in range(num_steps):
            reward, expected_reward = self.play_arm(arm)
            self.pseudo_regret.append(self.optimal_expected_reward - expected_reward)
            theta_hat_a += (arm / arm_variance) * reward
        self.remaining_steps -= num_steps
        return V_l_a, theta_hat_a
    
    def phase(self):
        phase_start_remaining_steps = self.remaining_steps
        support, self.pi_l = find_optimal_design(self.A_l)
        V_l = np.zeros((self.d, self.d))
        self.theta_hat = np.zeros((self.d, 1))
        for arm_index in support:
            if self.remaining_steps <= 0:
                break
            V_l_a, theta_hat_a = self.play_arm_t_l_a(arm_index)
            V_l += V_l_a
            self.theta_hat += theta_hat_a
    
        print("Calculating theta hat")
        V_l_inv = np.linalg.inv(V_l)
        self.theta_hat = V_l_inv @ self.theta_hat
    
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
    num_steps = 2000000
    k, d, delta = 1000, 60, 0.00001
    A = sample_matrix_in_unit_ball(k, d)
    
    theta_star = uniform_vector_unit_ball(d)
    r = 1 - np.linalg.norm(theta_star)
    cov_matrix = np.eye(d) * (r**2) / (d+2)

    phase_elimination = PhasedElimination(A, delta, theta_star, cov_matrix, r, num_steps)
    regret, phase_lengths, time_to_one_arm, time_to_ucb = phase_elimination.phase_elimination_alg()

    # Plot regret
    worst_case = np.array([np.max(A @ theta_star) - np.min(A @ theta_star)] * num_steps)
    num_phases = len(phase_lengths)
    delta_m = delta / (k * num_phases * (num_phases + 1))
    plot_regret(regret, worst_case, phase_lengths, d, delta_m, cov_matrix, time_to_one_arm, time_to_ucb)

if __name__ == '__main__':
    main()
