import math
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
from ucb import UCB1, run_ucb


def uniform_vector_unit_sphere(d):
    # Sample a random direction by drawing standard normals and normalizing
    direction = np.random.uniform(low=-1.0, high=1.0, size=d)
    direction /= np.linalg.norm(direction)
    return direction

def uniform_vector_unit_ball(d):
    # Sample radius uniformly in [0,1]
    radius = np.random.uniform(size=1)

    return radius * uniform_vector_unit_sphere(d)

def sample_matrix_in_unit_ball(k, d):
    """
    Sample a k x d matrix where each row is uniformly drawn from the unit ball in R^d.
    """
    return np.array([uniform_vector_unit_ball(d) for _ in range(k)])

def get_max_V_inv_norm(A, V_inv):
    AV_inv = A @ V_inv

    scores = np.sum(AV_inv * A, axis=1)

    idx = np.argmax(scores)

    return idx, scores[idx]


def frank_wolfe(A):
    print("Finding approximate optimal design")
    k, d = A.shape
    pi = np.ones(k) / k
    
    for _ in range(10 * math.ceil(d * math.log2(math.log2(k)) + d)):
        V_pi = sum(pi[i] * np.outer(A[i] , A[i]) for i in range(k))
        a_max, a_max_norm = get_max_V_inv_norm(A, np.linalg.inv(V_pi)) 
        gamma = ((1/d) * a_max_norm - 1) / (a_max_norm - 1)

        a_max_one_hot = np.zeros(k)
        a_max_one_hot[a_max] = 1

        pi = (1 - gamma) * pi + gamma * a_max_one_hot

    support = np.where(pi > 1e-4)[0]
    print(f"Support size is {len(support)}")
    return support, pi

def find_optimal_design(A):
    print("Finding optimal design")
    k, d = A.shape
    
    # 2) Define weights on the simplex
    w = cp.Variable(k, nonneg=True)

    # 3) Form the information matrix M = sum_i w_i x_i x_i^T
    M = sum(w[i] * np.outer(A[i] , A[i]) for i in range(k))

    # 4) Set up D-optimal objective and simplex constraint
    obj = cp.Maximize(cp.log_det(M))
    constraints = [cp.sum(w) == 1]

    # 5) Solve as an SDP
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)  # SCS, MOSEK, CVXOPT, etc. :contentReference[oaicite:4]{index=4}

    # 6) Extract support of the design
    w_val = w.value
    support = np.where(w_val > 1e-5)[0]
    print(f"Support size is {len(support)}")
    return support, w_val

def get_t_l_a(pi_arm, d, eps, k, l, delta, arm_variance):
    return math.ceil(((16 * d * pi_arm) / (math.pow(eps, 2))) * (math.log(k * l * (l  +1) / delta)) * arm_variance)

def sample_theta_s(theta_star, cov_matrix):
    return np.random.multivariate_normal(theta_star, cov_matrix)

def sample_shifted_uniform(theta_star, r, d):
    theta_star = np.array(theta_star)
    if np.linalg.norm(theta_star) + r > 1:
        raise ValueError("The ball of radius r around mu must lie inside the unit ball.")

    # Sample direction uniformly on the unit sphere
    direction = uniform_vector_unit_sphere(d)
    
    return theta_star + r * direction

def play_arm(arm, theta_star, r):
    theta_s = sample_shifted_uniform(theta_star, r, arm.shape[0])
    return (theta_s @ arm).item(), (theta_star @ arm).item()

def should_eliminate_arm(theta_hat, A_l, arm, eps_l):
    max_gap = 0
    for b in A_l:
        curr_gap = (theta_hat.T @ (b - arm)).item()
        if curr_gap > max_gap:
            max_gap = curr_gap
    if max_gap > 2 * eps_l:
        return True
    return False

def play_arm_t_l_a(arm_index, arm, pi_arm, eps_l, l, delta, A, theta_star, r, remaining_steps, arm_variance):
    arm_expected_rewards = []
    k, d = A.shape
    V_l = np.zeros((d, d))
    theta_hat = np.zeros((d, 1))
    arm = np.expand_dims(arm, axis=1)
    T_l_a = get_t_l_a(pi_arm, d, eps_l, k, l, delta, arm_variance)
    V_l += T_l_a * (arm @ arm.T / arm_variance)

    print(f"Running arm {arm_index} for {T_l_a} steps")
    for _ in range(min(T_l_a, remaining_steps)):
        reward, expected_reward = play_arm(arm, theta_star, r)
        arm_expected_rewards.append(expected_reward)
        theta_hat += (arm / arm_variance) * reward
    return V_l, theta_hat, arm_expected_rewards

def phase(A, l, delta, theta_star, r, variances, remaining_steps):
    k, d = A.shape
    support, pi_l = find_optimal_design(A)
    eps_l = math.pow(2, -l)
    V_l = np.zeros((d, d))
    variances_hat = np.maximum(variances, eps_l)
    theta_hat = np.zeros((d, 1))
    phase_rewards = []
    for arm_index in support:
        if remaining_steps <= 0:
            break
        V_l_a, theta_hat_a, phase_rewards_a = play_arm_t_l_a(arm_index, A[arm_index],
                                                             pi_l[arm_index], eps_l, l, delta, A, theta_star,
                                                             r, remaining_steps, variances_hat[arm_index])
        remaining_steps -= len(phase_rewards_a)
        phase_rewards += phase_rewards_a
        V_l += V_l_a
        theta_hat += theta_hat_a

    print("Calculating theta hat")
    V_l_inv = np.linalg.inv(V_l)
    theta_hat = V_l_inv @ theta_hat

    print("Eliminating arms")
    vec_fn = np.vectorize(
        lambda A_arm: should_eliminate_arm(theta_hat, A, A_arm, eps_l),
        otypes=[bool],
        signature='(d)->()'
    )
    new_A = A[~vec_fn(A)]
    print(f"Eliminated {k - new_A.shape[0]} arms")
    return new_A, variances[~vec_fn(A)], phase_rewards

def phase_elimination_alg(A, delta, theta_star, r, variances, num_steps):
    time_to_one_arm = 0
    k, d = A.shape
    l = 1
    A_l = A
    variances_l = variances
    total_rewards = []
    remaining_steps = num_steps
    phase_lengths = []
    while remaining_steps > 0:
        print(f"\n\nRunning phase {l}"
              f"\nNumber of arms: {A_l.shape[0]}"
              f"\nRemaining steps: {remaining_steps}")
        print(f"Optimal arm eliminated: {A[np.argmax(A @ theta_star)] not in A_l}")
        k_l = A_l.shape[0]
        delta_l = delta / (k * l * (l+1))

        if k_l < d:
            # Play UCB algo
            print("Running UCB")
            reward_function = lambda a: play_arm(A[a], theta_star, r)
            ucb = UCB1(n_arms=k_l, reward_function=reward_function)
            phase_rewards = run_ucb(ucb, remaining_steps)
        else:
            A_l, variances_l, phase_rewards = phase(A_l, l, delta_l, theta_star, r, variances_l, remaining_steps)
        phase_lengths.append(len(phase_rewards))
        remaining_steps -= len(phase_rewards)
        if A_l.shape[0] == 1 and time_to_one_arm == 0:
            time_to_one_arm = num_steps - remaining_steps
        total_rewards += phase_rewards
        l += 1
    return total_rewards, phase_lengths, time_to_one_arm

def plot_cumulative_sum(data, phase_lengths, d, delta, cov_matrix, time_to_one_arm):
    # Plot sqrt
    n_values = np.arange(1, len(data))
    f_n = 8 * np.sqrt(d * n_values * math.log2(1/delta) * np.linalg.trace(cov_matrix) * np.log2(n_values))
    plt.plot(n_values,  f_n, color='green', label='Regret Bound')

    # Plot regret
    regret = np.cumsum(data)
    plt.plot(regret, color='blue', label='Actual Regret')
    plt.title('Regret as a function of time')
    plt.xlabel('Timestep')
    plt.ylabel('Regret')
    plt.grid(True)
    
    # Add vertical lines for phases
    cumpos = np.cumsum(phase_lengths)
    for x0 in cumpos:
        plt.axvline(x=x0, linestyle='--', linewidth=1, color='tab:orange')

    # Add vertical line for time to one arm
    if time_to_one_arm > 0:
        plt.axvline(x=time_to_one_arm, linestyle='--', linewidth=1, color='green')
        plt.gca().text(
            time_to_one_arm,           # x in data coordinates
            0.95,                       # y in axes-fraction (0 at bottom, 1 at top)
            'one-arm time',            # your annotation text
            color='green',
            ha='right',                 # horizontal alignment
            va='top',                   # vertical alignment
            rotation=90,
            transform=plt.gca().get_xaxis_transform()  # blend data x with axes y :contentReference[oaicite:1]{index=1}
        )



    plt.legend()
    plt.show()

def main():
    num_steps = 1000000
    k, d, delta = 1000, 50, 0.00001
    A = sample_matrix_in_unit_ball(k, d)
    
    theta_star = uniform_vector_unit_ball(d)
    r = 1 - np.linalg.norm(theta_star)
    cov_matrix = np.eye(d) * (r**2) / (d+2)

    variances = np.array([arm.T @ cov_matrix @ arm for arm in A])
    rewards, phase_lengths, time_to_one_arm = phase_elimination_alg(A, delta, theta_star, r, variances, num_steps)

    a_star = A[np.argmax(A @ theta_star)]
    optimal_reward = theta_star.T @ a_star
    
    regret = optimal_reward - np.array(rewards)
    num_phases = len(phase_lengths)
    plot_cumulative_sum(regret, phase_lengths, d, delta / (k * num_phases * (num_phases + 1)), cov_matrix, time_to_one_arm)

if __name__ == '__main__':
    main()
