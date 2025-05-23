import math
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

def sample_in_ball(k, d):
    # 1) random directions
    V = np.random.randn(k, d)
    V /= np.linalg.norm(V, axis=1, keepdims=True)

    # 2) random radii with PDF ~ r^{d-1}
    R = np.random.rand(k) ** (1.0/d)

    return V * R[:, None]

def find_optimal_design(A, variances):
    k, d = A.shape
    
    # 2) Define weights on the simplex
    w = cp.Variable(k, nonneg=True)

    # 3) Form the information matrix M = sum_i w_i x_i x_i^T
    if k < d:
        return np.arange(k), (np.ones(k) / k) * variances

    M = sum(w[i] * np.outer(A[i] * variances[i], A[i]) for i in range(k))

    # 4) Set up D-optimal objective and simplex constraint
    obj = cp.Maximize(cp.log_det(M))
    constraints = [cp.sum(w) == 1]

    # 5) Solve as an SDP
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)  # SCS, MOSEK, CVXOPT, etc. :contentReference[oaicite:4]{index=4}

    # 6) Extract support of the design
    w_val = w.value
    support = np.where(w_val > 1e-5)[0]
    return support, w_val

def get_t_l_a(pi_arm, d, eps, k, l, delta, arm_variance):
    return ((2 * d * pi_arm) / (math.pow(eps, 2))) * (math.log(k * l * (l  +1) / delta)) * arm_variance

def sample_theta_s(theta_star, cov_matrix):
    return np.random.multivariate_normal(theta_star, cov_matrix)

def play_arm(arm, theta_star, cov_matrix):
    theta_s = sample_theta_s(theta_star, cov_matrix)
    return theta_s @ arm

def should_eliminate_arm(theta_hat, A_l, arm, eps_l):
    max_gap = 0
    for b in A_l:
        curr_gap = (theta_hat.T @ (b - arm)).item()
        if curr_gap > max_gap:
            max_gap = curr_gap
    if max_gap > 2 * eps_l:
        return True
    return False

def play_arm_t_l_a(arm_index, arm, pi_arm, eps_l, l, delta, A, theta_star, cov_matrix, remaining_steps, arm_variance):
    arm_rewards = []
    k, d = A.shape
    V_l = np.zeros((d, d))
    theta_hat = np.zeros((d, 1))
    arm = np.expand_dims(arm, axis=1)
    T_l_a = math.ceil(get_t_l_a(pi_arm, d, eps_l, k, l, delta, arm_variance))
    V_l += T_l_a * (arm @ arm.T)

    print(f"Running arm {arm_index} for {T_l_a} steps")
    for _ in range(min(T_l_a, remaining_steps)):
        reward = play_arm(arm, theta_star, cov_matrix)
        arm_rewards.append(reward)
        theta_hat += arm * reward
    return V_l, theta_hat, arm_rewards

def phase(A, l, delta, theta_star, cov_matrix, variances, remaining_steps):
    k, d = A.shape

    support, pi_l = find_optimal_design(A, variances)
    eps_l = math.pow(2, -l)
    V_l = np.eye(d)
    theta_hat = np.zeros((d, 1))
    phase_rewards = []
    for arm_index in support:
        if remaining_steps <= 0:
            break
        V_l_a, theta_hat_a, phase_rewards_a = play_arm_t_l_a(arm_index, A[arm_index],
                                                             pi_l[arm_index], eps_l, l, delta, A, theta_star,
                                                             cov_matrix, remaining_steps, variances[arm_index])
        remaining_steps -= len(phase_rewards_a)
        phase_rewards += phase_rewards_a
        V_l += V_l_a
        theta_hat += theta_hat_a

    print("Calculating theta hat")
    theta_hat = np.linalg.inv(V_l) @ theta_hat

    vec_fn = np.vectorize(
        lambda A_arm: should_eliminate_arm(theta_hat, A, A_arm, eps_l),
        otypes=[bool],
        signature='(d)->()'
    )
    return A[~vec_fn(A)], variances[~vec_fn(A)], phase_rewards

def phase_elimination_alg(A, delta, theta_star, cov_matrix, variances, num_steps):
    time_to_one_arm = 0
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
        A_l, variances_l, phase_rewards = phase(A_l, l, delta, theta_star, cov_matrix, variances_l, remaining_steps)
        phase_lengths.append(len(phase_rewards))
        remaining_steps -= len(phase_rewards)
        if A_l.shape[0] == 1 and time_to_one_arm == 0:
            time_to_one_arm = num_steps - remaining_steps
        total_rewards += phase_rewards
        l += 1
    return total_rewards, phase_lengths, time_to_one_arm

def plot_cumulative_sum(data, phase_lengths, d, k, time_to_one_arm):
    # Plot sqrt
    n_values = np.arange(1, len(data))
    f_n = np.sqrt(d * n_values * np.log(k))
    plt.plot(n_values, 8 * f_n, color='green', label='8 * sqrt(dnlog(k))')

    # Plot regret
    regret = np.cumsum(data)
    plt.plot(regret, color='blue', label='regret')
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

def random_cov_unit_ball_dirichlet(n):
    """
    Generate an n×n PSD covariance matrix Σ with trace ≤ 1
    via Dirichlet + Uniform scaling of eigenvalues and
    random orthogonal eigenvectors.
    """
    # 1) Sample v ~ Dirichlet(1,...,1) so sum(v)=1
    v = np.random.dirichlet(alpha=np.ones(n))            # :contentReference[oaicite:4]{index=4}

    # 2) Draw u ~ Uniform(0,1) to scale the trace
    u = np.random.uniform(0.0, 1.0)                      # :contentReference[oaicite:5]{index=5}

    # 3) Compute eigenvalues λ_i = u * v_i (sum ≤ 1)
    lambdas = u * v

    # 4) Generate random orthogonal matrix Q via QR of Gaussian matrix
    A = np.random.randn(n, n)
    Q, _ = np.linalg.qr(A)                               # :contentReference[oaicite:6]{index=6}

    # 5) Assemble Σ = Q diag(lambdas) Q^T
    Sigma = Q @ np.diag(lambdas) @ Q.T

    return Sigma

def main():
    num_steps = 50000
    k, d, delta = 1000, 5, 0.00001
    A = sample_in_ball(k, d)
    
    theta_star = np.random.uniform(0, 1, d)

    cov_matrix = random_cov_unit_ball_dirichlet(d)

    variances = np.array([arm.T @ cov_matrix @ arm for arm in A])
    rewards, phase_lengths, time_to_one_arm = phase_elimination_alg(A, delta, theta_star, cov_matrix, variances, num_steps)

    a_star = A[np.argmax(A @ theta_star)]
    optimal_reward = theta_star.T @ a_star
    
    regret = optimal_reward - np.array(rewards)
    plot_cumulative_sum(regret, phase_lengths, d, k, time_to_one_arm)

if __name__ == '__main__':
    main()
