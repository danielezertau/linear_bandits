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

def find_optimal_design(X):
    m, d = X.shape
    # 2) Define weights on the simplex
    w = cp.Variable(m, nonneg=True)

    # 3) Form the information matrix M = sum_i w_i x_i x_i^T
    M = sum(w[i] * np.outer(X[i], X[i]) for i in range(m))

    # 4) Set up D-optimal objective and simplex constraint
    obj = cp.Maximize(cp.log_det(M))
    constraints = [cp.sum(w) == 1]

    # 5) Solve as an SDP
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK, verbose=True)  # SCS, MOSEK, CVXOPT, etc. :contentReference[oaicite:4]{index=4}

    # 6) Extract support of the design
    w_val = w.value
    support = np.where(w_val > 1e-5)[0]
    return support, w_val

def get_t_l_a(a, pi, d, eps, k, l, delta):
    return ((2 * d * pi[a]) / (math.pow(eps, 2))) * math.log(k * l * (l  +1) / delta)

def play_arm(arm, theta_star):
    return theta_star @ arm + np.random.normal(0, 1, 1)

def should_eliminate_arm(theta_hat, A_l, arm, eps_l):
    max_gap = 0
    for b in A_l:
        curr_gap = (theta_hat.T @ (b - arm)).item()
        if curr_gap > max_gap:
            max_gap = curr_gap
    if max_gap > 2 * eps_l:
        return True
    return False

def phase(A, l, delta, theta_star):
    k = A.shape[0]
    d = A.shape[1]
    support, pi_l = find_optimal_design(A)
    eps_l = math.pow(2, -l)
    V_l = np.zeros((d, d))
    theta_hat = np.zeros((d, 1))
    phase_rewards = []
    for a in support:
        arm = np.expand_dims(A[a], axis=1)
        T_l_a = math.ceil(get_t_l_a(a, pi_l, d, eps_l, k, l, delta))
        V_l += T_l_a * (arm @ arm.T)
        for _ in range(T_l_a):
            reward = play_arm(arm, theta_star)
            phase_rewards.append(reward)
            theta_hat += arm * reward
    theta_hat = np.linalg.inv(V_l) @ theta_hat

    vec_fn = np.vectorize(
        lambda A_arm: should_eliminate_arm(theta_hat, A, A_arm, eps_l),
        otypes=[bool],
        signature='(d)->()'
    )
    return A[~vec_fn(A)], phase_rewards

def phase_elimination_alg(A, delta, theta_star, num_steps):
    l = 1
    A_l = A
    d = A.shape[1]
    total_rewards = []
    while l <= num_steps:
        A_l, phase_rewards = phase(A_l, l, delta, theta_star)
        total_rewards += phase_rewards
        if len(A_l) < d:
            return A_l, total_rewards
        l += 1
    return [], total_rewards

def plot_cumulative_sum(data, d, k):
    # Plot sqrt
    n_values = np.arange(1, len(data))
    f_n = np.sqrt(d * n_values * np.log(k))
    plt.plot(n_values, 10 * f_n, color='green', label='10 * sqrt(dnlog(k))')

    # Plot regret
    regret = np.cumsum(data)
    plt.plot(regret, color='blue', label='regret')
    plt.title('Regret as a function of time')
    plt.xlabel('Timestep')
    plt.ylabel('Regret')
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    num_steps = 10
    k, d, delta = 1000, 5, 0.00001
    A = sample_in_ball(k, d)
    theta_star = np.random.uniform(0, 1, d)
    a_star = A[np.argmax(A @ theta_star)]
    optimal_reward = theta_star.T @ a_star

    result, rewards = phase_elimination_alg(A, delta, theta_star, num_steps)
    
    
    regret = optimal_reward - np.array(rewards)
    plot_cumulative_sum(regret, d, k)

    print(f"Optimal arm: {a_star}")
    print(f"\n\nResult: {result}")

if __name__ == '__main__':
    main()
