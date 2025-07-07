import math
import matplotlib.pyplot as plt
import cvxpy as cp
import numpy as np

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

def vertical_line_with_text(x, color, text):
    plt.axvline(x=x, linestyle='--', linewidth=1, color=color)
    plt.gca().text(
        x,
        0.95,
        text,
        color=color,
        ha='right',
        va='top',
        rotation=90,
        transform=plt.gca().get_xaxis_transform()
    )


def plot_regret(pseudo_regret, worst_case, phase_lengths, d, delta, cov_matrix, time_to_one_arm, time_to_ucb):
    # Plot sqrt
    T_values = np.arange(1, len(pseudo_regret))
    f_n = 2 * math.sqrt(2) * np.sqrt(d * T_values * math.log2(1/delta) * np.linalg.trace(cov_matrix) * np.log2(T_values))
    plt.plot(f_n, color='green', label='Regret Bound')

    # Plot worst case
    worst_case_regret = np.cumsum(worst_case)
    plt.plot(worst_case_regret, color='red', label='Worst Case Regret')

    # Plot regret
    regret = np.cumsum(pseudo_regret)
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
        vertical_line_with_text(time_to_one_arm, color='orange', text="One Arm")

    # Add vertical line for time to UCB
    if time_to_ucb > 0:
        vertical_line_with_text(time_to_ucb, color='orange', text="UCB")

    plt.legend()
    plt.show()
