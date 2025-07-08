import numpy as np
from utils import uniform_vector_unit_sphere
from distributions.dist import Distribution

class UniformBallDist(Distribution):
    def sample_theta_s(self):
        if np.linalg.norm(self.theta_star) + self.l2_dist_from_sphere > 1:
            raise ValueError("The ball of radius r around theta_star must lie inside the unit ball.")

        # Sample direction uniformly on the unit sphere
        direction = uniform_vector_unit_sphere(self.d)

        return self.theta_star + self.l2_dist_from_sphere * direction

    def get_cov_matrix(self):
        return np.eye(self.d) * (self.l2_dist_from_sphere ** 2) / (self.d + 2)
