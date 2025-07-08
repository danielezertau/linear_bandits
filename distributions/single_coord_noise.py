from utils import one_hot_vector
from distributions.dist import Distribution
import numpy as np

class SingleCoordNoise(Distribution):
    def sample_theta_s(self):
        # Uniform noise in the first coordinate
        low = min(self.theta_star[0] - self.l2_dist_from_sphere, 0)
        high = max(self.theta_star[0] + self.l2_dist_from_sphere, 1)
        noise = np.random.uniform(low=low, high=high, size=1)

        return self.theta_star + noise * one_hot_vector(0, self.d)        

    def get_cov_matrix(self):
        mat = np.zeros((self.d, self.d)) 
        mat[0,0] = (1/12) * ((2 * self.l2_dist_from_sphere) ** 2)
        return mat