from distributions.dist import Distribution
import numpy as np

class MultivariateNormal(Distribution):
    def __init__(self, l2_dist_from_sphere, theta_star, cov_matrix):
        super().__init__(theta_star, l2_dist_from_sphere)
        self.cov_matrix = cov_matrix
        
    def sample_theta_s(self):
        return np.random.multivariate_normal(self.theta_star, self.cov_matrix)

    def get_cov_matrix(self):
        return self.cov_matrix
