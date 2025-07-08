from abc import abstractmethod, ABC


class Distribution(ABC):
    def __init__(self, theta_star, l2_dist_from_sphere):
        self.theta_star = theta_star
        self.l2_dist_from_sphere = l2_dist_from_sphere
        self.d = theta_star.shape[0]

    @abstractmethod
    def sample_theta_s(self):
        pass

    @abstractmethod
    def get_cov_matrix(self):
        pass