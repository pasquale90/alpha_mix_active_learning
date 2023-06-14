import numpy as np
from .strategy import Strategy


class RandomSampling(Strategy):
	def __init__(self, X, Y, idxs_lb, X_val, Y_val, model, args, device, writer):
		super(RandomSampling, self).__init__(X, Y, idxs_lb, X_val, Y_val, model, args, device, writer)

	def query(self, n):
		# randomly selects n out of #self.idxs_lb(== the number of samples in the unlabeled pool)
		return np.random.choice(np.where(self.idxs_lb==0)[0], n, replace=False), None, None, None, None, None
