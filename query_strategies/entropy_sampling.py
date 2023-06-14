import numpy as np
import torch
from .strategy import Strategy

class EntropySampling(Strategy):
	def __init__(self, X, Y, idxs_lb, X_val, Y_val, model, args, device, writer):
		super(EntropySampling, self).__init__(X, Y, idxs_lb, X_val, Y_val, model, args, device, writer)

	def query(self, n):
		import pdb
		pdb.set_trace()
		# n = *n_query, defined in the arguments

		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb] # == list of the form [0,1,.....,#of_samples_in_the_unlabeled_pool]
		'''
		in the 
		probs, embeddings = self.predict_prob_embed(self.X[idxs_unlabeled], self.Y[idxs_unlabeled]) ,
			eval=True

		'''
		probs, embeddings = self.predict_prob_embed(self.X[idxs_unlabeled], self.Y[idxs_unlabeled]) # eval=True, 

		# Calculate Entropy --> https://en.wikipedia.org/wiki/Entropy_(information_theory) 
		log_probs = torch.log(probs)
		U = (probs*log_probs).sum(1)
		selected = U.sort()[1][:n]
		return idxs_unlabeled[selected], embeddings, probs.max(1)[1], probs, selected, None
