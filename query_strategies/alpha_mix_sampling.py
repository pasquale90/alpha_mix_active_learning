import copy
import math

import numpy as np
from torch.autograd import Variable

from .strategy import Strategy
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
import query_strategies.research as research

class AlphaMixSampling(Strategy):
	def __init__(self, X, Y, idxs_lb, X_val, Y_val, model, args, device, writer):
		super(AlphaMixSampling, self).__init__(X, Y, idxs_lb, X_val, Y_val, model, args, device, writer)

	def query(self, n, idxs_prohibited=None):

		# idxs_prohibited=None at the first round

		self.query_count += 1 # increments query_count by 1

		''' idxs, idxs_unlabeled

			idxs : the number of samples that are used for training. These are separated from the unlabeled pool.
				type:bool
				len:60.000
				shape:(60.000,)
				(Pdb) len(np.where(idxs==True)[0]) ---> 100  === n_init_lb (bash arg) 
				
			idxs_unlabeled : the labeled samples that are considered to be unlabeled. They are separated from the dataset
				array 
				type:int64
				len:59.900
				shape:(59.900,)

		'''
		idxs = self.idxs_lb if idxs_prohibited is None else (self.idxs_lb + idxs_prohibited) 
		idxs_unlabeled = np.arange(self.n_pool)[~idxs]

		''' predict_prob_embed FOR THE UNLABELED SAMPLES

        input:
            X: unlabeled images ,torch tensor of shape (samples,width,height) || (samples,height,width) of type torch.uint8
            Y: actual labels    ,torch tensor of shape (samples) of type torch.int64 -----------------------> these labels are used for validating the alphamix-framework
        output:
            ulb_probs : embeddings that are passed through the fully connected + softmax, 
                    torch tensor of shape (samples,{#classes}}), type torch.float32
            org_ulb_embedding : embeddings exctracted from the backbone. 
                        torch tensor of shape (samples,{train_params['emb_size']}), type torch.float32
            
        '''
		ulb_probs, org_ulb_embedding = self.predict_prob_embed(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])

		probs_sorted, probs_sort_idxs = ulb_probs.sort(descending=True) # sort the unlabeled probabilities
		pred_1 = probs_sort_idxs[:, 0] 									# get the maximum probabilities for all unlabeled samples

		lb_probs, org_lb_embedding = self.predict_prob_embed(self.X[self.idxs_lb], self.Y[self.idxs_lb]) # predict_prob_embed FOR THE LABELED SAMPLES

		ulb_embedding = org_ulb_embedding 	# copy unlabeled embeddings
		lb_embedding = org_lb_embedding 	# copy labeled embeddings

		unlabeled_size = ulb_embedding.size(0) # Number of unlabeled samples
		embedding_size = ulb_embedding.size(1) # Number of dims of the feature descriptor

		min_alphas = torch.ones((unlabeled_size, embedding_size), dtype=torch.float) 	# torch tensor of ones , of shape (#numsamples,featureVec_size)
		candidate = torch.zeros(unlabeled_size, dtype=torch.bool)						# torch tensor of zeros, of shape(#numsamples)
		
		if self.args.alpha_closed_form_approx:											# if alpha_closed_form_approx scenario is chosen, then
			var_emb = Variable(ulb_embedding, requires_grad=True).to(self.device)			# wrap unlabeled_features with Variable for computing gradients
			out, _ = self.model.clf(var_emb, embedding=True)								# pass unlabeled features through the model - these come without the Softmax.
			loss = F.cross_entropy(out, pred_1.to(self.device))								# calculate the loss between 
			
			"""grads = torch.autograd.grad(loss, var_emb)[0].data.cpu()
				Compute the grads for the each unlabeled sample.
				The differential equation is between :
					unlabeled features passed throught the model, and
					unlabeled features passed throught the same model with the difference that these were not passed through all the linear layers as before.			
			"""
			grads = torch.autograd.grad(loss, var_emb)[0].data.cpu()						# COPMPUTE THE GRADIENTS W.R.T. the unlabeled embeddings?
																							# https://pytorch.org/docs/stable/_modules/torch/autograd.html#grad
																							# https://stackoverflow.com/questions/69148622/difference-between-autograd-grad-and-autograd-backward

																							# 					https://realpython.com/python-mutable-vs-immutable-types/ : mutable == in_place 
																						# IS THAT ASSUMPTION ACCURATE?--> model's layers are not freezed anymore. The weights and biases of the model will get updated in a forward pass

															# https://github.com/AminParvaneh/alpha_mix_active_learning/issues/10
			del loss, var_emb, out
		else:																			# scenario of alpha_learnable
			grads = None

		alpha_cap = 0.
		while alpha_cap < 1.0:
			alpha_cap += self.args.alpha_cap											# default vaj self.args.alpha_cap=0.03125
			
			"""
			We consider interpolating an unlabelled instance with all
			the anchors representing different classes to uncover the
			sufficiently distinct features by considering how the model’s
			prediction changes.
			"""
			# returns potential candidates and minimum values for alpha.
			tmp_pred_change, tmp_min_alphas = \
						self.find_candidate_set(	
							lb_embedding, ulb_embedding, pred_1, ulb_probs, alpha_cap=alpha_cap,
							Y=self.Y[self.idxs_lb],
							grads=grads)

			is_changed = min_alphas.norm(dim=1) >= tmp_min_alphas.norm(dim=1)

			min_alphas[is_changed] = tmp_min_alphas[is_changed]
			candidate += tmp_pred_change

			print('With alpha_cap set to %f, number of inconsistencies: %d' % (alpha_cap, int(tmp_pred_change.sum().item())))

			if candidate.sum() > n:
				break

		if candidate.sum() > 0:
			print('Number of inconsistencies: %d' % (int(candidate.sum().item())))

			print('alpha_mean_mean: %f' % min_alphas[candidate].mean(dim=1).mean().item())
			print('alpha_std_mean: %f' % min_alphas[candidate].mean(dim=1).std().item())
			print('alpha_mean_std %f' % min_alphas[candidate].std(dim=1).mean().item())

			self.writer.add_scalar('stats/candidate_set_size', candidate.sum().item(), self.query_count)
			self.writer.add_scalar('stats/alpha_mean_mean', min_alphas[candidate].mean(dim=1).mean().item(), self.query_count)
			self.writer.add_scalar('stats/alpha_std_mean', min_alphas[candidate].mean(dim=1).std().item(), self.query_count)
			self.writer.add_scalar('stats/alpha_mean_std', min_alphas[candidate].std(dim=1).mean().item(), self.query_count)

			c_alpha = F.normalize(org_ulb_embedding[candidate].view(candidate.sum(), -1), p=2, dim=1).detach()
# IN THIS PARTH OF THE CODE WE DO OUR RESEARCH
			# import pdb
			# pdb.set_trace()
			# selected_idxs = self.sample(min(n, candidate.sum().item()), feats=c_alpha)							# clustering
			# selected_idxs = self.sample_newMethod(min(n, candidate.sum().item()), feats=c_alpha)							# clustering
			selected_idxs = research.sample_newMethod1(min(n, candidate.sum().item()), feats=c_alpha)
			lenQuery=len(selected_idxs)
			# import pdb
			# pdb.set_trace()
			# selected_idxs = self.sample_DBSCAN(min(n, candidate.sum().item()), feats=c_alpha)							# clustering
			u_selected_idxs = candidate.nonzero(as_tuple=True)[0][selected_idxs]
			selected_idxs = idxs_unlabeled[candidate][selected_idxs]
		else:
			selected_idxs = np.array([], dtype=int)

		if len(selected_idxs) < lenQuery:
			remained = lenQuery - len(selected_idxs)
			idx_lb = copy.deepcopy(self.idxs_lb)
			idx_lb[selected_idxs] = True
			selected_idxs = np.concatenate([selected_idxs, np.random.choice(np.where(idx_lb == 0)[0], remained)])

			print('picked %d samples from RandomSampling.' % (remained))

		return lenQuery,np.array(selected_idxs), ulb_embedding, pred_1, ulb_probs, u_selected_idxs, idxs_unlabeled[candidate]
	
############################################################################################################################### is find_alpha called anywhere?????????
	def find_alpha(self):																							
		assert self.args.alpha_num_mix <= self.args.n_label - (
			0 if self.args.alpha_use_highest_class_mix else 1), 'c_num_mix should not be greater than number of classes'

		self.query_count += 1

		idxs_unlabeled = np.arange(self.n_pool)[self.idxs_lb]

		ulb_probs, ulb_embedding = self.predict_prob_embed(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])

		probs_sorted, probs_sort_idxs = ulb_probs.sort(descending=True)
		pred_1 = probs_sort_idxs[:, 0]
		gt_lables = self.Y[idxs_unlabeled]
		preds = pred_1 == gt_lables

		ulb_embedding = ulb_embedding[preds]
		ulb_probs = ulb_probs[preds]
		pred_1 = pred_1[preds]

		lb_embedding = self.get_embedding(self.X[self.idxs_lb], self.Y[self.idxs_lb])

		alpha_cap = 0.
		for i in range(self.args.alpha_alpha_scales if self.args.alpha_alpha_growth_method == 'exponential' else int(pow(2, self.args.alpha_alpha_scales) - 1)):
			if self.args.alpha_alpha_growth_method == 'exponential':
				alpha_cap = self.args.alpha_cap / (pow(2, self.args.alpha_alpha_scales - i - 1))
			else:
				#alpha_cap *= float(n * (1 if self.args.alpha_max_changes <= 0 else self.args.alpha_max_changes)) / pred_change.sum().item()
				alpha_cap += self.args.alpha_cap / (pow(2, self.args.alpha_alpha_scales - 1))

			tmp_pred_change, tmp_pred_change_sum, tmp_min_alphas, tmp_min_added_feats, tmp_cf_probs, _, tmp_min_mixing_labels, tmp_min_cf_feats = \
				self.find_candidate_set(
					lb_embedding, ulb_embedding, pred_1, ulb_probs, probs_sort_idxs, alpha_cap=alpha_cap,
					Y=self.Y[self.idxs_lb])

			if tmp_pred_change.sum() > 0:
				print('selected alpha_max %f' % alpha_cap)
				self.writer.add_scalar('stats/alpha_cap', alpha_cap, self.query_count)

				return alpha_cap

		print('no labelled sample change!!!')
		return 0.5

	def find_candidate_set(self, lb_embedding, ulb_embedding, pred_1, ulb_probs, alpha_cap, Y, grads):
		'''
		input:
			lb_embedding 	: the labeled features
			ulb_embedding 	: the unlabeled features
			pred_1			: the MAXIMUM classification scores of all unlabeled samples (Argmax is applied)
			ulb_probs		: the UNSORTED classification scores of all unlabeled samples (for all available classes)
			alpha_cap 		: alpha_cap
			Y
			grads ??? ==None

		output:
			pred_change 	: the potential candidates out of the unlabeled pool
			min_alphas		: the minimum values of alpha.
		'''
		unlabeled_size = ulb_embedding.size(0)										# Number of unlabeled samples
		embedding_size = ulb_embedding.size(1)										# Number of dims of the feature descriptor

		min_alphas = torch.ones((unlabeled_size, embedding_size), dtype=torch.float)
		pred_change = torch.zeros(unlabeled_size, dtype=torch.bool)

		if self.args.alpha_closed_form_approx:										# if closed_form_approx method is chosen, then
			alpha_cap /= math.sqrt(embedding_size)										# maximum value for alpha is divided with the square root of embedding size
			grads = grads.to(self.device)												# model's layers are not freezed
			
		for i in range(self.args.n_label):														# for each class

			emb = lb_embedding[Y == i]																# emb = embeddings of labeled samples for class==i									
			if emb.size(0) == 0:	
				emb = lb_embedding																	# ????										
			anchor_i = emb.mean(dim=0).view(1, -1).repeat(unlabeled_size, 1)						# anchor of class i = the mean of emb (that is the mean of all labeled features for that specific class)
																									# here the ".repeat(unlabeled_size, 1)" makes a copy of the anchor_i in #num_unlabeled_samples positions

			if self.args.alpha_closed_form_approx:													# scenario of alpha_closed_form_approx
				embed_i, ulb_embed = anchor_i.to(self.device), ulb_embedding.to(self.device)		
				alpha = self.calculate_optimum_alpha(alpha_cap, embed_i, ulb_embed, grads)			# dual norm calculation of the alpha --> refers to eq (5)
				embedding_mix = (1 - alpha) * ulb_embed + alpha * embed_i							# the mix
				out, _ = self.model.clf(embedding_mix, embedding=True)								# 
				out = out.detach().cpu()
				alpha = alpha.cpu()

				pc = out.argmax(dim=1) != pred_1

				# import pdb
				# pdb.set_trace()

			else:																					# scenario of alpha_learnable	--> SKILLPOINT find_candidate_set
				alpha = self.generate_alpha(unlabeled_size, embedding_size, alpha_cap)				# generate_alpha creates an α value for each unlabeled feature vector, in the range [1e-8, alpha_cap]. Both mean & std of all α == alpha_cap/2

				if self.args.alpha_opt:
					alpha, pc = self.learn_alpha(ulb_embedding, pred_1, anchor_i, alpha, alpha_cap,	# ARGS:alpha carries the minimum a, and pc holds all the potential candidates, OUT: alpha is defined(learned),pc holds all the inconsistencies
												 log_prefix=str(i))
				else:
					embedding_mix = (1 - alpha) * ulb_embedding + alpha * anchor_i
					out, _ = self.model.clf(embedding_mix.to(self.device), embedding=True)
					out = out.detach().cpu()

					pc = out.argmax(dim=1) != pred_1
					
					# find the inconsistencies - get the classes
					
				# import pdb
				# pdb.set_trace()

			torch.cuda.empty_cache()
			self.writer.add_scalar('stats/inconsistencies_%d' % i, pc.sum().item(), self.query_count)


			alpha[~pc] = 1.
			pred_change[pc] = True
			is_min = min_alphas.norm(dim=1) > alpha.norm(dim=1)
			min_alphas[is_min] = alpha[is_min]

			self.writer.add_scalar('stats/inconsistencies_%d' % i, pc.sum().item(), self.query_count)
			
		return pred_change, min_alphas

	def calculate_optimum_alpha(self, eps, lb_embedding, ulb_embedding, ulb_grads):
		'''
		input:
			eps 			: the maximum value of alpha which is defined differently in the alpha_closed_form_approx mode
			lb_embedding	: the class-specific anchor
			ulb_embedding	: the unlabeled embeddings
			ulb_grads		: the gradients computed for the backprop 
		'''
		z = (lb_embedding - ulb_embedding) #* ulb_grads									# (z*-zu) : element-wise subtraction between anchor of a specific class with the feature of the unlabeled sample. Do it for all unlabeled samples.
		alpha = (eps * z.norm(dim=1) / ulb_grads.norm(dim=1)).unsqueeze(dim=1).repeat(1, z.size(1)) * ulb_grads / (z + 1e-8)	# eq (5) 

		# 				eps * z.norm(dim=1) (
      	# 			/ulb_grads.norm(dim=1)
		# 		)
		# 		.unsqueeze(dim=1).repeat(1, z.size(1)) 
		# 	* ulb_grads 
		# /(z + 1e-8)
		# alpha_cap, embed_i, ulb_embed, grads)
		# import pdb
		# pdb.set_trace()
		return alpha
	
	def sample_DBSCAN(self, n, feats):
		'''
		input:
			n		: the number of samples requested
			feats	: the candidate feature embeddings
		
		N clusters are made. 
		The feature that is closer to the center of each cluster is selected.
		
		output:
			n number of samples out of the candidates.
		'''
		
		
		''' HDBSCAN
		from sklearn.cluster import DBSCAN

		import hdbscan
		from sklearn.datasets import make_blobs

		feats = feats.numpy()
		
		clustering = DBSCAN(eps=0.4, min_samples=1).fit(feats) # eps=0.5
		labels=clustering.labels_
		Nteams=len(np.unique(labels))
		print(f"{Nteams} num teams")

		clusterer = hdbscan.HDBSCAN(min_cluster_size=60, min_samples=1).fit(feats)
		hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True).fit(feats)
		import pdb
		pdb.set_trace()
		'''


		# cluster_learner = KMeans(n_clusters=n)
		# cluster_learner.fit(feats)

		# cluster_idxs = cluster_learner.predict(feats)
		# centers = cluster_learner.cluster_centers_[cluster_idxs]
		# dis = (feats - centers) ** 2
		# dis = dis.sum(axis=1)
		# return np.array(
		# 	[np.arange(feats.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in range(n) if
		# 	 (cluster_idxs == i).sum() > 0])

		import pdb 
		pdb.set_trace()
		

	def sample(self, n, feats):
		'''
		input:
			n		: the number of samples requested
			feats	: the candidate feature embeddings
		
		N clusters are made. 
		The feature that is closer to the center of each cluster is selected.
		
		output:
			n number of samples out of the candidates.
		'''
		feats = feats.numpy()
		cluster_learner = KMeans(n_clusters=n)
		cluster_learner.fit(feats)

		cluster_idxs = cluster_learner.predict(feats)
		centers = cluster_learner.cluster_centers_[cluster_idxs]
		dis = (feats - centers) ** 2
		dis = dis.sum(axis=1)

		return np.array([np.arange(feats.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in range(n) if (cluster_idxs == i).sum() > 0])

	def retrieve_anchor(self, embeddings, count):
		return embeddings.mean(dim=0).view(1, -1).repeat(count, 1)

	def generate_alpha(self, size, embedding_size, alpha_cap):
		'''
		size 			: Number of unlabeled samples
		embedding_size 	: Number of dims of the feature descriptor
		'''
		# torch.normal : Returns a tensor of random numbers drawn from separate normal distributions whose mean and standard deviation are given.
		alpha = torch.normal(
			mean=alpha_cap / 2.0,
			std=alpha_cap / 2.0,
			size=(size, embedding_size))
		alpha[torch.isnan(alpha)] = 1										# if nan in any of the (size x embedding_size), then make 1.
		i=self.clamp_alpha(alpha, alpha_cap)				
		return i							# clamps alpha into range [1e-8, alpha_cap] - alpha_cap is the maximum value

	def clamp_alpha(self, alpha, alpha_cap):
		'''
		torch.clamp:
			Clamps all elements in input into the range [ min, max ]. Letting min_value and max_value be min and max, respectively, this returns:
				yi​=min(max(xi​,min_valuei​),max_valuei​)
			Note: If min is greater than max torch.clamp(..., min, max) sets all elements in input to the value of max.

			It simply trims the range of values by setting a lower and an upper threshold.
			Each value lower than the lower_thres becomes equal to the lower_thres.
			Each value greater than the upper_thres becomes equal to the upper_thres.
		'''
		return torch.clamp(alpha, min=1e-8, max=alpha_cap)
					  # ulb_embedding, pred_1, anchor_i, alpha, alpha_cap,log_prefix=str(i))
	def learn_alpha(self, org_embed, labels, anchor_embed, alpha, alpha_cap, log_prefix=''): # ALFA IS A LEARNABLE.
		
		'''
		org_embed		: all unlabeled embeddings						size:(#unl_samples, embedding_dim)
		labels			: The predicted class of unlabeled classes		size:(#unl_samples)
		anchor_embed	: class-specific anchor for each un_sample		size:(#unl_samples, embedding_dim)			* all in all there are #unl_samples copies of the class-specifi anchor. I.e. simply print $anchor_embed[0] && $anchor_embed[-1]
		alpha			: alpha for each unlabeled sample				size:(#unl_samples, embedding_dim)			* for each value in each embedding, an alpha is generated
		alpha_cap		: the maximum value of α
		log_prefix=''	: string representing the AL round

		for each alpha_learning_iteration
			for each batch

				l <- copy alpha to l
				mix features
				pass mixed_features batch through the model
				calculate loss between pseudo-labels and predictions
				update the min_alpha(array of ones) only for the samples that presented inconsistency in model's predictions. Update with the default values generated by the generate_alpha. Therefore, only some of the generated alpha will be kept.
		'''
		labels = labels.to(self.device)
		min_alpha = torch.ones(alpha.size(), dtype=torch.float)				# ones of shape (#unl_samples, embedding_dim). min_alpha globally contains the min values for alpha in all iterations. Before the first iteration these are set to the max value αε[0,1]
		pred_changed = torch.zeros(labels.size(0), dtype=torch.bool)		# Falses of size (#unl_samples). Before 1st iteration, no sample has changed

		loss_func = torch.nn.CrossEntropyLoss(reduction='none')

		self.model.clf.eval()												# ----> in eval mode --> https://discuss.pytorch.org/t/does-gradients-change-in-model-eval/127732/2
																			# https://stackoverflow.com/questions/57323023/pytorch-loss-backward-and-optimizer-step-in-eval-mode-with-batch-norm-laye


		for i in range(self.args.alpha_learning_iters):						# self.args.alpha_learning_iters == The number of iterations for learning alpha
			tot_nrm, tot_loss, tot_clf_loss = 0., 0., 0.
			for b in range(math.ceil(float(alpha.size(0)) / self.args.alpha_learn_batch_size)):	# for each batch
				self.model.clf.zero_grad()
				start_idx = b * self.args.alpha_learn_batch_size
				end_idx = min((b + 1) * self.args.alpha_learn_batch_size, alpha.size(0))

				l = alpha[start_idx:end_idx]																	 	# batch-alpha
				l = torch.autograd.Variable(l.to(self.device), requires_grad=True)									# alpha learnable
				opt = torch.optim.Adam([l], lr=self.args.alpha_learning_rate / (1. if i < self.args.alpha_learning_iters * 2 / 3 else 10.)) # define optimizer
				e = org_embed[start_idx:end_idx].to(self.device)													# all unlabeled embeddings
				c_e = anchor_embed[start_idx:end_idx].to(self.device)												# class-specific anchor for each un_sample
				embedding_mix = (1 - l) * e + l * c_e																# eq(1) => z̃α = αz* + (1 − α)zu

				# comment:
				# 	Since l(alpha) is small (i.e. torch.mean(l)=0.0156 , torch.std(l)=0.0112),
				# 	... the 1-l (i.e. =0.9844) factor multiplied by the embedding of the unlabeled sample (e),
				#	... indicates that the mix (embedding_mix) primarily preserves the features of the unlabeled sample ...
				#	... whereas a smaller portion of the class-avg-feature (anchor) is contributing to the final fused embedding.

				out, _ = self.model.clf(embedding_mix, embedding=True)												# pass through the model								

				label_change = out.argmax(dim=1) != labels[start_idx:end_idx]										# Comparison of predictions
				# comment:
				# 	label_change compares the predictions of the same model on two different inputs
				# 		- unlabeled sample as it is
				# 		- unlabeled sample fused with the class-achor

				tmp_pc = torch.zeros(labels.size(0), dtype=torch.bool).to(self.device)								# copies label changes to tmp_pc
				tmp_pc[start_idx:end_idx] = label_change
				pred_changed[start_idx:end_idx] += tmp_pc[start_idx:end_idx].detach().cpu()							# stores the results in pred_changed
				
				tmp_pc[start_idx:end_idx] = tmp_pc[start_idx:end_idx] * (l.norm(dim=1) < min_alpha[start_idx:end_idx].norm(dim=1).to(self.device))
				''' tmp_pc[start_idx:end_idx] = tmp_pc[start_idx:end_idx] * (l.norm(dim=1) < min_alpha[start_idx:end_idx].norm(dim=1).to(self.device))

				eq (5) in the paper
				
				l.norm(dim=1)	 											--> finds the norm of the alpha for each unlabeld sample
				min_alpha[start_idx:end_idx].norm(dim=1).to(self.device)	--> Get the aggregated min alphas sample-wise.
					--> norm => sqrt(sum(1^2+1^2+... (embedding_size_times) ...+1^2)) 

				Therefore:
					> (l.norm(dim=1) < min_alpha[start_idx:end_idx].norm(dim=1).to(self.device))
						returns a tensor of size #num_unlabeled_samples, with the minimum values of the alpha-norms are stored. 
						In each alpha_learning_iteration these are updated.
 					> tmp_pc is updated for the indexes where the min alphas are 

				'''
				min_alpha[tmp_pc] = l[tmp_pc[start_idx:end_idx]].detach().cpu() 								# update the min_alpha(array of ones) only for the samples changed with the default values generated by the generate_alpha

				clf_loss = loss_func(out, labels[start_idx:end_idx].to(self.device))							# calc loss -->this determines whether new novel features have been found

				l2_nrm = torch.norm(l, dim=1)																	# calc norm of l --> this determines the model's sensitivity on those features
				
				clf_loss *= -1																					# invert the loss to maximize it

				loss = self.args.alpha_clf_coef * clf_loss + self.args.alpha_l2_coef * l2_nrm					# eq (2) and (3) from the paper. => 1* (-loss) + 0.01*norm_of_alpha
				loss.sum().backward(retain_graph=True)															# back-propagate the alpha
																												# https://stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method
																												# https://discuss.pytorch.org/t/how-to-confirm-parameters-of-frozen-part-of-network-are-not-being-updated/142482/2?u=paschalis_m --> model does not update its weights, since self.model.clf.lm1.weight.grad==None && self.model.clf.lm2.weight.grad==None || before and after optim.step() 
				opt.step()

				l = self.clamp_alpha(l, alpha_cap)																# filter max values of alpha (truncate values that exceed the value of alpha_cap)

				alpha[start_idx:end_idx] = l.detach().cpu()														# update alpha values

				tot_clf_loss += clf_loss.mean().item() * l.size(0)
				tot_loss += loss.mean().item() * l.size(0)
				tot_nrm += l2_nrm.mean().item() * l.size(0)

				del l, e, c_e, embedding_mix
				torch.cuda.empty_cache()
		
		count = pred_changed.sum().item()
		if pred_changed.sum() > 0:
			self.writer.add_scalar('stats/inconsistencies_%s' % (log_prefix), count, self.query_count)
			self.writer.add_scalar('alpha_change/mean_%s' % (log_prefix), min_alpha[pred_changed].mean().item(), self.query_count)
			self.writer.add_scalar('alpha_change/std_%s' % (log_prefix), min_alpha[pred_changed].std().item(), self.query_count)
			self.writer.add_scalar('alpha_change/norm_%s' % (log_prefix), min_alpha[pred_changed].norm(dim=1).mean(), self.query_count)
			self.writer.add_scalar('alpha_change/clf_loss_%s' % (log_prefix), tot_clf_loss / min_alpha.size(0), self.query_count)
			self.writer.add_scalar('alpha_change/loss_%s' % (log_prefix), tot_loss / min_alpha.size(0), self.query_count)
		else:
			self.writer.add_scalar('stats/inconsistencies_%s' % (log_prefix), 0, self.query_count)
			self.writer.add_scalar('alpha_change/mean_%s' % (log_prefix), 0, self.query_count)
			self.writer.add_scalar('alpha_change/std_%s' % (log_prefix), 0, self.query_count)
			self.writer.add_scalar('alpha_change/norm_%s' % (log_prefix), 0, self.query_count)
			self.writer.add_scalar('alpha_change/clf_loss_%s' % (log_prefix), 0, self.query_count)
			self.writer.add_scalar('alpha_change/loss_%s' % (log_prefix), 0, self.query_count)

		return min_alpha.cpu(), pred_changed.cpu()