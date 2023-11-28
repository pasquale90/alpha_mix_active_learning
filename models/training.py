import os
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class Training(object):
    def __init__(self, net, net_args, handler, args, writer, device, round, dataset, model, sampling, weights_dir):
        self.net = net
        self.net_args = net_args
        self.handler = handler

        self.args = args
        self.writer = writer
        self.device = device
        self.weights_dir = weights_dir

        self.currRound = round
        self.model_name = model
        self.dataset_name = dataset
        self.sampling_name = sampling
        

        # if round==1:
        self.clf = self.net(**self.net_args).to(self.device)
        self.initial_state_dict = copy.deepcopy(self.clf.state_dict())

        try:
            os.makedirs(self.weights_dir)
        except:
            pass
    
    def store_weights(self):
        
        self.clf.eval()
                
        # weights_filename = model + "_" + dataset + "_" + sampling + "_" + str(round) + "_weights.pth"
        # weights_filepath = os.path.join(self.weights_dir, weights_filename)
        weights_filename = self.model_name + "_" + self.dataset_name + "_" + self.sampling_name + "_" + str(self.currRound) + "_weights.pth"
        weights_filepath = os.path.join(self.weights_dir, weights_filename)

        # print(f"\n\n Round {round} Model parameters when STORING ... /n{[param for param in self.clf.parameters()][0]}\n{[param for param in self.clf.parameters()][-1]}\n\n")
        # print(f"\n STORING --> Round {round} Model parameters when STORING ... /n{[param for param in self.clf.parameters()][-1]}\n")

        if self.scheduler is not None:
            torch.save({
                'round': round,
                # 'model_state_dict': self.net.get_model().state_dict(),
                'model_state_dict': self.clf.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss' : self.loss,
                'scheduler_state_dict': self.scheduler.state_dict()
                }, weights_filepath)
        else:
            torch.save({
                'round': round,
                # 'model_state_dict': self.net.get_model().state_dict(),
                'model_state_dict': self.clf.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss' : self.loss
                }, weights_filepath)



    def load_weights(self,round, dataset, model, sampling):
        
        self.clf.eval()

        if round>0:
            # weights_filename = model + "_" + dataset + "_" + sampling + "_" + str(round) + "_weights.pth"
            # weights_filepath = os.path.join(self.weights_dir, weights_filename)
            weights_filename = self.model_name + "_" + self.dataset_name + "_" + self.sampling_name + "_" + str(self.currRound-1) + "_weights.pth"
            weights_filepath = os.path.join(self.weights_dir, weights_filename)

    # import pdb
    # pdb.set_trace()
            assert os.path.exists(weights_filepath)
        
            print(f"Loading weights {weights_filename}")

            checkpoint = torch.load(weights_filepath)
            # self.net.model.load_state_dict(checkpoint['model_state_dict'])
            self.clf.load_state_dict(checkpoint['model_state_dict'])
        elif round<0:
            raise ValueError

        self.args['continue_training']=True
        for params in self.clf.parameters():
            params.requires_grad = True

# print(f" Print the weights of the model to ensure that weights are loaded ")
# import pdb
# pdb.set_trace()

    def _load_weights(self,round, dataset, model, sampling):
        
        self.clf.eval()

        # weights_filename = self.model + "_" + dataset + "_" + sampling + "_" + str(round-1) + "_weights.pth"
        # weights_filepath = os.path.join(self.weights_dir, weights_filename)
        weights_filename = self.model_name + "_" + self.dataset_name + "_" + self.sampling_name + "_" + str(self.currRound-1) + "_weights.pth"
        weights_filepath = os.path.join(self.weights_dir, weights_filename)

        assert os.path.exists(weights_filepath)

        print(f"Loading weights {weights_filename}")

        checkpoint = torch.load(weights_filepath)
        # self.net.model.load_state_dict(checkpoint['model_state_dict'])
        self.clf.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss = checkpoint['loss']

        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.args['continue_training']=True
        for params in self.clf.parameters():
            params.requires_grad = True

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        # self.clf.eval()
        # print(f"\n\n Round {round} Model parameters when LOADING ... /n{[param for param in self.clf.parameters()][0]}\n{[param for param in self.clf.parameters()][-1]}\n\n")
        # print(f"\n LOADING --> Round {round} Model parameters when LOADING ... /n{[param for param in self.clf.parameters()][-1]}\n")

        print(f" Print the weights of the model to ensure that weights are loaded ")
        # import pdb
        # pdb.set_trace()
    
    def _validate(self, X_val, Y_val, name, epoch):
        if X_val is None or len(X_val) <= 0:
            return

        P = self.predict(X_val, Y_val)
        acc = 1.0 * (Y_val == P).sum().item() / len(Y_val)

        self.writer.add_scalar('vaidation_accuracy/%s' % name, acc, epoch)
        #print('%s validation accuracy at epoch %d: %f' % (name, epoch, acc))

        return acc

    def _mixup_train(self, epoch, loader_tr, optimizer, name):
        self.clf.train()
        criterion = nn.CrossEntropyLoss()

        # import ipdb; ipdb.set_trace()

        accFinal, tot_loss, iters = 0., 0., 0
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            if y.size(0) <= 1:
                continue
            self.optimizer.zero_grad()

            inputs, targets_a, targets_b, lam = self.mixup_data(x, y)
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

            out, e1 = self.clf(inputs)
            self.los = self.mixup_criterion(criterion, out, targets_a, targets_b, lam)

            tot_loss += self.los.item()
            accFinal += torch.sum((torch.max(out, 1)[1] == y).float()).data.item()

            self.los.backward()
            self.optimizer.step()

            self.iter += 1
            iters += 1

        self.writer.add_scalar('training_loss/%s' % name, tot_loss / iters, epoch)
        return accFinal / len(loader_tr.dataset.X)

    def mixup_data(self, x, y):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        alpha = self.args['mixup_alpha']
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        if self.args['mixup_max_lambda']:
            lam = max(lam, 1 - lam)

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def _mixup_hidden_train(self, epoch, loader_tr, optimizer, name):
        self.clf.train()
        criterion = nn.CrossEntropyLoss()

        # import ipdb; ipdb.set_trace()

        accFinal, tot_loss, iters = 0., 0., 0
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            if y.size(0) <= 1:
                continue
            self.optimizer.zero_grad()

            #inputs, targets_a, targets_b, lam = self.mixup_data(x, y)
            #inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

            _, embedding = self.clf(x)

            inputs, targets_a, targets_b, lam = self.mixup_data(embedding, y)

            out, embedding = self.clf(inputs, embedding=True)

            self.los = self.mixup_criterion(criterion, out, targets_a, targets_b, lam)

            tot_loss += self.los.item()
            accFinal += torch.sum((torch.max(out, 1)[1] == y).float()).data.item()

            self.los.backward()
            self.optimizer.step()

            self.iter += 1
            iters += 1

        self.writer.add_scalar('training_loss/%s' % name, tot_loss / iters, epoch)
        return accFinal / len(loader_tr.dataset.X)

    def _train(self, epoch, loader_tr, optimizer, name, scheduler=None):
        self.clf.train()
        dt_size = len(loader_tr)
        accFinal, tot_loss, iters = 0., 0., 0

        
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            
            # try:
            x, y = x.to(self.device), y.to(self.device)
            if y.size(0) <= 1:
                continue
            self.optimizer.zero_grad()
            out, e1 = self.clf(x)
            
            self.loss = F.cross_entropy(out, y)

            tot_loss += self.loss.item()
            accFinal += torch.sum((torch.max(out, 1)[1] == y).float()).data.item()
            self.loss.backward()
            self.optimizer.step()

            # if self.iter > 0 and self.iter % n_val_iter == 0:
            #    self._validate(name)

            self.iter += 1
            iters += 1

            # except:
            #     import pdb
            #     pdb.set_trace()
            
            if scheduler is not None:
                scheduler.step(epoch - self.args['lr_warmup'] + iters/float(dt_size))

        self.writer.add_scalar('training_loss/%s' % name, tot_loss / iters, epoch)

        # print(f"Epoch {epoch} (end of epoch) Model parameters when TRAINING ... \n{[param for param in self.clf.parameters()][-1]}")

        return accFinal / len(loader_tr.dataset.X)

    def train(self, name, X, Y, idxs_lb, X_val, Y_val, train_epoch_func=None):
            
        n_epoch = 2000 if self.args['n_epoch'] <= 0 else self.args['n_epoch']
        #n_val_iter = self.args['n_val_iter']
        if not self.args['continue_training']:
            self.clf = self.net(**self.net_args).to(self.device)
            self.clf.load_state_dict(copy.deepcopy(self.initial_state_dict))

        if self.args['optimizer'] == 'Adam':
            print('Adam optimizer...')
            self.optimizer = optim.Adam(self.clf.parameters(), **self.args['optimizer_args'])
        else:
            print('SGD optimizer...')
            self.optimizer = optim.SGD(self.clf.parameters(), **self.args['optimizer_args'])
        self.optimizer.zero_grad()

        if self.args['lr_schedule']:
            for param in self.optimizer.param_groups:
                param['initial_lr'] = param['lr']

            #scheduler = CosineAnnealingWarmRestarts(optimizer,
            #                                        T_0=min(self.args['lr_T_0'], n_epoch - self.args['lr_warmup']),
            #                                        T_mult=self.args['lr_T_mult'])
            self.scheduler = CosineAnnealingLR(self.optimizer, n_epoch, eta_min=0)
        else:
            self.scheduler = None

        # Load previous state
        al_round = int(name)
        if (al_round != 0): #self,round, dataset, model, sampling):
            self._load_weights(al_round, dataset="MNIST",model="mlp", sampling='AlphaMixSampling')
        
        idxs_train = np.arange(len(Y))[idxs_lb]
        loader_tr = DataLoader(self.handler(X[idxs_train], Y[idxs_train], transform=self.args['transform']),
                               shuffle=True, **self.args['loader_tr_args'])
        
        self.iter = 0
        self.best_model = None
        best_acc, best_epoch, n_stop = 0., 0, 0

        lr = self.args['optimizer_args']['lr']
        print('Training started...')
        for epoch in tqdm(range(n_epoch)):

            # print(f"\n\n Training for the epoch {epoch}\n\n")
            if self.args['lr_decay_epochs'] is not None and epoch in self.args['lr_decay_epochs']:
                for param in self.optimizer.param_groups:
                    lr /= 10.
                    param['lr'] = lr
                    param['initial_lr'] = lr

                    if self.args['lr_schedule']:
                        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=self.args['lr_T_0'], T_mult=self.args['lr_T_mult'])

            if epoch < self.args['lr_warmup']:
                learning_rate = lr * (epoch + 1) / float(self.args['lr_warmup'])
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = learning_rate

            for param in self.optimizer.param_groups:
                self.writer.add_scalar('learning_rate/%s' % name, param['lr'], epoch)

            if train_epoch_func is not None:
                accCurrent = train_epoch_func(epoch, loader_tr, self.optimizer, name,
                                     scheduler=(None if epoch < self.args['lr_warmup'] else self.scheduler))
            else:
                accCurrent = self._train(epoch, loader_tr, self.optimizer, name,
                                         scheduler=(None if epoch < self.args['lr_warmup'] else self.scheduler))

            self.writer.add_scalar('training_accuracy/%s' % name, accCurrent, epoch)

            if X_val is not None and len(X_val) > 0:
                val_acc = self._validate(X_val, Y_val, name, epoch)
                if val_acc is not None:
                    if val_acc > best_acc:
                        best_acc = val_acc
                        best_epoch = epoch
                        n_stop = 0
                        if self.args['choose_best_val_model']:
                            self.best_model = copy.deepcopy(self.clf)
                    else:
                        n_stop += 1

                    if n_stop > self.args['n_early_stopping']:
                        print('Early stopping at epoch %d ' % epoch)
                        break

            if not self.args['train_to_end'] and accCurrent >= 0.99:      # and int(epoch / self.args['lr_change_period']) >= (math.ceil(n_epoch / self.args['lr_change_period']) - 1):
                print('Reached max accuracy at epoch %d ' % epoch)
                break

            #if self.scheduler is not None:
            #    self.scheduler.step()

        if self.best_model is not None:
            self.clf = self.best_model
            print('Best model based on validation accuracy (%f) selected in epoch %d.' % (best_acc, best_epoch))
        
        # import pdb
        # pdb.set_trace()
        self.store_weights()
        return epoch

    def predict(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['test_transform']),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        P = torch.zeros(len(Y), dtype=Y.dtype).to(self.device)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)

                pred = out.max(1)[1]
                P[idxs] = pred

        return P.cpu()

    def predict_prob(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['test_transform']),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        probs = torch.zeros([len(Y), self.clf.n_label])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()

        return probs

    def predict_prob_embed(self, X, Y, eval=True):
        '''
        input:
            X: unlabeled images ,torch tensor of shape (samples,width,height) || (samples,height,width) of type torch.uint8
            Y: actual labels    ,torch tensor of shape (samples) of type torch.int64 -----------------------> (not applicable here, but : these labels are used for validating the alphamix-framework)
        output:
            probs : embeddings that are passed through the fully connected + softmax, 
                    torch tensor of shape (samples,{#classes}}), type torch.float32
            embeddings : embeddings exctracted from the backbone. 
                        torch tensor of shape (samples,{train_params['emb_size']}), type torch.float32
            
        '''
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['test_transform']),
                               shuffle=False, **self.args['loader_te_args'])

        probs = torch.zeros([len(Y), self.clf.n_label])
        embeddings = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        if eval:
            self.clf.eval()
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)                                   # e1 is the embeddings, out is the results of classification used for obtaining scores. 
                    prob = F.softmax(out, dim=1)
                    probs[idxs] = prob.cpu()
                    embeddings[idxs] = e1.cpu()
                    # import pdb
                    # pdb.set_trace()
        else:
            self.clf.train()
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
                embeddings[idxs] = e1.cpu()

        return probs, embeddings

    def predict_all_representations(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['test_transform']),
                               shuffle=False, **self.args['loader_te_args'])

        probs = torch.zeros([len(Y), self.clf.n_label])
        all_reps = [None for i in range(self.clf.get_n_representations())]
        embeddings = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        if eval:
            self.clf.eval()
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1, reps = self.clf(x, return_reps=True)
                    for i in range(self.clf.get_n_representations()):
                        if all_reps[i] is None:
                            all_reps[i] = reps[i]
                        else:
                            all_reps[i] = np.concatenate([all_reps[i], reps[i]], axis=0)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] = prob.cpu()
                    embeddings[idxs] = e1.cpu()

        return probs, embeddings, all_reps

    def predict_embedding_prob(self, X_embedding):
        loader_te = DataLoader(SimpleDataset(X_embedding),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        probs = torch.zeros([X_embedding.size(0), self.clf.n_label])
        with torch.no_grad():
            for x, idxs in loader_te:
                x = x.to(self.device)
                out, e1 = self.clf(x, embedding=True)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()

        return probs

    def predict_prob_dropout(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['test_transform']),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([len(Y), self.clf.n_label])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i + 1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop

        return probs

    def predict_prob_dropout_split(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['test_transform']),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([n_drop, len(Y), self.clf.n_label])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i + 1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()

        return probs

    def predict_prob_embed_dropout_split(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['test_transform']),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([n_drop, len(Y), self.clf.n_label])
        embeddings = torch.zeros([n_drop, len(Y), self.clf.get_embedding_dim()])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i + 1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
                    embeddings[i][idxs] = e1.cpu()

        return probs, embeddings

    def get_embedding(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['test_transform']),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        embedding = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                embedding[idxs] = e1.cpu()

        return embedding

    def get_grad_embedding(self, X, Y, is_embedding=False):
        model = self.clf
        embDim = model.get_embedding_dim()
        model.eval()
        nLab = len(np.unique(Y))
        embedding = np.zeros([len(Y), embDim * nLab])
        #import ipdb; ipdb.set_trace()
        if is_embedding:
            loader_te = DataLoader(SimpleDataset2(X, Y),
                                   shuffle=False, **self.args['loader_te_args'])
        else:
            loader_te = DataLoader(self.handler(X, Y, transform=self.args['test_transform']),
                                   shuffle=False, **self.args['loader_te_args'])
        with torch.no_grad():
            print('Creating gradient embeddings:')
            for x, y, idxs in tqdm(loader_te):
                x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
                cout, out = self.clf(x, embedding=is_embedding)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c: embDim * (c + 1)] = copy.deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c: embDim * (c + 1)] = copy.deepcopy(out[j]) * (-1 * batchProbs[j][c])
            return torch.Tensor(embedding)


class SimpleDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __getitem__(self, index):
        x = self.X[index]
        return x, index

    def __len__(self):
        return len(self.X)


class SimpleDataset2(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)