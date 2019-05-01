# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from logging import getLogger
import scipy
import scipy.linalg
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.modules.distance import CosineSimilarity

from .utils import get_optimizer, load_embeddings, normalize_embeddings, export_embeddings
from .utils import clip_parameters
from .dico_builder import build_dictionary
from .evaluation.word_translation import DIC_EVAL_PATH, load_identical_char_dico, load_dictionary
from .rcsls_loss import RCSLS

logger = getLogger()


class Trainer(object):

    def __init__(self, src_emb, tgt_emb, mapping, params):
        """
        Initialize trainer script.
        """
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.src_dico = params.src_dico
        self.tgt_dico = getattr(params, 'tgt_dico', None)
        self.mapping = mapping
        self.params = params

        # optimizers
        if hasattr(params, 'mapping_optimizer'):
            optim_fn, optim_params = get_optimizer(params.mapping_optimizer)
            self.mapping_optimizer = optim_fn(mapping.parameters(), **optim_params)
        self.criterion = CosineSimilarity() #ContrastiveLoss(margin=0.0, measure='cosine')
        self.criterionRCSLS = RCSLS()

        # best validation score
        self.best_valid_metric = -1e12

        self.decrease_lr = False

    def get_xy(self, volatile=True):
        """
        Get transofrmation input batch / output target.
        """
        # select random word IDs
        bs = self.params.batch_size
        ids = torch.LongTensor(bs).random_(len(self.dico))
        if self.params.cuda:
            ids = ids.cuda()
        
        # get word embeddings
        with torch.no_grad():
            
            dico_src_emb = self.src_emb(self.dico[:,0])
            dico_tgt_emb = self.tgt_emb(self.dico[:,1])
            
            src_emb = dico_src_emb[ids]
            tgt_emb = dico_tgt_emb[ids]

            src_emb = Variable(src_emb.data)
            tgt_emb = Variable(tgt_emb.data)
        
            neg_src_emb = Variable(self.src_emb.weight)
            neg_tgt_emb = Variable(self.tgt_emb.weight)

        if self.params.cuda:
            tgt_emb = tgt_emb.cuda()
        return src_emb, tgt_emb, neg_src_emb, neg_tgt_emb

    def mapping_step(self, stats):
        """
        Train the source embedding mappingation.
        """
        self.mapping.train()
        loss=0
        for _ in range(int(5000/self.params.batch_size)):
          #loss
          src_emb, tgt_emb, neg_src_emb, neg_tgt_emb = self.get_xy()
        
          src_emb_trans = self.mapping(Variable(src_emb.data))
          neg_src_emb_trans = self.mapping(Variable(neg_src_emb.data))

          loss += self.criterionRCSLS(src_emb, src_emb_trans, tgt_emb, neg_src_emb, neg_src_emb_trans, neg_tgt_emb)
       
        stats['MAP_COSTS'].append(loss.data.item())

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (discriminator)")
            exit()

        # optim
        self.mapping_optimizer.zero_grad()
        loss.backward()
        self.mapping_optimizer.step()
        clip_parameters(self.mapping, self.params.clip_weights)

        return self.params.batch_size

    def load_training_dico(self, dico_train):
        """
        Load training dictionary.
        """
        word2id1 = self.src_dico.word2id
        word2id2 = self.tgt_dico.word2id

        # identical character strings
        if dico_train == "identical_char":
            self.dico = load_identical_char_dico(word2id1, word2id2)
        # use one of the provided dictionary
        elif dico_train == "default":
            filename = '%s-%s.0-5000.txt' % (self.params.src_lang, self.params.tgt_lang)
            self.dico = load_dictionary(
                os.path.join(DIC_EVAL_PATH, filename),
                word2id1, word2id2
            )
        # dictionary provided by the user
        else:
            self.dico = load_dictionary(dico_train, word2id1, word2id2)

        # cuda
        if self.params.cuda:
            self.dico = self.dico.cuda()

    def build_dictionary(self):
        """
        Build a dictionary from aligned embeddings.
        """
        src_emb = self.mapping(self.src_emb.weight).data
        tgt_emb = self.tgt_emb.weight.data
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        self.dico = build_dictionary(src_emb, tgt_emb, self.params)

    def procrustes(self, stats):
        """
        Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """
        A = self.src_emb.weight.data[self.dico[:, 0]]
        B = self.tgt_emb.weight.data[self.dico[:, 1]]
        W = self.mapping.layers[1].weight.data
        M = B.transpose(0, 1).mm(A).cpu().detach().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        _W = U.dot(V_t)
        W.copy_(torch.from_numpy(_W).type_as(W))

    def orthogonalize(self):
        """
        Orthogonalize the mapping.
        """
        if self.params.map_beta > 0:
            W = self.mapping.layers[1].weight.data
            beta = self.params.map_beta
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

    def update_lr(self, to_log, metric):
        """
        Update learning rate when using SGD.
        """
        if 'sgd' not in self.params.mapping_optimizer:
            return
        old_lr = self.mapping_optimizer.param_groups[0]['lr']
        new_lr = max(self.params.min_lr, old_lr * self.params.lr_decay)
        if new_lr < old_lr:
            logger.info("Decreasing learning rate: %.8f -> %.8f" % (old_lr, new_lr))
            self.mapping_optimizer.param_groups[0]['lr'] = new_lr

        if self.params.lr_shrink < 1 and to_log[metric] >= -1e7:
            if to_log[metric] < self.best_valid_metric:
                logger.info("Validation metric is smaller than the best: %.5f vs %.5f"
                            % (to_log[metric], self.best_valid_metric))
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
                if self.decrease_lr:
                    old_lr = self.mapping_optimizer.param_groups[0]['lr']
                    self.mapping_optimizer.param_groups[0]['lr'] *= self.params.lr_shrink
                    logger.info("Shrinking the learning rate: %.5f -> %.5f"
                                % (old_lr, self.mapping_optimizer.param_groups[0]['lr']))
                self.decrease_lr = True

    def save_best(self, to_log, metric, mapping=True):
        """
        Save the best model for the given validation metric.
        """
        #import pdb; pdb.set_trace()
        # best mapping for the given validation criterion
        if mapping:
            if to_log[metric] > self.best_valid_metric:
                # new best mapping
                self.best_valid_metric = to_log[metric]
                logger.info('* Best value for "%s": %d' % (metric, to_log[metric]))
                logger.info('* P@1: %.5f' % (to_log['precision_at_1-csls_knn_10']))
                logger.info('* P@5: %.5f' % (to_log['precision_at_5-csls_knn_10']))
                logger.info('* P@10: %.5f' % (to_log['precision_at_10-csls_knn_10']))
                # save the mapping
                W = self.mapping.layers[1].weight.data.cpu().numpy()
                path = os.path.join(self.params.exp_path, 'best_mapping.pth')
                logger.info('* Saving the mapping to %s ...' % path)
                torch.save(W, path)
            #else: logger.info('* Best value for "%s": %.5f' % (metric, self.best_valid_metric))
        else:
            pass

    def reload_best(self, mapping=True):
        """
        Reload the best mapping.
        """
        if mapping:
            path = os.path.join(self.params.exp_path, 'best_mapping.pth')
            logger.info('* Reloading the best model from %s ...' % path)
            # reload the model
            assert os.path.isfile(path)
            to_reload = torch.from_numpy(torch.load(path))
            W = self.mapping.layers[1].weight.data
            assert to_reload.size() == W.size()
            W.copy_(to_reload.type_as(W))
        else:
            pass
    def export(self):
        """
        Export embeddings.
        """
        params = self.params

        # load all embeddings
        logger.info("Reloading all embeddings for mapping ...")
        params.src_dico, src_emb = load_embeddings(params, source=True, full_vocab=True)
        params.tgt_dico, tgt_emb = load_embeddings(params, source=False, full_vocab=True)

        # apply same normalization as during training
        normalize_embeddings(src_emb, params.normalize_embeddings, mean=params.src_mean)
        normalize_embeddings(tgt_emb, params.normalize_embeddings, mean=params.tgt_mean)

        # map source embeddings to the target space
        bs = 4096
        logger.info("Map source embeddings to the target space ...")
        for i, k in enumerate(range(0, len(src_emb), bs)):
            with torch.no_grad():
                x = Variable(src_emb[k:k + bs])
                src_emb[k:k + bs] = self.mapping(x.cuda() if params.cuda else x).data.cpu()

        # write embeddings to the disk
        export_embeddings(src_emb, tgt_emb, params)
