# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import json
import time
import argparse
from collections import OrderedDict
import torch
import numpy as np

from src.utils import bool_flag, initialize_exp
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator

VALIDATION_METRIC_UNSUP = 'induced_dico_size' # mean cos sim doesn't work well here
VALIDATION_METRIC_SUP = 'precision_at_1-csls_knn_10'

# main
parser = argparse.ArgumentParser(description='Supervised training')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=1, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
parser.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")

# data
parser.add_argument("--src_lang", type=str, default='en', help="Source language")
parser.add_argument("--tgt_lang", type=str, default='es', help="Target language")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
parser.add_argument("--dropout", type=float, default=0.0, help="dropout")
parser.add_argument("--input_dropout", type=float, default=0.0, help="input dropout")
parser.add_argument("--non_lin", type=str, default=None, help="non linear function")

# training refinement
parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--epoch_size", type=int, default=10000, help="Iterations per epoch")
parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
parser.add_argument("--map_beta", type=float, default=0.001, help="Beta for orthogonalization")
parser.add_argument("--mapping_optimizer", type=str, default="adam,lr=0.0005", help="Reconstruction optimizer")
parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
parser.add_argument("--lr_shrink", type=float, default=0.5,
                    help="Shrink the learning rate if the validation metric decreases (1 to disable)")
parser.add_argument("--clip_weights", type=float, default=0, help="Clip discriminator weights (0 to disable)")

# dictionary creation parameters (for refinement)
parser.add_argument("--dico_train", type=str, default="default",
                    help="Path to training dictionary (default: use identical character strings)")
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
parser.add_argument("--dico_method", type=str, default='csls_knn_10',
                    help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=10000, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
parser.add_argument("--boot", type=bool_flag, default=False, help="Use bootsrapped self-learning")
parser.add_argument("--orthogonal", type=bool_flag, default=False, help="Orthogonalize mapping")

# reload pre-trained embeddings
parser.add_argument("--src_emb", type=str, default='', help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default='', help="Reload target embeddings")
parser.add_argument("--normalize_embeddings", type=str, default="renorm", help="Normalize embeddings before training")

# parse parameters
params = parser.parse_args()

# check parameters
assert not params.cuda or torch.cuda.is_available()
assert params.dico_train in ["identical_char", "default"] or os.path.isfile(params.dico_train)
assert params.dico_build in ["S2T", "T2S", "S2T|T2S", "S2T&T2S"]
assert params.dico_max_size == 0 or params.dico_max_size < params.dico_max_rank
assert params.dico_max_size == 0 or params.dico_max_size > params.dico_min_size
assert os.path.isfile(params.src_emb)
assert os.path.isfile(params.tgt_emb)
assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)
assert params.export in ["", "txt", "pth"]

# build logger / model / trainer / evaluator
logger = initialize_exp(params)
src_emb, tgt_emb, mapping = build_model(params)
trainer = Trainer(src_emb, tgt_emb, mapping, params)
evaluator = Evaluator(trainer)

# load a training dictionary. if a dictionary path is not provided, use a default
# one ("default") or create one based on identical character strings ("identical_char")
trainer.load_training_dico(params.dico_train)

# define the validation metric
VALIDATION_METRIC = VALIDATION_METRIC_UNSUP if params.dico_train == 'identical_char' else VALIDATION_METRIC_SUP
logger.info("Validation metric: %s" % VALIDATION_METRIC)


"""
Learning loop for crosslingual training
"""
for n_epoch in range(params.n_epochs):

    # induce new training dico
    if ( n_epoch > 0 and params.boot ) or not hasattr(trainer, 'dico'):
        trainer.build_dictionary() 

    stats = {'MAP_COSTS': []}
    # perform initial procrustes fitting
    if n_epoch == 0:
        trainer.procrustes(stats)
        # embeddings / discriminator evaluation
        to_log = OrderedDict({'n_epoch': n_epoch})
        evaluator.all_eval(to_log)
        logger.info('Procrustes fitting completed.\n\n')  
      
    tic = time.time()
    n_words_proc = 0
    logger.info('Starting training epoch %i...' % n_epoch)
    for n_iter in range(0, params.epoch_size, params.batch_size):
        # updating mapping
        n_words_proc += trainer.mapping_step(stats)
        if params.orthogonal:
            trainer.orthogonalize()
        # log stats
        if n_iter % 1000 == 0 and n_iter != 0 :
            stats_str = [('MAP_COSTS', 'Crosslingual mapping loss')]
            stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                         for k, v in stats_str if len(stats[k]) > 0]
            stats_log.append('%i samples/s' % int(n_words_proc / (time.time() - tic)))
            logger.info(('%06i - ' % n_iter) + ' - '.join(stats_log))

            # reset
            tic = time.time()
            n_words_proc = 0
            for k, _ in stats_str:
                del stats[k][:]
    
    trainer.mapping.eval()
    to_log = OrderedDict({'n_epoch': n_epoch})
    evaluator.all_eval(to_log)

    # JSON log / save best model / end of epoch
    logger.info("__log__:%s" % json.dumps(to_log))
    trainer.save_best(to_log, VALIDATION_METRIC)
    logger.info('End of epoch %i.\n\n' % n_epoch)

    # update the learning rate (stop if too small)
    trainer.update_lr(to_log, VALIDATION_METRIC)
    if trainer.mapping_optimizer.param_groups[0]['lr'] < params.min_lr:
        logger.info('Learning rate < 1e-6. BREAK.')
        break

# export embeddings
if params.export:
    trainer.reload_best()
    trainer.export()
