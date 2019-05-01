# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn
from torch.autograd import Variable
import os

from .utils import load_embeddings, normalize_embeddings
from .evaluation.word_translation import load_dictionary, DIC_EVAL_PATH

class Mapping(nn.Module):

    def __init__(self, params):
        super(Mapping, self).__init__()

        self.emb_dim = params.emb_dim
        self.dropout = params.dropout
        self.input_dropout = params.input_dropout
        
        layers = [nn.Dropout(self.input_dropout)]
        input_dim = self.emb_dim
        output_dim = self.emb_dim
        layers.append(nn.Linear(input_dim, output_dim, bias=False))
        layers[-1].weight.data.copy_(torch.diag(torch.ones(params.emb_dim)))
        if params.non_lin == 'tanh':
            layers.append(nn.Tanh())
        elif params.non_lin == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif params.non_lin == 'relu':
             layers.append(nn.ReLU())
        
        self.layers = nn.Sequential(*layers) 
        
    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.layers(x)

def build_model(params):
    """
    Build all components of the model.
    """
    # source embeddings
    src_dico, _src_emb = load_embeddings(params, source=True)
    params.src_dico = src_dico
    src_emb = nn.Embedding(len(src_dico), params.emb_dim, sparse=True)
    src_emb.weight.data.copy_(_src_emb)

    # target embeddings
    if params.tgt_lang:
        tgt_dico, _tgt_emb = load_embeddings(params, source=False)
        params.tgt_dico = tgt_dico
        tgt_emb = nn.Embedding(len(tgt_dico), params.emb_dim, sparse=True)
        tgt_emb.weight.data.copy_(_tgt_emb)
    else:
        tgt_emb = None

    mapping = Mapping(params)

    # cuda
    if params.cuda:
        src_emb.cuda()
        if params.tgt_lang:
            tgt_emb.cuda()
            mapping.layers.cuda() 

    # normalize embeddings
    params.src_mean = normalize_embeddings(src_emb.weight.data, params.normalize_embeddings)
    if params.tgt_lang:
        params.tgt_mean = normalize_embeddings(tgt_emb.weight.data, params.normalize_embeddings)

    return src_emb, tgt_emb, mapping
