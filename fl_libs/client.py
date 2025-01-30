# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import time
import numpy as np

class Client:
    def __init__(self, cid, x_train, y_train, x_test, y_test, device_tier=0, comm_prune_rate=0.0, comm_quantize_bits=0, comm_quantize_separate_sign=False, channel_multiplier=1.0, args=None):
        self.cid = cid
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.device_tier = device_tier
        self.args = args # For debugging
        
        # Optimization params
        self.comm_prune_rate = comm_prune_rate
        self.comm_quantize_bits = comm_quantize_bits
        self.comm_quantize_separate_sign = comm_quantize_separate_sign

        self.channel_multiplier = channel_multiplier

    def init_client(self, model, device, optimizer):
        self.model = model
        self.device = device
        self.optimizer = optimizer

    def get_feature_values(self, feat_name):
        return self.x_train[feat_name].flatten(), self.x_train[feat_name].shape[0]

    def remove_features_except(self, feat_name, vals):
        isin = np.isin(self.x_train[feat_name], vals)
        #print(vals)
        #print(self.x_train[feat_name])
        #print(isin)
        for k in self.x_train:
            #print(len(self.x_train[k]))
            self.x_train[k] = self.x_train[k][isin]
            #print(len(self.x_train[k]))
        #print(len(self.y_train))
        self.y_train = self.y_train[isin]
        #print(len(self.y_train))
        return len(isin), isin.sum()

    def replace_features_w_default_except(self, feat_name, vals):
        # Use 0 as a default (TODO: Only the hist features has zero as a default.
        assert("hist_" in feat_name)
        replaced = np.invert(np.isin(self.x_train[feat_name], np.append(vals, 0)))
        if replaced.sum() > 0:
            self.x_train[feat_name][replaced] = 0
        return

    def train(self, epochs, bs, lr):
        # Init optimizer
        if self.optimizer == "sgd":
            optim = torch.optim.SGD(self.model.parameters(), lr=lr)
        elif self.optimizer == "adagrad":
            optim = torch.optim.Adagrad(self.model.parameters(), lr=lr)
        elif self.optimizer == "adam":
            optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.optim = optim

        # To calculate each gradient.
        # Calculating and sending gradients instead of
        # the raw model value for future extensibility.
        grad = {}
        for name, param in self.model.state_dict().items():
            grad[name] = param.data.clone().detach()

        # Run training locally
        if bs == 0: # Full-batch training
            bs = len(self.y_train)
        self.model.fit(self.x_train, self.y_train, batch_size=bs, epochs=epochs, verbose=0, validation_split=0.0)

        # Calculate gradient
        for name, param in self.model.state_dict().items():
            grad[name] -= param.data.clone().detach()

        num_train_samples = self.y_train.shape[0]

        return num_train_samples, grad
