# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import copy

import torch
import torch.distributed as dist

from sklearn.metrics import log_loss, roc_auc_score, recall_score, precision_score, f1_score, average_precision_score
import time

import math
import random
import pickle

from .utils import set_optimizer

class Server:
    def __init__(self, rank, world_size, train_clients, test_data, model, device, optimizer, server_lr, total_train_cids, test_metrics, seed, staleness=0, dp_oram_params=[], dp_oram_eps=0.0, dp_max_feats=0):
        # Make clients a dict. That is more handy
        # TODO: This needs some cleanup
        self.local_train_clients = {c.cid: c for c in train_clients}
        self.local_total_clients = {c.cid: c for c in train_clients}

        self.total_train_cids = total_train_cids

        self.test_data = test_data

        self.model = copy.deepcopy(model)
        self.device = device
        self.total_weight = 0
        self.server_lr = server_lr
        self.optim = set_optimizer(self.model, optimizer, server_lr)
        self.rank = rank
        self.world_size = world_size

        self.seed = seed
        self.test_metrics = [x.strip() for x in test_metrics.split(",")]

        self.staleness = staleness
        self.stale_models = []
        for _ in range(staleness + 1):
            self.stale_models.append(copy.deepcopy(model))
        self.rd_ptr = 0
        self.wr_ptr = 0

        self.dp_oram_params = dp_oram_params + []
        self.dp_oram_eps = dp_oram_eps
        self.dp_max_feats = dp_max_feats

    def init_clients(self, model, device, optimizer):
        for cid, c in self.local_total_clients.items():
            c.init_client(model, device, optimizer)


    def select_clients(self, clients_per_round, r, client_selection):
        client_size = min(int(clients_per_round), len(self.total_train_cids))

        if client_selection == "random":
            np.random.seed(r)
            selected_clients = np.random.choice(self.total_train_cids, client_size, replace=False)
        elif client_selection == "sequential":
            selected_clients = self.total_train_cids[r * client_size : (r + 1) * client_size]

        if len(selected_clients) == 0:
            return None # All clients used
        return [self.local_train_clients[cid] for cid, _ in list(filter(lambda x: x[0] in self.local_train_clients, selected_clients))]


    def train(self, clients_per_round, epochs, bs, aggr_method, test_freq, client_lr, client_selection="random", dp_oram_method="fcfs", dp_loss_mitigation="default"):
        timestamps = []
        print("Start training", flush=True)

        r = 0

        print(self.model)

        total_reads_baseline = 0
        total_reads = 0
        wasted_reads = 0
        lost_reads = 0
        max_feat_per_user = 0

        tot_sample_cnt = 0
        lost_sample_cnt = 0
        while True:
            timestamps.append(time.time())
            if ((r + 1) % test_freq == 0) and self.rank == 0:
                print(f"========== Round {r} ===========", flush=True)
            # Model to train mode if not
            self.model = self.model.train()

            # Select clients
            selected_clients = self.select_clients(clients_per_round, r, client_selection)

            if selected_clients is None:
                # All clients used
                print(f"All clients used", flush=True)
                return

            #print(f"Round {r}: selected clients for rank {self.rank}: {[c.cid for c in selected_clients]}")
            # For the feature to use DP-ORAM,
            # collect all the feature value requested and find the union.
            for fname in self.dp_oram_params:
                vals = np.array([])
                for c in selected_clients:
                    c_vals, num_samples = c.get_feature_values(fname)
                    #vals = np.concatenate((vals, c.get_feature_values(fname)))
                    # Each user only sends one request per duplicating features
                    tot_sample_cnt += num_samples
                    unique_c_vals = np.unique(c_vals)
                    if self.dp_max_feats > 0 and len(unique_c_vals) > self.dp_max_feats:
                        # If there is a limit to the number of features for each user, only use a subset of the features.
                        unique_c_vals = np.random.choice(unique_c_vals, self.dp_max_feats, False)
                    vals = np.concatenate((vals, unique_c_vals))
                    max_feat_per_user = max(max_feat_per_user, len(unique_c_vals))
                    #print(c_vals)
                    #print(vals)
                    #print(num_samples)
                    #exit(0)
                unique_vals = np.unique(vals)
                #print(fname, len(vals), len(unique_vals))
                # Count the total number of requests (K for non-hist features)
                if self.dp_max_feats > 0:
                    baseline_read_per_round = len(selected_clients) * self.dp_max_feats
                else:
                    baseline_read_per_round = len(vals)
                total_reads_baseline += baseline_read_per_round
                #print(vals)
                total_reads += len(unique_vals)
                #print(total_reads_baseline, total_reads, tot_sample_cnt)
                #exit(0)
                
                if self.dp_oram_eps > 0:
                    # 1. Choose total number to read ORAM
                    # Assume number of samples are known.
                    # TODO: Currently assume Ys are uniform.
                    #p = torch.arange(len(vals))
                    p = torch.arange(baseline_read_per_round)
                    p = (- self.dp_oram_eps * (len(unique_vals) - p).abs() / 2).exp()
                    p /= p.sum()
                    num_oram_reads = p.multinomial(1)
                    #print(len(selected_clients) * self.dp_max_feats, len(vals))
                    #print(f"Number of reads with eps {self.dp_oram_eps}, {len(unique_vals)} {num_oram_reads}, diff {num_oram_reads - len(unique_vals)}")
                    if num_oram_reads - len(unique_vals) > 0:
                        wasted_reads += (num_oram_reads - len(unique_vals))
                    else:
                        lost_reads += (len(unique_vals) - num_oram_reads)

                    # 2. If the total number is larger than the number of unique values, simply do nothing.
                    if num_oram_reads < len(unique_vals):
                        if dp_oram_method == "fcfs":
                            unique_vals = unique_vals[:num_oram_reads.item()]
                        elif dp_oram_method == "random":
                            #print(unique_vals, num_oram_reads)
                            unique_vals = np.random.choice(unique_vals, num_oram_reads.item(), False)
                        else:
                            raise AssertionError()

                if self.dp_oram_eps > 0 or self.dp_max_feats > 0:
                    '''
                    # Mitigation 1: Simply not use the samples with the removed features
                    if dp_loss_mitigation == "remove":
                        # For now, let's try simply not using the samples with the removed features
                        for c in selected_clients:
                            orig_samples, remaining_samples = c.remove_features_except(fname, vals)
                            lost_sample_cnt += (orig_samples - remaining_samples)
                            tot_sample_cnt += orig_samples
                    # Mitigation 2: Use a default vector (e.g., unknown) instead
                    '''
                    if dp_loss_mitigation == "default":
                        for c in selected_clients:
                            c.replace_features_w_default_except(fname, unique_vals)

            # Temporary dict to accumulate gradients
            accum = {name: torch.zeros(param.shape if len(param.shape) > 0 else 1, dtype=param.dtype).to(self.device) for name, param in self.model.state_dict().items()}

            # For sparse embedding features, we count the # samples per row (for FSL).
            # We can optimize this further, as all dense features share the same count. However, we do not do such an optimization for now.
            total_samples = {name: torch.zeros(param.shape[0]).to(self.device) if ("embedding_dict" in name and aggr_method == "fsl") else torch.zeros(1).to(self.device) for name, param in self.model.state_dict().items()}

            # Remove selected clients with zero samples left.
            selected_clients = list(filter(lambda x: len(x.y_train) > 0, selected_clients))
            if len(selected_clients) == 0:
                continue

            for c in selected_clients:
                # Send the model to the users.
                # If staleness > 0, send a previous model to effectively
                # mimic FL using stale models.
                #c.model.load_state_dict(self.model.state_dict())
                c.model.load_state_dict(self.stale_models[self.rd_ptr].state_dict())

                # Run local training
                num_samples, update = c.train(epochs, bs, client_lr)

                for name in accum:
                    # Collect the gradients
                    accum[name] += update[name] * num_samples

                    # Count total # of samples, handling sparse features with care
                    if "embedding_dict" in name and aggr_method == "fsl":
                        total_samples[name] += (update[name][:, 0] != 0.0) * num_samples
                    else:
                        total_samples[name] += num_samples
            self.rd_ptr = (self.rd_ptr + 1) % (self.staleness + 1)

            # Total_samples zero to 1 to avoid divide by zero
            if aggr_method == "fsl":
                for name in total_samples:
                    if "embedding_dict" in name:
                        total_samples[name][total_samples[name] == 0.0] = 1.0

            # All-reduce the accumulated gradients and the sample stats across GPUs
            reqs = []
            for name in accum:
                reqs.append(dist.all_reduce(total_samples[name], async_op=True))
                reqs.append(dist.all_reduce(accum[name], async_op=True))
            for req in reqs:
                req.wait()

            # Normalize the gradients
            for ii, name in enumerate(accum):
                if "embedding_dict" in name and aggr_method == "fsl":
                    accum[name] /= total_samples[name][:, None]
                # Handling BN (I handle it naively)
                elif "num_batches_tracked" in name:
                    accum[name] //= total_samples[name].long()
                else:
                    accum[name] /= total_samples[name]

            # Feed the aggregated gradient to the server optimizer
            # When staleness > 0, update based on the previous stale model, not the
            # most up-to-date model.
            # TODO: Different write mode must be explored (overwrite / accumulate on).
            # Currently updating with the up-to-date model anyways. Is the below line necessary?
            self.model.load_state_dict(self.stale_models[self.wr_ptr - 1].state_dict())
            updated = set()
            for name, param in self.model.named_parameters():
                param.grad = accum[name].detach().clone()
                updated.add(name)

            # Step
            self.optim.step()
            self.stale_models[self.wr_ptr].load_state_dict(self.model.state_dict())
            self.wr_ptr = (self.wr_ptr + 1) % (self.staleness + 1)

            # Update other states not handled by the optimizer, such as batch statistics
            with torch.no_grad():
                for name, param in self.model.state_dict().items():
                    if name not in updated:
                        if "num_batches_tracked" in name:
                            param += (self.server_lr * accum[name][0]).type(param.dtype)
                        else:
                            param += self.server_lr * accum[name]

            if ((r + 1) % test_freq == 0):
                if self.rank == 0:
                    print(f"Average time/round", np.mean([timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]))
                    timestamps = []
                    print(f"Total requests {total_reads_baseline}, Total reads {total_reads}, Wasted reads {wasted_reads}, Lost reads {lost_reads}, Total samples {tot_sample_cnt}, Max # feat per user {max_feat_per_user}")
                    self.test()

            r += 1


    def test(self, save_result=False):
        total_ans = np.array([])
        total_y = np.array([])
        t = time.time()

        x_test = self.test_data["x_test"]
        y_test = self.test_data["y_test"]
        pred_ans = self.model.predict(x_test, 256)
        total_ans = np.concatenate([total_ans, pred_ans.squeeze()])
        total_y = np.concatenate([total_y, y_test.squeeze()])

        self.print_metrics(total_y, total_ans, f"Total")
        print(f"Total test loss", round(log_loss(total_y, total_ans, labels=[0, 1]), 4))


    def print_metrics(self, y, ans, prefix=""):
        for metric in self.test_metrics:
            if metric == "loss":
                print(f"{prefix} test loss", round(log_loss(y, ans, labels=[0, 1]), 4), flush=True)
            elif metric == "auc":
                if 0 < np.sum(y) < len(y):
                    print(f"{prefix} test AUC", round(roc_auc_score(y, ans), 4), flush=True)
                else:
                    print("Cannot calculate AUC")
            elif metric == "recall":
                # TODO: Assuming 0.5 cutoff
                print(f"{prefix} test Recall", round(recall_score(y, np.round(ans, 0), labels=[0, 1]), 4), flush=True)
            elif metric == "precision":
                print(f"{prefix} test Precision", round(precision_score(y, np.round(ans, 0), labels=[0, 1]), 4), flush=True)
            elif metric == "f1":
                print(f"{prefix} test f1", round(f1_score(y, np.round(ans, 0), labels=[0, 1]), 4), flush=True)
            elif metric == "ap":
                print(f"{prefix} test ap", round(average_precision_score(y, ans), 4), flush=True)
            else:
                print(f"{prefix} Unknown metric {metric}")
