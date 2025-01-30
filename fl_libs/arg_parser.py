# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

def get_args():
    ### parse arguments ###
    parser = argparse.ArgumentParser()

    ## common model-related parameters
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--model", type=str, default="din")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hist-len", type=int, default=10)
    parser.add_argument("--emb-dim", type=int, default=16)
    parser.add_argument("--l2-reg-dnn", type=float, default=0.0)
    parser.add_argument("--l2-reg-emb", type=float, default=1e-6)
    parser.add_argument("--l2-reg-linear", type=float, default=0.0)
    parser.add_argument("--combiner", type=str, default="mean")
    parser.add_argument("--dnn-dropout", type=float, default=0.0)

    # Common system related params
    parser.add_argument("--ngpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--port", type=int, default=29500)
    parser.add_argument("--metrics", type=str, default="loss,auc,ap")

    # For DNN-base models (for DLRM, this is the top MLP)
    parser.add_argument("--dnn-hidden-units", type=str, default="256")

    # For DIN/DIEN
    parser.add_argument("--att-hidden-units", type=str, default="64-16")
    parser.add_argument("--dice-norm-type", type=str, default=None)
    parser.add_argument("--att-activation", type=str, default="Dice")
    parser.add_argument("--att-weight-normalization", action="store_true", default=False)

    # For DCNv1/v2
    parser.add_argument("--cross-num", type=int, default=2)

    # For DLRM
    parser.add_argument("--bot-dnn-hidden-units", type=str, default="16")

    ## FL-related params
    parser.add_argument("--fl", action="store_true", default=False)
    parser.add_argument("--aggr-method", type=str, default="fedavg")
    parser.add_argument("--min-sample-size", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=10)
    parser.add_argument("--clients-per-round", type=int, default=10)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--server-optimizer", type=str, default="adagrad")
    parser.add_argument("--client-optimizer", type=str, default="sgd")
    parser.add_argument("--data-split", type=str, default="niid")
    parser.add_argument("--num-clients", type=int, default=0)
    parser.add_argument("--server-lr", type=float, default=0.01)
    parser.add_argument("--num-global-epochs", type=int, default=1)

    # Input preprocessing
    parser.add_argument("--dataset", type=str, default="taobao")
    parser.add_argument("--logarithm-input", action="store_true", default=False)
    parser.add_argument("--standardize-input", action="store_true", default=False)
    parser.add_argument("--target-ctr", type=float, default=-1)

    # Exclude a certain feature. For DIN/DLRM, we exclude s0 (user id)
    parser.add_argument("--exclude", type=str, default="s0")

    # Staleness-related params
    parser.add_argument("--staleness", type=int, default=0)

    # DP-ORAM related params
    parser.add_argument("--dp-oram-params", type=str, default="")
    parser.add_argument("--dp-oram-eps", type=float, default=0.0)
    parser.add_argument("--dp-oram-method", type=str, default="fcfs")
    parser.add_argument("--dp-max-feats", type=int, default=0) # 0 means no limit

    return parser.parse_args()
