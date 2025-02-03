# FL: MovieLens-20
python3 run.py --dataset movielens-20 --model dlrm --logarithm-input --att-weight-normalization --fl --learning-rate 1.0 --client-optimizer sgd --batch-size 0 --server-lr 1e-1 --test-freq 100 --ngpus 1 --clients-per-round 100 --seed 123 --l2-reg-emb 0 --staleness 0 --exclude "s0;hist_s1" --dp-oram-params "" --dp-oram-eps 3.0 --dnn-dropout 0.5 --seed 123 | tee movielens-pub.log
python3 run.py --dataset movielens-20 --model dlrm --logarithm-input --att-weight-normalization --fl --learning-rate 1.0 --client-optimizer sgd --batch-size 0 --server-lr 1e-1 --test-freq 100 --ngpus 1 --clients-per-round 100 --seed 123 --l2-reg-emb 0 --staleness 0 --exclude "s0" --dp-oram-params "hist_s1" --dp-oram-eps 0.0 --dnn-dropout 0.5 --port 29501 | tee movielens-hide_priv_val-eps_inf.log
python3 run.py --dataset movielens-20 --model dlrm --logarithm-input --att-weight-normalization --fl --learning-rate 1.0 --client-optimizer sgd --batch-size 0 --server-lr 1e-1 --test-freq 100 --ngpus 1 --clients-per-round 100 --seed 123 --l2-reg-emb 0 --staleness 0 --exclude "s0" --dp-oram-params "hist_s1" --dp-oram-eps 1.0 --dnn-dropout 0.5 --port 29501 | tee movielens-hide_priv_val-eps_1.0.log
python3 run.py --dataset movielens-20 --model dlrm --logarithm-input --att-weight-normalization --fl --learning-rate 1.0 --client-optimizer sgd --batch-size 0 --server-lr 1e-1 --test-freq 100 --ngpus 1 --clients-per-round 100 --seed 123 --l2-reg-emb 0 --staleness 0 --exclude "s0" --dp-oram-params "hist_s1" --dp-oram-eps 0.1 --dnn-dropout 0.5 --port 29501 | tee movielens-hide_priv_val-eps_0.1.log
python3 run.py --dataset movielens-20 --model dlrm --logarithm-input --att-weight-normalization --fl --learning-rate 1.0 --client-optimizer sgd --batch-size 0 --server-lr 1e-1 --test-freq 100 --ngpus 1 --clients-per-round 100 --seed 123 --l2-reg-emb 0 --staleness 0 --exclude "s0" --dp-oram-params "hist_s1" --dp-oram-eps 0.0 --dnn-dropout 0.5 --port 29501 --dp-max-feats 100 | tee movielens-hide_number_of_priv_val-eps_inf.log
python3 run.py --dataset movielens-20 --model dlrm --logarithm-input --att-weight-normalization --fl --learning-rate 1.0 --client-optimizer sgd --batch-size 0 --server-lr 1e-1 --test-freq 100 --ngpus 1 --clients-per-round 100 --seed 123 --l2-reg-emb 0 --staleness 0 --exclude "s0" --dp-oram-params "hist_s1" --dp-oram-eps 0.01 --dnn-dropout 0.5 --port 29501 --dp-max-feats 100 | tee movielens-hide_number_of_priv_val-eps_1.0.log
python3 run.py --dataset movielens-20 --model dlrm --logarithm-input --att-weight-normalization --fl --learning-rate 1.0 --client-optimizer sgd --batch-size 0 --server-lr 1e-1 --test-freq 100 --ngpus 1 --clients-per-round 100 --seed 123 --l2-reg-emb 0 --staleness 0 --exclude "s0" --dp-oram-params "hist_s1" --dp-oram-eps 0.001 --dnn-dropout 0.5 --port 29501 --dp-max-feats 100 | tee movielens-hide_number_of_priv_val-eps_0.1.log

# FL: Taobao
# python3 run.py --dataset taobao --model dlrm --logarithm-input --att-weight-normalization --fl --learning-rate 1.0 --client-optimizer sgd --batch-size 0 --server-lr 1e-2 --test-freq 100 --ngpus 1 --clients-per-round 100 --seed 123 --l2-reg-emb 0 --staleness 0 --exclude "s0; hist_s9" --dp-oram-params "" --dp-oram-eps 0.0 --seed 123 | tee taobao-pub.log
# python3 run.py --dataset taobao --model dlrm --logarithm-input --att-weight-normalization --fl --learning-rate 1.0 --client-optimizer sgd --batch-size 0 --server-lr 1e-2 --test-freq 100 --ngpus 1 --clients-per-round 100 --seed 123 --l2-reg-emb 0 --staleness 0 --exclude "s0" --dp-oram-params "hist_s9" --dp-oram-eps 0.0 --seed 123 | tee movielens-hide_priv_val-eps_inf.log
# python3 run.py --dataset taobao --model dlrm --logarithm-input --att-weight-normalization --fl --learning-rate 1.0 --client-optimizer sgd --batch-size 0 --server-lr 1e-2 --test-freq 100 --ngpus 1 --clients-per-round 100 --seed 123 --l2-reg-emb 0 --staleness 0 --exclude "s0" --dp-oram-params "hist_s9" --dp-oram-eps 1.0 --seed 123 | tee movielens-hide_priv_val-eps_1.0.log
# python3 run.py --dataset taobao --model dlrm --logarithm-input --att-weight-normalization --fl --learning-rate 1.0 --client-optimizer sgd --batch-size 0 --server-lr 1e-2 --test-freq 100 --ngpus 1 --clients-per-round 100 --seed 123 --l2-reg-emb 0 --staleness 0 --exclude "s0" --dp-oram-params "hist_s9" --dp-oram-eps 0.1 --seed 123 | tee movielens-hide_priv_val-eps_0.1.log
# python3 run.py --dataset taobao --model dlrm --logarithm-input --att-weight-normalization --fl --learning-rate 1.0 --client-optimizer sgd --batch-size 0 --server-lr 1e-2 --test-freq 100 --ngpus 1 --clients-per-round 100 --seed 123 --l2-reg-emb 0 --staleness 0 --exclude "s0" --dp-oram-params "hist_s9" --dp-oram-eps 0.0 --seed 123 --dp-max-feats 100 | tee movielens-hide_number_of_priv_val-eps_inf.log
# python3 run.py --dataset taobao --model dlrm --logarithm-input --att-weight-normalization --fl --learning-rate 1.0 --client-optimizer sgd --batch-size 0 --server-lr 1e-2 --test-freq 100 --ngpus 1 --clients-per-round 100 --seed 123 --l2-reg-emb 0 --staleness 0 --exclude "s0" --dp-oram-params "hist_s9" --dp-oram-eps 0.01 --seed 123 --dp-max-feats 100 | tee movielens-hide_number_of_priv_val-eps_1.0.log
# python3 run.py --dataset taobao --model dlrm --logarithm-input --att-weight-normalization --fl --learning-rate 1.0 --client-optimizer sgd --batch-size 0 --server-lr 1e-2 --test-freq 100 --ngpus 1 --clients-per-round 100 --seed 123 --l2-reg-emb 0 --staleness 0 --exclude "s0" --dp-oram-params "hist_s9" --dp-oram-eps 0.001 --seed 123 --dp-max-feats 100 | tee movielens-hide_number_of_priv_val-eps_0.1.log
