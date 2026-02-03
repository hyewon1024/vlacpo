#!/bin/bash
# ##############################################
# # Multi-sample handcrafted
# ##############################################
GROUP_NAME=safetygym_failure_pred
BASE_PATH=/home/hyewon/cpon_vla/safety-gym/dataset_collect/safetygym_point

for SEED in 0 1 2; do
    python -m failure_prob.train -m \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=safetygym_point \
        dataset.data_path_prefix=${BASE_PATH} \
        dataset.unseen_task_ratio=0.5 \
        dataset.load_to_cuda=False \
        model=lstm \
        model.batch_size=64 \
        model.lr=1e-4,3e-4,1e-3 \
        model.lambda_reg=1e-3,1e-2,1e-1,1 \
        train.seed=${SEED} \
        train.exp_suffix=lstm_multi_task
done
done

# ##############################################
# # LSTM and MLP
# ##############################################
# for ENV_NAME in point; do
# for SEED in 0 1 2; do

#     python -m failure_prob.train --multirun \
#         train.wandb_group_name=${GROUP_NAME} \
#         dataset=safetygym_${ENV_NAME} \
#         dataset.data_path_prefix=${DATA_ROOT} \
#         dataset.load_to_cuda=False \
#         model=lstm \
#         model.batch_size=64 \
#         model.lr=1e-4,3e-4,1e-3 \
#         model.lambda_reg=1e-3,1e-2,1e-1,1 \
#         train.seed=${SEED} \
#         train.exp_suffix=lstm

#     python -m failure_prob.train --multirun \
#         train.wandb_group_name=${GROUP_NAME} \
#         dataset=safetygym_${ENV_NAME} \
#         dataset.data_path_prefix=${DATA_ROOT} \
#         dataset.load_to_cuda=False \
#         model=indep \
#         model.batch_size=64 \
#         model.lr=1e-4,3e-4,1e-3 \
#         model.lambda_reg=1e-3,1e-2,1e-1,1 \
#         train.seed=${SEED} \
#         train.exp_suffix=mlp
# done
# done

# ##############################################
# # Embedding-based
# ##############################################
# for ENV_NAME in point; do

#     python -m failure_prob.train --multirun \
#         train.wandb_group_name=${GROUP_NAME} \
#         dataset=safetygym_${ENV_NAME} \
#         dataset.data_path_prefix=${DATA_ROOT} \
#         dataset.load_to_cuda=False \
#         model=embed \
#         model.n_epochs=1 \
#         model.distance=cosine,euclid \
#         model.use_success_only=False \
#         model.topk=1,5,10 \
#         model.cumsum=False,True \
#         train.seed=0-1-2 \
#         train.exp_suffix=embed

#     python -m failure_prob.train --multirun \
#         train.wandb_group_name=${GROUP_NAME} \
#         dataset=safetygym_${ENV_NAME} \
#         dataset.data_path_prefix=${DATA_ROOT} \
#         dataset.load_to_cuda=False \
#         model=embed \
#         model.distance=mahala \
#         model.use_success_only=False \
#         model.cumsum=False,True \
#         train.seed=0-1-2 \
#         train.exp_suffix=embed
# done

# ##############################################
# # RND / LogpZO
# ##############################################
# for ENV_NAME in point; do

#     python -m failure_prob.train --multirun \
#         train.wandb_group_name=${GROUP_NAME} \
#         dataset=safetygym_${ENV_NAME} \
#         dataset.data_path_prefix=${DATA_ROOT} \
#         dataset.load_to_cuda=False \
#         model=rnd \
#         train.roc_every=50 \
#         model.batch_size=32 \
#         model.use_success_only=False \
#         train.seed=0-1-2 \
#         train.exp_suffix=chen

#     python -m failure_prob.train --multirun \
#         train.wandb_group_name=${GROUP_NAME} \
#         dataset=safetygym_${ENV_NAME} \
#         dataset.data_path_prefix=${DATA_ROOT} \
#         dataset.load_to_cuda=False \
#         model=logpzo \
#         train.roc_every=50 \
#         model.batch_size=32 \
#         model.forward_chunk_size=512 \
#         model.use_success_only=False \
#         train.seed=0-1-2 \
#         train.exp_suffix=chen
# done

# ##############################################
# # Handcrafted
# ##############################################
# for ENV_NAME in point; do
#     python -m failure_prob.train --multirun \
#         train.wandb_group_name=${GROUP_NAME} \
#         dataset=safetygym_${ENV_NAME} \
#         dataset.data_path_prefix=${DATA_ROOT} \
#         train.log_precomputed_only=True \
#         train.seed=0-1-2 \
#         train.exp_suffix=handcrafted
# done

# ##############################################
# # Multi-sample handcrafted
# ##############################################
# for ENV_NAME in point; do
#     python -m failure_prob.train --multirun \
#         train.wandb_group_name=${GROUP_NAME} \
#         dataset=safetygym_${ENV_NAME}_multi \
#         dataset.data_path_prefix=${DATA_ROOT} \
#         train.log_precomputed_only=True \
#         train.seed=0-1-2 \
#         train.exp_suffix=handcrafted_multi
# done

