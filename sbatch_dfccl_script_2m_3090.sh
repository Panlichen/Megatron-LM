#!/bin/bash
#SBATCH -N 2
#SBATCH --gres=gpu:8
#SBATCH --qos=gpugpu
#SBATCH -x g0048
#SBATCH -x g0022
module purge
module load anaconda/2021.05 cuda/12.1 gcc/11.2 cudnn/8.9.6_cuda12.x

module load cmake/3.22.0

source activate torch221_cuda121
# export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_bond_0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_GID_INDEX=3
### 获取每个节点的 hostname
for i in `scontrol show hostnames`
do
  let k=k+1
  host[$k]=$i
  echo $k
  echo ${host[$k]}
done





GPUS_PER_NODE=8
export DP_DFCCL=1
export TP_DFCCL=0
export PD_PATH=/HOME/scz1075/run/Megatron-LM/dev/py_dfccl
export PYTHONPATH=/HOME/scz1075/run/Megatron-LM:$PYTHONPATH
TP_SIZE=2
PP_SIZE=2

# 无tp和pp
# MICRO_BATCH_SIZE 9
# GLOBAL_BATCH_SIZE 72

# 222
MICRO_BATCH_SIZE=18
GLOBAL_BATCH_SIZE=288

#MASTER_ADDR="${host[1]}"
MASTER_PORT=6008
NNODES=2
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
CHECKPOINT_PATH=/HOME/scz1075/run/Megatron-LM/experiments/codeparrot-small
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
DATA_PATH=codeparrot_content_document
GPT_ARGS="--num-layers 12
--hidden-size 768
--num-attention-heads 12
--seq-length 1024
--max-position-embeddings 1024
--micro-batch-size $MICRO_BATCH_SIZE
--global-batch-size $GLOBAL_BATCH_SIZE
--lr 0.0005
--train-iters 200
--lr-decay-iters 150000
--lr-decay-style cosine
--lr-warmup-iters 2000
--weight-decay .1
--adam-beta2 .999
--log-interval 1
--save-interval 2000
--eval-interval 200
--eval-iters 10
"
TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"
rm -rf experiments

# 主节点
python3 -m torch.distributed.launch --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank 0 --master_addr "${host[1]}" --master_port $MASTER_PORT \
        pretrain_gpt.py \
        --no-async-tensor-model-parallel-allreduce \
        --tensor-model-parallel-size $TP_SIZE \
        --pipeline-model-parallel-size $PP_SIZE \
        $GPT_ARGS \
        --vocab-file $VOCAB_FILE \
        --merge-file $MERGE_FILE \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        $TENSORBOARD_ARGS   >> slurm-${SLURM_JOB_ID}-rank0.log 2>&1 &


## 使用 srun 运行第二个节点
srun -N 1 --gres=gpu:${GPUS_PER_NODE} -w ${host[2]} \
python3 -m torch.distributed.launch --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank 1 --master_addr "${host[1]}" --master_port $MASTER_PORT \
        pretrain_gpt.py \
        --no-async-tensor-model-parallel-allreduce \
        --tensor-model-parallel-size $TP_SIZE \
        --pipeline-model-parallel-size $PP_SIZE \
        $GPT_ARGS \
        --vocab-file $VOCAB_FILE \
        --merge-file $MERGE_FILE \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        $TENSORBOARD_ARGS   >> slurm-${SLURM_JOB_ID}-rank1.log 2>&1 &

wait
