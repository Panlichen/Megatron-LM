GPUS_PER_NODE=8

export DP_DFCCL=0
export TP_DFCCL=0
# export PD_PATH=/HOME/scz1075/run/Megatron-LM/dev/py_dfccl
# export PYTHONPATH=/HOME/scz1075/run/Megatron-LM:$PYTHONPATH
# export TORCH_LOGS="+dynamo"
# export TORCHDYNAMO_VERBOSE=1

# 无tp和pp
# MICRO_BATCH_SIZE 9
# GLOBAL_BATCH_SIZE 72

# 222
MICRO_BATCH_SIZE=18
GLOBAL_BATCH_SIZE=288

MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
CHECKPOINT_PATH=/workspace/Megatron-LM/experiments/codeparrot-small
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
python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
        pretrain_gpt.py \
        --no-async-tensor-model-parallel-allreduce \
        --tensor-model-parallel-size 2 \
        --pipeline-model-parallel-size 2 \
        $GPT_ARGS \
        --vocab-file $VOCAB_FILE \
        --merge-file $MERGE_FILE \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        $TENSORBOARD_ARGS > a.out 2>&1