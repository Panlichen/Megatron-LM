#!/bin/bash
module load singularity/3.10.0 
singularity exec \
  --nv \
  --ipc \
  -B /HOME/scz1075/run/Megatron-LM:/workspace/Megatron-LM \
  -B /HOME/scz1075/run/ofccl:/workspace/ofccl \
  -B /HOME/scz1075/run/nccl-tests:/workspace/nccl-tests \
  -B /HOME/scz1075/run/dfccl:/workspace/dfccl \
  -B /HOME/scz1075/run/dfccl-tests:/workspace/dfccl-tests \
  -B /HOME/scz1075/run/oneflow:/workspace/oneflow \
  --env LD_LIBRARY_PATH=/workspace/dfccl/build/lib:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
  megatron_image_20240906.sif 
