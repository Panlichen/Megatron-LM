# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch

import os
import sys
import time
import warnings
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn.parameter import Parameter

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import (
    get_global_memory_buffer,
    get_tensor_and_expert_parallel_rank,
    get_tensor_and_expert_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from ..dist_checkpointing.mapping import ShardedStateDict
from ..transformer.utils import make_sharded_tensors_for_checkpoint
from ..utils import make_tp_sharded_tensor_for_checkpoint, prepare_input_tensors_for_wgrad_compute
from .mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from .random import get_cuda_rng_tracker, get_expert_parallel_rng_tracker_name
from .utils import VocabUtility, divide

dfccl_path = os.environ.get("PD_PATH", '/workspace/Megatron-LM/dev/py_dfccl')
# dfccl_path = '/workspace/Megatron-LM/dev/py_dfccl'
# sys.path.append(dfccl_path)
# from dfccl_wrapper import DfcclWrapper
# import dfccl_wrapper

_grad_accum_fusion_available = True
try:
    import fused_weight_gradient_mlp_cuda
except ImportError:
    _grad_accum_fusion_available = False

_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {
    'tensor_model_parallel': False,
    'partition_dim': -1,
    'partition_stride': 1,
}


def param_is_not_tensor_parallel_duplicate(param):
    """Returns true if the passed-in parameter is not a duplicate parameter
    on another TP rank."""
    return (hasattr(param, 'tensor_model_parallel') and param.tensor_model_parallel) or (
        get_tensor_model_parallel_rank() == 0
    )


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    """Sets tp attributes to tensor"""
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, 'tensor_model_parallel', is_parallel)
    setattr(tensor, 'partition_dim', dim)
    setattr(tensor, 'partition_stride', stride)


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
    def maybe_set(attribute, value):
        if not hasattr(tensor, attribute):
            setattr(tensor, attribute, value)

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute):
            setattr(destination_tensor, attribute, getattr(source_tensor, attribute))

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)


def _initialize_affine_weight_gpu(
    weight, init_method, partition_dim, stride=1, expert_parallel=False
):
    """Initialize affine weight for model parallel on GPU."""

    set_tensor_model_parallel_attributes(
        tensor=weight, is_parallel=True, dim=partition_dim, stride=stride
    )

    if not expert_parallel:
        with get_cuda_rng_tracker().fork():
            init_method(weight)
    else:
        with get_cuda_rng_tracker().fork(get_expert_parallel_rng_tracker_name()):
            init_method(weight)


def _initialize_affine_weight_cpu(
    weight,
    output_size,
    input_size,
    per_partition_size,
    partition_dim,
    init_method,
    stride=1,
    return_master_weight=False,
    *,
    params_dtype=torch.float32,
    rank=None,
    world_size=None,
):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    set_tensor_model_parallel_attributes(
        tensor=weight, is_parallel=True, dim=partition_dim, stride=stride
    )

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size, dtype=torch.float, requires_grad=False)
    init_method(master_weight)
    master_weight = master_weight.to(dtype=params_dtype)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size, dim=partition_dim)
    if rank is None:
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        # all tensors must live on the same device
        cpu_weight = torch.cat(my_weight_list, dim=partition_dim).to_dense()
        weight.data.copy_(cpu_weight)
    if return_master_weight:
        return master_weight
    return None


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        reduce_scatter_embeddings: Decides whether to perform ReduceScatter after embedding lookup

    Keyword Args:
        config: A megatron.core.ModelParallelConfig object
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        init_method: Callable,
        reduce_scatter_embeddings: bool = False,
        config: ModelParallelConfig,
    ):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.reduce_scatter_embeddings = reduce_scatter_embeddings
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        (self.vocab_start_index, self.vocab_end_index) = (
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings,
                get_tensor_model_parallel_rank(),
                self.tensor_model_parallel_size,
            )
        )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index
        self.deterministic_mode = config.deterministic_mode

        # Allocate weights and initialize.
        if config.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition, self.embedding_dim, dtype=config.params_dtype
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight,
                    self.num_embeddings,
                    self.embedding_dim,
                    self.num_embeddings_per_partition,
                    0,
                    init_method,
                    params_dtype=config.params_dtype,
                )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(self.weight, init_method, partition_dim=0, stride=1)

    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
        # Get the embeddings.
        if self.deterministic_mode:
            output_parallel = self.weight[masked_input]
        else:
            # F.embedding currently has a non-deterministic backward function
            output_parallel = F.embedding(masked_input, self.weight)
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0

        if self.reduce_scatter_embeddings:
            # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
            output_parallel = output_parallel.transpose(0, 1).contiguous()
            output = reduce_scatter_to_sequence_parallel_region(output_parallel)
        else:
            # Reduce across all the model parallel GPUs.
            output = reduce_from_tensor_model_parallel_region(output_parallel)
        return output

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Non-default implementation for embeddings due to `allow_shape_mismatch` param"""
        state_dict = self.state_dict(prefix='', keep_vars=True)

        weight_prefix = f'{prefix}weight'
        return {
            weight_prefix: make_tp_sharded_tensor_for_checkpoint(
                tensor=state_dict['weight'],
                key=weight_prefix,
                allow_shape_mismatch=True,
                prepend_offsets=sharded_offsets,
            )
        }


class LinearWithFrozenWeight(torch.autograd.Function):
    """Linear operator that does not calculate gradient for weight.
    This op and LinearWithGradAccumulationAndAsyncCommunication performs
    mathematically-identical forward and DGRAD.

    Conceptually this op is the same as torch.nn.functional.linear with
    weight.requires_grad==False, but in experiments they are not identical
    mathematically."""

    @staticmethod
    @custom_fwd
    def forward(ctx, input, weight, bias, allreduce_dgrad):
        ctx.save_for_backward(weight)
        ctx.allreduce_dgrad = allreduce_dgrad
        output = torch.matmul(input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        (weight,) = ctx.saved_tensors
        grad_input = grad_output.matmul(weight)

        if ctx.allreduce_dgrad:
            # 没有调用这里
            # global_rank = torch.distributed.get_rank()
            # local_rank = os.environ.get("LOCAL_RANK", 0)
            # group_size = torch.distributed.get_world_size(group=get_tensor_model_parallel_group())
            # group_rank = torch.distributed.get_rank(group=get_tensor_model_parallel_group())
            
            # print(f"TP global rank {global_rank}, local rank {local_rank}, ddp group rank {group_rank}/{group_size}, AR in LinearWithFrozenWeight")
            # All-reduce. Note: here async and sync are effectively the same.
            torch.distributed.all_reduce(grad_input, group=get_tensor_model_parallel_group())

        return grad_input, None, None, None


def linear_with_frozen_weight(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool,
    async_grad_allreduce: bool,
    sequence_parallel: bool,
    grad_output_buffer: Optional[List[torch.Tensor]] = None,
    wgrad_deferral_limit: None = None,
    allreduce_dgrad: bool = None,
) -> torch.Tensor:
    """Linear layer execution with weight.requires_grad == False.

    This function handles linear layers with weight frozen (untrainable).
    In the forward, it only saves weight and does not save input activations.
    In the backward, it does not perform weight gradient calculation, or
    weight gradient allreduce.

    Args:

    input (torch.Tensor required): input like torch.nn.functional.linear

    weight (torch.Tensor required): weight like torch.nn.functional.linear

    bias (torch.Tensor optional): bias like torch.nn.functional.linear

    gradient_accumulation_fusion (bool required): dummy argument, used to
    keep the API unified between all forward implementation functions.

    async_grad_allreduce (bool required): dummy argument, used to
    keep the API unified between all forward implementation functions.

    sequence_parallel (bool required): Indicates that sequence
        parallelism is used and thus in the forward pass the input is
        all gathered, and the backward pass the input gradients are
        reduce scattered.

    grad_output_buffer (List[torch.Tensor] optional): dummy argument, used to
    keep the API unified between all forward implementation functions.

    wgrad_deferral_limit (int optional): dummy argument, used to
    keep the API unified between all forward implementation functions.

    allreduce_dgrad (bool): Do the allreduce of input gradients.
        Here, async and sync allreduce are the same. If sequence_parallel is
        True, this must be False, as no all reduce is performed.

    """

    assert grad_output_buffer is None, (
        "grad_output_buffer kwarg is only supported with "
        "linear_with_grad_accumulation_and_async_allreduce"
    )

    assert wgrad_deferral_limit is None, (
        "This arg is only supported with " "linear_with_grad_accumulation_and_async_allreduce"
    )

    if sequence_parallel:
        input = gather_from_sequence_parallel_region(input, tensor_parallel_output_grad=True)
    else:
        input = input

    if allreduce_dgrad is None:
        warnings.warn(
            "`async_grad_allreduce` is deprecated and will be removed in a future release. "
            "Please ue `allreduce_dgrad` instead."
        )
        allreduce_dgrad = async_grad_allreduce

    args = [input, weight, bias, allreduce_dgrad]

    return LinearWithFrozenWeight.apply(*args)


class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """See linear_with_grad_accumulation_and_async_allreduce"""

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
    ):
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.sequence_parallel = sequence_parallel
        ctx.wgrad_deferral_limit = wgrad_deferral_limit
        ctx.grad_output_buffer = grad_output_buffer

        if sequence_parallel:
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            torch.distributed._all_gather_base(
                all_gather_buffer, input, group=get_tensor_model_parallel_group()
            )
            total_input = all_gather_buffer
        else:
            total_input = input

        output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        global_rank = torch.distributed.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        group_size = torch.distributed.get_world_size(group=get_tensor_model_parallel_group())
        group_rank = torch.distributed.get_rank(group=get_tensor_model_parallel_group())
        env_tp_dfccl = int(os.environ.get("TP_DFCCL", 0))
        coll_id = -1

        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        grad_output_buffer = ctx.grad_output_buffer
        wgrad_deferral_limit = ctx.wgrad_deferral_limit

        wgrad_compute = True
        if grad_output_buffer is not None:
            if wgrad_deferral_limit == 0 or len(grad_output_buffer) < wgrad_deferral_limit:
                grad_output_buffer.append(grad_output)
                wgrad_compute = False

        if wgrad_compute:
            if ctx.sequence_parallel:
                world_size = get_tensor_model_parallel_world_size()
                dim_size = list(input.size())
                dim_size[0] = dim_size[0] * world_size

                all_gather_buffer = get_global_memory_buffer().get_tensor(
                    dim_size, input.dtype, "mpu"
                )
                handle = torch.distributed._all_gather_base(
                    all_gather_buffer, input, group=get_tensor_model_parallel_group(), async_op=True
                )

                # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
                # gather is scheduled before the input gradient computation
                total_input = all_gather_buffer
            else:
                total_input = input
        grad_input = grad_output.matmul(weight)

        if ctx.sequence_parallel and wgrad_compute:
            handle.wait()

        if wgrad_compute:
            grad_output, total_input = prepare_input_tensors_for_wgrad_compute(
                grad_output, total_input
            )

        if env_tp_dfccl:

            # print(f"global rank {global_rank}, local rank {local_rank}, tp group rank {group_rank}/{group_size}, tp_global_tensor_counter is {dfccl_wrapper.get_tp_global_tensor_counter()}")

            if dfccl_wrapper.dfccl_ext is None:
                dfccl_wrapper.dfccl_wrapper_object = DfcclWrapper(global_rank, local_rank, -1, group_rank, group_size, get_tensor_model_parallel_group())
                dfccl_wrapper.dfccl_ext = dfccl_wrapper.dfccl_wrapper_object.init_dfccl_ext()
            
            coll_id = dfccl_wrapper.get_tp_global_tensor_counter()
            
            coll_already_init_nccl_comm = dfccl_wrapper.dfccl_wrapper_object.coll_already_init_nccl_comm.get(coll_id, False)
            if not coll_already_init_nccl_comm:
                # print(f"global rank {global_rank}, local rank {local_rank}, tp group rank {group_rank}/{group_size}, prepare for coll_id {coll_id}")
                dfccl_wrapper.dfccl_wrapper_object.prepare_dfccl_ar(coll_id=coll_id, parallel_type="TP", tensor=grad_input)

            if dfccl_wrapper.get_seen_all_tp_colls() and not dfccl_wrapper.get_tp_already_call_finalize():
                # print(f"global rank {global_rank}, local rank {local_rank}, tp group rank {group_rank}/{group_size}, call dfccl_finalize")
                dfccl_wrapper.dfccl_wrapper_object.dfccl_finalize()  # 发现seen_all_tp_colls是True了, 可以Finalize了, 只需要调用一次, 要保证仅仅调用一次
                dfccl_wrapper.set_tp_already_call_finalize()

        if ctx.allreduce_dgrad:
            time_start = time.time()
            # Asynchronous all-reduce
            
            # if global_rank == 6:
            #     print(f"TP global rank {global_rank}, local rank {local_rank}, ddp group rank {group_rank}/{group_size}, AR in LinearWithGradAccumulationAndAsyncCommunication")
            if env_tp_dfccl:
                if dfccl_wrapper.get_seen_all_tp_colls():
                    # print(f"global rank {global_rank}, local rank {local_rank}, tp group rank {group_rank}/{group_size}, call_dfccl_ar for coll_id {coll_id}")
                    dfccl_wrapper.dfccl_wrapper_object.call_dfccl_ar(coll_id=coll_id, tensor=grad_input)
                else:
                    handle = torch.distributed.all_reduce(
                        grad_input, group=get_tensor_model_parallel_group(), async_op=True
                    )
            else:
                handle = torch.distributed.all_reduce(
                    grad_input, group=get_tensor_model_parallel_group(), async_op=True
                )
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # all-reduce is scheduled before the weight gradient computation
            time_end = time.time()
            # print(f"global rank {global_rank}, local rank {local_rank}, tp group rank {group_rank}/{group_size}, call tp ar takes {time_end - time_start:.4f} s")

        if ctx.sequence_parallel:
            # print(f"ctx.sequence_parallel: {ctx.sequence_parallel}")
            assert not ctx.allreduce_dgrad
            dim_size = list(input.size())
            sub_grad_input = torch.empty(
                dim_size, dtype=input.dtype, device=torch.cuda.current_device(), requires_grad=False
            )
            # reduce_scatter
            handle = torch.distributed._reduce_scatter_base(
                sub_grad_input, grad_input, group=get_tensor_model_parallel_group(), async_op=True
            )
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # reduce scatter is scheduled before the weight gradient computation

        if ctx.gradient_accumulation_fusion:
            # if global_rank == 6:
            #     print(f"ctx.gradient_accumulation_fusion: {ctx.gradient_accumulation_fusion}")
            if wgrad_compute:
                if weight.main_grad.dtype == torch.float32:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                        total_input, grad_output, weight.main_grad
                    )
                elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                        total_input, grad_output, weight.main_grad
                    )
                else:
                    raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")

            if hasattr(weight, 'grad_added_to_main_grad'):
                # When overlap_grad_reduce is True, need to ensure that backward hooks
                # are all run on the main backprop thread to prevent deadlocks. Setup
                # dummy grad_weight tensor to prevent backward hooks from being run
                # in a background thread.
                if getattr(weight, 'zero_out_wgrad', False):
                    grad_weight = torch.zeros(
                        weight.main_grad.shape,
                        dtype=input.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    grad_weight = torch.empty(
                        weight.main_grad.shape,
                        dtype=input.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                weight.grad_added_to_main_grad = True
            else:
                grad_weight = None
        else:
            grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.sequence_parallel:
            handle.wait()
            # Need to return None's as gradient has to flow for all the input arguments
            # provided during forward
            return sub_grad_input, grad_weight, grad_bias, None, None, None, None, None

        if ctx.allreduce_dgrad:
            time_start = time.time()
            if env_tp_dfccl:
                if dfccl_wrapper.get_seen_all_tp_colls():
                    dfccl_wrapper.dfccl_wrapper_object.wait_dfccl_cqe_4_coll(coll_id)
                    # print(f"global rank {global_rank}, local rank {local_rank}, tp group rank {group_rank}/{group_size}, coll_id {coll_id} done")
                else:
                    handle.wait()
            else:
                handle.wait()
            time_end = time.time()
            # print(f"global rank {global_rank}, local rank {local_rank}, tp group rank {group_rank}/{group_size}, wait tp ar takes {time_end - time_start:.4f} s")

        if env_tp_dfccl:
            dfccl_wrapper.increase_tp_global_tensor_counter()  # 最后再更新coll_id

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


def linear_with_grad_accumulation_and_async_allreduce(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool,
    sequence_parallel: bool,
    allreduce_dgrad: bool,
    async_grad_allreduce: Optional[bool] = None,
    grad_output_buffer: Optional[List[torch.Tensor]] = None,
    wgrad_deferral_limit: Optional[int] = 0,
) -> torch.Tensor:
    """Linear layer execution with asynchronous communication and
    gradient accumulation fusion in backprop.

    This has the option to accumulate the result of backprop
    calculation into an existing gradient buffer, preventing the need
    to do an additional addition kernel after the gradient
    calculation.

    Additionally, the tensor parallel all reduce of the input
    gradients can be done asynchronously with the calculation of
    the weight gradients.

    In the case of sequence parallelism, the reduce scatter of the
    input gradients is done asynchronously with the calcluation of the
    weight gradients.

    Use of this module requires that the environment variable
    CUDA_DEVICE_MAX_CONNECTIONS=1. There are a few collective
    operations, noted in the code, that should be scheduled before
    compute kernels to overlap the communication with the computation,
    which is necessary for a speedup but not for correctness so that
    ordering isn't imposed by the scheduler. Setting
    CUDA_DEVICE_MAX_CONNECTIONS=1 forces the kernels to be scheduled
    in the order they are called.

    Args:
        input (torch.Tensor required): input like torch.nn.functional.linear

        weight (torch.Tensor required): weight like torch.nn.functional.linear

        bias (torch.Tensor optional): bias like torch.nn.functional.linear

        gradient_accumulation_fusion (bool required): Perform the gradient
            accumulation fusion, requires the custom CUDA extension
            fused_weight_gradient_mlp_cuda module. To use
            gradient_accumulation_fusion you must install APEX with
            --cpp_ext and --cuda_ext. For example: "pip install
            --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\"
            " Note that the extension requires CUDA>=11. Otherwise, you
            must turn off gradient accumulation fusion."

        allreduce_dgrad (bool required): Do the allreduce of input gradients.
            The allreduce is done asynchronously with the computation of weight
            gradients. If sequence_parallel is True, this must be
            False, as no all reduce is performed.

        async_grad_allreduce (bool optional): Do the allreduce of input
            gradients asyncronously with the computation of weight
            gradients. If sequence_parallel is True, this must be
            False, as no all reduce is performed. Will be deprecated with 0.10.0

        sequence_parallel (bool required): Indicates that sequence
            parallelism is used and thus in the forward pass the input is
            all gathered, and the backward pass the input gradients are
            reduce scattered.

        grad_output_buffer (List[torch.Tensor] optional): Buffer used to save
            output gradients when embedding table wgrad compute is deferred.
            Defaults to None.

        wgrad_deferral_limit (int optional): Limit on the number of
            micro-batches for which embedding weight gradient GEMM should be
            deferred. Disable by setting this to 0. Defaults to 0.

    """
    if async_grad_allreduce is not None:
        warnings.warn(
            "async_grad_allreduce is deprecated, not in use anymore and will"
            " be fully removed with 0.10.0. Please use allreduce_dgrad instead."
        )

    args = [
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
    ]

    if not linear_with_grad_accumulation_and_async_allreduce.warned:
        if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
            if sequence_parallel:
                warnings.warn(
                    "When using sequence parallelism it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup"
                )
                linear_with_grad_accumulation_and_async_allreduce.warned = True

            if allreduce_dgrad:
                warnings.warn(
                    "When using async grad allreduce it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup"
                )
                linear_with_grad_accumulation_and_async_allreduce.warned = True

    return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)


linear_with_grad_accumulation_and_async_allreduce.warned = False


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size:
            first dimension of matrix A.
        output_size:
            second dimension of matrix A.
        bias:
            If true, add bias
        gather_output:
            If true, call all-gather on output and make Y available to all GPUs,
            otherwise, every GPU will have its output which is Y_i = XA_i
        init_method:
            method to initialize weights. Note that bias is always set to zero.
        stride:
            For the strided linear layers.
        keep_master_weight_for_test:
            This was added for testing and should be set to False. It
            returns the master weights used for initialization.
        skip_bias_add:
            If True, do not add the bias term, instead return it to be added by the
            caller. This enables performance optimations where bias can be fused with other
            elementwise operations.
        skip_weight_param_allocation:
            If True, weight parameter is not allocated and must be passed
            as a keyword argument `weight` during the forward pass. Note that this does not
            affect bias, which will be allocated if bias is True. Defaults to False.
        embedding_activation_buffer:
            This buffer holds the input activations of the final embedding
            linear layer on the last pipeline stage when defer_embedding_wgrad_compute is enabled.
        grad_output_buffer:
            This buffer holds the gradient outputs of the final embedding linear
            layer on the last pipeline stage when defer_embedding_wgrad_compute is enabled.
        is_expert:
            If True, the layer is treated as an MoE expert layer.
        config:
            ModelParallelConfig object
        tp_comm_buffer_name:
            Communication buffer name is not used in non-Transformer-Engine modules.
        disable_grad_reduce:
            If True, reduction of output gradients across tensor-parallel ranks
            will be disabled. Defaults to False. This feature is used by Lora Adapter in Nemo to
            delay and fuse reduction along with other gradients for performance optimization.
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias=True,
        gather_output=False,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        skip_weight_param_allocation: bool = False,
        embedding_activation_buffer: Optional[List[torch.Tensor]] = None,
        grad_output_buffer: Optional[List[torch.Tensor]] = None,
        is_expert: bool = False,
        tp_comm_buffer_name: str = None,  # Not used
        disable_grad_reduce: bool = False,
    ):
        super(ColumnParallelLinear, self).__init__()  # 调用父类的初始化方法

        # Keep input parameters
        self.input_size = input_size  # 保存输入大小
        self.output_size = output_size  # 保存输出大小
        self.gather_output = gather_output  # 是否收集输出
        # Divide the weight matrix along the last dimension.
        self.skip_bias_add = skip_bias_add  # 是否跳过偏置加法
        self.is_expert = is_expert  # 是否为专家层
        self.expert_parallel = config.expert_model_parallel_size > 1  # 是否使用专家并行
        self.embedding_activation_buffer = embedding_activation_buffer  # 嵌入激活缓冲区
        self.grad_output_buffer = grad_output_buffer  # 梯度输出缓冲区
        self.config = config  # 配置对象
        self.disable_grad_reduce = disable_grad_reduce  # 是否禁用梯度归约

        self.explicit_expert_comm = self.is_expert and (  # 是否需要显式专家通信
            config.tensor_model_parallel_size > 1 or self.expert_parallel
        )
        if self.explicit_expert_comm and config.moe_extended_tp:  # 如果需要显式专家通信且使用扩展张量并行
            world_size = get_tensor_and_expert_parallel_world_size()  # 获取张量和专家并行的世界大小
            rank = get_tensor_and_expert_parallel_rank()  # 获取张量和专家并行的排名
        else:
            world_size = get_tensor_model_parallel_world_size()  # 获取张量模型并行的世界大小
            rank = get_tensor_model_parallel_rank()  # 获取张量模型并行的排名

        self.output_size_per_partition = divide(output_size, world_size)  # 计算每个分区的输出大小

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if not skip_weight_param_allocation:  # 如果不跳过权重参数分配
            if config.use_cpu_initialization:  # 如果使用CPU初始化
                self.weight = Parameter(  # 创建权重参数
                    torch.empty(
                        self.output_size_per_partition, self.input_size, dtype=config.params_dtype
                    )
                )
                if config.perform_initialization:  # 如果执行初始化
                    self.master_weight = _initialize_affine_weight_cpu(  # 初始化CPU上的仿射权重
                        self.weight,
                        self.output_size,
                        self.input_size,
                        self.output_size_per_partition,
                        0,
                        init_method,
                        stride=stride,
                        return_master_weight=keep_master_weight_for_test,
                        rank=rank,
                        world_size=world_size,
                    )
            else:
                self.weight = Parameter(  # 创建GPU上的权重参数
                    torch.empty(
                        self.output_size_per_partition,
                        self.input_size,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
                if config.perform_initialization:  # 如果执行初始化
                    _initialize_affine_weight_gpu(  # 初始化GPU上的仿射权重
                        self.weight,
                        init_method,
                        partition_dim=0,
                        stride=stride,
                        expert_parallel=(self.is_expert and self.expert_parallel),
                    )

            setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))  # 设置权重的allreduce属性
        else:
            self.weight = None  # 如果跳过权重参数分配，则权重为None

        if bias:  # 如果使用偏置
            if config.use_cpu_initialization:  # 如果使用CPU初始化
                self.bias = Parameter(  # 创建CPU上的偏置参数
                    torch.empty(self.output_size_per_partition, dtype=config.params_dtype)
                )
            else:
                self.bias = Parameter(  # 创建GPU上的偏置参数
                    torch.empty(
                        self.output_size_per_partition,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)  # 设置偏置的张量模型并行属性
            if config.perform_initialization:  # 如果执行初始化
                # Always initialize bias to zero.
                with torch.no_grad():
                    self.bias.zero_()  # 将偏置初始化为零
            setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))  # 设置偏置的allreduce属性
        else:
            self.register_parameter('bias', None)  # 如果不使用偏置，注册None作为偏置参数

        self.sequence_parallel = config.sequence_parallel  # 是否使用序列并行
        if self.sequence_parallel and world_size <= 1:  # 如果使用序列并行但世界大小小于等于1
            warnings.warn(  # 发出警告
                "`sequence_parallel` is set to `True`, but tensor model parallel size "
                f"is {world_size}. Disabling sequence parallel."
            )
            self.sequence_parallel = False  # 禁用序列并行

        self.allreduce_dgrad = (  # 是否执行梯度的allreduce操作
            world_size > 1 and not self.sequence_parallel and not self.disable_grad_reduce
        )

        if config.gradient_accumulation_fusion and not _grad_accum_fusion_available:  # 如果启用梯度累积融合但不可用
            raise RuntimeError(  # 抛出运行时错误
                "ColumnParallelLinear was called with gradient_accumulation_fusion set "
                "to True but the custom CUDA extension fused_weight_gradient_mlp_cuda "
                "module is not found. To use gradient_accumulation_fusion you must "
                "install APEX with --cpp_ext and --cuda_ext. For example: "
                "pip install --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\" "
                "Note that the extension requires CUDA>=11. Otherwise, you must turn off "
                "gradient accumulation fusion."
            )
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion  # 是否使用梯度累积融合

        if self.allreduce_dgrad and self.sequence_parallel:  # 如果同时启用allreduce_dgrad和sequence_parallel
            raise RuntimeError(  # 抛出运行时错误
                "`allreduce_dgrad` and `sequence_parallel` cannot be enabled at the same time."
            )

        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce  # 设置前向传播实现函数

        # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(  # 注册加载状态字典的预处理钩子
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
                f'{prefix}_extra_state'
            )
        )

    def forward(self, input_: torch.Tensor, weight: Optional[torch.Tensor] = None):
        """Forward of ColumnParallelLinear

        Args:
            input_:
                3D tensor whose order of dimension is [sequence, batch, hidden]
            weight (optional):
                weight tensor to use, compulsory when skip_weight_param_allocation is True.

        Returns:
            - output
            - bias

        """
        if weight is None:  # 如果没有提供权重
            if self.weight is None:  # 如果模块的权重也为None
                raise RuntimeError(  # 抛出运行时错误
                    "weight was not supplied to ColumnParallelLinear forward pass "
                    "and skip_weight_param_allocation is True."
                )
            weight = self.weight  # 使用模块的权重
        else:
            # Check the weight passed in is the correct shape
            expected_shape = (self.output_size_per_partition, self.input_size)  # 期望的权重形状
            if weight.shape != expected_shape:  # 如果提供的权重形状不正确
                raise RuntimeError(  # 抛出运行时错误
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {expected_shape} as expected"
                )

        if self.config._cpu_offloading_context is not None:  # 如果存在CPU卸载上下文
            if self.config._cpu_offloading_context.inside_context is True:  # 如果在上下文内部
                assert (  # 断言CPU卸载未启用
                    self.config.cpu_offloading is False
                ), "CPU Offloading cannot be enabled while using non-TE modules"

        bias = self.bias if not self.skip_bias_add else None  # 确定是否使用偏置

        if (  # 如果满足以下任一条件
            self.allreduce_dgrad
            or self.sequence_parallel
            or self.explicit_expert_comm
            or self.disable_grad_reduce
        ):
            input_parallel = input_  # 直接使用输入
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)  # 复制输入到张量模型并行区域

        if self.config.defer_embedding_wgrad_compute:  # 如果延迟嵌入权重梯度计算
            if (
                self.config.wgrad_deferral_limit == 0
                or len(self.embedding_activation_buffer) < self.config.wgrad_deferral_limit
            ):
                self.embedding_activation_buffer.append(input_parallel)  # 将输入添加到嵌入激活缓冲区

        # Matrix multiply.
        if not weight.requires_grad:  # 如果权重不需要梯度
            self._forward_impl = linear_with_frozen_weight  # 使用冻结权重的线性前向实现
        else:
            self._forward_impl = linear_with_grad_accumulation_and_async_allreduce  # 使用梯度累积和异步allreduce的线性前向实现

        allreduce_dgrad = False if self.explicit_expert_comm else self.allreduce_dgrad  # 确定是否执行梯度的allreduce

        output_parallel = self._forward_impl(  # 执行前向传播
            input=input_parallel,
            weight=weight,
            bias=bias,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=allreduce_dgrad,
            sequence_parallel=False if self.explicit_expert_comm else self.sequence_parallel,
            grad_output_buffer=(
                self.grad_output_buffer if self.config.defer_embedding_wgrad_compute else None
            ),
            wgrad_deferral_limit=(
                self.config.wgrad_deferral_limit
                if self.config.defer_embedding_wgrad_compute
                else None
            ),
            allreduce_dgrad=allreduce_dgrad,
        )
        if self.gather_output:  # 如果需要收集输出
            # All-gather across the partitions.
            assert not self.sequence_parallel  # 断言不使用序列并行
            output = gather_from_tensor_model_parallel_region(output_parallel)  # 从张量模型并行区域收集输出
        else:
            output = output_parallel  # 直接使用并行输出
        output_bias = self.bias if self.skip_bias_add else None  # 确定是否返回偏置
        return output, output_bias  # 返回输出和偏置

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias sharded"""
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {'weight': 0, 'bias': 0}, sharded_offsets
        )

    def set_extra_state(self, state: Any):
        """Extra state is ignored"""

    def get_extra_state(self) -> None:
        """Keep compatibility with TE state dict."""
        return None


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along its first dimension and X
    along its second dimension. A = transpose([A_1 .. A_p]) X = [X_1, ..., X_p]

    Args:
        input_size:
            first dimension of matrix A.
        output_size:
            second dimension of matrix A.
        bias:
            If true, add bias. Note that bias is not parallelized.
        input_is_parallel:
            If true, we assume that the input is already split across the GPUs
            and we do not split again.
        init_method:
            method to initialize weights. Note that bias is always set to zero.
        stride:
            For the strided linear layers.
        keep_master_weight_for_test:
            This was added for testing and should be set to False. It returns the master weights
            used for initialization.
        skip_bias_add:
            If True, do not add the bias term, instead return it to be added by the
            caller. This enables performance optimations where bias can be fused with other
            elementwise operations.
        is_expert:
            If True, the layer is treated as an MoE expert layer
        tp_comm_buffer_name:
            Communication buffer name. Not used in non-Transformer-Engine modules.
        config:
            ModelParallelConfig object

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        is_expert: bool = False,
        tp_comm_buffer_name: str = None,  # Not used
    ):
        super(RowParallelLinear, self).__init__()  # 调用父类的初始化方法

        # Keep input parameters
        self.input_size = input_size  # 存储输入大小
        self.output_size = output_size  # 存储输出大小
        self.input_is_parallel = input_is_parallel  # 存储输入是否已并行
        self.skip_bias_add = skip_bias_add  # 存储是否跳过偏置添加
        self.config = config  # 存储配置对象
        self.is_expert = is_expert  # 存储是否为专家层
        self.expert_parallel = config.expert_model_parallel_size > 1  # 判断是否使用专家并行
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion  # 存储是否使用梯度累积融合
        self.sequence_parallel = config.sequence_parallel  # 存储是否使用序列并行
        if self.sequence_parallel and not self.input_is_parallel:  # 检查序列并行和输入并行的一致性
            raise RuntimeError("To enable `sequence_parallel`, `input_is_parallel` must be `True`")  # 如果不一致，抛出运行时错误

        self.explicit_expert_comm = self.is_expert and (  # 判断是否需要显式专家通信
            config.tensor_model_parallel_size > 1 or self.expert_parallel
        )

        # Divide the weight matrix along the last dimension.
        if self.explicit_expert_comm and config.moe_extended_tp:  # 如果需要显式专家通信且使用扩展张量并行
            world_size = get_tensor_and_expert_parallel_world_size()  # 获取张量和专家并行的世界大小
            rank = get_tensor_and_expert_parallel_rank()  # 获取张量和专家并行的排名
        else:
            world_size = get_tensor_model_parallel_world_size()  # 获取张量模型并行的世界大小
            rank = get_tensor_model_parallel_rank()  # 获取张量模型并行的排名

        self.input_size_per_partition = divide(input_size, world_size)  # 计算每个分区的输入大小

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if config.use_cpu_initialization:  # 如果使用CPU初始化
            self.weight = Parameter(  # 创建权重参数
                torch.empty(
                    self.output_size, self.input_size_per_partition, dtype=config.params_dtype
                )
            )
            if config.perform_initialization:  # 如果执行初始化
                self.master_weight = _initialize_affine_weight_cpu(  # 初始化CPU上的仿射权重
                    self.weight,
                    self.output_size,
                    self.input_size,
                    self.input_size_per_partition,
                    1,
                    init_method,
                    stride=stride,
                    return_master_weight=keep_master_weight_for_test,
                    params_dtype=config.params_dtype,
                    rank=rank,
                    world_size=world_size,
                )
        else:
            self.weight = Parameter(  # 创建权重参数
                torch.empty(
                    self.output_size,
                    self.input_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:  # 如果执行初始化
                _initialize_affine_weight_gpu(  # 初始化GPU上的仿射权重
                    self.weight,
                    init_method,
                    partition_dim=1,
                    stride=stride,
                    expert_parallel=(self.is_expert and self.expert_parallel),
                )
        setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))  # 设置权重的allreduce属性

        if bias:  # 如果使用偏置
            if config.use_cpu_initialization:  # 如果使用CPU初始化
                self.bias = Parameter(torch.empty(self.output_size, dtype=config.params_dtype))  # 创建偏置参数
            else:
                self.bias = Parameter(  # 创建偏置参数
                    torch.empty(
                        self.output_size,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )

            if config.perform_initialization:  # 如果执行初始化
                # Always initialize bias to zero.
                with torch.no_grad():  # 禁用梯度计算
                    self.bias.zero_()  # 将偏置初始化为零
            setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))  # 设置偏置的allreduce属性
            setattr(self.bias, 'sequence_parallel', self.sequence_parallel)  # 设置偏置的sequence_parallel属性
        else:
            self.register_parameter('bias', None)  # 注册空的偏置参数

        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce  # 设置前向实现方法

        # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(  # 注册加载状态字典的预钩子
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
                f'{prefix}_extra_state'
            )
        )

    def forward(self, input_):
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """

        if self.config._cpu_offloading_context is not None:  # 如果存在CPU卸载上下文
            if self.config._cpu_offloading_context.inside_context is True:  # 如果在上下文内部
                assert (  # 断言CPU卸载未启用
                    self.config.cpu_offloading is False
                ), "CPU Offloading cannot be enabled while using non-TE modules"

        # Set up backprop all-reduce.
        if self.input_is_parallel:  # 如果输入已经并行
            input_parallel = input_  # 直接使用输入
        else:
            assert not self.sequence_parallel  # 断言不使用序列并行
            input_parallel = scatter_to_tensor_model_parallel_region(input_)  # 将输入分散到张量模型并行区域
        # Matrix multiply.
        if not self.weight.requires_grad:  # 如果权重不需要梯度
            self._forward_impl = linear_with_frozen_weight  # 使用冻结权重的线性前向实现
        else:
            self._forward_impl = linear_with_grad_accumulation_and_async_allreduce  # 使用梯度累积和异步allreduce的线性前向实现

        allreduce_dgrad = False  # 设置不执行梯度的allreduce

        output_parallel = self._forward_impl(  # 执行前向传播
            input=input_parallel,
            weight=self.weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=allreduce_dgrad,
            sequence_parallel=False,
            grad_output_buffer=None,
            allreduce_dgrad=allreduce_dgrad,
        )

        # All-reduce across all the partitions.
        if self.explicit_expert_comm:  # 如果需要显式专家通信
            assert self.skip_bias_add  # 断言跳过偏置添加
            output_ = output_parallel  # 直接使用并行输出
        elif self.sequence_parallel:  # 如果使用序列并行
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)  # 将并行输出reduce-scatter到序列并行区域
        else:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel)  # 从张量模型并行区域reduce输出
        if not self.skip_bias_add:  # 如果不跳过偏置添加
            output = (output_ + self.bias) if self.bias is not None else output_  # 添加偏置
            output_bias = None  # 设置输出偏置为None
        else:
            output = output_  # 直接使用输出
            output_bias = self.bias  # 设置输出偏置为偏置参数
        return output, output_bias  # 返回输出和输出偏置

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Sharding along axis 1, bias not sharded"""
        state_dict = self.state_dict(prefix='', keep_vars=True)  # 获取模块的状态字典
        return make_sharded_tensors_for_checkpoint(  # 为检查点创建分片张量
            state_dict, prefix, {'weight': 1}, sharded_offsets
        )

    def set_extra_state(self, state: Any):
        """Extra state is ignored"""
        pass  # 忽略额外状态

    def get_extra_state(self) -> None:
        """Keep compatibility with TE state dict."""
        return None  # 返回None以保持与TE状态字典的兼容性
