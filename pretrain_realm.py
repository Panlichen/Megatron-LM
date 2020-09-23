# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain BERT for Inverse Cloze Task"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

from megatron.checkpointing import load_ict_checkpoint
from megatron.data.realm_dataset import get_ict_dataset
from megatron.data.realm_index import BlockData, FaissMIPSIndex
from megatron import get_args
from megatron import get_timers
from megatron import mpu
from megatron import print_rank_0
from megatron.data.dataset_utils import build_train_valid_test_datasets
from megatron.model import REALMBertModel, REALMRetriever, ICTBertModel
from megatron.model.realm_model import general_ict_model_provider
from megatron.training import get_model, pretrain
from megatron.utils import reduce_losses, report_memory
from megatron import mpu
from megatron.indexer import initialize_and_run_async_megatron
from megatron.mpu.initialize import get_data_parallel_group

num_batches = 0


def model_provider():
    """Build the model."""
    args = get_args()
    print_rank_0('building REALM models ...')

    # query and block encoder models whose state dicts will be loaded from checkpoint
    model = get_model(lambda: general_ict_model_provider())

    try:
        ict_model = load_ict_checkpoint(model, from_realm_chkpt=True)
    except:
        ict_model = load_ict_checkpoint(model, from_realm_chkpt=False)

    # dataset and index over embeddings of blocks from that dataset
    ict_dataset = get_ict_dataset(use_titles=False)
    block_data = BlockData(args.block_data_path, load_from_path=True)
    faiss_mips_index = FaissMIPSIndex(embed_size=128, block_data=block_data, use_gpu=args.faiss_use_gpu)

    # retriever which gets data from the dataset based on the embedding index
    retriever = REALMRetriever(ict_model, ict_dataset, faiss_mips_index, args.block_top_k)

    # retrieval-augmented language model
    model = REALMBertModel(retriever)

    return model


def get_batch(data_iterator):
    # Items and their type.
    keys = ['tokens', 'labels', 'loss_mask', 'pad_mask', 'query_block_indices']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is None:
        data = None
    else:
        data = next(data_iterator)

    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens = data_b['tokens'].long()
    labels = data_b['labels'].long()
    loss_mask = data_b['loss_mask'].long()
    pad_mask = data_b['pad_mask'].long()
    query_block_indices = data_b['query_block_indices'].long()

    return tokens, labels, loss_mask, pad_mask, query_block_indices


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, loss_mask, pad_mask, query_block_indices = get_batch(data_iterator)
    timers('batch generator').stop()

    # Forward model.
    lm_logits, block_probs = model(tokens, pad_mask, query_block_indices)

    with torch.no_grad():
        max_retrieval_utility, top_retrieval_utility, avg_retrieval_utility, tokens_over_batch = mpu.checkpoint(
            get_retrieval_utility, lm_logits, block_probs, labels, loss_mask)

    # P(y|x) = sum_z(P(y|z, x) * P(z|x))
    null_block_probs = torch.mean(block_probs[:, block_probs.shape[1] - 1])

    # logits: [batch x top_k x 2 * seq_length x vocab_size]
    # labels: [batch x seq_length]
    relevant_logits = lm_logits[:, :, :labels.shape[1]].float()
    block_probs = block_probs.unsqueeze(2).unsqueeze(3).expand_as(relevant_logits)

    def get_log_probs(logits, b_probs):
        max_logits = torch.max(logits, dim=-1, keepdim=True)[0].expand_as(logits)
        logits = logits - max_logits

        softmaxed_logits = F.softmax(logits, dim=-1)
        marginalized_probs = torch.sum(softmaxed_logits * b_probs, dim=1)
        l_probs = torch.log(marginalized_probs)
        return l_probs

    def get_loss(l_probs, labs):
        vocab_size = l_probs.shape[2]
        loss = torch.nn.NLLLoss(ignore_index=-1)(l_probs.reshape(-1, vocab_size), labs.reshape(-1))
        return loss.float()

    lm_loss = get_loss(get_log_probs(relevant_logits, block_probs), labels)
    reduced_loss = reduce_losses([lm_loss, max_retrieval_utility, top_retrieval_utility, avg_retrieval_utility, null_block_probs, tokens_over_batch])
    return lm_loss, {'lm_loss': reduced_loss[0],
                     'max_ru': reduced_loss[1],
                     'top_ru': reduced_loss[2],
                     'avg_ru': reduced_loss[3],
                     'null_prob': reduced_loss[4],
                     'mask/batch': reduced_loss[5]}


def get_retrieval_utility(lm_logits_, block_probs, labels, loss_mask):
    """log P(y | z, x) - log P(y | null, x)"""

    # [batch x top_k x seq_len x vocab_size]
    lm_logits = lm_logits_[:, :, :labels.shape[1], :]
    batch_size, top_k = lm_logits.shape[0], lm_logits.shape[1]

    # non_null_block_probs = block_probs[:, :-1]
    # non_null_block_probs /= torch.sum(non_null_block_probs, axis=1, keepdim=True)
    # non_null_block_probs = non_null_block_probs.expand_as(lm_logits[:, :-1, :, :])

    null_block_lm_logits = lm_logits[:, -1, :, :]
    null_block_loss_ = mpu.vocab_parallel_cross_entropy(null_block_lm_logits.contiguous().float(),
                                                       labels.contiguous())
    null_block_loss = torch.sum(null_block_loss_.view(-1) * loss_mask.reshape(-1)) / batch_size

    retrieved_block_losses = []

    for block_num in range(top_k - 1):
        retrieved_block_lm_logits = lm_logits[:, block_num, :, :]
        retrieved_block_loss_ = mpu.vocab_parallel_cross_entropy(retrieved_block_lm_logits.contiguous().float(),
                                                                 labels.contiguous())

        # retrieved_block_loss_ *= non_null_block_probs[:, block_num].reshape(-1, 1)
        retrieved_block_loss = torch.sum(retrieved_block_loss_.view(-1) * loss_mask.reshape(-1)) / batch_size
        retrieved_block_losses.append(retrieved_block_loss)
    avg_retrieved_block_loss = torch.sum(torch.cuda.FloatTensor(retrieved_block_losses)) / (top_k - 1)

    max_retrieval_utility = null_block_loss - min(retrieved_block_losses)
    top_retrieval_utility = null_block_loss - retrieved_block_losses[0]
    avg_retrieval_utility = null_block_loss - avg_retrieved_block_loss

    tokens_over_batch = loss_mask.sum().float() / batch_size

    return max_retrieval_utility, top_retrieval_utility, avg_retrieval_utility, tokens_over_batch


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid and test datasets."""
    args = get_args()
    print_rank_0('> building train, validation, and test datasets '
                 'for BERT ...')

    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        max_seq_length=args.seq_length,
        masked_lm_prob=args.mask_prob,
        short_seq_prob=args.short_seq_prob,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        dataset_type='realm')
    print_rank_0("> finished creating BERT ICT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
             args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'},
             initializer_func=initialize_and_run_async_megatron)

