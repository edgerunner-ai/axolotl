"""Data handling specific to streaming datasets."""

import functools
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import torch
from datasets import Dataset
from torch.utils.data import RandomSampler
from transformers import PreTrainedTokenizerBase

from axolotl.utils.collators import PretrainingBatchSamplerDataCollatorForSeq2Seq
from axolotl.utils.logging import get_logger
from axolotl.utils.samplers import MultipackBatchSampler, get_dataset_lengths
from axolotl.utils.trainer import process_pretraining_datasets_for_packing

LOG = get_logger(__name__)


def encode_streaming(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
    text_column: str = "text",
    concatenate: bool = True,
) -> Dict[str, List]:
    """
    Encode streaming examples with auto-truncation support.

    This function tokenizes text without truncation, then automatically splits
    long sequences into max_tokens-sized chunks. This ensures no data is lost
    from long sequences (from PR #3081: pretraining-auto-truncate).

    Args:
        examples: Dictionary containing text samples
        tokenizer: The tokenizer to use
        max_tokens: Maximum sequence length for each chunk
        text_column: Name of the text column in examples
        concatenate: If True, concatenate all samples before chunking

    Returns:
        Dictionary with input_ids, labels, and attention_mask lists
    """
    # Tokenize without truncation to preserve all data
    full_inputs = tokenizer(
        examples[text_column],
        add_special_tokens=True,
    )

    # Convert input_ids and attention_mask to tensors
    full_inputs["input_ids"] = [
        torch.tensor(sample, dtype=torch.long) for sample in full_inputs["input_ids"]
    ]
    full_inputs["attention_mask"] = [
        torch.tensor(sample, dtype=torch.long)
        for sample in full_inputs["attention_mask"]
    ]

    # Resolve a safe pad token id for chunk padding
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else (tokenizer.eos_token_id or 0)
    )

    if tokenizer.pad_token_id is None:
        LOG.warning(
            "tokenizer.pad_token_id is None; falling back to %s for padding", pad_id
        )

    inputs_ids, target_ids, attention_mask = [], [], []

    # Concatenate all input_ids and attention masks into one tensor when concatenate is True
    if concatenate:
        full_inputs["input_ids"] = [torch.cat(full_inputs["input_ids"], dim=0)]
        full_inputs["attention_mask"] = [
            torch.cat(full_inputs["attention_mask"], dim=0)
        ]

    # Iterate through each sample and split into chunks of max_tokens
    for sample_index in range(len(full_inputs["input_ids"])):
        for text_index in range(
            0, len(full_inputs["input_ids"][sample_index]), max_tokens
        ):
            # Create partial tensors for inputs, targets, and attention masks with fill values
            partial_inputs_ids = torch.full((max_tokens,), pad_id, dtype=torch.long)
            partial_target_ids = torch.full((max_tokens,), -100, dtype=torch.long)
            partial_attention_mask = torch.zeros((max_tokens,), dtype=torch.long)

            # Determine the length of the text to copy
            text_length = min(
                max_tokens,
                len(full_inputs["input_ids"][sample_index]) - text_index,
            )

            # Copy the text into the partial tensors
            partial_inputs_ids[:text_length] = full_inputs["input_ids"][sample_index][
                text_index : text_index + text_length
            ]
            partial_target_ids[:text_length] = full_inputs["input_ids"][sample_index][
                text_index : text_index + text_length
            ]
            partial_attention_mask[:text_length] = full_inputs["attention_mask"][
                sample_index
            ][text_index : text_index + text_length]

            # Append the partial tensors to the lists
            inputs_ids.append(partial_inputs_ids)
            target_ids.append(partial_target_ids)
            attention_mask.append(partial_attention_mask)

    LOG.debug("Input IDs length: %s", len(inputs_ids))

    return {
        "input_ids": [input_id.tolist() for input_id in inputs_ids],
        "labels": [target_id.tolist() for target_id in target_ids],
        "attention_mask": [mask.tolist() for mask in attention_mask],
    }


def wrap_streaming_dataset(
    dataset,
    tokenizer,
    cfg,
    ds_wrapper_fn,
):
    if cfg.sample_packing:
        # For SFT (non-pretraining) datasets, always use multipack_attn=True to ensure
        # attention isolation between packed sequences
        multipack_attn = (
            True if not cfg.pretraining_dataset else cfg.pretrain_multipack_attn
        )

        collate_fn = PretrainingBatchSamplerDataCollatorForSeq2Seq(
            tokenizer,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=cfg.sequence_len,
            multipack_attn=multipack_attn,
        )
        encode = functools.partial(
            encode_packed_streaming,
            collate_fn,
            ds_wrapper_fn,
            max_seq_length=cfg.sequence_len,
            batch_size=cfg.micro_batch_size,
            multipack_attn=multipack_attn,
            bin_size=cfg.sample_packing_bin_size,
        )

        # Set this to 1 so downstream data_loader doesn't try to increase the batch size
        # again
        cfg.micro_batch_size = 1
    else:
        # NOTE: This is not reachable for SFT datasets since we use the pre-existing
        # loading function for non-packed streaming datasets. Refer to
        # _prepare_streaming_datasets in sft.py for that code path.
        text_column = (
            getattr(cfg.pretraining_dataset[0], "text_column", "text") or "text"
        )
        encode = functools.partial(
            encode_streaming,
            tokenizer=tokenizer,
            max_tokens=cfg.sequence_len,
            text_column=text_column,
            concatenate=cfg.pretraining_sample_concatenation is True,
        )

    if cfg.shuffle_merged_datasets:
        dataset = dataset.shuffle(
            seed=cfg.seed, buffer_size=cfg.streaming_multipack_buffer_size
        )
    else:
        LOG.debug("NOT shuffling merged pretraining datasets")

    # remove all the existing columns after mapping since they end up having
    # a different length than the encoded/tokenized column
    # this is empty during streaming/pretraining
    remove_columns = []
    if dataset.features is None:
        for first_row in dataset:
            remove_columns = list(first_row.keys())
            break
    else:
        remove_columns = list(dataset.features.keys())

    dataset = dataset.map(
        encode,
        batched=True,
        batch_size=cfg.streaming_multipack_buffer_size,
        remove_columns=remove_columns,
    )
    return dataset


def encode_packed_streaming(
    collate_fn,
    ds_wrapper: Callable,
    examples: Dict[str, List],
    bin_size: int,
    max_seq_length: int = 2048,
    batch_size: int = 4,
    multipack_attn: Optional[bool] = True,
) -> Dict[str, List]:
    # tokenize all the examples
    # rows get split with stride (overlap)
    train_dataset = ds_wrapper(dataset=Dataset.from_dict(examples))[0]

    train_dataset = process_pretraining_datasets_for_packing(
        train_dataset,
        max_seq_length,
        skip_position_ids=not multipack_attn,
        # FIXME using attention mask unpad/pad with trainer and packed pretraining is broken atm
        # workaround by using the position id logic for now in trainer
        drop_attention_mask=multipack_attn,
    )

    sampler = MultipackBatchSampler(
        sampler=RandomSampler(train_dataset),
        lengths=get_dataset_lengths(train_dataset),
        batch_size=1,
        batch_max_len=batch_size * max_seq_length,
        drop_last=True,
        num_processes=1,
        bin_size=bin_size,
    )

    chunked_data = defaultdict(list)

    for batch in sampler:
        for data in batch:
            features = train_dataset[data]
            if "num_truncated_tokens" in features:
                del features["num_truncated_tokens"]
            if "overflow_to_sample_mapping" in features:
                del features["overflow_to_sample_mapping"]
            if "labels" not in features:
                features["labels"] = features["input_ids"].copy()
            collated_features = collate_fn(features)

            for feature in features.keys():
                if feature == "length":
                    continue
                chunked_data[feature].append(collated_features[feature].squeeze(0))

    return chunked_data
