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
    Encode streaming examples with auto-chunking support.

    This function tokenizes text without truncation, then automatically splits
    long sequences into max_tokens-sized chunks. This ensures no data is lost
    from long sequences.

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
        sample_len = len(full_inputs["input_ids"][sample_index])

        for text_index in range(0, sample_len, max_tokens):
            # Create partial tensors for inputs, targets, and attention masks with fill values
            partial_inputs_ids = torch.full((max_tokens,), pad_id, dtype=torch.long)
            partial_target_ids = torch.full((max_tokens,), -100, dtype=torch.long)
            partial_attention_mask = torch.zeros((max_tokens,), dtype=torch.long)

            # Determine the length of the text to copy
            text_length = min(max_tokens, sample_len - text_index)

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

    LOG.debug("encode_streaming: created %d chunks", len(inputs_ids))

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
            is_pretraining=bool(cfg.pretraining_dataset),
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


def _chunk_long_sequences(
    train_dataset: Dataset,
    max_seq_length: int,
) -> Dataset:
    """
    Chunk sequences longer than max_seq_length into multiple smaller sequences.

    Instead of dropping long sequences (which loses data), this function splits
    them into max_seq_length-sized chunks. This is especially useful for pretraining
    datasets with very long samples (e.g., millions of tokens per example).

    Note: This should only be used for pretraining. For SFT, long sequences should
    be dropped to maintain complete instruction-response pairs.

    Args:
        train_dataset: Dataset with input_ids, attention_mask, and optionally labels
        max_seq_length: Maximum sequence length for each chunk

    Returns:
        Dataset with all sequences <= max_seq_length
    """
    columns = train_dataset.column_names
    has_labels = "labels" in columns
    has_attention_mask = "attention_mask" in columns

    total_samples = len(train_dataset)
    long_samples = sum(
        1 for i in range(total_samples)
        if len(train_dataset[i]["input_ids"]) > max_seq_length
    )

    if long_samples == 0:
        return train_dataset

    LOG.info(
        "Chunking %d/%d sequences that exceed max_seq_length=%d",
        long_samples,
        total_samples,
        max_seq_length,
    )

    new_data = defaultdict(list)
    total_chunks = 0

    for i in range(total_samples):
        sample = train_dataset[i]
        input_ids = sample["input_ids"]
        seq_len = len(input_ids)

        if seq_len <= max_seq_length:
            new_data["input_ids"].append(input_ids)
            if has_attention_mask:
                new_data["attention_mask"].append(sample["attention_mask"])
            if has_labels:
                new_data["labels"].append(sample["labels"])
            total_chunks += 1
        else:
            num_chunks = (seq_len + max_seq_length - 1) // max_seq_length
            for chunk_idx in range(num_chunks):
                start = chunk_idx * max_seq_length
                end = min(start + max_seq_length, seq_len)

                new_data["input_ids"].append(input_ids[start:end])
                if has_attention_mask:
                    new_data["attention_mask"].append(sample["attention_mask"][start:end])
                if has_labels:
                    new_data["labels"].append(sample["labels"][start:end])
                total_chunks += 1

    LOG.info("Chunking complete: %d samples -> %d chunks", total_samples, total_chunks)

    return Dataset.from_dict(dict(new_data))


def encode_packed_streaming(
    collate_fn,
    ds_wrapper: Callable,
    examples: Dict[str, List],
    bin_size: int,
    max_seq_length: int = 2048,
    batch_size: int = 4,
    multipack_attn: Optional[bool] = True,
    is_pretraining: bool = False,
) -> Dict[str, List]:
    """
    Encode examples for sample packing with streaming support.

    This function tokenizes examples, optionally chunks long sequences (for pretraining),
    and then packs them together efficiently using MultipackBatchSampler.

    For pretraining: long sequences are chunked to preserve data.
    For SFT: long sequences are dropped to maintain complete instruction-response pairs.

    Args:
        collate_fn: Collator function for batching
        ds_wrapper: Dataset wrapper function for tokenization
        examples: Raw examples to process
        bin_size: Bin size for multipack sampler
        max_seq_length: Maximum sequence length
        batch_size: Micro batch size
        multipack_attn: Whether to use multipack attention
        is_pretraining: If True, chunk long sequences; if False, let them be dropped
    """
    # Tokenize all the examples
    train_dataset = ds_wrapper(dataset=Dataset.from_dict(examples))[0]

    # Only chunk long sequences for pretraining (preserves data)
    # For SFT, we want to drop long sequences to keep complete examples
    if is_pretraining:
        train_dataset = _chunk_long_sequences(train_dataset, max_seq_length)

    # Process for packing - sequences are now all <= max_seq_length
    train_dataset = process_pretraining_datasets_for_packing(
        train_dataset,
        max_seq_length,
        skip_position_ids=not multipack_attn,
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
