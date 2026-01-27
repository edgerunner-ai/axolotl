"""Data handling specific to streaming datasets."""

import functools
import sys
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


def _get_memory_mb(obj) -> float:
    """Get approximate memory usage of an object in MB."""
    if isinstance(obj, torch.Tensor):
        return obj.element_size() * obj.nelement() / (1024 * 1024)
    if isinstance(obj, list):
        return sum(_get_memory_mb(item) for item in obj)
    if isinstance(obj, dict):
        return sum(_get_memory_mb(v) for v in obj.values())
    return sys.getsizeof(obj) / (1024 * 1024)


def _format_memory(mb: float) -> str:
    """Format memory size for display."""
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.2f} MB"


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
    num_examples = len(examples[text_column])
    LOG.info(
        "encode_streaming: Processing %d examples (max_tokens=%d, concatenate=%s)",
        num_examples,
        max_tokens,
        concatenate,
    )

    # Tokenize without truncation to preserve all data
    LOG.info("Tokenizing %d examples without truncation...", num_examples)
    full_inputs = tokenizer(
        examples[text_column],
        add_special_tokens=True,
    )

    # Convert input_ids and attention_mask to tensors
    LOG.info("Converting tokenized outputs to tensors...")
    full_inputs["input_ids"] = [
        torch.tensor(sample, dtype=torch.long) for sample in full_inputs["input_ids"]
    ]
    full_inputs["attention_mask"] = [
        torch.tensor(sample, dtype=torch.long)
        for sample in full_inputs["attention_mask"]
    ]

    # Calculate total tokens and memory usage after tokenization
    total_tokens = sum(len(ids) for ids in full_inputs["input_ids"])
    tokenized_mem = _get_memory_mb(full_inputs)
    LOG.info(
        "Tokenization complete: %d total tokens across %d samples, memory: %s",
        total_tokens,
        num_examples,
        _format_memory(tokenized_mem),
    )

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
        LOG.info(
            "Concatenating %d samples into single tensor (%d tokens)...",
            num_examples,
            total_tokens,
        )
        full_inputs["input_ids"] = [torch.cat(full_inputs["input_ids"], dim=0)]
        full_inputs["attention_mask"] = [
            torch.cat(full_inputs["attention_mask"], dim=0)
        ]
        concat_mem = _get_memory_mb(full_inputs)
        LOG.info(
            "Concatenation complete: single tensor of %d tokens, memory: %s",
            len(full_inputs["input_ids"][0]),
            _format_memory(concat_mem),
        )

    # Estimate output memory requirement
    num_chunks_estimate = (total_tokens + max_tokens - 1) // max_tokens
    # Each chunk has input_ids, labels, attention_mask (3 tensors of max_tokens int64)
    estimated_output_mem = num_chunks_estimate * 3 * max_tokens * 8 / (1024 * 1024)
    LOG.info(
        "Chunking into ~%d chunks of %d tokens each, estimated output memory: %s",
        num_chunks_estimate,
        max_tokens,
        _format_memory(estimated_output_mem),
    )

    # Iterate through each sample and split into chunks of max_tokens
    LOG.info("Splitting sequences into max_tokens-sized chunks...")
    for sample_index in range(len(full_inputs["input_ids"])):
        sample_len = len(full_inputs["input_ids"][sample_index])
        num_chunks_for_sample = (sample_len + max_tokens - 1) // max_tokens

        if not concatenate and num_chunks_for_sample > 1:
            LOG.debug(
                "Sample %d: %d tokens -> %d chunks",
                sample_index,
                sample_len,
                num_chunks_for_sample,
            )

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

    # Calculate final memory usage
    final_mem = _get_memory_mb(inputs_ids) + _get_memory_mb(target_ids) + _get_memory_mb(attention_mask)
    LOG.info(
        "Chunking complete: %d chunks created, output tensors memory: %s",
        len(inputs_ids),
        _format_memory(final_mem),
    )

    # Convert to lists for return
    LOG.info("Converting %d chunks to lists for output...", len(inputs_ids))
    result = {
        "input_ids": [input_id.tolist() for input_id in inputs_ids],
        "labels": [target_id.tolist() for target_id in target_ids],
        "attention_mask": [mask.tolist() for mask in attention_mask],
    }

    result_mem = _get_memory_mb(result)
    LOG.info(
        "encode_streaming complete: %d chunks, final output memory: %s",
        len(result["input_ids"]),
        _format_memory(result_mem),
    )

    return result


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


def _chunk_long_sequences(
    train_dataset: Dataset,
    max_seq_length: int,
) -> Dataset:
    """
    Chunk sequences longer than max_seq_length into multiple smaller sequences.

    Instead of dropping long sequences (which loses data), this function splits
    them into max_seq_length-sized chunks. This is especially useful for datasets
    with very long samples (e.g., millions of tokens per example).

    Args:
        train_dataset: Dataset with input_ids, attention_mask, and optionally labels
        max_seq_length: Maximum sequence length for each chunk

    Returns:
        Dataset with all sequences <= max_seq_length
    """
    # Get column names to process
    columns = train_dataset.column_names
    has_labels = "labels" in columns
    has_attention_mask = "attention_mask" in columns

    # Count sequences that need chunking
    total_samples = len(train_dataset)
    long_samples = sum(
        1 for i in range(total_samples)
        if len(train_dataset[i]["input_ids"]) > max_seq_length
    )

    if long_samples == 0:
        LOG.info("No sequences exceed max_seq_length=%d, skipping chunking", max_seq_length)
        return train_dataset

    LOG.info(
        "Chunking %d/%d sequences that exceed max_seq_length=%d (instead of dropping)",
        long_samples,
        total_samples,
        max_seq_length,
    )

    # Build new chunked data
    new_data = defaultdict(list)
    total_chunks = 0
    total_tokens_before = 0
    total_tokens_after = 0

    for i in range(total_samples):
        sample = train_dataset[i]
        input_ids = sample["input_ids"]
        seq_len = len(input_ids)
        total_tokens_before += seq_len

        if seq_len <= max_seq_length:
            # Keep as is
            new_data["input_ids"].append(input_ids)
            if has_attention_mask:
                new_data["attention_mask"].append(sample["attention_mask"])
            if has_labels:
                new_data["labels"].append(sample["labels"])
            total_chunks += 1
            total_tokens_after += seq_len
        else:
            # Chunk into max_seq_length pieces
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
                total_tokens_after += (end - start)

    LOG.info(
        "Chunking complete: %d samples -> %d chunks, %d tokens preserved (was %d)",
        total_samples,
        total_chunks,
        total_tokens_after,
        total_tokens_before,
    )

    # Create new dataset from chunked data
    return Dataset.from_dict(dict(new_data))


def encode_packed_streaming(
    collate_fn,
    ds_wrapper: Callable,
    examples: Dict[str, List],
    bin_size: int,
    max_seq_length: int = 2048,
    batch_size: int = 4,
    multipack_attn: Optional[bool] = True,
) -> Dict[str, List]:
    """
    Encode examples for sample packing with streaming support.

    This function tokenizes examples, chunks any long sequences (instead of dropping),
    and then packs them together efficiently using MultipackBatchSampler.
    """
    LOG.info(
        "encode_packed_streaming: Processing %d examples (max_seq_length=%d)",
        len(examples.get("text", examples.get("input_ids", []))),
        max_seq_length,
    )

    # tokenize all the examples
    # rows get split with stride (overlap)
    LOG.info("Tokenizing examples via ds_wrapper...")
    train_dataset = ds_wrapper(dataset=Dataset.from_dict(examples))[0]
    LOG.info("Tokenization complete: %d sequences", len(train_dataset))

    # Chunk long sequences instead of dropping them
    # This preserves all data from very long samples
    train_dataset = _chunk_long_sequences(train_dataset, max_seq_length)

    # Now process for packing - this will no longer drop sequences since they're all chunked
    train_dataset = process_pretraining_datasets_for_packing(
        train_dataset,
        max_seq_length,
        skip_position_ids=not multipack_attn,
        # FIXME using attention mask unpad/pad with trainer and packed pretraining is broken atm
        # workaround by using the position id logic for now in trainer
        drop_attention_mask=multipack_attn,
    )

    LOG.info("Creating MultipackBatchSampler for %d sequences...", len(train_dataset))
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
    num_batches = 0

    for batch in sampler:
        num_batches += 1
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

    LOG.info(
        "encode_packed_streaming complete: %d batches, %d packed sequences",
        num_batches,
        len(chunked_data.get("input_ids", [])),
    )

    return chunked_data
