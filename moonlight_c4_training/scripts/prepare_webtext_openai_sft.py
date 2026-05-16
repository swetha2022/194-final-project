#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""
Build OpenAI-messages JSONL for NeMo RL ``openai_format`` SFT (same layout as
``RL/scripts/prepare_driving_sft.py`` in celinetan's setup: ``messages`` +
``gt_answer``).

We stream a **large public web-text corpus** that is representative of general
LLM pretraining mixes (English web crawl), defaulting to HuggingFace
``allenai/c4`` (config ``en``). Each row is a **prefix → continuation**
supervised example (continuation style), which is a standard pretraining-style
objective expressed as chat SFT.

Output layout::

    <output_dir>/
        train.jsonl
        val.jsonl

Example::

    uv run python scripts/prepare_webtext_openai_sft.py \\
        --output_dir data/moonlight_c4_sft \\
        --max_train_examples 200000 \\
        --max_val_examples 4000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys


SYSTEM_PROMPT = (
    "You are a helpful assistant. Given an excerpt from public English web text, "
    "write a natural, coherent continuation that matches the topic and tone."
)


def _normalize_text(text: str) -> str:
    t = " ".join(text.split())
    return t.strip()


def format_row(prefix: str, suffix: str) -> dict:
    user_content = (
        "Continue the following passage. Do not repeat the excerpt; only provide "
        "the continuation.\n\nExcerpt:\n" + prefix
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": suffix},
        ],
        "gt_answer": suffix,
    }


def iter_c4_texts(
    *,
    hf_dataset: str,
    subset: str | None,
    split: str,
    streaming: bool,
):
    from datasets import load_dataset

    if subset:
        ds = load_dataset(hf_dataset, subset, split=split, streaming=streaming)
    else:
        ds = load_dataset(hf_dataset, split=split, streaming=streaming)
    for row in ds:
        text = row.get("text")
        if not text:
            continue
        yield _normalize_text(str(text))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare web-text OpenAI-format JSONL for NeMo RL SFT "
        "(openai_format dataset)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/moonlight_c4_sft",
        help="Directory for train.jsonl and val.jsonl (created if missing).",
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="allenai/c4",
        help="HuggingFace dataset id (default: allenai/c4, typical web crawl).",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="en",
        help="Dataset config/subset name (use '' for single-config datasets).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to stream for building train+val (default train).",
    )
    parser.add_argument(
        "--no_streaming",
        action="store_true",
        help="Disable streaming (materializes full split; not recommended for c4).",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.02,
        help="Fraction of collected rows reserved for val (after max caps).",
    )
    parser.add_argument(
        "--max_train_examples",
        type=int,
        default=200_000,
        help="Cap number of training rows.",
    )
    parser.add_argument(
        "--max_val_examples",
        type=int,
        default=4_000,
        help="Cap number of validation rows.",
    )
    parser.add_argument(
        "--prefix_chars",
        type=int,
        default=400,
        help="Approximate character length of user excerpt (prefix).",
    )
    parser.add_argument(
        "--suffix_chars",
        type=int,
        default=400,
        help="Approximate character length of assistant continuation (suffix).",
    )
    parser.add_argument(
        "--min_doc_chars",
        type=int,
        default=900,
        help="Skip documents shorter than this many characters after normalization.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed before val split.",
    )
    args = parser.parse_args()

    if args.prefix_chars < 32 or args.suffix_chars < 32:
        print("prefix_chars and suffix_chars should be >= 32.", file=sys.stderr)
        sys.exit(1)

    subset = args.subset or None
    streaming = not args.no_streaming
    total_target = args.max_train_examples + args.max_val_examples
    # Collect a pool then shuffle/split (same pattern as driving script).
    pool: list[dict] = []
    skipped_short = 0
    skipped_split = 0

    print(
        f"Streaming {args.hf_dataset} subset={subset!r} split={args.split!r} "
        f"streaming={streaming} …"
    )
    for text in iter_c4_texts(
        hf_dataset=args.hf_dataset,
        subset=subset,
        split=args.split,
        streaming=streaming,
    ):
        if len(text) < args.min_doc_chars:
            skipped_short += 1
            continue
        cut = min(args.prefix_chars, len(text) // 2)
        if cut >= len(text) - 32:
            skipped_split += 1
            continue
        prefix = text[:cut]
        suffix = text[cut : cut + args.suffix_chars]
        if len(suffix) < 32:
            skipped_split += 1
            continue
        pool.append(format_row(prefix, suffix))
        if len(pool) >= total_target:
            break

    if len(pool) < 10:
        print(
            "Too few examples collected. Check network/HF access or lower "
            "--min_doc_chars.",
            file=sys.stderr,
        )
        sys.exit(2)

    rng = random.Random(args.seed)
    rng.shuffle(pool)

    n_val = max(1, int(round(len(pool) * args.val_fraction)))
    n_val = min(n_val, args.max_val_examples, len(pool) - 1)
    val_rows = pool[:n_val]
    train_rows = pool[n_val:]
    train_rows = train_rows[: args.max_train_examples]
    val_rows = val_rows[: args.max_val_examples]

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")

    def _write(path: str, rows: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _write(train_path, train_rows)
    _write(val_path, val_rows)

    print(
        f"Wrote {len(train_rows)} train / {len(val_rows)} val → {args.output_dir}\n"
        f"  skipped_short={skipped_short} skipped_split={skipped_split} seed={args.seed}"
    )


if __name__ == "__main__":
    main()
