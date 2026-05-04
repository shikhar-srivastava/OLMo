#!/usr/bin/env python3
"""
Download a proportional subset of OLMo-tiny training data to a local directory.

The tiny model configs (60M/150M/300M) contain ~1.82T tokens spread across
13 data sources in a flat `data.paths` list. Sampling is token-proportional
(IterableDataset shuffles uniformly over all tokens). This script:

  1. Parses training + eval URLs from a public config YAML.
  2. Issues parallel HEAD requests to get each shard's file size (= tokens × 2
     bytes, since dtype is uint16).
  3. Selects a proportional prefix of shards from each source so that the total
     downloaded token count reaches `--target-tokens` (default 40B, which is
     20B training budget × 2 safety buffer).
  4. Downloads the selected files in parallel, skipping already-complete ones.
  5. Writes `*-local.yaml` configs next to the public ones with all
     `https://olmo-data.org/` paths rewritten to the local directory.

Usage:
    python scripts/download_olmo_data.py \\
        --config configs/tiny/OLMo-60M-public.yaml \\
        --data-dir /dev/shm/ssrivastava/datasets/olmo-data \\
        --target-tokens 40_000_000_000 \\
        --max-workers 16

The generated local configs are written to:
    configs/tiny/OLMo-{60M,150M,300M}-local.yaml

All three model configs share identical data paths, so running this once for
the 60M config is sufficient — the other two local configs are derived
automatically.
"""

import argparse
import math
import os
import sys
import time
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import requests
import yaml

BASE_URL = "https://olmo-data.org/"
DTYPE_BYTES = 2  # uint16


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def head_content_length(url: str, retries: int = 3, timeout: int = 20) -> Optional[int]:
    """Return the Content-Length of *url* in bytes, or None on failure."""
    for attempt in range(retries):
        try:
            r = requests.head(url, allow_redirects=True, timeout=timeout)
            cl = r.headers.get("content-length")
            if cl is not None:
                return int(cl)
            # Some servers don't return Content-Length on HEAD; fall back to GET
            # with stream=True just to read the headers.
            r2 = requests.get(url, stream=True, timeout=timeout)
            cl2 = r2.headers.get("content-length")
            r2.close()
            if cl2 is not None:
                return int(cl2)
            return None
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def download_file(url: str, dest: Path, retries: int = 3, timeout: int = 120) -> bool:
    """Download *url* to *dest*, skipping if already complete. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded with the correct size.
    if dest.exists():
        cl = head_content_length(url, retries=2, timeout=30)
        if cl is not None and dest.stat().st_size == cl:
            return True  # already complete

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MiB chunks
                        f.write(chunk)
            tmp.rename(dest)
            return True
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [ERROR] Failed to download {url}: {exc}", file=sys.stderr)
    return False


# ---------------------------------------------------------------------------
# YAML parsing helpers
# ---------------------------------------------------------------------------

def extract_training_urls(config: dict) -> List[str]:
    """Return the ordered list of training paths (may contain duplicates)."""
    data = config.get("data", {})
    paths = data.get("paths")
    if paths:
        return [str(p) for p in paths]
    datasets = data.get("datasets")
    if datasets:
        result = []
        for label in sorted(datasets.keys()):
            result.extend(str(p) for p in datasets[label])
        return result
    raise ValueError("Config has neither data.paths nor data.datasets")


def extract_eval_urls(config: dict) -> List[str]:
    """Return all eval shard URLs (typically small validation files)."""
    urls: List[str] = []
    for evaluator in config.get("evaluators", []):
        data = evaluator.get("data", {})
        # perplexity / memmap evaluators
        for p in data.get("paths", []):
            urls.append(str(p))
        for _, paths in data.get("datasets", {}).items():
            for p in paths:
                urls.append(str(p))
    return urls


def source_key(url: str) -> str:
    """Return the source-group key: everything between BASE_URL and the filename."""
    rel = url.replace(BASE_URL, "")
    parts = rel.split("/")
    # e.g. preprocessed/starcoder/v0_decontaminated_doc_only/gpt-neox-olmo-dolma-v1_5
    return "/".join(parts[:-1])


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def plan_downloads(
    training_urls: List[str],
    target_tokens: int,
    max_workers: int,
) -> Tuple[Set[str], Dict[str, int], int]:
    """
    Issue HEAD requests for every unique training shard and decide which ones
    to download to reach *target_tokens*.

    Returns:
        selected_urls: set of unique URLs to download
        sizes: url -> bytes
        total_tokens: total tokens across all shards
    """
    unique_urls: List[str] = list(dict.fromkeys(training_urls))  # deduplicated, ordered

    print(f"Fetching sizes for {len(unique_urls)} unique training shards "
          f"(~{max_workers} parallel) …")

    sizes: Dict[str, int] = {}
    failed: List[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(head_content_length, url): url
                         for url in unique_urls}
        done = 0
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            size = future.result()
            if size is None:
                failed.append(url)
                sizes[url] = 0
            else:
                sizes[url] = size
            done += 1
            if done % 50 == 0:
                print(f"  … {done}/{len(unique_urls)} sizes fetched")

    if failed:
        print(f"  WARNING: could not get size for {len(failed)} URLs; "
              f"they will be excluded from the download plan.")

    total_bytes = sum(sizes.values())
    total_tokens = total_bytes // DTYPE_BYTES
    fraction = min(1.0, target_tokens / max(total_tokens, 1))

    print(f"\nTotal dataset: {total_tokens / 1e9:.1f}B tokens "
          f"({total_bytes / 1e12:.2f} TB)")
    print(f"Target:        {target_tokens / 1e9:.1f}B tokens  "
          f"(fraction = {fraction:.4f})")

    # Group unique URLs by source (directory prefix)
    source_groups: Dict[str, List[str]] = defaultdict(list)
    for url in unique_urls:
        if sizes[url] > 0:
            source_groups[source_key(url)].append(url)

    print(f"\nSource groups ({len(source_groups)}):")
    selected_urls: Set[str] = set()
    for src, urls in sorted(source_groups.items()):
        src_bytes = sum(sizes[u] for u in urls)
        src_tokens = src_bytes // DTYPE_BYTES
        n_select = max(1, math.ceil(len(urls) * fraction))
        n_select = min(n_select, len(urls))
        selected = urls[:n_select]
        selected_bytes = sum(sizes[u] for u in selected)
        selected_tokens = selected_bytes // DTYPE_BYTES
        selected_urls.update(selected)
        src_label = "/".join(src.split("/")[1:3])  # brief label
        print(f"  {src_label:55s}  {n_select:4d}/{len(urls):4d} shards  "
              f"  {selected_tokens / 1e9:6.2f}B / {src_tokens / 1e9:6.2f}B tokens")

    planned_bytes = sum(sizes[u] for u in selected_urls)
    planned_tokens = planned_bytes // DTYPE_BYTES
    print(f"\nPlanned download: {len(selected_urls)} shards, "
          f"{planned_tokens / 1e9:.1f}B tokens, "
          f"{planned_bytes / 1e9:.1f} GB")
    return selected_urls, sizes, total_tokens


def download_all(
    urls: List[str],
    data_dir: Path,
    max_workers: int,
) -> int:
    """Download all *urls* into *data_dir*, preserving path structure. Returns success count."""
    print(f"\nDownloading {len(urls)} files to {data_dir} …")
    total = len(urls)
    successes = 0
    failures: List[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {}
        for url in urls:
            rel = url.replace(BASE_URL, "")
            dest = data_dir / rel
            future_to_url[executor.submit(download_file, url, dest)] = url

        done = 0
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            ok = future.result()
            done += 1
            if ok:
                successes += 1
            else:
                failures.append(url)
            if done % 20 == 0 or done == total:
                print(f"  … {done}/{total}  ({successes} ok, {len(failures)} failed)")

    if failures:
        print(f"\nWARNING: {len(failures)} downloads failed:")
        for u in failures[:20]:
            print(f"  {u}")
        if len(failures) > 20:
            print(f"  … and {len(failures) - 20} more")
    return successes


def rewrite_yaml_paths(
    config_path: Path,
    data_dir: Path,
    selected_train_urls: Set[str],
    all_eval_urls: Set[str],
) -> Path:
    """
    Read *config_path*, replace all https://olmo-data.org/ prefixes with local
    paths, drop training shards that were not downloaded, and write a *-local.yaml.

    Returns the path to the written local config.
    """
    # We do a line-by-line rewrite to preserve comments and formatting exactly.
    src_lines = config_path.read_text().splitlines(keepends=True)
    out_lines: List[str] = []

    local_prefix = str(data_dir) + "/"

    for line in src_lines:
        stripped = line.lstrip()
        # Is this a YAML list item with a URL?
        if stripped.startswith("- https://olmo-data.org/"):
            url = stripped[2:].strip()
            local_path = url.replace(BASE_URL, local_prefix)
            # Decide whether to keep this line
            if url in selected_train_urls or url in all_eval_urls:
                indent = line[: len(line) - len(line.lstrip())]
                out_lines.append(f"{indent}- {local_path}\n")
            # else: omit this shard from the local config
            continue
        out_lines.append(line)

    if "-public.yaml" in config_path.name:
        local_name = config_path.name.replace("-public.yaml", "-local.yaml")
    else:
        local_name = config_path.stem + "-local.yaml"
    local_cfg_path = config_path.with_name(local_name)

    # Replace max_duration with a sentinel: the run script must pass --max_duration on
    # the CLI, and scripts/train.py refuses to start without it. This guarantees the
    # token budget cannot be silently set by a generated config file.
    import re as _re
    text = "".join(out_lines)
    text = _re.sub(
        r"^max_duration:.*$",
        "max_duration: MUST_BE_SET_BY_RUN_SCRIPT  # sentinel: real value is required via --max_duration on the CLI",
        text,
        flags=_re.MULTILINE,
    )

    local_cfg_path.write_text(text)
    return local_cfg_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download a proportional OLMo-tiny data subset and generate local configs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/tiny/OLMo-60M-public.yaml",
        help="Path to a public *-public.yaml config to use as the URL manifest.",
    )
    parser.add_argument(
        "--data-dir",
        default="/dev/shm/ssrivastava/datasets/olmo-data",
        help="Root directory to download shards into. Subdirectory structure "
             "mirrors the olmo-data.org URL path.",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=40_000_000_000,
        help="Total tokens to download (training shards only). Default is 40B = "
             "20B training budget × 2 safety buffer.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Maximum parallel HTTP connections for HEAD requests and downloads.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the download and generate local configs, but do not actually "
             "fetch any files.",
    )
    parser.add_argument(
        "--skip-size-check",
        action="store_true",
        help="Skip HEAD requests for sizes and use equal-shard-count proportional "
             "selection instead. Faster but less accurate.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    data_dir = Path(args.data_dir)

    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    training_urls = extract_training_urls(config)
    eval_urls = extract_eval_urls(config)

    print(f"Config:          {cfg_path}")
    print(f"Training shards: {len(training_urls)} entries "
          f"({len(set(training_urls))} unique)")
    print(f"Eval shards:     {len(set(eval_urls))} unique")
    print(f"Data directory:  {data_dir}")
    print(f"Target tokens:   {args.target_tokens / 1e9:.1f}B")
    print()

    # -----------------------------------------------------------------------
    # Phase 1: Plan
    # -----------------------------------------------------------------------
    if args.skip_size_check:
        # Fallback: select proportionally by shard count (assumes equal shard sizes)
        unique_train = list(dict.fromkeys(training_urls))
        total_shards = len(unique_train)
        fraction = min(1.0, args.target_tokens / (total_shards * 500_000_000))  # ~500M tok/shard est.
        source_groups: Dict[str, List[str]] = defaultdict(list)
        for url in unique_train:
            source_groups[source_key(url)].append(url)
        selected_train: Set[str] = set()
        for urls in source_groups.values():
            n = max(1, math.ceil(len(urls) * fraction))
            selected_train.update(urls[:n])
        sizes = {}
    else:
        selected_train, sizes, _ = plan_downloads(
            training_urls, args.target_tokens, args.max_workers
        )

    all_eval: Set[str] = set(eval_urls)

    print(f"\nEval shards to download (all): {len(all_eval)}")

    all_to_download = sorted(selected_train | all_eval)

    if args.dry_run:
        print("\n[DRY RUN] Would download:")
        for url in all_to_download[:10]:
            print(f"  {url}")
        if len(all_to_download) > 10:
            print(f"  … and {len(all_to_download) - 10} more")
    else:
        # -----------------------------------------------------------------------
        # Phase 2: Download
        # -----------------------------------------------------------------------
        ok = download_all(all_to_download, data_dir, args.max_workers)
        print(f"\nDownloaded {ok}/{len(all_to_download)} files successfully.")

    # -----------------------------------------------------------------------
    # Phase 3: Generate local config(s)
    # For tiny models (60M/150M/300M) all three share the same data; generate
    # all three from a single download run.  For 1B/7B, generate only the
    # config that was actually passed as --config.
    # -----------------------------------------------------------------------
    print("\nGenerating local configs …")
    config_dir = cfg_path.parent

    # Build the list of configs to rewrite.
    # Tiny model run: if any of the three public tiny configs is given, rewrite all three.
    tiny_names = ["OLMo-60M-public.yaml", "OLMo-150M-public.yaml", "OLMo-300M-public.yaml"]
    if cfg_path.name in tiny_names:
        configs_to_rewrite = [config_dir / n for n in tiny_names]
    else:
        configs_to_rewrite = [cfg_path]

    for pub_cfg in configs_to_rewrite:
        if not pub_cfg.exists():
            print(f"  SKIP {pub_cfg} (not found)")
            continue
        local_cfg = rewrite_yaml_paths(pub_cfg, data_dir, selected_train, all_eval)
        with open(local_cfg) as f:
            content = f.read()
        n_local = content.count(str(data_dir))
        print(f"  Wrote {local_cfg}  ({n_local} local paths)")

    print("\nDone.")
    print("\nTo use the local data, run e.g.:")
    print(f"  ./run_olmo_60m.sh pre   (will auto-detect configs/tiny/OLMo-60M-local.yaml)")


if __name__ == "__main__":
    main()
