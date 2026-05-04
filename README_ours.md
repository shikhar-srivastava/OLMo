# OLMo — Our Modifications

## Environment Setup

Create the `olmo` conda environment with Python 3.11 and install all dependencies:

```bash
# 1. Create the base environment
conda create -n olmo python=3.11 -c conda-forge
conda activate olmo

# 2. Install PyTorch with CUDA 12.4 (matches H100/L40S driver stack ≥ 525)
pip install torch==2.6.0+cu124 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# 3. Install the OLMo package in editable mode (from repo root)
pip install -e '.[all]'

# 4. Install remaining runtime dependencies
pip install \
    wandb==0.26.0 \
    omegaconf==2.3.0 \
    rich==13.9.4 \
    cached_path==1.8.10 \
    tokenizers==0.22.2 \
    transformers==5.5.4 \
    datasets==4.8.4 \
    boto3==1.42.89 \
    requests==2.33.1 \
    tqdm==4.67.3 \
    pyyaml==6.0.3
```

Key versions pinned in this environment:

| Package | Version |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.6.0+cu124 |
| CUDA toolkit | 12.4 (driver ≥ 525, supports H100 and L40S) |
| ai2-olmo | 0.6.0 (editable install) |
| transformers | 5.5.4 |
| wandb | 0.26.0 |
| omegaconf | 2.3.0 |
| triton | 3.2.0 |

## LayerNorm Scaling (LNS)

A new normalization variant `lns` was added to `olmo/config.py` (`LayerNormType.lns`) and implemented in `olmo/model.py`. LNS uses RMSNorm as its base but applies a depth-dependent scale factor of `1 / sqrt(layer_id + 1)` to the normalized output before both the attention and feed-forward sublayers in `OLMoSequentialBlock` and `OLMoLlamaBlock`. This is selected via `--model.layer_norm_type=lns`.

## Training Scripts (run / dispatch)

`run_olmo_{60m,150m,300m,1b}.sh` launch 4-GPU training via `torchrun --nproc_per_node 4`. `run_olmo_7b.sh` launches **8-GPU** training on a single node (`torchrun --nproc_per_node 8`) for **8× A100 80GB**. Each script takes a `norm_type` argument (`pre` = RMSNorm baseline, `lns` = LNS variant) and sets consistent overrides: `wandb.project=olmo-runs`, `wandb.entity=null` (uses your logged-in W&B account), `max_duration=2e10T` (20B token cap), and a local `checkpoints/<run_name>` save folder with no remote upload. Five corresponding `dispatch_olmo_{size}.sh` scripts call each run script twice — once for `pre` and once for `lns` — to launch both variants in sequence.

## Per-device microbatch and wall-clock ETA (20B tokens)

`device_train_microbatch_size` is overridden in each `run_olmo_*.sh` (not the YAML defaults). Gradient accumulation fills out the configured `global_train_batch_size`; effective training dynamics are unchanged.

| Model | `device_train_microbatch_size` | GPUs (this setup) | ETA (wall clock) |
|-------|-------------------------------:|-------------------|------------------|
| 60M   | 32 | 4× H100 | unknown (expected under 12 h) |
| 150M  | 8  | 4× L40S | ~26 h |
| 300M  | 8  | 4× H100 | ~18 h |
| 1B    | 8  | 4× H100 | ~32 h |
| 7B    | 4  | 8× A100 (single node) | ~6 days (~144 h) |

ETAs are approximate observed or projected totals for a full 20B-token run with local cached data; actual time varies with I/O, checkpointing, and cluster load.

## Public Configs for Tiny Models

The default `configs/tiny/OLMo-{60M,150M,300M}.yaml` files reference `s3://ai2-llm/` data paths requiring AWS credentials. The pre-existing `*-public.yaml` variants (`OLMo-60M-public.yaml`, etc.) in the same directory use `https://olmo-data.org/` instead and are what the run scripts use. `remote_save_folder` was set to `null` in all three to prevent any remote checkpoint uploads. A stale data path (`redpajama_stackexchange_only/v1_decontaminated`) present in these files was corrected to `redpajama_v1_decon_fix/stackexchange` per [allenai/OLMo#878](https://github.com/allenai/OLMo/issues/878).

## Local Data Download

`scripts/download_olmo_data.py` downloads a proportional subset of training shards locally and generates `*-local.yaml` config files that point to local paths instead of streaming from `olmo-data.org`. It sends parallel HEAD requests to determine shard sizes, selects shards proportionally across all data source groups (preserving the original mixture), downloads them with resume support, and rewrites the YAML. Default data directory is `/dev/shm/ssrivastava/datasets/olmo-data`. Run once per model family before training:

```bash
# Tiny models (60M / 150M / 300M share the same data)
python scripts/download_olmo_data.py \
    --config configs/tiny/OLMo-60M-public.yaml \
    --data-dir /dev/shm/ssrivastava/datasets/olmo-data \
    --target-tokens 40_000_000_000

# 1B
python scripts/download_olmo_data.py \
    --config configs/official-0724/OLMo-1B.yaml \
    --data-dir /dev/shm/ssrivastava/datasets/olmo-data \
    --target-tokens 40_000_000_000

# 7B
python scripts/download_olmo_data.py \
    --config configs/official-0724/OLMo-7B.yaml \
    --data-dir /dev/shm/ssrivastava/datasets/olmo-data \
    --target-tokens 40_000_000_000
```

All five model sizes can share the same `--data-dir` without conflicts; their training shards live under non-overlapping subdirectories (`olmo-mix/v1_6-decontaminated`, `olmo-mix/v1_5`, `olmo-mix/v1_5-sample`), and the shared eval shards are simply skipped on the second download.

## Local Config Auto-Detection

All five run scripts check for a `*-local.yaml` config at startup. If found (generated by the download script above), it is used for training with no network I/O. Otherwise the script falls back to the `*-public.yaml` (tiny models) or base YAML (1B/7B) to stream data from `olmo-data.org`. The active config and mode are printed in the startup banner.

## Running Training

All commands below assume the `olmo` conda environment is active and you are at the repo root. Each `run_olmo_*.sh` takes a required `norm_type` (`pre` or `lns`) and an optional `master_port` (default `29500`). Each `dispatch_olmo_*.sh` runs `pre` and `lns` back-to-back on the same port.

### 60M (4× H100)

```bash
./run_olmo_60m.sh pre              # RMSNorm baseline
./run_olmo_60m.sh lns               # LayerNorm Scaling
./run_olmo_60m.sh lns 29515         # custom torchrun master port
./dispatch_olmo_60m.sh              # both variants in sequence
```

### 150M (4× L40S)

```bash
./run_olmo_150m.sh pre
./run_olmo_150m.sh lns
./dispatch_olmo_150m.sh
```

### 300M (4× H100)

```bash
./run_olmo_300m.sh pre
./run_olmo_300m.sh lns
./dispatch_olmo_300m.sh
```

### 1B (4× H100)

```bash
./run_olmo_1b.sh pre
./run_olmo_1b.sh lns
./dispatch_olmo_1b.sh
```

### 7B (8× A100 80GB, single node)

```bash
./run_olmo_7b.sh pre
./run_olmo_7b.sh lns
./dispatch_olmo_7b.sh
```

### W&B routing

By default the run scripts pass `--wandb.entity=null`, which sends runs to your logged-in W&B account. To route to a team, export `WANDB_ENTITY` before invoking the script:

```bash
export WANDB_ENTITY=klab-shikhar
./run_olmo_60m.sh lns
```

### Concurrent runs on the same node

Use distinct `master_port` values when launching multiple `torchrun` jobs simultaneously (e.g. while one run is still up):

```bash
./run_olmo_60m.sh  pre 29500 &
./run_olmo_150m.sh pre 29501 &
```

## Token-Budget Fail-Safes

The 20B-token cap is enforced by a single CLI flag (`--max_duration='2e10T'`) that every `run_olmo_*.sh` passes explicitly. To prevent silent overrides, [scripts/train.py](scripts/train.py) refuses to start unless:

1. `--max_duration=<value>` is present on the CLI (the YAMLs carry a sentinel string `MUST_BE_SET_BY_RUN_SCRIPT` that has no valid parse).
2. After CLI merge, `cfg.max_duration` is no longer the sentinel.
3. None of the competing termination knobs (`stop_at`, `stop_after`, `time_limit`, `early_stopping_factor`) are set — these would otherwise be the trainer's actual loop terminator and silently override `max_duration`.

If any check fails, the run aborts with a clear error before the first optimizer step. Any reintroduction of these knobs in a config or CLI override has to be intentional.
