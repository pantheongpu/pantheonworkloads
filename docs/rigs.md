# Rented rigs: launch, cap, mirror the weights, run a batch, tear down

Stage 1 to 2a launched about 70 workloads' worth of rigs with ad-hoc scripts: each agent re-invented the launch and the teardown, one lost
its SSH key to a shared scratchpad, and the shutdown-timer guardrail was cancelled by hand. Stage 2b puts about 56 workloads (110 manifests with
their bench twins) on **one** 8 x A100 80 GB node, whose hours are expensive and whose runs are dominated by weight downloads. `tools/rig/` is the
reusable, tested version. Everything here is driven by the `aws` CLI and `ssh`; the tests (`tests/test_rig*.py`) run all of it against a fake `aws`,
`ssh`, `scp`, `curl` and `nvidia-smi` (`tests/rig_fakes/`) and need no AWS account.

**What has and has not been exercised against real AWS is listed at the end. Read it before trusting a rig to this tooling.**

| Script | Does |
| --- | --- |
| `tools/rig/launch.sh` | one tagged on-demand instance, key pair, SSH-only security group, self-terminate timer; prints the connection command |
| `tools/rig/extend.sh <tag> <hours> --reason ...` | the only sanctioned way to move a rig's deadline; logged |
| `tools/rig/status.sh` | every `pw-*` instance and volume with elapsed hours and running cost vs cap; exit 1 when anything is over |
| `tools/rig/teardown.sh <tag>` | terminate, wait, verify, delete key pair and group, log the cost; idempotent |
| `tools/rig/stage_weights.py` | CPU-box job: Hub -> sha256-verified -> `s3://<bucket>/pw-weights/<repo>/<commit>/` |
| `tools/rig/mirror.py` | the consumer: copy from `PW_MODEL_MIRROR` and verify (used by `lc_fetch_shards` and `workloads/_pytorch/pinned.py`) |
| `tools/rig/run_batch.sh <tag> ...` | run workloads on the rig, sequentially, results copied back, nothing pushed |

The shell files are one-line wrappers; the logic is in `rig.py`, `batch.py` (Python 3, standard library plus the repository's own PyYAML use).

## State: where things live

`~/.cache/pw-rigs/` (override `PW_RIGS_HOME`; **refused** when the path has a `scratchpad` component or is under `/tmp/claude-*`, because that is how a key was lost):

```
~/.cache/pw-rigs/ledger.tsv          append-only audit trail: launch, extend, teardown (and timer-unverified, launch-failed)
~/.cache/pw-rigs/<tag>/key.pem       the private key, mode 0600, in a 0700 directory; deleted by teardown
~/.cache/pw-rigs/<tag>/rig.json      instance id, AZ, rate, caps, absolute deadline, extensions, state
~/.cache/pw-rigs/<tag>/results/<batch>/   what run_batch.sh copied back
```

A tag is `pw-` plus lower-case letters, digits and `-` (the prefix is added when missing). `launch.sh` refuses a tag whose `rig.json` already exists, so a second
agent cannot overwrite a running rig's key; pick a tag per agent (`pw-stage2b-<who>`).

## Launching

```bash
tools/rig/launch.sh --tag stage2b --instance-type p4de.24xlarge \
    --disk-gib 2000 --iops 16000 --throughput 1000 \
    --azs us-east-1a,us-east-1b,us-east-1c,us-east-1d,us-east-1f \
    --max-hours 24 --max-dollars 1200            # add --rate USD_PER_HOUR when you know the real rate
tools/rig/launch.sh ... --dry-run                # prints the plan and the user-data; no AWS call
```

* **AMI**: `--ami ami-...`, or `--ami-name` (default `Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) *`, Amazon-owned, newest `CreationDate` wins). The login user is `ubuntu`.
* **Disk**: root gp3, `--disk-gib`, `--iops`, `--throughput`, delete-on-termination. gp3 defaults (125 MB/s, 3000 IOPS) made model loads and sha256 checks crawl in tier B (`docs/model-registry.md`, "Rig traps"); ask for more at launch instead of `modify-volume` later.
* **AZ fallback**: the AZs are tried in order, each in its default subnet; `InsufficientInstanceCapacity` (and `Unsupported`) moves to the next AZ, any other error stops, and when no AZ has capacity the key pair, group and local directory are released again. One security group serves every attempt.
* **Security**: group with TCP 22 from `<your public IP>/32` only (`--my-ip` or `checkip.amazonaws.com`), IMDSv2 required, instance-initiated shutdown behaviour **terminate**, volumes tagged.
* **Tags** on the instance: `Name`, `pw-rig-tool=1`, `pw-deadline` (epoch), `pw-max-dollars`, `pw-hourly-usd`, `pw-owner`. `status.sh` and `teardown.sh` work from these tags if `rig.json` is missing; `teardown.sh` refuses an instance without `pw-rig-tool` (shared accounts: other sessions' instances) unless `--force-foreign`.

### Caps

`--max-hours` and `--max-dollars` are both required. The hourly figure is `rate + disk` where the rate is `--rate` or the built-in table (`RATES` in `rig.py`) and the disk is the gp3 monthly price /730 for the size, IOPS above 3000 and MB/s above 125. The rig lives for `min(max_hours, max_dollars / hourly)` and that is printed as the cap and its reason. The deadline is an **absolute epoch** computed at launch.

**The rate table is typed from memory of the on-demand list prices, not read from AWS**, and for p4/p5 deliberately the older, higher figures (a rate that is too high ends a rig early; one that is too low overspends). The gp3 prices are from memory as well. Both are only used for caps and estimates; pass `--rate` when you know better. The cost in the ledger is `elapsed x (rate + disk)`, **an estimate, not a bill**.

### The timer, and what it does not stop

`user_data` writes `/etc/pw-rig/deadline` (epoch, `root:root`, 0644) and a systemd timer that runs a watchdog every minute. At or after the deadline the watchdog runs `shutdown -h now`, which terminates the instance (shutdown behaviour = terminate). It **fails closed**: a missing or unparseable deadline file counts as expired. Fifteen minutes before, it broadcasts a warning. The deadline file, the watchdog and both unit files are made immutable (`chattr +i`), so `rm`, an `echo >` or `systemctl disable` by a non-root process, or by root who has not noticed, fails. `launch.sh` then reads the file back over ssh and checks the timer is active (`--ssh-wait`, default 300 s; exit 3 and a ledger line `timer-unverified` if it cannot).

Extending needs `tools/rig/extend.sh <tag> <hours> --reason "<at least 10 characters>"` **from the launching machine**: it ssh-es in, lifts the flag, writes the new deadline, sets it again, updates `rig.json` and the instance tags, and appends the old and new deadline, the projected worst-case cost and your reason to `ledger.tsv`. It refuses when the projected cost passes the cap (`--max-dollars N` raises the cap, logged), when the deadline has already passed, and for rigs not launched from this machine.

Limits, stated plainly: anyone with root on the instance (the `ubuntu` user has passwordless sudo on these AMIs) can still run `chattr -i` and stop the timer. The design stops silent and accidental extension and makes the legitimate path leave a record; it does not stop a determined root user. The second layer is outside the box: `status.sh` compares against `rig.json` and the instance tags, neither of which the instance can edit, and flags `PAST-DEADLINE` ten minutes after the deadline. **Run `status.sh` from a cron or a loop while a rig is up** (it exits 1 when something is over, so it composes with an alert), and always `teardown.sh` when done.

## Status and teardown

`status.sh` lists `pw-*` instances (pending, running, stopping, stopped) and volumes in `--region` (comma list ok). Flags: `OVER-CAP` (running cost above the dollar cap), `PAST-DEADLINE`, `NO-CAP-RECORD` (an instance with no cap in `rig.json` or tags is uncapped, so it counts), `ORPHAN-VOLUME` (an `available` volume nobody is paying attention to), `STOPPED-EBS-BILLING` and `FOREIGN` (informational). Exit 1 on any of the first four.

`teardown.sh <tag>` terminates, waits until `describe-instances` says terminated, waits for `describe-volumes` to show no volume with the rig's `Name` tag (deleting any that stayed `available`), deletes the key pair, deletes the security group (retrying `DependencyViolation` while the network interface goes away), removes the local private key, and writes the elapsed hours and estimated cost to the ledger once. It is safe to run again (it re-checks AWS, writes nothing new to the ledger), exits 1 with `PROBLEM:` lines when anything is left, and never touches a rig it did not launch.

## Weights: stage once, mirror, copy at rig speed

Model weights are the long pole: the stage 2b set is **55 distinct repositories, 9,683 GiB** (sum of the pinned files' sizes in `tools/catalog/hub.json` for the `2b` tier below; from the catalog, not downloaded here), the largest being Kimi-K2.7-Code-GGUF and Kimi-K2.6-GGUF at 543.6 GiB, DeepSeek-V4.1-Flash 475.3 GiB, Qwen3-VL-235B 439.0 GiB, GLM-5.3 435.2 GiB. Most workloads come as a functional/bench pair on one repository, and `lc_fetch_shards` kept one directory per workload, so before this change both twins downloaded the same shards; the mirror path keeps one directory per repository and commit.

**Tiers** (computed from the manifests by `tools/rig/wl.py`): `A` one GPU of at most 22 GiB; `B` one GPU of more than 22 and at most 44 GiB; `2b` everything else that fits 8 x 80 GiB (`gpus <= 8`, `gpu_memory_gb <= 80`) and has not graduated (notes not starting with `VERIFIED ON` or `BLOCKED`); `big` needs more than 8 x 80 GiB.

### Producer: a cheap CPU box

```bash
# on any cheap instance in the SAME region as the rig, with an instance profile or credentials allowed to write the bucket
python3 tools/rig/stage_weights.py --tier 2b --mirror s3://MY-BUCKET/pw-weights --dry-run   # plan: files, GiB, what is staged
python3 tools/rig/stage_weights.py --tier 2b --mirror s3://MY-BUCKET/pw-weights                        # do it (resumable)
python3 tools/rig/stage_weights.py --tier 2b --mirror s3://MY-BUCKET/pw-weights --check                # exit 1 unless every model is complete
HF_TOKEN=... python3 tools/rig/stage_weights.py llamacpp-foo ...                                       # gated repositories need a token
```

For each distinct `<repo> @ <commit>` (the header comment of `model.sha256`; a workload and its bench twin share one) it plans: every pinned file, plus the small config/tokenizer files the loaders also fetch (`*.json *.txt *.model *.jinja *.tiktoken *.yaml *.py tokenizer* *.vocab *.bpe`, each at most 64 MiB), cross-checking every pin against the Hub's LFS oid in `hub.json` (a disagreement is an error and nothing is downloaded). Each file is fetched with `curl -C -` (resumable; the partial file is kept in `--scratch`, default `~/.cache/pw-stage`), its sha256 checked, uploaded with `aws s3 cp`, and recorded in `manifest.json`, which is rewritten after every file with `complete: false` until the last one. A corrupt download is deleted and never uploaded. Files already in the manifest with the right size and sha256 are skipped, so a second run, or a run after a crash, only does what is missing. Files are fetched one at a time (to go faster, run several boxes on disjoint lists of workloads; never two on the same model at once, because `manifest.json` is rewritten whole and the second writer would drop the first's entries). Scratch disk needed: the largest single file (shards are a few GiB to tens of GiB; check `--dry-run`).

Layout (`PW_MODEL_MIRROR` is the prefix that holds `<repo>/<commit>/`, here `s3://MY-BUCKET/pw-weights`):

```
s3://MY-BUCKET/pw-weights/unsloth/Kimi-K2.6-GGUF/47c7cab1.../manifest.json   {"schema": "pw-mirror/0", "complete": true, "files": {"<path>": {"size", "sha256", "pinned", "source"}}}
s3://MY-BUCKET/pw-weights/unsloth/Kimi-K2.6-GGUF/47c7cab1.../Q4_K_M/....gguf
```

A local directory works the same way (`--mirror /mnt/weights`), which is also how the tests run.

### Consumer: the rig

```bash
export PW_MODEL_MIRROR=s3://MY-BUCKET/pw-weights   # or: tools/rig/run_batch.sh <tag> ... --mirror s3://...
export PW_MIRROR_JOBS=16                           # parallel copies (default 8)
export PW_MODEL_MIRROR_STRICT=1                    # a model missing from the mirror SKIPs instead of using the Hub
```

* `lc_fetch_shards` (all `llamacpp-*`) copies the shards the workload's `model.sha256` lists into `$PW_CACHE/models/mirror/<repo>-<commit>/` and sets `LC_MODEL` to the first shard.
* `workloads/_pytorch/pinned.py snapshot()` replaces `huggingface_hub.snapshot_download` in `catalog_common.fetch_snapshot` (transformers workloads), `diffusion.py` and `vlm.py`, and `workloads/_shared/vllm_catalog.py` does the same inline for vLLM. Files land in `$PW_CACHE/hf/pw-mirror/<repo>/<commit>/` and the loaders point at that directory, exactly as with a snapshot.
* **Verification stays on.** Every copied file is hashed and compared with `manifest.json`; the pins from `model.sha256` are compared with the manifest before anything is copied (a mirror that lies in its manifest is caught) and the existing post-check (`verify_snapshot`, the `sha256sum` loop of `lc_fetch_shards`) still runs on the Hub path. A mismatch deletes the copy and **fails the run (exit 1); it is never retried from the Hub and never turned into a SKIP**. A file already on the rig's disk is trusted only if it was verified earlier and its size and mtime are unchanged.
* **Misses**: no manifest, an incomplete manifest lacking a wanted file, or a wanted file absent from it is a *miss*. By default a miss warns and downloads from the Hub as before; with `PW_MODEL_MIRROR_STRICT=1` it is a SKIP (`run_batch.sh --strict-mirror` sets it). Use strict on a metered rig so a typo cannot start a 500 GiB Hub download.
* **Not covered**: `whisper.cpp` ggml files, Ollama, and ONNX/spaCy assets still download as before; they are small. An unpinned run (`PW_HF_REVISION` set) uses the Hub.
* `python3 tools/rig/mirror.py fetch --mirror M --repo R --rev V --dest D [--sums model.sha256] [--flat]` is the same code for a manual copy; exit 3 = miss, 1 = mismatch or copy error.

### Cost and speed: what is known and what is not

Known (arithmetic on the catalog): the 2b set is 9,683 GiB, the 110 manifests share 55 repositories, the largest is 543.6 GiB. **Not measured by anyone here**: Hub download speed from a CPU box, `aws s3 cp` speed from S3 to a p4de, the cost of sha256 on the rig's cores, or whether the instance-store NVMe of a p4de is a better `PW_CACHE` than gp3 (it is not used by this tooling; set `PW_CACHE` to it yourself). So there is no speed claim: time to copy a model is `size / sustained throughput`, and the throughput must be measured on the first model (`time tools/rig/mirror.py fetch ...` on the rig, `sha256sum` on one shard) before planning rig-hours around it. Sizing arithmetic that does not need a measurement: S3 Standard storage at the list price I remember ($0.023 per GiB-month, unverified) is about $223 per month for 9,683 GiB; a bucket in the rig's region avoids inter-region transfer (also from memory); `PW_MIRROR_JOBS` and the CLI's `max_concurrent_requests` (`aws configure set default.s3.max_concurrent_requests 64`) are the knobs if one stream is slow. Disk on the rig: each model's weights are held once under `PW_CACHE`; the largest set is 543.6 GiB, and a mirror copy plus an old copy of another model is what to budget for (`--disk-gib`).

## Running a batch

```bash
tools/rig/run_batch.sh stage2b --tier 2b --mirror s3://MY-BUCKET/pw-weights --strict-mirror      # sequential
tools/rig/run_batch.sh stage2b llamacpp-glm52 llamacpp-bench-glm52 --bench --repeat 5
tools/rig/run_batch.sh stage2b --tier 2b --pack                                                   # concurrent on disjoint GPU sets
tools/rig/run_batch.sh stage2b --tier 2b --dry-run                                               # what would run / be skipped
```

The rig needs this repository checked out at `--remote-dir` (default `~/pantheonworkloads`, same commit as your local one: `git bundle` or clone it, run `docs/first-gpu-run-a10g.md`'s setup for the venvs and builds; this tooling does not provision runtimes). Then, per workload:

* **requires**: the rig's GPUs (`nvidia-smi` over ssh) are checked against the manifest with `bin/pw`'s own `unmet_requires`; a workload that cannot run is **not started**, recorded as `SKIP-REQUIRES` with the reason, as is one with no `gpu` target.
* **one job per GPU**: each job gets exactly the GPUs it `requires` (`CUDA_VISIBLE_DEVICES`; `--all-gpus` leaves all visible), and takes `flock -n` on `<lock-dir>/pw-gpu-<n>.lock` for each, so a second batch, or another agent's job, gets `BUSY` instead of silently sharing the card (the stage 2a false failures). Without `--pack` jobs run strictly one after the other. With `--pack`, jobs run concurrently on **disjoint** GPU sets handed out from a free list, among the GPUs that meet each job's `requires`.
* **timeout**: `PW_TIMEOUT_S` for `bin/pw` is the manifest's `timeout_s` (or `--timeout`), and an outer `timeout -k` (limit + 120 s) kills anything that ignored it; result `TIMEOUT`. The batch continues.
* **results**: after each job the remote `~/pw-batch/<batch>/` is copied to `~/.cache/pw-rigs/<tag>/results/<batch>/` (`<w>.json`, `<w>.log`, `<w>.stdout`, `<w>.exit`, `results.tsv`, and `bench/` with `--bench`), and `batch.tsv` summarises every workload (PASS/FAIL/SKIP/TIMEOUT/BUSY/NO-RESULT/SKIP-REQUIRES/NOT-RUN). Exit 1 when anything FAILed, timed out, was busy, or was not run.
* **deadline guard**: a job is not started unless it can finish (timeout + grace + 5 minutes) before the rig's deadline; the rest are `NOT-RUN`. Extend (with a reason) or relaunch.
* **it pushes nothing**: no git, no `gh`, nothing is recorded into a repository. Copy `bench/` and the references into a checkout yourself, and commit them as the usual flow says.

`--env K=V` (repeatable) adds environment to the job, e.g. `--env PW_CACHE=/mnt/nvme/pw --env PW_LLAMACPP_NO_BUILD=1`.

## A stage 2b session, start to finish

1. `stage_weights.py --tier 2b --mirror s3://... --dry-run`, then run it on a cheap CPU box; `--check` until it exits 0.
2. `launch.sh --dry-run`, then `launch.sh` with the caps for the whole session; note the printed deadline.
3. ssh in, clone the repository at the commit to test, install runtimes (`docs/first-gpu-run-a10g.md` traps: Python 3.12 via `uv`, `env -u LD_LIBRARY_PATH` for PyTorch, `PW_CUDA_ARCHS=80` for llama.cpp), measure the mirror copy speed on the smallest 2b model.
4. `run_batch.sh <tag> --tier 2b --mirror ... --strict-mirror` (add `--pack` only for workloads that need few GPUs and when the lock bookkeeping is wanted).
5. `status.sh` in a loop on your machine; `extend.sh` only with a reason.
6. `teardown.sh <tag>`; read the ledger line; confirm `status.sh` shows nothing.

## Tests, and what they do not prove

`tests/test_rig.py` (cap and deadline arithmetic, key placement and mode, the scratchpad refusal, AZ fallback and cleanup, user-data contents, timer verification, ledger lines, teardown idempotence, leaked volumes, security-group retries, foreign instances, status over-cap / past-deadline / no-cap / orphan volume, extend), `tests/test_rig_mirror.py` (plan, resumable staging, corrupt download, sha256 mismatch on every consumer path, miss vs mismatch, `lc_fetch_shards` end to end, `pinned.snapshot`, the real catalog's 2b pins) and `tests/test_rig_batch.py` (requires skips, one-at-a-time, disjoint `--pack` sets, timeout, busy GPU, results copy-back, deadline guard). They run in CI with `python3 -m unittest discover -s tests`.

**Exercised on real AWS (2026-10-13, us-east-1, a g5.xlarge for ~7 minutes and a t3.small for ~8 minutes, about $0.03 in all):**

* a dry run, a real launch (AMI search, key pair, security group limited to the caller's address, `run-instances`, AZ loop, wait for the address, ssh), the on-box timer check, `status.sh`, the extend refusals (reason too short; projected cost over the cap), and a full `teardown.sh` that verified terminated / no volumes / key pair and group gone;
* the on-box watchdog: a rig launched with `--max-hours 0.1` terminated itself about a minute after its absolute deadline, AWS reported `Client.InstanceInitiatedShutdown: Instance initiated shutdown`, and CloudTrail showed only `RunInstances` (no `TerminateInstances` call), so the `shutdown -h now` with shutdown behaviour `terminate` path works, including `chattr +i`, `systemctl enable --now` in cloud-init and `/sbin/shutdown` on the Deep Learning AMI (Ubuntu 22.04, 20261009);
* two bugs only the real CLI could show, both fixed with regression tests (`TestRealAwsShapes`): `--metadata-options` used `HttpEndpointEnabled` (the key is `HttpEndpoint`), and a multi-line CLI error kept only its first line ("Parameter validation failed:") and hid the cause. The AMI also needs a root volume of at least 75 GiB; `launch.sh` now reports that error in full.

**Still untested against real AWS or real hardware:**

* the `InsufficientInstanceCapacity` / `Unsupported` AZ fallback and the quota errors (`VcpuLimitExceeded`, `MaxSpotInstanceCountExceeded`) with real codes: no launch hit a capacity error in these tests, and `p4de.24xlarge` and the other P types have never been launched with this tooling;
* `shutdown -h now` terminating a p4de specifically (it does for g5 and t3 here);
* `StateTransitionReason` timestamps for the teardown cost when an instance had already terminated itself (the t3.small had; the ledger recorded it, but the exact cost it computed is an estimate from the rate table, not a bill);
* the rate table: typed from memory of the on-demand list prices, and only used for caps and estimates;
* `aws s3 cp` against a real bucket (multi-part behaviour for 50 GiB shards, credentials on the rig and the CPU box), real Hub downloads with `curl -C -` (xet-backed files, redirects, rate limits, `HF_TOKEN` for gated repositories);
* the consumers on real runtimes: `pinned.snapshot` replaced four `snapshot_download` call sites and was exercised only with a fake `huggingface_hub`; `lc_fetch_shards` was exercised with tiny files;
* `run_batch.sh` against a real bin/pw run on a real rig: the `flock`, `timeout -k` and `$HOME` quoting were run on this machine's bash only; the real `bin/pw` writes `<w>.json` as the fake does (read from `cmd_run`), but no real batch was run.
