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
| `tools/rig/fastfetch.py` | **current path**: direct, parallel, resumable Hub download on the rig itself (`PW_FAST_FETCH=1`) |
| `tools/rig/nvme-setup.sh` | instance-store NVMe -> RAID0 -> `/mnt/nvme`, prints the cache exports |
| `tools/rig/stage_weights.py` | DEPRECATED (S3 bucket deleted, no AWS storage): CPU-box job Hub -> sha256-verified -> `s3://<bucket>/pw-weights/<repo>/<commit>/` |
| `tools/rig/mirror.py` | DEPRECATED, unused: the consumer of that mirror (`PW_MODEL_MIRROR`) |
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

## Direct download on a rig (current path)

The S3 mirror below is **unused and deprecated**: the bucket was deleted and no standing AWS storage is kept. Its `PW_MODEL_MIRROR`,
`PW_MIRROR_JOBS` and `PW_MODEL_MIRROR_STRICT` settings still work if you build a mirror yourself, but nothing here depends on them.
Instead the rig pulls the weights straight from the Hub, fast enough that staging is not worth its storage bill.

**Measured** (2026-10-30, a `c6in.xlarge` in us-east-1, Hub files served by `us.aws.cdn.hf.co`, written to `/dev/null`; not measured on g7e or p5):

| parallel ranged streams | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|
| MiB/s | 145 | 437 | 843 | 1,133 | 1,557 (13 Gbit/s) |

Throughput scales almost linearly with streams, so one `curl` (what `lc_fetch_shards` used) leaves most of the link idle. The same download to the
default **gp3 root volume reached only 170 MiB/s**: the disk, not the network, is then the limit, so also put the cache on fast storage.
Arithmetic from those numbers, not a measurement of a real model: 100 GiB at 1.5 GiB/s is about 70 s, at gp3's 170 MiB/s about 10 minutes.

### Turn it on

```bash
sudo tools/rig/nvme-setup.sh            # instance-store NVMe -> /mnt/nvme (g7e: 6 TB; p4d/p5 more); safe where there is none
source /mnt/nvme/pw-env.sh              # PW_CACHE=/mnt/nvme/pw-cache  HF_HOME=/mnt/nvme/hf  UV_CACHE_DIR=/mnt/nvme/uv-cache  TMPDIR=/mnt/nvme/tmp
export PW_FAST_FETCH=1                  # default OFF: without it nothing below changes
export PW_FAST_STREAMS=48               # parallel range requests per file (default 48; on a g7e the rate was 780 MiB/s at 24, 1,067 at 48, 960-990 at 64 to 96)
tools/rig/run_batch.sh <tag> ... --env PW_FAST_FETCH=1 --env PW_CACHE=/mnt/nvme/pw-cache --env HF_HOME=/mnt/nvme/hf
```

`nvme-setup.sh` (idempotent, needs root, Deep Learning AMI / Ubuntu 22.04): finds NVMe disks whose model is `Amazon EC2 NVMe Instance Storage`
(never EBS, never a mounted disk), makes an `mdadm` RAID0 `/dev/md/pwnvme` when there are several, `mkfs.xfs` (ext4 if xfs is missing) labelled `pwnvme`,
mounts it at `/mnt/nvme` with `noatime`, and `chown`s it to `$SUDO_USER`/`ubuntu`. Run again: already mounted means nothing changes; after a reboot
the labelled filesystem is mounted again, not recreated. **With no instance store** (g5, g6, c6in) it says so, exits 0 and tells you to stay on gp3
(raise the volume's throughput and IOPS at launch if the download is the bottleneck). Instance-store data is gone when the instance stops or terminates;
it is a cache, never the only copy. Environment overrides exist only for the tests (`PW_NVME_MNT`, `PW_NVME_PROC_MOUNTS`, `PW_NVME_SKIP_ROOT_CHECK`, `PW_NVME_ID`, `PW_NVME_USER`).

### What `PW_FAST_FETCH=1` does

| Consumer | Replaces |
|---|---|
| `lc_fetch_shards` (all `llamacpp-*`, `tools/llamacpp/common.sh`) | `curl -C -`, one file at a time, one stream; files go to `$PW_CACHE/models/<workload>/` as before |
| `workloads/_pytorch/pinned.py snapshot()` (pytorch, diffusers, vlm, asr: `catalog_common.fetch_snapshot`, `diffusion.py`, `vlm.py`) | `huggingface_hub.snapshot_download`; files go to `<cache>/pw-fast/<repo>/<commit>/` |
| `workloads/_shared/vllm_catalog.py snapshot()` (all `vllm-*` and `vllm-bench-*`) | `snapshot_download`, same layout |

Not covered (small, unchanged): whisper.cpp ggml files, Ollama pulls, ONNX/spaCy assets, the two workloads that call `snapshot_download` directly
(`sdxl-base-diffusers`, `blip2-opt-2p7b-pytorch`). Precedence when both are set: `PW_MODEL_FILE`, then `PW_MODEL_MIRROR`, then `PW_FAST_FETCH`, then the old Hub path.

`tools/rig/fastfetch.py` is standard-library Python (`http.client` plus a thread pool; no aria2c, no hf_transfer, so nothing to install on the rig):

* **Resolve once.** `resolve/<commit>/<file>` redirects to a signed CDN URL. It is resolved once; when a chunk gets 401/403 (the signature expired) one worker re-resolves and all reuse the new URL. A 404/401/403 on the Hub itself is permanent (gated repository: set `HF_TOKEN`; the token goes only to the host of the first URL, never to the CDN).
* **Streams and memory.** `PW_FAST_STREAMS` (48) workers each fetch a `PW_FAST_CHUNK_MIB` (48 MiB) range and `pwrite()` it into a preallocated `<file>.part`, through a 1 MiB buffer: memory use is about `streams` MiB.
* **Resumable, never a corrupt "complete" file.** `<file>.part.map` has one byte per chunk, set only after the data file was `fsync`ed (batched every `PW_FAST_FLUSH_S`, 2 s). On resume, the 2 highest-numbered recorded chunks are fetched again anyway (the earlier staging run lost data to a reboot), and the final file name exists only after the whole file hashed to the pin and was renamed. A sidecar for a different size, chunk size or pin is discarded.
* **Verified, always.** sha256 is computed in order by a thread that follows the downloaders (no second pass over a 100 GiB file) and compared with `model.sha256` (or the Hub's LFS oid for `files`/`snapshot` of unpinned LFS files; small unpinned configs are checked by size only). A file already present is hashed first and kept only if it matches. A mismatch deletes the file and **fails the run (exit 1), never a SKIP**; if the download had been resumed, the file is fetched once more from scratch first. After a fast `snapshot()` the callers skip their own re-hash of those files in the same process (same size and mtime) and otherwise still verify as before.
* **Retries.** Connection resets, short reads, 429 (honours `Retry-After`) and 5xx retry with exponential backoff and jitter, `PW_FAST_RETRIES` (8) per chunk. When they run out, the run SKIPs with the reason (as an unreachable Hub did before) and the partial file stays for the next try.
* CLI: `fastfetch.py file|files|snapshot ...` (exit 0 ok, 1 mismatch, 4 transfer failed); progress goes to stderr every 10 s unless `PW_FAST_QUIET=1`. `PW_HF_ENDPOINT`/`HF_ENDPOINT` change the Hub URL (used by the tests).

### Untested at scale (read before planning rig-hours around it)

* The 1,557 MiB/s figure is raw network to `/dev/null` from one 4-vCPU box. **Not yet measured**: this code's end-to-end rate on g7e or p5 (Python threads doing TLS decryption, `pwrite` and sha256 may top out below what `curl` ranges reach; if so, 16 streams and bigger chunks, or `curl --range` workers, are the next step), the NVMe rate on g7e, and `nvme-setup.sh` on a real instance store (it is tested only against fake `lsblk`/`mdadm`/`mkfs`/`mount`).
* Hub rate limits and xet/CAS throttling for hundreds of GiB from one IP are unknown; 24 streams on every file of a 500 GiB model may draw 429s (they are retried with backoff, which would show up as lower throughput, not failures). An `HF_TOKEN` may raise limits.
* sha256 at 1.5 GiB/s needs one fast core (SHA extensions); on a slow core hashing, not the network, is the bound (the hash thread follows the download, so it overlaps but does not disappear).
* Real Hub behaviour of `Range` requests on xet-backed files through the redirect, and the exact signed-URL lifetime, are assumed from the Hub's documented redirect scheme and the earlier staging run, not tested against a 100 GiB file. The tests use a local HTTP server with Range support, a redirecting `resolve/` endpoint, expiring tokens, a SIGKILL mid-download and planted corrupt partials.
* Tests: `tests/test_rig_fastfetch.py` (assembled bytes, one stream, empty file, concurrency limit, retries, persistent failure then resume, SIGKILL then resume, corrupt tail, corruption beyond the redo window, sha mismatch, expired URL, token not sent to the CDN, `files`/`snapshot`, the wiring with the flag off and on, `nvme-setup.sh`).

## Weights: stage once, mirror, copy at rig speed (DEPRECATED: the bucket is deleted; see Direct download above)

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

### Staging from a shared box (polite mode)

`stage_weights.py` is built to run for days on a machine other people use (a CI host). Everything below is on by default.

| Option | Default | Meaning |
|---|---|---|
| `--nice` / `--no-nice` | on | curl, aws and the python process itself run under `nice -n 19`; curl and aws also under `ionice -c3` (idle class; the kernel honours it only with the BFQ scheduler, so on `none`/`mq-deadline` disks `nice` and the rate cap are what limit the load) |
| `--threads N` | 4 | files in flight (download, hash, upload pipeline); the pool works through one global queue of files, models in `--order` |
| `--rate-limit MBPS` | 20 | MiB/s over all downloads (each curl gets `--limit-rate` of rate/threads) and, separately, over all uploads (the generated `~/.pw-stage/aws-config` sets `s3.max_bandwidth` to rate/`--aws-max-concurrent` per upload process). 0 = unlimited. With fewer files in flight than threads the total is lower |
| `--aws-max-concurrent N` | 2 | simultaneous `aws s3 cp` processes (each also limited to 4 parallel requests) |
| `--max-load L` | 12 | pause while the 1-minute load average is above L |
| `--min-free-gb G` | 500 | pause while the filesystem holding `--scratch` has less than G GiB free |
| `--busy-rate MBPS` | 8 | while any process named exactly `Runner.Worker` exists (`pgrep -x`: a GitHub Actions job is running) downloads and uploads are limited to this instead of `--rate-limit`; 0 = no throttle. **CI throttles, it does not pause** |
| `--strict-ci-pause` | off | the old behaviour: stop transfers while a `Runner.Worker` exists (starves the run when CI is back to back) |
| `--retries N`, `--retry-base-s`, `--retry-max-s` | 20, 5, 300 | per-file retries of a transient error with exponential backoff |
| `--batch-gb G` | 1500 | at most G GiB downloaded and not yet uploaded; a file larger than that runs alone |
| `--order` | `smallest-first` | models by total size ascending (more models fully staged early), or `manifest` (sorted by repository name) |

Pause gates (load average, free space; and CI with `--strict-ci-pause`) are checked before every file, while waiting every `--poll-s` (30) seconds, and by a running transfer every min(`--poll-s`, 5) seconds. A closed gate stops curl (the partial file is kept and `curl -C -` resumes it) and freezes aws with SIGSTOP (an upload in flight resumes where it was; a pause of many minutes can make S3 drop the connection, which the retry logic covers). Each pause and resume is logged with its reason (`PAUSE: ...` / `RESUME after 312s (was: ...)`).

Throttle (the default for CI): a monitor checks for `Runner.Worker` every 5 s and logs each transition (`THROTTLE busy: 1 Runner.Worker process(es) running; downloads and uploads limited to 8 MiB/s (was 20 MiB/s)` / `THROTTLE free: ...`). The new rate takes effect live: a running curl is stopped and restarted with the new `--limit-rate` and `-C -` (it resumes the partial file; the cost is one reconnect), and uploads started afterwards use an AWS config with the new `max_bandwidth`. An `aws s3 cp` already running cannot change its cap, and killing it would lose its multipart progress, so it is put on a SIGSTOP/SIGCONT duty cycle of new/old rate (1 s period) until it ends (logged once as `THROTTLE: in-flight upload slowed to a 40% duty cycle`). With `--rate-limit 0` an in-flight upload is not slowed. `--status` shows `throttle: BUSY ...` or `free`.

Retries: a transient failure of one file's download or upload is retried in the process, `--retry-base-s` doubling to `--retry-max-s`, up to `--retries` tries, without failing the run (`RETRY download <file> (try 2/20) in 10s: ...`). Transient: curl exits 6, 7, 16, 18, 28, 35, 52, 55, 56 and 92 (the Hub CDN cancelling an HTTP/2 stream), HTTP 408, 429 and 5xx, a download of the wrong size, any S3 upload error other than the permanent ones, and an S3 object whose size does not match after upload. Every retry asks the Hub's `resolve/` URL again (the CDN redirect is signed and expires; no URL is reused) and resumes the partial file. Permanent, never retried: HTTP 404, 401 and 403 (gated repository: set `HF_TOKEN`), a sha256 that does not match its pin (the download is deleted), S3 AccessDenied or a missing bucket, a local write error (disk full). A file that exhausts its tries or hits a permanent error is an error for its model; the run exits 1 after the others finish and the supervisor restarts it.

Resuming after a crash: a reboot or power loss can leave a partial file whose size is right but whose tail is zeroed or stale (the page cache was not flushed), and `curl -C -` would continue after the damage. So the first time a pass touches a partial file left by an earlier process, its **last 64 MiB** (the new size rounded down to a 4 MiB boundary; `--resume-trim-mb`, 0 turns it off) are cut and fetched again; in-process restarts (throttle, retries) do not trim. The state dir records `/proc/sys/kernel/random/boot_id`; when it differs from the last run (or was never recorded) every such partial loses its **last 4 GiB** instead, because writeback delay, not the file size, bounds the damage. If a file that was resumed still fails its sha256, it is deleted and downloaded again from scratch, up to 2 times (`SHA256 MISMATCH on a resumed file ...`); a mismatch after one clean pass (one curl, no partial found) is permanent at once. The corrupt file is always deleted, never left to be re-hashed.

Permanent failures: a file that fails permanently (404/401/403, sha256, AccessDenied, local write error) is listed by name and reason in the final `SUMMARY` (`FAILED FILE <repo>@<rev> <file>: <reason>`), in `status.json` and `--status`, and on stderr; the run finishes everything else first, then exits **5**. The supervisor does not restart on 5 (it would only repeat the failure): it prints the list to its log and stderr and exits 5. Exhausted transient retries stay exit 1 (restarted).

Disk discipline: a file is hashed in 4 MiB chunks (never read whole into memory) and compared with its pin before upload; after `aws s3 cp` the object's size is read back with `s3api head-object` and must equal the local size, and only then is it recorded in `manifest.json` and the local copy deleted. A size mismatch keeps the local copy, records nothing, and is retried by the next pass. `manifest.json` stays `complete: false` until the last file of the model is in.

Progress: `stage_weights.py --status` (reads `~/.pw-stage/status.json`, rewritten every 10 s) prints staged models, GiB done/total, the files in flight, the pause state, the **measured** sustained rate (bytes verified in the mirror by this run divided by wall time, pauses included) and the rate over the last 10 minutes, and an ETA computed from the sustained rate (unknown until the first file finishes). The same numbers are logged at every file (`DONE ... sustained X MiB/s`) and at the end (`SUMMARY`). One staging run per state dir (`~/.pw-stage/stage.lock`).

`tools/rig/stage_supervisor.sh` keeps it going without root or systemd system units:

```bash
S=tools/rig/stage_supervisor.sh
$S start --tier 2b --mirror s3://BUCKET/pw-weights      # detached; flock makes a second start refuse (exit 9)
$S status                                              # supervisor up? + stage_weights.py --status
$S stop                                                # SIGTERM; partial files are kept
$S run --tier 2b --mirror s3://BUCKET/pw-weights       # same loop in the foreground (what cron runs)
$S install-cron --tier 2b --mirror s3://BUCKET/pw-weights   # @reboot user crontab entry, marked "# pw-stage-supervisor"
$S remove-cron                                         # removes exactly that line
```

It restarts `stage_weights.py` on failure with a growing delay (30 s doubling to 30 min; reset after a run longer than 10 minutes), exits 0 when the plan is complete, and does not retry exit 2 (usage or catalog error) or 4 (run lock held). Logs: `~/.pw-stage/logs/stage.log` (events, rotated at 10 MB, 5 kept), `supervisor.log`, `stage.stderr.log` (rotated at 5 MB). If `~/.pw-stage/credentials` exists it is used as the AWS credentials file with profile `pw-stage`; `aws` is taken from `~/.local/bin` when it is not on the PATH. When `systemd-run --user --scope` works the process runs in a scope with `CPUQuota=400% MemoryMax=6G MemorySwapMax=0`, otherwise only nice/ionice and the rate caps apply; the supervisor log says which (`limits: ...`).

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

## server2 as a CPU test host

server2 (192.168.1.216, Ryzen 9 9950X, Zen 5, 32 threads, `avx2 avx512f avx512_vnni avx_vnni`, no AMX; 29 GiB RAM of which about 14 GiB is free; Ubuntu 24.04) is where CPU-side checks run instead of the WSL machine or server1. **The box is shared**: the multi-day weight-staging job (`~/pw-stage`, `tools/rig/stage_supervisor.sh`) runs there at low priority and pauses when the 1-minute load average exceeds 12 (and throttles to 8 MiB/s while a CI job runs), and two GitHub runner services run CI jobs. Your work must not starve either. Rules that worked (2026-10-15, about 40 minutes in all, load never above 5):

* One directory only: `~/pw-cpu`. The repo goes over ssh with no GitHub credentials on the box: `git archive --prefix=repo/ HEAD | ssh server2 'cd ~/pw-cpu && rm -rf repo && tar x'`. Do not touch `/home/ghrunner`, `~/pw-stage`, `~/.pw-stage`, system units, crontabs; no sudo is needed. No AWS calls.
* Everything sequential, never two workloads at once, inside one limited scope (`~/pw-cpu/limit.sh`, the same limits as the staging supervisor, with a little more memory):

  ```
  export XDG_RUNTIME_DIR=/run/user/$(id -u); export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
  systemd-run --user --scope -q -p CPUQuota=400% -p MemoryMax=8G -p MemorySwapMax=0 nice -n 19 ionice -c3 "$@"
  ```

* Etiquette between workloads: `while [ "$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)" -ge 8 ]; do sleep 30; done`. Look at `~/pw-stage/repo/tools/rig/stage_supervisor.sh status` before and after; the staging log (`~/.pw-stage/logs/stage.log`) shows pauses and `THROTTLE` lines, which are caused by CI jobs, not by you.
* Python: the system `python3` is 3.12.3 and has `venv`, so `python3 -m venv` works (no uv needed, and `tests/test_arch_coverage.py`'s unreachable-index test runs). The workloads' cp313-only wheel (`sentencepiece-test-model`) needs 3.13: install uv into `~/pw-cpu/uv` (`UV_INSTALL_DIR=... UV_NO_MODIFY_PATH=1`, standalone installer), `uv venv --python 3.13`, install `tools/ort-requirements.txt` into it and run with `PW_ORT_PYTHON=<that python>`.
* Environment used (`~/pw-cpu/env.sh`): `PW_VENV_ROOT=~/pw-cpu/venvs PW_CACHE=~/pw-cpu/cache PW_PYTHON=$(PW_TORCH_CPU_PREFIX=~/pw-cpu/venv-torch tools/torch-cpu-env.sh) PW_ORT_THREADS=1` (the default: the references were recorded with one ORT thread, so numbers stay comparable with `docs/int8-and-nondeterminism.md`), `TMPDIR=~/pw-cpu/tmp`; llama.cpp builds with `PW_BUILD_JOBS=4`; the Ollama workload with `OLLAMA_MODELS=~/pw-cpu/ollama-models PW_PULL=1` (it starts its own `ollama serve` on a random port and does not touch the box's Ollama service).
* Unit tests: `$PW_PYTHON -m unittest discover -s tests` (torch 2.14.1+cpu). To run without skips also `pip install opencv-python==5.0.0.93 gguf==0.19.0` into that venv. Result: 468 tests, 74 s, no skips, no failures. With the venv lacking those two packages: 447 tests, 2 skipped (OpenCV, numpy+gguf), which is expected.
* Workloads: `bin/pw run <w> --target cpu --out ~/pw-cpu/results/<w>`. Never record benchmark numbers here (shared box), and no `-bench` workload has a cpu reference anyway.
* Tidy up: keep `~/pw-cpu/venv-torch`, `venvs`, `venvs313`, `uv`; delete `cache` (3.3 GB after a full run: weights, ORT assets, llama.cpp and whisper.cpp builds), `uvcache`, `ollama-models`. Kept state is about 3.8 GB.

Results of the first full pass (commit `main` + this change, torch 2.14.1+cpu, transformers 5.19.0, onnxruntime 1.30.0, spaCy 3.8.16, llama.cpp b11447): 63 workloads run, 63 PASS, 0 FAIL, 0 unexpected SKIP. Seconds are wall time under the limits (the first workload of each family includes making its environment or building; downloads included).

| Workload | Result | Seconds | Detail |
| --- | --- | ---: | --- |
| `selftest`, `loadgen-plumbing-check`, `py3langid-wheel` | PASS | 0.0, 4.3, 1.9 | |
| `sentencepiece-test-model` | PASS (on Python 3.13) | 0.7 | SKIPs on 3.12 by design: the pinned wheel is cp313 only (`SKIP: the pinned sentencepiece wheel cannot be imported here`) |
| `spacy-en-core-web-sm`, `-md`, `spacy-multilingual-sm` | PASS | 14.0, 2.3, 4.8 | spaCy 3.8.16 |
| `onnx-zoo-bertsquad-int8`, `kokoro-tts-int8-onnx`, `moonshine-tiny-en-onnx` | PASS (3 runs x 2 modes) | 0.8 to 1.2, 4.7, 1.0 | `docs/int8-and-nondeterminism.md`, section 4 |
| `onnx-zoo-bidaf`, `-efficientnet-lite4`, `-mnist`, `-mobilenetv2`, `-shufflenet-v2`, `-ssd-mobilenetv1` | PASS | 2.1, 5.3, 1.0, 2.2, 2.2, 3.8 | zoo output diffs 0 to 3.9e-09 |
| `crnn-text-recognition`, `glove-wiki-gigaword-50-knn`, `gtcrn-enhance-onnx`, `kws-zipformer-gigaspeech-onnx`, `minilm-l6-v2-onnx`, `nanodet-object-detection`, `pphumanseg-person-segmentation`, `ppocr-rapidocr`, `sface-face-embedding`, `silero-vad-onnx`, `wespeaker-resnet34-onnx`, `yolox-object-detection`, `yunet-face-detection`, `zipformer-audio-tagging-onnx` | PASS | 1.7, 6.3, 1.3, 4.7, 3.7, 1.8, 1.0, 8.0, 1.6, 1.4, 1.6, 2.0, 0.8, 11.5 | |
| `arch-encdec`, `arch-llama-family`, `arch-moe`, `arch-ssm`, `arch-vision` | PASS | 3.0, 3.0, 2.6, 2.7, 2.7 | margins 3.0e-3 to 5.1e-2; fp64 deviations at 1e-6 |
| `bert-base-uncased-pytorch`, `gpt2-small-pytorch` | PASS | 64.7, 25.2 | includes the Hub download |
| `gpt-train-fp32`, `gpt-train-bf16` | PASS | 2.0, 1.8 | loss 3.583 to 2.976 / 2.978 |
| `lib-attention-precision`, `-composite-blocks`, `-fft-linalg`, `-rnn-conv`, `-sparse-embedding` | PASS | 1.3, 1.2, 1.2, 1.2, 1.2 | worst error 8.4e-2, 3.9e-2, 7.2e-6, 2.6e-2, 1.3e-6 of rms |
| `pytorch-microsuite`, `resnet18-randinit-pytorch` | PASS | 1.3, 2.6 | |
| `llamacpp-synth-llama` | PASS | 322.7 | includes building llama.cpp b11447 (cmake, `-j4`) |
| the other 12 `llamacpp-synth-*` | PASS | 0.3 to 7.5 | |
| `llamacpp-qwen25-0p5b`, `llamacpp-smollm2-135m` | PASS | 20.6, 4.8 | model download included |
| `whisper-cpp-tiny-en` | PASS | 29.1 | builds whisper.cpp; WER 0.000 |
| `smollm2-135m-ollama` | PASS | 14.1 | pulls the model into `~/pw-cpu/ollama-models` |

Not run: the cpu `-bench` workloads (`arch-bench`, `gpt-train-bench`, `llamacpp-bench-*`, `llamacpp-synth-bench-*`: benchmarks, no reference, and no numbers from a shared box), and every gpu / sim target.
