# Phase 2 plan: stage 2b and Tier F

Written 2026-10-12. **Research and plan only.** Nothing was rented, launched, downloaded or run for this document; no AWS API was called.
Prices come from web pages read on 2026-10-12 (URLs in section 2). Most are third-party price trackers, not the provider's own
page; each is marked **official** (an AWS page fetched directly) or **tracker** (secondary, may be stale or wrong). Where nothing
could be read it says **unverified**. No price here has been charged to anyone: treat every dollar figure as an estimate to be
re-checked on the provider's page at the moment of launch.

Scope. Stage 2a (Tier A, Tier B) is done (docs/model-registry.md, "First real runs"). Stage 2b is every remaining catalog workload
that fits 8 x 80 GiB; Tier F is the five workloads whose `requires.gpu_memory_gb` is above 80. The staging approach (weights fetched
once into object storage, rigs copy from it) is being built in `tools/rig` by a parallel change and is not designed here; this
document only prices it.

## 1. Tier F: what each workload really needs

`requires` is one pair (gpus, GiB on **each** GPU). A card passes when the driver reports at least 98% of the declared GiB
(`tools/hwcheck.py`, `SLACK = 0.98`), and a host with *more* GPUs than declared also passes (the GPU count is a minimum). The
declared figure is the smallest class that holds the arithmetic need (docs/model-registry.md, "Hardware need"):
llama.cpp `weights/n * 1.05 + 3.0`; vLLM `(weights/n + 6.0) / 0.90`.

Workloads with `requires.gpu_memory_gb > 80` (found with a scan of `workloads/*/manifest.yaml`; the `-bench` twins repeat the same
numbers and share the same weights): exactly five.

| Workload | Runtime | Weights GiB | Declared | Arithmetic need per GPU | Alternatives that also fit |
| --- | --- | ---: | --- | ---: | --- |
| `llamacpp-kimi-k2-thinking` | llama.cpp b11447 | 578.6 | 4 x 180 | 578.6/4*1.05 + 3 = 154.9 | 8 x 140: 578.6/8*1.05 + 3 = 78.9 |
| `llamacpp-deepseek-v4-pro-0813` | llama.cpp | 791.3 | 4 x 256 | 791.3/4*1.05 + 3 = 210.7 | 8 x 140: 791.3/8*1.05 + 3 = 106.9 (also 8 x 180) |
| `llamacpp-qwen38-2p4t-a95b` | llama.cpp | 1220.8 | 8 x 180 | 1220.8/8*1.05 + 3 = 163.2 | none (8 x 140 = 163.2 > 140) |
| `llamacpp-kimi-k3` | llama.cpp | 1405.1 | 8 x 192 | 1405.1/8*1.05 + 3 = 187.4 | none (8 x 180 = 187.4 > 180) |
| `vllm-hy4-preview` | vLLM 0.30.0, TP 8 | 1452.8 | 8 x 256 | (1452.8/8 + 6)/0.9 = 208.5 | none (a 180/192 GiB card cannot hold 208.5 at 0.90 utilisation) |

Sum of the five weights: 5448.6 GiB (about 5.85 TB). No other manifest declares more than 80 GiB per GPU; the only other
catalog row above 8 x 80 is MiniMax-H3 (2 x 256, diffusers), which is blocked (own inference code), so it is not in scope.

### Which cards pass, exactly

Per-GPU memory as drivers report it (from the task brief; the marketing figure is in brackets). The "needs" column is 98% of the
declared GiB.

| Card | Reported GiB | 4 x 180 (>= 176.4) | 8 x 180 (>= 176.4) | 8 x 192 (>= 188.2) | 4 x 256 / 8 x 256 (>= 250.9) |
| --- | ---: | --- | --- | --- | --- |
| A100 / H100 80 GB | 79.6 | no | no | no | no |
| H200 (141 GB) | ~140 | no | no | no | no |
| B200 (180 GB class) | ~178-179 | yes, by about 2 GiB | yes, by about 2 GiB | no | no |
| B300 / GB300 (288 GB) | ~268-279 | yes | yes | yes | yes |
| MI300X (192 GB) | ~192 | yes | yes | yes, by about 4 GiB | no |
| MI325X (256 GB) | ~256 | yes | yes | yes | yes, by about 5 GiB |
| MI355X (288 GB) | ~288 | yes | yes | yes | yes |

Per workload:

| Workload | Passes `requires` as written | Passes only with an `alternatives` extension |
| --- | --- | --- |
| `llamacpp-kimi-k2-thinking` | B200, B300, MI300X, MI325X, MI355X (8 GPUs in the box; 4 is the minimum) | H200 8 x 140 |
| `llamacpp-deepseek-v4-pro-0813` | B300, MI325X, MI355X | H200 8 x 140, B200 8 x 180 |
| `llamacpp-qwen38-2p4t-a95b` | B200, B300, MI300X, MI325X, MI355X | none |
| `llamacpp-kimi-k3` | B300, MI300X, MI325X, MI355X | none |
| `vllm-hy4-preview` | B300, MI325X, MI355X (ROCm vLLM is an unverified path, see section 2) | none |

Two margins are thin and should be checked with `tools/hwcheck.py` on the first minute of a rig, before any weights are copied:
B200 reports about 178-179 GiB against 176.4 required (AWS's own page gives 1,432 GB for 8 cards, which is 179 per card), and MI300X
reports about 192 against 188.2 required while Kimi-K3 needs 187.4 of it (4.6 GiB of slack; a real llama.cpp compute buffer may not
fit, the 3 GiB margin is an assumption, not a measurement). AWS's page gives B300 as 2,144 GB for 8 cards, 268 per card; the brief
says about 279 GiB. Either number passes every row above, so the B300 conclusion does not depend on which is right.

AWS `p6e-gb200` (GB200 UltraServer) is not considered further: 4 GPUs per instance at about 185 GB, ARM host, sold only as
UltraServers through Capacity Blocks in the Dallas Local Zone at 381 USD/h per 36-GPU unit (official), and 185 GB is close
enough to the 176.4 line that it is not a safe 180 class.

### Proposed schema extension (sketch only, not implemented)

`requires.alternatives`: an optional list of `{gpus, gpu_memory_gb}` pairs. The top-level pair stays the primary. A host passes when
it satisfies the primary **or any** alternative (each with the same 98% rule and "at least this many GPUs" semantics).

```yaml
requires:
  gpus: 4
  gpu_memory_gb: 180
  alternatives:
    - {gpus: 8, gpu_memory_gb: 140}
  notes: "..."
```

Change list (each one small; nothing here has been edited):

1. `docs/manifest.md`: document the key; say a host reports SKIP only if it fails the primary and every alternative, and that the SKIP
   message lists all shapes.
2. `tools/validate.py`: accept the key, check each entry like the primary (`gpus` integer >= 1, `gpu_memory_gb` > 0), and require
   every alternative to be a different shape from the primary.
3. `tools/hwcheck.py` and the matching code in `bin/pw`: evaluate the primary, then each alternative, pass on the first match.
4. `tools/llamacpp/common.sh` `lc_require_gpus` and the `run.sh` template: take several `n mem` pairs (`lc_require_gpus 4 180 8 140`). vLLM
   workloads keep one shape per alternative as long as `run.sh` hard-codes `--tensor-parallel-size`; a vLLM alternative may only change
   the per-GPU memory, not the GPU count, unless `run.sh` derives TP from the host.
5. `tools/catalog/generate.py`: emit alternatives from the same formulas (every n in {1,2,4,8} and class in the widely rented list
   that holds weights plus margin, dropping any that a smaller already-listed pair dominates); `tests/test_model_catalog.py` recomputes
   them; the coverage table's Needs column shows `4 x 180 or 8 x 140`.
6. Effect: only `kimi-k2-thinking` and `deepseek-v4-pro-0813` gain a shape (8 x 140, i.e. H200). The other three have no alternative.
   Also worth a pass over the 8 x 80 and 4 x 80 stage-2b rows, where a lone 8 x 48 or 4 x 140 might fit; that is not needed for the cost plan.

## 2. Where the machines can be rented (2026-10)

How to read this: `$/h` is the whole 8-GPU node unless stated; "API" means it can be launched from a CLI or API without a human in
a console. The workloads' runtime is NVIDIA CUDA for the llama.cpp ones (`tools/llamacpp/common.sh` also has a HIP build, picked when
`/dev/kfd` exists, so llama.cpp can run on AMD) and vLLM 0.30.0 for `vllm-hy4-preview` (the run script only checks `import vllm`;
whether a ROCm vLLM 0.30.0 loads Hy4-preview is **unverified** and nothing in the repo has run it).

### 2a. AWS

| SKU | GPUs | vCPU | $/h on-demand | Capacity Block $/h | Min commitment | Spot | API | Notes |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| p4de.24xlarge | 8 x A100 80 GB (640 GB) | 96 | 27.45 us-east-1 (tracker: vantage, holori, Feb 2026); other trackers 32.77-32.94 (region/date unclear) | 20.369 us-east-1/us-west-2 (**official**) | none on-demand; Capacity Block 1 day or more | not verified for this type | yes | Stage 2b target. Needs 96 vCPUs of P quota; quota is 64 with 192 requested (case 179035780000124). |
| p5e.48xlarge | 8 x H200 141 GB | 192 | no clean figure found; one tracker says no on-demand at all | 54.924 (**official**), regions us-east-2, us-west-1/2, eu-west-2, eu-north-1, ap-northeast-1, sa-east-1, ... (not us-east-1) | Capacity Block | tracker: about 15 in us-east-2 (unreliable) | yes | Supply reported tight (tracker). |
| p5en.48xlarge | 8 x H200 141 GB | 192 | 63.296 us-east-1 (tracker: holori, Feb 2026) | 63.158 (**official**) us-east-1/2, us-west-1/2, eu-west-2, ... | none on-demand | unverified | yes | |
| p6-b200.48xlarge | 8 x B200, 1,432 GB total per AWS | 192 | 113.93 us-east-1 (tracker: vantage, holori) | 113.666 (**official**) | none on-demand | trackers disagree: 23, 30, 40-42 | yes | 2,048 GiB RAM, 8 x 3.84 TB NVMe (**official** spec page). One search snippet quoted a Capacity Block of 82.4; the official pricing page fetched afterwards says 113.666, which is used. |
| p6-b300.48xlarge | 8 x B300, 2,144 GB total per AWS | 192 | 142.42 us-east-1 (tracker) | 129.168 (**official**) us-east-1, us-west-2, ap-* | none on-demand | tracker: about 21 (unverified, looks implausibly low) | yes | 4,096 GiB RAM, 8 x 3.84 TB NVMe. GA 2025-11-18 (tracker). |
| p6e-gb200 | 4 x GB200 per instance, UltraServer only | 144 | none | 380.95 per 36-GPU UltraServer (**official**), Dallas Local Zone only | Capacity Block | no | yes | Not viable (above). |

Quota: AWS "Running On-Demand P instances" is counted in vCPUs. 64 today cannot launch p4de (96); 192 once approved fits exactly one
192-vCPU node (p5en/p6) **or** one p4de, never two at once. Spot P quota is a separate quota. Neither p4de nor any Tier F node should be
requested before the quota case is answered. Earlier approvals recorded in project notes (H100 x2 Spot, B200 Spot all 8 GPUs) were for
other workloads and are not assumed to cover this plan.

Official sources: https://aws.amazon.com/ec2/capacityblocks/pricing (fetched 2026-10-12; the page says prices are "scheduled to be updated
next in January, 2027") and https://aws.amazon.com/ec2/instance-types/p6/ (specs). On-demand prices are not on a page the fetch tool
could read, so on-demand = tracker values (vantage.sh, holori, cloudprice.net, aws-pricing.com via web search 2026-10-12).

### 2b. Other NVIDIA providers (trackers only; none of these were opened on the provider's own page)

| Provider | SKU | $/GPU-h -> 8-GPU $/h | Commitment | Interruptible | API | Fits Tier F? | Source and caveat |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Lambda | 8 x B200 SXM (180 GB) | 6.69 -> 53.5 (Sept 2026; one older source 5.29) | on-demand, per-minute (tracker) | no | yes | k2-thinking, qwen38 | computeprices.com compare page updated 2026-09-18, usagepricing.com, costbench. lambda.ai/pricing not read. Lambda H200 8x: not found. Project notes say Lambda instances on the shared account may belong to other jobs: only terminate ones this plan launches. |
| CoreWeave | 8 x B200 / 8 x H200 | 8.60 -> 68.8 / 6.31 -> 50.4 | contract-style; CFO said "largely sold out" (tracker) | no | yes | B200: k2, qwen38; H200: with extension | cloudzero, computeprices, spheron blog, Mar-Jul 2026. Sales-led access; unverified for a one-off. |
| Nebius | B200; H200 | 5.50 -> 44 on-demand (Mar 2026); 2.69 preemptible; H200 2.45 | on-demand and reserved | yes (preemptible) | yes | same as B200 / H200 | single sources, conflicting dates; unverified. |
| Crusoe | H200; B200 | H200 4.29 -> 34; B200 contact sales | | spot "contact sales" | yes | H200 with extension | computeprices.com. |
| RunPod | B200; H200 | 5.98 -> 47.8; 3.59 -> 28.7 | per-second | community vs secure cloud | yes | B200: k2, qwen38 | trackers only; whether an 8-GPU pod is in stock is unverified. |
| Vast.ai | H200; B200 | H200 about 3.55 -> 28; B200 5.6-6.9 -> 45-55 | none | yes (interruptible, bid) | yes | H200 with extension; B200 k2/qwen38 | Prices set by hosts; 8-GPU B200 not verified to exist; host bandwidth varies a lot. |
| Together, Fireworks (dedicated endpoints) | model endpoints, not rentable boxes | n/a | | | | no | No root access, so llama.cpp and vLLM at pinned versions with `llama-bench` cannot be run. Not viable. |

Project constraint to respect: the bench campaign driver limits parallel Vast cards to 2 and RunPod pods to 2 at once. An 8-GPU Vast or RunPod
rental breaks that limit; it needs an explicit exception from the user (decision 5).

### 2c. AMD (ROCm; only llama.cpp HIP is known to work in this repo)

| Provider | SKU | $/GPU-h -> 8-GPU $/h | Commitment | Notes |
| --- | --- | --- | --- | --- |
| Vultr | MI300X; MI325X; MI355X | 1.75-1.85 -> 14.8; 2.00 -> 16.0; 2.59 -> 20.7 (tracker; July 2026) | on-demand (tracker) | Cheapest everywhere; 8-GPU availability and whether the MI355X price is real are **unverified** (DigitalOcean's MI355X is listed at 8.00, which suggests the two trackers measure different things). |
| TensorWave | MI300X 1.71; MI325X 2.25 | 13.7; 18 | on-demand (tracker) | MI355X price not found. |
| Hot Aisle | MI300X | 1.99 -> 15.9 | | MI325X / MI355X not found. |
| DigitalOcean | MI300X 1.99; MI325X 3.80 (1.69 on contract); MI355X 8.00 | | contract for the low rate | |
| Crusoe | MI300X | 3.45 -> 27.6 | | |
| RunPod | MI300X | 1.99 -> 15.9 | | |

Sources: spheron.network AMD pricing posts, computeprices.com, gpuperhour.com, thundercompute tracker, all July-Oct 2026, none the
provider's own page. For AMD there are two problems besides price. (1) `vllm-hy4-preview` has not been shown to work on ROCm. (2) A
`reference.json` recorded on AMD would be compared exactly (3 greedy tokens) when someone later runs the workload on NVIDIA; the
project's own experience (docs/int8-and-nondeterminism.md) is that exact-token references can differ across hardware. Bench records are
per device anyway; the open question is only which vendor records the reference.

### What could not be verified

The providers' own pricing pages (Lambda, CoreWeave, Nebius, Crusoe, RunPod, Vast, Vultr, TensorWave, Hot Aisle) were not read; AWS on-demand
prices (not Capacity Block) are from trackers; AWS Spot prices for P-family are from disagreeing trackers and are not used for the estimates
below; no provider's current stock of 8-GPU nodes; per-GPU memory as the driver reports it on any of these cards (the registry numbers are
taken as given); whether p4de, p5e or p6 have Spot capacity at all; p4de local NVMe size (AWS lists 8 x 1 TB, not re-fetched); S3 to instance
throughput (assumed, below).

## 3. Cost model

### 3a. What a rig-hour is spent on, per workload

Stage 2a's procedure, applied to a pair (functional workload plus its `-bench` twin, same weights): record the reference (1 load), 3 consecutive
confirmation runs (3 loads), then the bench twin at 5 repeats (5 loads, each running `llama-bench` pp512 + tg128 with 3 internal repetitions).
That is **9 loads** of the same weights per pair, plus the one-time copy. So:

```
rig_h(workload) = W/s3_rate            copy weights S3 -> local NVMe
                + W/cold_load_rate     first load from NVMe
                + 8 * W/warm_load_rate the other 8 loads (page cache or NVMe)
                + fixed                process start, CUDA init, the 3-token decode, 5 x llama-bench compute, retries
rig_total = sum(rig_h) + setup (driver check, CUDA toolkit, llama.cpp build for the arch, venvs) + boot (about 10 min)
```

Weights are fetched once into object storage (the `tools/rig` approach), so a restart or a second rig never refetches from
Hugging Face. Hugging Face to S3 happens on a cheap CPU instance and is priced separately below. Copying from S3 inside the same AWS
region is free; to any non-AWS provider S3 egress is 0.09 USD/GB (AWS public price, not re-read today), which is 56 USD for
Kimi-K2-Thinking and about 520 USD for all five: for non-AWS rigs the weights should instead come straight from Hugging Face or from a
bucket in the provider's own region.

Assumptions (the arithmetic is in the scratchpad script used for this document and is reproduced by the formulas above):

| | low | likely | high |
| --- | ---: | ---: | ---: |
| S3 -> NVMe (GB/s) | 6 | 3 | 1 |
| first (cold) load (GB/s) | 3 | 2 | 1 |
| warm loads (GB/s) | 8 | 5 | 2 |
| fixed per workload (min), stage 2b | 15 | 25 | 45 |
| fixed per workload (min), Tier F (x1.6 for bigger models) | 24 | 40 | 72 |
| setup (h) | 1 | 2 | 3 |
| retry/debug overhead on the whole | 0% | +15% | +35% |

"High" is meant to cover a first run that fails once (the registry says none of these were ever run; stage 2a found run-script fixes) and a
slow object store, not a catastrophe. Anything beyond it is what the stop-loss is for.

### 3b. Stage 2b on one p4de.24xlarge (8 x A100 80 GB)

Scope: the 55 unverified functional workloads (and their 55 twins) with `requires` within 8 x 80 / 4 x 44 / 8 x 48, i.e. everything above
except the five Tier F ones: 17 x (1 x 80), 11 x (4 x 80), 10 x (8 x 80), 7 x (2 x 80), 5 x (1 x 48), 2 x (4 x 44), 2 x (2 x 48),
1 x (8 x 48). Total weights 9,586.7 GiB (10.3 TB), all distinct repositories. A100 80 GB reports 79.6 GiB, so all 8 x 80 rows pass. The
8 x 80 llama.cpp rows are the tightest (52.6 GiB needed per GPU for the 685B class, comfortable on 79.6).

Blockers: P quota 64 < 96 vCPUs (open request for 192); the node's local NVMe (about 8 TB) is smaller than 10.3 TB, so staging must be in
two to three batches grouped by size (copy a batch, run it, delete it), and the largest single set (543.6 GiB) is small enough that this is
easy. Order the batches 1 x 80 first (most workloads, cheapest to debug), then 2 and 4 x 80, then the 8 x 80 giants.

| | low | likely | high |
| --- | ---: | ---: | ---: |
| rig hours | 19 | 37 | 83 |
| at 27.45 USD/h on-demand (tracker) | 523 | 1,006 | 2,275 |
| at 32.94 USD/h (higher tracker) | 627 | 1,207 | 2,730 |
| S3 storage 10.3 TB for the stage, EBS scratch, one HF-to-S3 staging instance | about 90 | about 95 | about 100 |
| **stage 2b total** | **about 610-720** | **about 1,100-1,300** | **about 2,375-2,830** |

(Capacity Block at 20.37 USD/h would need whole days: 2 days = 977 USD for the likely case, cheaper only if the work is certain to fill
48 h; it also removes the quota-and-capacity gamble for a 96-vCPU node, so it is worth pricing again once the quota lands.)

HF -> S3 staging (not in the rig hours): about 10.3 TB at an assumed 0.5-1 GB/s from Hugging Face on a c6in-class instance is 3-6 h,
roughly 10-30 USD of compute; it can overlap with the rig running earlier batches.

### 3c. Tier F, per workload and per rig (one run-through each: reference + 3 confirmations + bench x5)

Hours (likely, including a 40-min fixed share): k2-thinking 1.2, dsv4-pro 1.4, qwen38 1.8, k3 1.9, hy4 2.0; plus about 2.2 h of setup and
boot if a rig is rented for that workload alone (setup is shared when several run on one rig). Dollars below are standalone, "likely", rounded;
low is about 0.55 times, high about 1.8 times.

| Workload | AWS B300 (142.42) | AWS B200 (113.93) | AWS H200 p5en (63.30), needs extension | Lambda 8 x B200 (53.5, tracker) | Vast 8 x H200 (28, tracker, needs extension) | AMD 8 x MI300X / MI325X (14.8 / 16.0, tracker; ROCm) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| kimi-k2-thinking | 487 | 390 | 213 | 180 | 96 | 51 / 55 |
| deepseek-v4-pro-0813 | 512 | n/a (needs ext.) | 226 | n/a | 101 | n/a / 58 |
| qwen38-2p4t-a95b | 563 | 451 | n/a | 212 | n/a | 59 / 63 |
| kimi-k3 | 585 | n/a | n/a | n/a | n/a | 61 / 66 |
| hy4-preview (vLLM) | 591 | n/a | n/a | n/a | n/a | not supported yet (vLLM ROCm) / 66, unverified |

All five on one rented node (setup paid once): AWS B300 low 724 (Capacity Block rate) to 799 (on-demand), **likely about 1,360-1,500**,
high 3,150-3,480 (the 3,480 is on-demand). All five on one 8 x MI325X: likely about 170, high about 390, with the two caveats above. The
two-model set that B200 can do (k2 and qwen38) is about 590 likely on AWS B200 and about 280 on Lambda B200; k2 plus dsv4-pro on an
H200 box (extension needed) is about 290 on p5en.

Tier F storage: 5.85 TB in S3 is about 135 USD/month (0.023 USD/GB-month, AWS public price), roughly 20-35 USD for the week this takes.
Non-AWS rigs add egress or a second copy as noted in 3a.

### 3d. Cheapest viable provider per Tier F workload

"Viable" means the workload's `requires` passes today without schema changes unless stated, and the software path is known to work.

| Workload | Cheapest NVIDIA-only (CUDA) | Cheapest overall (adds ROCm) |
| --- | --- | --- |
| kimi-k2-thinking | Lambda 8 x B200, about 180 (tracker, 53.5/h); AWS B200 about 390 is the verifiable-price option. With the extension: Vast 8 x H200 about 96 | Vultr/TensorWave 8 x MI300X, about 51 (tracker) |
| deepseek-v4-pro-0813 | AWS B300 about 512 as written. With the extension: Vast H200 about 101, AWS p5en about 226 | 8 x MI325X, about 58 |
| qwen38-2p4t-a95b | Lambda 8 x B200 about 212 (tracker); AWS B200 about 451 | 8 x MI300X about 59 |
| kimi-k3 | AWS B300 about 585 (the only NVIDIA option) | 8 x MI300X about 61 (thin 4.6 GiB slack) or MI325X about 66 |
| vllm-hy4-preview | AWS B300 about 591 (the only verified software path) | 8 x MI325X about 66, but vLLM ROCm unverified |

Several of these are 3 to 10 times cheaper than AWS because the AMD and neocloud figures are tracker prices for on-demand lists that
may not exist as a node today; they only become real when a person confirms stock and price on the provider's page.

## 4. Recommendation

**Order.** (1) Quota first: nothing below launches until case 179035780000124 is answered (192 vCPUs). (2) Stage 2b before Tier F: it is
cheaper per workload, it exercises the unverified run scripts (the 8 x 80 giants share their loader paths with Tier F), and what it breaks
informs Tier F. (3) Tier F in the order that buys the most coverage per dollar: k2-thinking, then dsv4-pro, then qwen38, then k3 and hy4
together on the largest-memory node.

**Caps and stop-losses.** A cap is where a person re-approves; a stop-loss is where the rig is terminated whatever state the run is in.

| Stage | Cap | Hard stop-loss | Also stop at |
| --- | ---: | ---: | --- |
| Stage 2b (p4de, on-demand) | 1,500 | 2,500 | 60 rig hours, or 3 rig hours with no new PASS |
| Tier F, k2 + dsv4-pro (cheapest rig that fits) | 450 | 750 | 8 rig hours |
| Tier F, qwen38 (B200 class) | 500 | 800 | 6 rig hours |
| Tier F, k3 + hy4 (B300, or MI325X) | 1,100 | 1,800 (B300) | 10 rig hours |
| Tier F total, if only one B300 node is rented for all five | 1,600 | 3,500 | 25 rig hours |

Mechanics: tag every rig with a self-terminate (a shutdown timer set at launch for the stop-loss hours), a budget alarm at the cap, and the
rig terminated by the run script on exit. Never leave an 8-GPU node idle overnight: a B300 idle costs about 3,400 USD/day.

**Minimal Tier F set for the most coverage per dollar.** If the user wants only some: (a) `llamacpp-kimi-k2-thinking`: cheapest (about 180-490
USD), covers the 4 x 180 llama.cpp path, and doubles as the pilot for the whole 8-GPU large-model flow; (b) `llamacpp-qwen38-2p4t-a95b`:
the only one other than K3 that nothing but a 180+ GiB card can run, about 210-560; (c) `llamacpp-kimi-k3` on a B300 (about 585) because it
also proves the 192 GiB class and the 1.4 TB load path. `vllm-hy4-preview` is the most expensive and the only vLLM one: do it last and only if
the B300 is already rented. `deepseek-v4-pro-0813` is cheap if the schema extension is accepted (H200) and costs a B300 otherwise. The
single most economical plan if the extension is approved: one H200 box (k2, dsv4-pro, about 290) plus one B300 box for qwen38, k3 and
hy4 (about 1,100 likely together, 2,300 high) = about 1,400 likely. Without the extension: one B300 box for all five, about 1,400 likely.
Either way the total is in the 1.4k range, which is why the AMD alternative (about 170 likely) is the single biggest lever, and also the one with
the two unresolved software questions.

**Pilot before committing.** Whatever the provider, spend the first 30-45 minutes of the first rig on: `tools/hwcheck.py` (does it report
the GiB assumed), one workload end to end, then extrapolate the measured load rate against section 3a's assumptions before continuing. If
the measured S3 or load rate is at the "high" column, stop at the cap rather than at the stop-loss.

## 5. Decisions needed from the user

1. Approve the `requires.alternatives` schema extension (section 1)? It lets H200 hosts run `kimi-k2-thinking` and `deepseek-v4-pro-0813`
   and roughly halves their cost; without it only B200/B300/MI3xx cards qualify.
2. May AMD be used at all for Tier F (llama.cpp HIP for four workloads), and if so which vendor records the reference (an AMD-recorded
   exact-token reference may not reproduce on NVIDIA)? Hy4-preview on ROCm vLLM is an unverified path.
3. Approve the caps and stop-losses in section 4 (stage 2b 1,500 cap / 2,500 stop; Tier F per row), or give different numbers.
4. Rent AWS stage 2b on-demand (about 27-33 USD/h) or as a Capacity Block (20.37 USD/h, whole days)? On-demand is assumed.
5. May non-AWS providers (Lambda, Vast, RunPod, Nebius, Vultr, TensorWave, Hot Aisle) be used for 8-GPU nodes? Vast and RunPod would exceed the
   current 2-card / 2-pod concurrency rule; and S3 staging then needs a per-provider copy or direct Hugging Face download.
6. Which Tier F subset: the minimal three (k2-thinking, qwen38, k3), all five, or only what fits the cheapest rig?
7. Spot or interruptible allowed for Tier F? (Staging in S3 makes a restart cheap, but a reclaim in the middle of a 1.4 TB load wastes up to an
   hour; the numbers above are all on-demand or fixed-price.)
8. Who answers the AWS quota case, and is 192 vCPUs enough (one node at a time)? Stage 2b and Tier F cannot overlap on AWS under 192.
