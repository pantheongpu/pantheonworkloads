"""vLLM greedy generation of a few tokens: the functional workload's Python half.

Writes the contract's JSON object to the file named by argv[1] (vLLM and its libraries print on
stdout too, so run.sh prints the file last). Exits 77 with a reason when this machine cannot run it.
The engine settings are pantheonsim's (amd/tests/e2e/run_vllm_amd.sh): eager mode (no CUDA graphs,
no torch.compile), one short sequence, fp16, so a simulated GPU can run it in minutes.
"""
import json
import os
import sys


def skip(why):
    print(f"SKIP: {why}")
    sys.exit(77)


out_file = sys.argv[1]
model = os.environ.get("PW_MODEL", "HuggingFaceTB/SmolLM2-135M")
revision = os.environ.get("PW_MODEL_REVISION") or None
prompt = os.environ.get("PW_PROMPT", "The capital of France is")
sim = os.environ.get("PW_TARGET", "").startswith("sim:")

try:
    import torch
except ImportError as e:
    skip(f"torch is not installed ({e})")
try:
    import vllm
    from vllm import LLM, SamplingParams
except ImportError as e:
    skip(f"vLLM is not installed ({e})")
if not torch.cuda.is_available():
    skip(f"torch {torch.__version__} sees no CUDA/ROCm GPU: vLLM has no CPU path here")

util = float(os.environ.get("PW_GPU_MEM_UTIL", "0.5" if sim else "0.3"))
llm = LLM(model=model, revision=revision, enforce_eager=True, max_model_len=128, max_num_seqs=1,
          max_num_batched_tokens=128, gpu_memory_utilization=util, dtype="float16", seed=0)
res = llm.generate([prompt], SamplingParams(max_tokens=8, temperature=0))[0].outputs[0]
with open(out_file, "w") as f:
    json.dump({"output": res.text,
               "detail": f"{model}@{revision or 'unpinned'}, vllm {vllm.__version__}, torch {torch.__version__}, "
                         f"token ids {list(res.token_ids)}",
               "metrics": {}}, f)
