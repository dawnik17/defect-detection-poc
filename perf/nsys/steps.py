"""Break an Nsight Systems trace of a vLLM server into engine steps, and show where their GPU time goes.

    python perf/nsys/steps.py runs/perf/<run>/nsys/server.sqlite

vLLM labels each engine step with an NVTX range such as execute_context_1(733)_generation_3(3): one 733-token
prefill batched with three decodes. A kernel belongs to the step that launched it; under load the GPU runs behind the
CPU, so it may execute during a later step. For prefill and decode steps this prints the step's CPU wall time (its
NVTX range), GPU busy time, kernel count, host-sync wait and kernel time by type. A prefill step with images runs the
vision encoder first, then the language model: kernels launched before LLM_START (below) count as vision.
"""
import bisect
import re
import sqlite3
import statistics as st
import sys
from collections import defaultdict

STEP = re.compile(r":?execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")
LLM_START = re.compile(r"reshape_and_cache|causal_conv1d")

KERNEL_TYPES = [
    ("attention (Triton unified)", r"kernel_unified_attention"),
    ("attention (FlashInfer TRT-LLM)", r"fmhaSm"),
    ("attention (FlashAttention)", r"flash_fwd|flash::"),
    ("attention (cuDNN)", r"cudnn.*sdpa"),
    ("attention (PyTorch SDPA)", r"fmha|attention_kernel|efficient_attention|cutlass.*Attention"),
    ("MoE experts", r"moe|^bmm_"),
    ("Mamba (conv1d, SSM scan)", r"causal_conv1d|selective_scan|_chunk_|state_passing|mamba"),
    ("GEMM (cuBLAS/CUTLASS)", r"gemm|nvjet|s16816|s1688|cutlass|xmma|ampere_|sm80_|cublas"),
    ("fused (Inductor triton_*)", r"^triton_"),
    ("Triton other", r"^_|triton"),
    ("norm / softmax / reduce", r"norm|softmax|reduce|Reduce"),
    ("index / gather / scatter / select", r"index|gather|scatter|Select|select|nonzero|masked|one_hot|embedding"),
    ("copy / cat / fill", r"[Cc]opy|cat|Cat|fill|Fill|memset"),
    ("elementwise (ATen)", r"elementwise|Elementwise|gelu|Gelu|tanh|mul|add|pow"),
]


def kernel_type(name):
    return next((label for label, pattern in KERNEL_TYPES if re.search(pattern, name)), "other")


def covered(intervals):
    """Time covered by (start, end) intervals, counting overlaps once."""
    total, cur_start, cur_end = 0, None, None
    for start, end in sorted(intervals):
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    return total + (cur_end - cur_start if cur_end is not None else 0)


def load(path):
    """Engine steps (start, end, kind, prefill requests), kernels (launch, start, end, name), host syncs; in ns."""
    db = sqlite3.connect(path)
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    steps = []
    for start, end, text, text_id in db.execute(
            "SELECT start, end, text, textId FROM NVTX_EVENTS WHERE end IS NOT NULL"):
        m = STEP.match((text if text is not None else strings.get(text_id)) or "")
        if m:
            prefills, decodes = int(m.group(1)), int(m.group(3))
            steps.append((start, end, "prefill" if prefills else "decode" if decodes else "empty", prefills))
    # Each kernel joins its launch call (cudaLaunchKernel, cudaGraphLaunch, ...) by correlation id within a process;
    # globalTid and globalPid carry the process id above bit 24.
    kernels = [(launch, s, e, strings.get(n, str(n))) for launch, s, e, n in db.execute(
        "SELECT r.start, k.start, k.end, coalesce(k.demangledName, k.shortName) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId AND r.globalTid >> 24 = k.globalPid >> 24 "
        "ORDER BY r.start, k.start")]
    syncs = db.execute(
        "SELECT r.start, r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId = s.id "
        "WHERE s.value LIKE 'cudaStreamSynchronize%' OR s.value LIKE 'cudaDeviceSynchronize%'").fetchall()
    return sorted(steps), kernels, syncs


def report(kind, steps, kernels, syncs):
    """Print medians over the steps of one kind (prefill or decode), and their kernel time by type."""
    starts, launches = [s for s, *_ in steps], [k[0] for k in kernels]
    walls, busys, counts, sync_waits, prefills_per_step = [], [], [], [], []
    vision_times, vision_busys, llm_times = [], [], []
    time_by_type, vision_by_type = defaultdict(float), defaultdict(float)
    for start, end, step_kind, prefills in steps:
        if step_kind != kind:
            continue
        # The step's kernels: those launched from its start until the next step starts.
        later = bisect.bisect_right(starts, start)
        next_start = starts[later] if later < len(starts) else end + 50_000_000
        ks = kernels[bisect.bisect_left(launches, start):bisect.bisect_left(launches, next_start)]
        if not ks:
            continue
        walls.append((end - start) / 1e6)
        busys.append(covered([(s, e) for _, s, e, _ in ks]) / 1e6)
        counts.append(len(ks))
        prefills_per_step.append(prefills)
        sync_waits.append(sum(min(e, end) - max(s, start) for s, e in syncs if e > start and s < end) / 1e6)
        for _, s, e, name in ks:
            time_by_type[kernel_type(name)] += (e - s) / 1e6

        # Split a prefill step at the language model's first kernel: vision encoder before it, language model after.
        first_llm = next((i for i, k in enumerate(ks) if LLM_START.search(k[3])), None)
        if kind == "prefill" and first_llm:  # None or 0: no vision encoder ran in this step
            gpu_start, llm_start = min(s for _, s, _, _ in ks), ks[first_llm][1]
            vision_times.append((llm_start - gpu_start) / 1e6)
            vision_busys.append(covered([(s, e) for _, s, e, _ in ks[:first_llm]]) / 1e6)
            llm_times.append((max(e for _, _, e, _ in ks) - llm_start) / 1e6)
            for _, s, e, name in ks[:first_llm]:
                vision_by_type[kernel_type(name)] += (e - s) / 1e6
    if not walls:
        return

    n = len(walls)
    requests = f" with {st.mean(prefills_per_step):.1f} prefill requests on average" if kind == "prefill" else ""
    print(f"\n{kind}: {n} steps{requests}, CPU wall median {st.median(walls):.1f} ms, GPU busy median "
          f"{st.median(busys):.1f} ms, kernels per step median {st.median(counts):.0f}, host-sync wait median "
          f"{st.median(sync_waits):.1f} ms")
    if vision_times:
        print(f"  vision encoder: GPU time median {st.median(vision_times):.1f} ms, busy {st.median(vision_busys):.1f} ms; "
              f"then the LLM: GPU time median {st.median(llm_times):.1f} ms ({len(vision_times)} steps with images)")
    total = sum(time_by_type.values())
    print("  kernel time per step by type (ms, mean):" + ("   [vision encoder share]" if vision_by_type else ""))
    for label, t in sorted(time_by_type.items(), key=lambda x: -x[1]):
        share = f"   [{vision_by_type[label] / n:6.2f}]" if vision_by_type else ""
        print(f"    {label:36s} {t / n:7.2f}  {t / total:5.1%}{share}")


def main(path):
    steps, kernels, syncs = load(path)
    if not steps or not kernels:
        sys.exit("No vLLM step ranges or kernels in this capture")
    span_start, span_end = steps[0][0], max(end for _, end, *_ in steps)
    busy = covered([(max(s, span_start), min(e, span_end)) for _, s, e, _ in kernels if e > span_start and s < span_end])
    print(f"capture: {(span_end - span_start) / 1e9:.2f} s of engine steps, GPU busy {busy / (span_end - span_start):.0%}")
    for kind in ("prefill", "decode"):
        report(kind, steps, kernels, syncs)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
