"""Fused low-rank branch for SVDQuant checkpoints: ``y + (x @ l2.T) @ l1.T``.

`add_low_rank` pays for the branch in three pieces: two GEMM launches, a
full-size intermediate, and a full-size add kernel. Measured on the 3090 at
4096 tokens, the add alone is ~40-60 ms per step, and cuBLAS runs these
skinny GEMMs (N = rank) at roughly 20-35% of fp16 peak.

This module computes the same fp16 math (fp32 accumulation) with less of
each:

* the add is folded into a Triton GEMM epilogue -- y is read once, the
  result written once, no intermediate tensor, no add kernel;
* the first GEMM picks between cuBLAS and a fixed-config Triton GEMM.

Neither choice is a constant: at 4096 tokens Triton wins the first GEMM for
some shapes and cuBLAS for others, and `torch.addmm` (cuBLASLt's beta path)
measures no better than the separate add. So the first call for a shape
benches its candidates once, caches the winner, and every later call is a
dict lookup. The bench and the one-time Triton compiles land on the first
image, which already pays ~50 s of compilation when a TorchCompileModel is
in the graph; eager pays ~1-2 s there too.

The math is unchanged -- fp16 in, fp32 accumulate, fp16 out, exactly the
rounding the stock path does per GEMM -- so fidelity sits inside the render
noise floor. The usual LPIPS harness still applies before shipping a claim.

Everything degrades to the stock path: no Triton, no CUDA, a non-2D or
non-contiguous input, a dtype/device mismatch, `KREA2_FAST_BRANCH=0`, or the
op registration failing. `attach_branch` keeps `add_low_rank` as the
fallback for exactly those calls.
"""

from __future__ import annotations

import logging
import os
import time

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_IMPORT = True
except ImportError:  # pragma: no cover - triton ships with the cu130 wheel
    triton = None
    tl = None
    _TRITON_IMPORT = False

DISABLED = os.environ.get("KREA2_FAST_BRANCH") == "0"

# shape key (in_features, rank, out_features, dtype) -> (g1, g2) plan, where
# g1 is "cublas" | "triton" and g2 is "triton" | "addmm" | "mm_add".
_PLAN: dict[tuple, tuple[str, str]] = {}

_SM_BUDGET: dict[str, int | None] = {}
_TFITS: dict[tuple, bool] = {}
_OP = None
_OP_ERROR = ""


if _TRITON_IMPORT:

    # OUT = (C +) A @ B^T with A [M, K] row-major and B stored as [N, K]
    # row-major (i.e. the factors l1 [N, R] / l2 [R, K] are read directly,
    # never transposed into a side copy). fp16/bf16 inputs, fp32
    # accumulation, one fixed config: the per-shape bench below decides
    # whether Triton is worth using at all, so a second config would only
    # buy compile time for a choice that is often "cuBLAS".
    @triton.jit
    def _branch_gemm(A, B, C, OUT, M, N, K,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     HAS_C: tl.constexpr):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BM)
        num_pid_n = tl.cdiv(N, BN)
        # Grouped ordering so consecutive pids reuse A's rows in L2.
        num_in_group = 8 * num_pid_n
        group_id = pid // num_in_group
        first_pid_m = group_id * 8
        group_size_m = min(num_pid_m - first_pid_m, 8)
        pid_m = first_pid_m + ((pid % num_in_group) % group_size_m)
        pid_n = (pid % num_in_group) // group_size_m
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        a_ptrs = A + rm[:, None] * K + rk[None, :]
        b_ptrs = B + rk[:, None] + rn[None, :] * K
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            a = tl.load(a_ptrs, mask=rm[:, None] < M, other=0.0)
            b = tl.load(b_ptrs, mask=rn[None, :] < N, other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BK
            b_ptrs += BK
        if HAS_C:
            c = tl.load(C + rm[:, None] * N + rn[None, :],
                        mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)
            acc += c.to(tl.float32)
        tl.store(OUT + rm[:, None] * N + rn[None, :], acc.to(OUT.dtype.element_ty),
                 mask=(rm[:, None] < M) & (rn[None, :] < N))


def _pow2(n: int, lo: int = 16, hi: int = 128) -> int:
    return min(max(triton.next_power_of_2(n), lo), hi)


def _smem_budget(device) -> int | None:
    """Shared memory per Triton block on this device, cached. None = unknown."""
    key = str(device)
    if key not in _SM_BUDGET:
        try:
            _SM_BUDGET[key] = int(
                torch.cuda.get_device_properties(device).shared_memory_per_block_optin)
        except Exception:
            _SM_BUDGET[key] = None
    return _SM_BUDGET[key]


def _triton_usable(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether `_launch`'s fixed tile fits this device's shared memory.

    Tiles grow with rank; at rank 256 the pipeline exceeds the 99 KB per-block
    limit on Ampere and the Triton launch raises OutOfResources mid-bench, which
    kills the prompt. cuBLAS takes over in that case.

    A shared-memory formula is not a reliable way to tell: Triton's own staging
    does not follow it (the rank 256 launch asked for exactly 131072 B where the
    formula said ~426 KB), and an over-estimate at rank 64 -- where Triton wins
    the bench by 1.3-1.7x -- costs real step time. So the formula is only a
    pre-screen; the verdict is a one-time probe launch of the real kernel at a
    tiny token count, cached per tile config.
    """
    if not _TRITON_IMPORT:
        return False
    M, K = a.shape
    N = b.shape[0]
    bn = _pow2(N)
    bk = _pow2(min(K, N)) if K < N else 64
    key = (str(a.device), K, N, a.dtype)
    hit = _TFITS.get(key)
    if hit is not None:
        return hit
    budget = _smem_budget(a.device)
    fits = budget is None or (128 * bk + bk * bn) * 2 * 3 <= budget
    if fits:
        try:
            pm = min(M, 16)
            pa = torch.zeros(pm, K, device=a.device, dtype=a.dtype)
            pc = torch.zeros(pm, N, device=a.device, dtype=a.dtype)
            _launch(pa, b, pc, pc)
        except Exception as exc:
            fits = "OutOfResources" in type(exc).__name__ or \
                "shared memory" in str(exc)
    _TFITS[key] = fits
    return fits


def _launch(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor | None,
            out: torch.Tensor) -> None:  # pragma: no cover - needs _TRITON_IMPORT
    M, K = a.shape
    N = b.shape[0]
    has_c = c is not None
    # BN covers the small N (rank) tightly; BK matches the smaller K.
    bn = _pow2(N)
    bk = _pow2(min(K, N)) if K < N else 64
    num_m = (M + 127) // 128
    num_n = (N + bn - 1) // bn
    grid = (num_m * num_n,)
    _branch_gemm[grid](a, b, c if has_c else a, out, M, N, K,
                       BM=128, BN=bn, BK=bk, HAS_C=has_c,
                       num_warps=8, num_stages=3)


def _triton_gemm1(x, l2, h):
    _launch(x, l2, None, h)


def _triton_gemm2(h, l1, y, out):
    _launch(h, l1, y, out)


def _time_ms(fn, iters: int = 4) -> float:
    fn()  # warm, then time
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def _pick_timed(cands: list, rounds: int = 5) -> tuple[str, dict[str, float]]:
    """The fastest of `[(name, fn), ...]`, measured round-robin, with medians.

    Round-robin rather than back-to-back: a one-shot sweep times the first
    candidate cold and the last one warm, and at these magnitudes (~0.3-0.8 ms
    per call) that ordering noise is as large as the real differences -- which
    is how a single noisy run picked a plan that then measured 0.63x. The
    median over rounds keeps an outlier round from deciding.
    """
    stats: dict[str, list[float]] = {name: [] for name, _ in cands}
    for _ in range(rounds):
        for name, fn in cands:
            stats[name].append(_time_ms(fn))
    medians = {name: sorted(t)[len(t) // 2] for name, t in stats.items()}
    return min(medians, key=medians.get), medians


def _bench_plan(x, y, l1, l2) -> tuple[tuple[str, str], dict]:
    """One-time per shape: bench the candidates and keep the winner, with the times."""
    M, K = x.shape
    N, R = l1.shape
    h = torch.empty(M, R, device=x.device, dtype=x.dtype)
    y0 = y.clone()
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    tmp = torch.empty(M, N, device=x.device, dtype=x.dtype)

    g1_cands = [
        ("cublas", lambda: torch.mm(x, l2.t(), out=h)),
    ]
    if _triton_usable(x, l2):
        # first call compiles; the rounds below time it warm
        _triton_gemm1(x, l2, h)
        g1_cands.append(("triton", lambda: _triton_gemm1(x, l2, h)))
    # g2's probe may compile its kernel variant here, before any timing round.
    _triton_usable(h, l1)
    g1, g1_times = _pick_timed(g1_cands)

    def mm_add():
        torch.mm(h, l1.t(), out=tmp)
        return tmp.add(y0)

    g2_cands = [
        ("addmm", lambda: torch.addmm(y0, h, l1.t())),
        ("mm_add", mm_add),
    ]
    if _triton_usable(h, l1):
        g2_cands.insert(0, ("triton", lambda: _triton_gemm2(h, l1, y0, out)))
    g2, g2_times = _pick_timed(g2_cands)
    return (g1, g2), {"g1": g1_times, "g2": g2_times}


def _branch_impl(x, y, l1, l2) -> torch.Tensor:
    """`y + (x @ l2.T) @ l1.T`; x [..., K], y [..., N], l1 [N, R], l2 [R, K].

    The GEMM is over the last dimension. Krea 2's blocks feed the linears 3D
    [B, S, C] tensors (FLUX-style, never flattened), so the leading dims are
    collapsed to a token count here -- a free view for the contiguous inputs
    the caller requires -- and the result is shaped back to y's.
    """
    K = x.shape[-1]
    N, R = l1.shape
    m2 = x.reshape(-1, K)
    y2 = y.reshape(-1, N)
    key = (K, R, N, x.dtype)
    plan = _PLAN.get(key)
    if plan is None:
        # Logged rather than just cached: if the fast path ever turns out not
        # to help end to end, these lines are the only record of which plan
        # was chosen and what the alternatives measured on this machine.
        plan, times = _bench_plan(m2, y2, l1, l2)
        _PLAN[key] = plan
        g1t, g2t = times["g1"], times["g2"]
        logging.info("[krea2-svdquant] fast branch in%d r%d out%d: plan %s/%s "
                     "(g1 triton %.3f / cublas %.3f ms; g2 triton %.3f / addmm %.3f "
                     "/ mm_add %.3f ms)",
                     K, R, N, plan[0], plan[1],
                     g1t.get("triton", -1), g1t.get("cublas", -1),
                     g2t.get("triton", -1), g2t["addmm"], g2t["mm_add"])
    g1, g2 = plan

    M = m2.shape[0]
    h = torch.empty(M, R, device=x.device, dtype=x.dtype)
    if g1 == "triton" and _triton_usable(m2, l2):
        _triton_gemm1(m2, l2, h)
    else:
        torch.mm(m2, l2.t(), out=h)
    if g2 == "triton" and _triton_usable(h, l1):
        out = torch.empty(M, N, device=x.device, dtype=x.dtype)
        _triton_gemm2(h, l1, y2, out)
        return out.reshape(y.shape)
    if g2 == "addmm":
        return torch.addmm(y2, h, l1.t()).reshape(y.shape)
    return (y2 + torch.mm(h, l1.t())).reshape(y.shape)


def _get_op():
    """The registered `krea2::branch_linear` op, or None.

    Registered under `torch.library` so a compiled step emits one node for the
    whole branch instead of tracing the Triton dispatch -- the same mechanism
    as `krea2::w4a4_linear` in the loader.
    """
    global _OP, _OP_ERROR
    if _OP is not None or _OP_ERROR:
        return _OP
    try:
        @torch.library.custom_op("krea2::branch_linear", mutates_args=())
        def branch_linear(x: torch.Tensor, y: torch.Tensor, l1: torch.Tensor,
                          l2: torch.Tensor) -> torch.Tensor:
            return _branch_impl(x, y, l1, l2)

        @branch_linear.register_fake
        def _(x, y, l1, l2):
            return x.new_empty(y.shape)

        _OP = branch_linear
    except Exception as exc:
        _OP_ERROR = "{}: {}".format(type(exc).__name__, exc)
    return _OP


def is_available(device) -> bool:
    return (not DISABLED and _TRITON_IMPORT and device.type == "cuda"
            and _get_op() is not None)


def make_applier():
    """The per-branch call `apply(x, y, l1, l2)`, or None to stay on `add_low_rank`.

    Eager calls skip the `torch.library` dispatch (it exists for Dynamo, not
    for speed); compiled calls go through the op so the branch stays in the
    graph. Per-call validity (device/dtype/layout) is checked by the caller.
    """
    if not is_available(torch.device("cuda")):
        return None
    op = _get_op()

    def apply(x, y, l1, l2):
        if torch.compiler.is_compiling():
            return op(x, y, l1, l2)
        return _branch_impl(x, y, l1, l2)

    return apply


def status() -> str:
    if DISABLED:
        return "fast branch: off (KREA2_FAST_BRANCH=0)"
    if not _TRITON_IMPORT:
        return "fast branch: off (no triton)"
    if _get_op() is None:
        return "fast branch: off ({})".format(_OP_ERROR)
    if not _PLAN:
        return "fast branch: on (no shapes seen yet)"
    desc = ", ".join("{}x{}x{}: {}/{}".format(k[2], k[1], k[0], g1, g2)
                     for k, (g1, g2) in sorted(_PLAN.items()))
    return "fast branch: on ({})".format(desc)
