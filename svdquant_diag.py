"""Diagnostics for Krea2 SVDQuant / W4A4 checkpoints.

Answers the questions a performance or OOM report needs answered before anything
else can be said:

* which acceleration path each quantized layer's matmul runs on -- W4A4 on the
  int4 tensor cores, W4A8 and stock INT8 on the int8 tensor cores, Triton as
  a software emulation, and plain bf16 matmul (eager) as the unpack-and-matmul
  fallback that is *slower* than fp8;

The first one matters more than it looks: ComfyUI disables comfy_kitchen's CUDA backend
outright when torch was built against CUDA < 13 (``comfy/quant_ops.py``), which silently
turns every quantized checkpoint into the eager path. That one fact explains most
"quantized is slower than fp8" reports and is invisible without this node.

Everything here is read-only apart from ``bench``, which needs the weights resident and
will ask ComfyUI to load the model the same way sampling would.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time

import torch
import torch.nn.functional as F

import comfy.model_management as mm
# Importing this is what applies ComfyUI's own backend gating -- notably the
# `ck.registry.disable("cuda")` it performs when torch was built against CUDA < 13.
# Without it we would query a registry ComfyUI has not finished configuring and happily
# report a CUDA backend that never runs in practice.
_QUANT_OPS_ERROR = None
try:
    import comfy.quant_ops  # noqa: F401
except Exception as exc:  # pragma: no cover - depends on the ComfyUI build
    # Guarded so a build without quant_ops still gets *this* module, and therefore still gets
    # Env Check -- the one node whose entire job is explaining a missing quantization backend.
    # Unguarded, the ImportError would propagate through __init__.py and remove all seven.
    _QUANT_OPS_ERROR = "{}: {}".format(type(exc).__name__, exc)

# Buffer names for the low-rank factors, chosen to match the on-disk keys exactly: a buffer
# called ``svdq_l1`` on ``blocks.0.attn.wq`` produces the state-dict key
# ``blocks.0.attn.wq.svdq_l1``, byte-identical to what quantize_krea2.py writes. So a model
# re-saved out of ComfyUI round-trips straight back into the loader. Defined here because
# svdquant_w4a4 imports this module (not the other way round) and both need them.
BUF_L1 = "svdq_l1"
BUF_L2 = "svdq_l2"

_CK_IMPORT_ERROR = None
try:
    from comfy_kitchen.registry import registry as ck_registry
    from comfy_kitchen.tensor.convrot_w4a4 import TensorCoreConvRotW4A4Layout
    from comfy_kitchen.tensor.w4a8_int8 import AsymW4A8Int8Layout
except Exception as exc:  # pragma: no cover - depends on the install
    ck_registry = None
    TensorCoreConvRotW4A4Layout = None
    AsymW4A8Int8Layout = None
    _CK_IMPORT_ERROR = "{}: {}".format(type(exc).__name__, exc)

# 1024x1024 with Krea2's patch size lands on (1024/16)^2 tokens. Sampling shape drives
# every shape-dependent kernel decision (see `_convrot_int4_fused_shared_memory_fits` in
# comfy_kitchen), so probing at the wrong token count can report the wrong path.
_DEFAULT_TOKENS = (1024 // 16) * (1024 // 16)

_FUNC = "convrot_w4a4_linear"
_W4A8_FUNC = "w4a8_int8_linear"
_INT8_FUNC = "int8_tensorwise"
# Human-readable names for the acceleration paths the matmuls run on. These are
# what the loader, the env check, and the diagnostics reports print: plain
# English first, the machine name kept in parentheses for bug reports.
# The mapping matches the installed comfy_kitchen registry: W4A4 matmuls run on
# the int4 tensor cores, W4A8 matmuls on the int8 tensor cores (the 4-bit weight
# is expanded to int8 for the MMA), and the Triton backend implements the same
# matmul in software. Stock int8_tensorwise checkpoints are not in the registry
# at all -- ComfyUI routes them through torch._int_mm, which needs
# supports_int8_compute().
_PATH_NAMES = {
    (_FUNC, "cuda"): "INT4 cores (convrot_w4a4)",
    (_FUNC, "eager"): "plain bf16 matmul (eager)",
    (_FUNC, "triton"): "Triton emulation (triton)",
    (_W4A8_FUNC, "cuda"): "INT8 cores (w4a8_int8)",
    (_W4A8_FUNC, "eager"): "plain bf16 matmul (eager)",
    (_W4A8_FUNC, "triton"): "Triton emulation (triton)",
}
_INT8_PATH = "INT8 cores (int8_tensorwise)"


def _known_kitchen_op(func: str | None) -> bool:
    return func in (_FUNC, _W4A8_FUNC)


def acceleration_path(func: str | None, backend: str | None) -> str:
    """The plain-English name of the matmul path that will run."""
    if func == _INT8_FUNC:
        return _INT8_PATH if mm.supports_int8_compute() else "plain bf16 matmul"
    if not _known_kitchen_op(func):
        return "no acceleration path"
    return _PATH_NAMES.get((func, backend or ""), "no acceleration path")


# One category for every node in this pack. Under `advanced/loaders` they were scattered
# among ComfyUI's own dozen-plus loaders; class names are what saved workflows match on, so
# moving them here does not break existing graphs.
_CATEGORY = "Krea2/SVDQuant"


def _is_convrot_quantized(weight) -> bool:
    """True for a QuantizedTensor carrying a convrot layout (W4A4 or W4A8).

    The two layouts share ``convrot_groupsize`` and differ in the rest: W4A4's Params
    record ``linear_dtype``, W4A8's record ``group_size``. int8-convrot carries a
    ``convrot`` flag instead of either, so it is not matched here.
    """
    params = getattr(weight, "_params", None)
    if params is None:
        return False
    return hasattr(params, "convrot_groupsize") and (
        hasattr(params, "linear_dtype") or hasattr(params, "group_size"))


def quantized_linears(diffusion_model):
    """(name, module) for every Linear whose weight is a convrot-quantized QuantizedTensor."""
    for name, module in diffusion_model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is not None and _is_convrot_quantized(weight):
            yield name, module


def branch_buffers(module):
    """(name, tensor) for every low-rank factor buffer on a module, for summing bytes.

    Use `branch_factors` when you need l1 and l2 individually -- these come back in
    registration order, which is not something to index positionally.
    """
    for name, buf in module.named_buffers(recurse=False):
        if buf is not None and name.startswith("svdq_l"):
            yield name, buf


def branch_factors(module):
    """The (l1, l2) pair attached to a module, or None if it has no branch.

    Keyed by name deliberately. Reading `list(branch_buffers(module))[0]` as l1 happens to
    work today only because the loader registers l1 first; a third factor or a reordering
    would silently swap the two and every derived number would look plausible and be wrong.
    """
    bufs = getattr(module, "_buffers", {})
    l1, l2 = bufs.get(BUF_L1), bufs.get(BUF_L2)
    return None if l1 is None or l2 is None else (l1, l2)


def _probe_op(module) -> str | None:
    """The comfy_kitchen op this module's matmul dispatches to, by layout."""
    params = getattr(module.weight, "_params", None)
    if params is None:
        return None
    if hasattr(params, "linear_dtype"):  # W4A4
        return _FUNC
    if hasattr(params, "group_size"):  # W4A8
        return _W4A8_FUNC
    return None


def _probe_kwargs(module, tokens: int, device) -> dict:
    """Kwargs shaped exactly like a real quantized-linear call, for validation.

    The activation and the weight tensors are `torch.empty` rather than copies: the
    registry validates dtype, ndim, device and divisibility, never contents, and a real
    copy of a 6144x16384 packed weight is 50 MB we would rather not move on the card
    that is already OOMing.
    """
    params = module.weight._params
    x_dtype = params.orig_dtype if params.orig_dtype in (
        torch.float32, torch.float16, torch.bfloat16) else torch.bfloat16
    in_features = int(params.orig_shape[1])
    if hasattr(params, "linear_dtype"):  # W4A4
        qweight, wscales = TensorCoreConvRotW4A4Layout.get_plain_tensors(module.weight)
        return {
            "x": torch.empty((tokens, in_features), dtype=x_dtype, device=device),
            "qweight": torch.empty(tuple(qweight.shape), dtype=qweight.dtype, device=device),
            "wscales": torch.empty(tuple(wscales.shape), dtype=wscales.dtype, device=device),
            "bias": None,
            "convrot_groupsize": params.convrot_groupsize,
            "quant_group_size": params.quant_group_size,
            "linear_dtype": params.linear_dtype,
        }
    # W4A8
    qdata, s_rel, s_channel, correction, codebook = \
        AsymW4A8Int8Layout.get_plain_tensors(module.weight)

    def _empty(t):
        return None if t is None else torch.empty(tuple(t.shape), dtype=t.dtype, device=device)

    return {
        "x": torch.empty((tokens, in_features), dtype=x_dtype, device=device),
        "qdata": _empty(qdata),
        "s_rel": _empty(s_rel),
        "s_channel": _empty(s_channel),
        "codebook": _empty(codebook),
        "correction": _empty(correction),
        "bias": None,
        "group_size": params.group_size,
        "convrot_groupsize": params.convrot_groupsize,
        "out_dtype": x_dtype,
    }


def resolve_dispatch(module, tokens: int = _DEFAULT_TOKENS, device=None):
    """Which backend would run this layer? -> (backend, impl_path, failures).

    `failures` maps backend name to why it was skipped, so "ComfyUI disabled cuda" and
    "this shape violates a kernel constraint" do not look alike in a bug report.
    """
    if ck_registry is None:
        return None, None, {"__import__": _CK_IMPORT_ERROR}

    func = _probe_op(module)
    if func is None:
        return None, None, {"__probe__": "unsupported quantized layout"}

    device = device or mm.get_torch_device()
    try:
        kwargs = _probe_kwargs(module, tokens, device)
    except Exception as exc:
        return None, None, {"__probe__": "{}: {}".format(type(exc).__name__, exc)}

    failures = {}
    for name in ("cuda", "triton", "eager"):
        result = ck_registry.validate_backend_for_call(name, func, kwargs)
        if not result.success:
            failures[name] = "{}: {}".format(result.failed_param, result.failure_reason)

    try:
        backend = ck_registry.get_capable_backend(func, kwargs)
        impl = ck_registry.get_implementation(func, kwargs=kwargs)
        impl_path = "{}.{}".format(impl.__module__, impl.__qualname__)
    except Exception as exc:
        return None, None, failures or {"__dispatch__": str(exc)}
    finally:
        del kwargs

    return backend, impl_path, failures


def dispatch_warning(backend: str | None, failures: dict, func: str = _FUNC) -> str | None:
    """The message the loader prints when the fast kernel is not in play.

    Deliberately not caveman-terse and not abbreviated: this text ends up pasted into
    issue threads by people who have not read the README.
    """
    if backend == "cuda":
        return None

    cuda_reason = (failures or {}).get("cuda", "unknown")
    path = acceleration_path(func, backend)
    lines = [
        "[krea2-svdquant] WARNING: this checkpoint will run on {}, not the "
        "tensor-core path.".format(path),
        "  The fast tensor-core path is not in use, so this checkpoint runs "
        "slower than it should.",
        "  cuda backend was rejected because: {}".format(cuda_reason),
    ]
    if backend == "eager":
        lines.append(
            "  The eager path unpacks the quantized weights back to bf16 and runs "
            "an ordinary matmul. It is slower than fp8, and can be slower than "
            "plain bf16.")
    elif backend == "triton":
        lines.append(
            "  The Triton path runs a software emulation of the matmul on the GPU. "
            "It has no dedicated tensor-core path.")

    cuda_build = torch.version.cuda
    if cuda_build is None or tuple(int(p) for p in str(cuda_build).split(".")[:1]) < (13,):
        lines.append(
            "  Most likely cause: ComfyUI disables comfy_kitchen's CUDA backend when torch "
            "is built against CUDA < 13 (see comfy/quant_ops.py). You have torch {} "
            "(cuda {}). Install a cu130 or newer torch build.".format(
                torch.__version__, cuda_build)
        )
    lines.append("  Run the 'Krea2 SVDQuant Diagnostics' node (mode=dispatch) for detail.")
    return "\n".join(lines)


def _capability(device=None) -> tuple[int, int] | None:
    """(major, minor) compute capability of the device we would sample on, or None."""
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_capability(device or mm.get_torch_device())
    except Exception:
        return None


def architecture_note(device=None) -> str | None:
    """The speed caveat that applies even when the cuda backend *is* live.

    Turing (SM 7.5) is the case that generates support traffic. The backend validates, the
    kernel runs, nothing looks wrong -- and the checkpoint is barely faster than int8, so
    people conclude their install is broken and spend an evening trying to compile
    comfy_kitchen by hand. It is not broken and compiling changes nothing:

    * The int4 tensor-core instruction the Ampere+ path is built around (`mma.m16n8k64`,
      s4*s4->s32) does not exist on SM 7.5. comfy_kitchen's default convrot path targets
      Sm89 and gates that instruction behind `__CUDA_ARCH__ >= 800`.
    * SM 7.5 is served instead by separate kernels (`turing_int4.cu`, `turing_int8.cu`)
      built on a smaller MMA instruction tile -- `GemmShape<8, 8, 32>` against the default
      int8 path's `GemmShape<16, 8, 32>` -- and without `cp.async` prefetch, which is also
      SM80+. Both formats lose, so switching to int8 does not dodge it either.

    We have no Turing hardware to measure on, so this says "expect little or no speedup"
    rather than quoting a ratio; community reports on 2060/2070/2080 Ti are consistent with
    roughly parity against int8. The value here is telling someone their setup is fine.
    """
    cap = _capability(device)
    if cap is None or cap[0] != 7:
        return None
    return (
        "[krea2-svdquant] NOTE: this GPU is compute capability {}.{} (Turing). The int4 MMA "
        "instruction the fast path is built around (mma.m16n8k64) is Ampere and newer, so "
        "comfy_kitchen falls back to its separate SM75 kernels, which use a smaller "
        "instruction tile and no cp.async prefetch.\n"
        "  Expect little or no speedup from int4 over int8 on this card. That is a hardware "
        "and upstream-kernel limitation, not a misconfiguration -- rebuilding comfy_kitchen "
        "will not change it. The int4 checkpoint is still worth it for the smaller VRAM "
        "footprint.".format(cap[0], cap[1])
    )


def log_dispatch(diffusion_model) -> str:
    """One line on every load, plus a loud block when the fast kernel is not available.

    Returns the same text it logs so the loader node can show it in the UI; returns an empty
    string when there is nothing to say.
    """
    try:
        first = next(iter(quantized_linears(diffusion_model)), None)
        if first is None:
            return ""
        func = _probe_op(first[1])
        if func is None:
            return ""
        backend, impl_path, failures = resolve_dispatch(first[1])
        if backend is None:
            logging.warning("[krea2-svdquant] could not resolve a %s backend: %s",
                            func, failures)
            return "could not resolve a {} backend: {}".format(func, failures)
        path = acceleration_path(func, backend)
        logging.info("[krea2-svdquant] Using %s", path)
        warning = dispatch_warning(backend, failures, func)
        if warning:
            logging.warning("%s", warning)
        # Reported even when the backend is 'cuda': on Turing the int4 kernel is live and
        # still slow, which is exactly the case `dispatch_warning` stays silent about. It
        # is a W4A4-specific note -- W4A8 runs on the int8 path, which Turing serves at
        # full rate.
        arch = architecture_note() if func == _FUNC else None
        if arch:
            logging.warning("%s", arch)
        return "\n".join(x for x in (
            "Using {} ({})".format(path, impl_path),
            warning, arch) if x)
    except Exception:
        # A diagnostic must never be the reason a model fails to load.
        logging.debug("[krea2-svdquant] dispatch probe failed", exc_info=True)
        return ""


def report_backend_status() -> str:
    """The part of the dispatch story that needs no model loaded.

    Answers "is the int4 tensor-core kernel even available on this install?" -- worth being
    able to ask *before* downloading an 8 GB checkpoint, which is why both `diagnose.py
    --no-load` and the env-check node come through here.
    """
    lines = ["torch {}  (cuda build {})".format(torch.__version__, torch.version.cuda), ""]
    if _QUANT_OPS_ERROR:
        lines.append("comfy.quant_ops failed to import: {}".format(_QUANT_OPS_ERROR))
        lines.append("No quantized checkpoint can load on this build, and this repo's "
                     "quantizer has no formats available. Update ComfyUI.")
        lines.append("")
    if ck_registry is None:
        lines.append("comfy_kitchen unavailable: {}".format(_CK_IMPORT_ERROR))
        return "\n".join(lines)

def acceleration_paths_table() -> list[str]:
    """Plain-English status for each acceleration path.

    Shared by the env check (no model loaded) and the diagnostics dispatch report
    (model loaded), so both say the same thing for the same machine. The labels
    mirror `_PATH_NAMES` -- the same names the loader prints on every load.
    """
    if ck_registry is None:
        return []
    backends = ck_registry.list_backends()

    def _live(backend: str, funcs: tuple[str, ...]) -> bool:
        info = backends.get(backend)
        if info is None:
            return False
        return info["available"] and not info["disabled"] and any(f in info["capabilities"] for f in funcs)

    rows = [
        ("INT4 cores (convrot_w4a4)",
         "live" if _live("cuda", (_FUNC,)) else "off",
         "W4A4 / SVDQuant",
         "the matmul runs on the int4 tensor cores"),
        ("INT8 cores (w4a8_int8)",
         "live" if _live("cuda", (_W4A8_FUNC,)) else "off",
         "W4A8 / SVDQuant8",
         "the matmul runs on the int8 tensor cores"),
        ("INT8 cores (int8_tensorwise)",
         "live" if mm.supports_int8_compute() else "off",
         "stock INT8",
         "the matmul runs on the int8 tensor cores (torch._int_mm)"),
        ("Triton emulation (triton)",
         "live" if _live("triton", (_FUNC, _W4A8_FUNC)) else "off",
         "W4A4 / W4A8",
         "the matmul runs in software, no tensor cores"),
        ("plain bf16 matmul (eager)", "always available", "all",
         "weights are unpacked to bf16, then an ordinary matmul"),
    ]
    lines = ["{:24} {:15} {:17} what it does".format("path", "status", "applies to")]
    for path, status, applies, what in rows:
        lines.append("{:24} {:15} {:17} {}".format(path, status, applies, what))
    return lines


def report_backend_status() -> str:
    """The part of the dispatch story that needs no model loaded.

    Answers "is the int4 tensor-core kernel even available on this install?" -- worth being
    able to ask *before* downloading an 8 GB checkpoint, which is why both `diagnose.py
    --no-load` and the env-check node come through here.
    """
    lines = ["torch {}  (cuda build {})".format(torch.__version__, torch.version.cuda), ""]
    if _QUANT_OPS_ERROR:
        lines.append("comfy.quant_ops failed to import: {}".format(_QUANT_OPS_ERROR))
        lines.append("No quantized checkpoint can load on this build, and this repo's "
                     "quantizer has no formats available. Update ComfyUI.")
        lines.append("")
    if ck_registry is None:
        lines.append("comfy_kitchen unavailable: {}".format(_CK_IMPORT_ERROR))
        return "\n".join(lines)

    lines.extend(acceleration_paths_table())

    backends = ck_registry.list_backends()
    active = [n for n, i in backends.items()
             if i["available"] and not i["disabled"]
             and (_FUNC in i["capabilities"] or _W4A8_FUNC in i["capabilities"])]
    lines.append("")
    if "cuda" in active:
        lines.append("The tensor-core paths are live. W4A4 runs on INT4 cores, "
                     "W4A8 and stock INT8 run on INT8 cores.")
        # "Available" is not "fast" on pre-Ampere for the int4 path, and this report is
        # the one people read before downloading 8 GB.
        arch = architecture_note()
        if arch:
            lines.append("")
            lines.append(arch)
    else:
        lines.append(
            "The tensor-core paths are NOT live. Quantized layers will fall back to "
            "{} -- expect the checkpoint to be slower than fp8.".format(active or "nothing"))
        cuda_build = torch.version.cuda
        if cuda_build is None or int(str(cuda_build).split(".")[0]) < 13:
            lines.append(
                "ComfyUI disables comfy_kitchen's CUDA backend on torch built against "
                "CUDA < 13 (comfy/quant_ops.py). Install a cu130 or newer torch build.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- reports


def _comfyui_version() -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    try:
        out = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"],
                             cwd=root, capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    try:
        import comfyui_version
        return getattr(comfyui_version, "__version__", "unknown")
    except Exception:
        return "unknown"


def report_env(patcher) -> list[str]:
    device = mm.get_torch_device()
    lines = ["== environment =="]
    lines.append("torch                : {}  (cuda build {})".format(
        torch.__version__, torch.version.cuda))
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(device)
        lines.append("device               : {}  (sm_{}{})".format(
            torch.cuda.get_device_name(device), major, minor))
    else:
        lines.append("device               : CUDA not available")
    lines.append("ComfyUI              : {}".format(_comfyui_version()))
    lines.append("comfy_kitchen        : {}".format(
        _CK_IMPORT_ERROR or _package_version("comfy-kitchen")))

    lines.append("")
    lines.append("== memory ==")
    lines.append("vram_state           : {}".format(mm.vram_state))
    lines.append("free / total         : {:.2f} / {:.2f} GiB".format(
        mm.get_free_memory(device) / 1024 ** 3, mm.get_total_memory(device) / 1024 ** 3))
    lines.append("max pinned           : {:.2f} GiB (currently {:.2f} GiB)".format(
        max(0, getattr(mm, "MAX_PINNED_MEMORY", 0)) / 1024 ** 3,
        getattr(mm, "TOTAL_PINNED_MEMORY", 0) / 1024 ** 3))

    lines.append("")
    lines.append("== patcher ==")
    lines.append("class                : {}".format(type(patcher).__name__))
    try:
        lines.append("is_dynamic           : {}".format(patcher.is_dynamic()))
    except Exception:
        lines.append("is_dynamic           : n/a")
    lines.append("model_size           : {:.3f} GiB".format(patcher.model_size() / 1024 ** 3))
    lines.append("loaded_size          : {:.3f} GiB".format(patcher.loaded_size() / 1024 ** 3))
    lines.append("lowvram              : {} ({} patched modules)".format(
        getattr(patcher.model, "model_lowvram", None), patcher.lowvram_patch_counter()))

    diffusion_model = patcher.model.diffusion_model
    devices: dict[str, int] = {}
    branch_bytes = 0
    branch_devices: dict[str, int] = {}
    count = 0
    for _name, module in quantized_linears(diffusion_model):
        count += 1
        key = str(module.weight.device)
        devices[key] = devices.get(key, 0) + 1
        for _bname, buf in branch_buffers(module):
            branch_bytes += buf.numel() * buf.element_size()
            bkey = str(buf.device)
            branch_devices[bkey] = branch_devices.get(bkey, 0) + 1

    lines.append("")
    lines.append("== quantized layers ==")
    lines.append("convrot linears      : {}".format(count))
    lines.append("weight devices       : {}".format(devices or "none"))
    lines.append("low-rank factors     : {} tensors, {:.1f} MiB".format(
        sum(branch_devices.values()), branch_bytes / 1024 ** 2))
    lines.append("factor devices       : {}".format(branch_devices or "none"))
    return lines


def _package_version(dist: str) -> str:
    try:
        from importlib.metadata import version
        return version(dist)
    except Exception:
        return "unknown"


def _distinct_shapes(diffusion_model):
    """One representative module per (in, out) shape -- constraints are shape-dependent."""
    seen: dict[tuple, tuple] = {}
    for name, module in quantized_linears(diffusion_model):
        shape = tuple(module.weight._params.orig_shape)
        seen.setdefault(shape, (name, module))
    return seen


def report_dispatch(patcher, tokens: int) -> list[str]:
    lines = ["== acceleration paths =="]
    lines.extend(acceleration_paths_table())

    lines.append("")
    lines.append("== dispatch per weight shape (tokens={}) ==".format(tokens))
    shapes = _distinct_shapes(patcher.model.diffusion_model)
    if not shapes:
        lines.append("no convrot-quantized layers found")
        return lines

    for shape, (name, module) in sorted(shapes.items()):
        backend, impl_path, failures = resolve_dispatch(module, tokens)
        lines.append("{}  [{} -> {}]".format(name, shape[1], shape[0]))
        lines.append("    path    : {}".format(acceleration_path(_probe_op(module), backend)))
        lines.append("    backend : {}".format(backend or "NONE"))
        if backend != "cuda":
            for bname, reason in sorted(failures.items()):
                lines.append("    rejected {}: {}".format(bname, reason))

    backend, _impl, failures = resolve_dispatch(next(iter(shapes.values()))[1], tokens)
    warning = dispatch_warning(backend, failures)
    if warning:
        lines.append("")
        lines.append(warning)
    return lines


def _timed(fn, reps: int = 20, warmup: int = 3, label: str = "", failures: list | None = None):
    """Milliseconds per call, CUDA-synchronised. Returns nan if the call raises.

    A failure is logged at *warning* and recorded in `failures`, not swallowed to DEBUG:
    ComfyUI logs at INFO by default, so a row of bare `nan`s used to arrive with its reason
    discarded -- the opposite of what a diagnostics tool is for.
    """
    try:
        for _ in range(warmup):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(reps):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000.0 / reps
    except Exception as exc:
        logging.warning("[krea2-svdquant] bench call failed (%s): %s", label or "?", exc,
                        exc_info=True)
        if failures is not None:
            failures.append((label, "{}: {}".format(type(exc).__name__, exc)))
        return float("nan")


def report_bench(patcher, tokens: int) -> list[str]:
    """Per-shape timing of the quantized path against a bf16 reference.

    The number that settles a "no speedup" report is the last column: quant/bf16 above
    1.0 means the quantized kernel is losing to the format it was meant to beat, which
    is only possible on the eager path.
    """
    lines = ["== per-layer benchmark (tokens={}) ==".format(tokens)]
    if not torch.cuda.is_available():
        lines.append("CUDA not available; nothing to measure")
        return lines

    # Timing offloaded weights would measure PCIe, not the kernel.
    mm.load_models_gpu([patcher], force_full_load=True)
    device = mm.get_torch_device()

    lines.append("{:<44} {:>9} {:>9} {:>9} {:>9}".format(
        "layer [in -> out]", "quant ms", "branch ms", "bf16 ms", "quant/bf16"))

    failures: list[tuple[str, str]] = []
    for shape, (name, module) in sorted(_distinct_shapes(patcher.model.diffusion_model).items()):
        out_features, in_features = int(shape[0]), int(shape[1])
        dtype = module.weight._params.orig_dtype
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            dtype = torch.bfloat16
        x = torch.randn((tokens, in_features), dtype=dtype, device=device)

        quant_ms = _timed(lambda: F.linear(x, module.weight),
                          label="{} quant".format(name), failures=failures)

        factors = branch_factors(module)
        if factors is None:
            branch_ms = float("nan")
            failures.append(("{} branch".format(name), "no svdq_l1/svdq_l2 on this module"))
        else:
            l1, l2 = factors
            a1 = l1.to(device=device, dtype=dtype)
            a2 = l2.to(device=device, dtype=dtype)
            branch_ms = _timed(lambda: F.linear(F.linear(x, a2), a1),
                               label="{} branch".format(name), failures=failures)
            del a1, a2

        try:
            dense = module.weight.dequantize().to(dtype)
            bf16_ms = _timed(lambda: F.linear(x, dense),
                             label="{} bf16".format(name), failures=failures)
            del dense
        except Exception as exc:
            logging.warning("[krea2-svdquant] could not dequantize %s for the bf16 reference: %s",
                            name, exc, exc_info=True)
            failures.append(("{} bf16".format(name),
                             "dequantize failed: {}: {}".format(type(exc).__name__, exc)))
            bf16_ms = float("nan")

        ratio = quant_ms / bf16_ms if bf16_ms == bf16_ms and bf16_ms > 0 else float("nan")
        lines.append("{:<44} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.2f}".format(
            "{} [{} -> {}]".format(name, in_features, out_features)[:44],
            quant_ms, branch_ms, bf16_ms, ratio))

        del x
        mm.soft_empty_cache()

    lines.append("")
    lines.append("quant/bf16 < 1.0 means the int4 kernel is winning. Above 1.0 means the "
                 "quantized path is being emulated -- check mode=dispatch.")
    if failures:
        # Every nan above has to be accountable here. A bench table pasted into an issue is
        # useless if a column of nans carries no reason with it.
        lines.append("")
        lines.append("failed measurements ({}):".format(len(failures)))
        for label, reason in failures:
            lines.append("  {:<40} {}".format(label[:40], reason))
    return lines


def report_profile(patcher, tokens: int) -> list[str]:
    """One sampled step under torch.profiler, attributed by CUDA time."""
    lines = ["== profile =="]
    if not torch.cuda.is_available():
        lines.append("CUDA not available; nothing to profile")
        return lines

    mm.load_models_gpu([patcher], force_full_load=True)
    device = mm.get_torch_device()
    shapes = _distinct_shapes(patcher.model.diffusion_model)
    if not shapes:
        lines.append("no convrot_w4a4 layers found")
        return lines

    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=False) as prof:
        for shape, (_name, module) in sorted(shapes.items()):
            dtype = module.weight._params.orig_dtype
            if dtype not in (torch.float16, torch.bfloat16, torch.float32):
                dtype = torch.bfloat16
            x = torch.randn((tokens, int(shape[1])), dtype=dtype, device=device)
            for _ in range(4):
                module(x)
            del x
        torch.cuda.synchronize()

    lines.append(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
    return lines


def report_compile(patcher, tokens: int) -> list[str]:
    """How many graph breaks a compiled step pays, and why.

    `torch.compile` is the largest single speed lever left on this model -- a profiled step
    is only 37% GEMM against 34% small elementwise/norm/RoPE kernels, and that 34% is exactly
    what inductor fuses. But it can only fuse *within* a graph, and today every quantized
    Linear is a deliberate break (`svdquant_w4a4._shield_from_dynamo`), so the model compiles
    as ~225 fragments instead of one.

    Rather than compile the whole diffusion model -- which needs a real timestep/context
    signature and takes a minute -- this traces one representative layer wrapped in the kind
    of elementwise work that surrounds it in a block. That is enough to answer the only
    question that matters: does control leave the graph at the kernel call? The whole-model
    figure is that count times the number of quantized layers, which is printed alongside so
    the number in a bug report is the model's, not one layer's.
    """
    lines = ["== compile (tokens={}) ==".format(tokens)]
    if not torch.cuda.is_available():
        lines.append("CUDA not available; nothing to trace")
        return lines

    try:
        from torch._dynamo import explain, reset as dynamo_reset
    except Exception as exc:  # pragma: no cover - depends on the torch build
        lines.append("torch._dynamo unavailable: {}: {}".format(type(exc).__name__, exc))
        return lines

    mm.load_models_gpu([patcher], force_full_load=True)
    device = mm.get_torch_device()
    shapes = _distinct_shapes(patcher.model.diffusion_model)
    if not shapes:
        lines.append("no convrot_w4a4 layers found")
        return lines

    layers = sum(1 for _ in quantized_linears(patcher.model.diffusion_model))
    lines.append("{:<44} {:>8} {:>8} {:>14}".format(
        "layer [in -> out]", "graphs", "breaks", "breaks/model"))

    reasons: list[str] = []
    for shape, (name, module) in sorted(shapes.items()):
        in_features = int(shape[1])
        dtype = module.weight._params.orig_dtype
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            dtype = torch.bfloat16
        x = torch.randn((tokens, in_features), dtype=dtype, device=device)

        # Scale/add either side so there is real fusable work for inductor to lose. A bare
        # `module(x)` would report the same break count while proving nothing about what the
        # break costs.
        def traced(t, _m=module):
            return torch.nn.functional.silu(_m(t * 1.0001)) + 1.0

        try:
            dynamo_reset()
            result = explain(traced)(x)
            graphs, breaks = result.graph_count, result.graph_break_count
            for entry in getattr(result, "break_reasons", [])[:1]:
                text = str(getattr(entry, "reason", entry)).replace("\n", " ")[:110]
                if text not in reasons:
                    reasons.append(text)
        except Exception as exc:
            logging.warning("[krea2-svdquant] dynamo explain failed on %s: %s", name, exc,
                            exc_info=True)
            lines.append("{:<44} {:>8} {:>8} {:>14}".format(
                "{} [{}]".format(name, in_features)[:44], "-", "-",
                "{}: {}".format(type(exc).__name__, exc)[:14]))
            del x
            continue
        finally:
            dynamo_reset()

        lines.append("{:<44} {:>8} {:>8} {:>14}".format(
            "{} [{} -> {}]".format(name, in_features, int(shape[0]))[:44],
            graphs, breaks, breaks * layers))
        del x
        mm.soft_empty_cache()

    lines.append("")
    lines.append("{} quantized layers in this model. 0 breaks per layer means a compiled "
                 "step is one graph and inductor can fuse across the whole block; anything "
                 "above 0 multiplies by the layer count.".format(layers))
    for reason in reasons:
        lines.append("  break reason: {}".format(reason))
    return lines


REPORTS = {
    "env": report_env,
    "dispatch": report_dispatch,
    "bench": report_bench,
    "profile": report_profile,
    "compile": report_compile,
}


def run_report(patcher, mode: str, tokens: int = _DEFAULT_TOKENS) -> str:
    fn = REPORTS[mode]
    lines = fn(patcher) if mode == "env" else fn(patcher, tokens)
    return "\n".join(lines)


class Krea2SVDQuantDiagnostics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Output of the Krea2 SVDQuant W4A4 Loader."}),
                "mode": (["dispatch", "env", "bench", "profile", "compile"], {
                    "tooltip": "dispatch: which kernel actually runs (start here). "
                               "env: versions and memory accounting. "
                               "bench: quantized vs bf16 per layer. "
                               "profile: torch.profiler table. "
                               "compile: graph breaks a TorchCompileModel run would pay.",
                }),
                "tokens": ("INT", {
                    "default": _DEFAULT_TOKENS, "min": 64, "max": 65536, "step": 64,
                    "tooltip": "Sequence length to probe with. 4096 = 1024x1024. Kernel "
                               "selection is shape-dependent, so match your real run.",
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Same reasoning as Env Check: this measures the live process (mode=bench times real
        # kernels), so a cached report would be a report about a previous run.
        return float("nan")

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    OUTPUT_TOOLTIPS = ("The model, unchanged - wire it on to your sampler.",
                       "The report as text, for pasting into a bug report.")
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = _CATEGORY
    TITLE = "Krea2 SVDQuant Diagnostics"
    DESCRIPTION = ("Passthrough node that reports which acceleration path the "
                   "quantized layers' matmuls run on, plus memory accounting and "
                   "timings. Paste the output into bug reports.")

    def run(self, model, mode, tokens):
        try:
            report = run_report(model, mode, int(tokens))
        except Exception as exc:
            logging.exception("[krea2-svdquant] diagnostics failed")
            report = "diagnostics failed: {}: {}".format(type(exc).__name__, exc)
        logging.info("\n%s", report)
        return {"ui": {"text": [report]}, "result": (model, report)}


class Krea2SVDQuantEnvCheck:
    """Backend status with no model wired in.

    The single most common report against this pack is "the quantized checkpoint is slower
    than fp8", and the answer is almost always that ComfyUI disabled comfy_kitchen's CUDA
    backend because torch was built against CUDA < 13. That is answerable without loading
    anything, so it should be answerable before an 8 GB download rather than after.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Backend availability is process state, not graph state, so never serve a cached
        # result -- the whole point is to report what is true right now.
        return float("nan")

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    OUTPUT_TOOLTIPS = ("Backend status as text, for pasting into a bug report.",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = _CATEGORY
    TITLE = "Krea2 SVDQuant Env Check"
    DESCRIPTION = ("Which acceleration paths are available on this install -- "
                   "INT4 cores, INT8 cores, Triton, or only plain bf16 -- "
                   "with no model loaded, so you can ask before downloading one.")

    def run(self):
        try:
            report = report_backend_status()
        except Exception as exc:
            logging.exception("[krea2-svdquant] env check failed")
            report = "env check failed: {}: {}".format(type(exc).__name__, exc)
        logging.info("\n%s", report)
        return {"ui": {"text": [report]}, "result": (report,)}


NODE_CLASS_MAPPINGS = {"Krea2SVDQuantDiagnostics": Krea2SVDQuantDiagnostics,
                       "Krea2SVDQuantEnvCheck": Krea2SVDQuantEnvCheck}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2SVDQuantDiagnostics": "Krea2 SVDQuant Diagnostics",
                              "Krea2SVDQuantEnvCheck": "Krea2 SVDQuant Env Check"}
