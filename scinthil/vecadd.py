"""Python wrapper for the TVM-FFI CUDA vecadd kernel."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import tvm_ffi


_ENV_LIBRARY = "SCINTHIL_VECADD_LIBRARY"
_CANDIDATE_NAMES = (
    "libscinthil_vecadd.so",
    "scinthil_vecadd.so",
)
_MODULE_CACHE: dict[str, Any] = {}


def load_library(path: str | os.PathLike[str] | None = None) -> Any:
    """Load and cache the compiled TVM-FFI vecadd shared library."""
    package_dir = Path(__file__).resolve().parent
    candidate_paths = [
        root / name
        for root in (package_dir, package_dir.parent)
        for name in _CANDIDATE_NAMES
    ]

    if path is not None:
        resolved = Path(path).expanduser()
        if resolved.is_file():
            resolved = resolved.resolve()
        else:
            raise RuntimeError(
                f"vecadd library not found at {resolved}. Pass a valid library_path "
                f"or set {_ENV_LIBRARY}."
            )
    elif env_path := os.environ.get(_ENV_LIBRARY):
        resolved = Path(env_path).expanduser()
        if resolved.is_file():
            resolved = resolved.resolve()
        else:
            raise RuntimeError(
                f"{_ENV_LIBRARY} points to {resolved}, but that file does not exist. "
                "Pass library_path explicitly or update the environment variable."
            )
    else:
        for candidate in candidate_paths:
            if candidate.is_file():
                resolved = candidate.resolve()
                break
        else:
            candidate_text = ", ".join(str(candidate) for candidate in candidate_paths)
            raise RuntimeError(
                "Unable to locate the compiled vecadd TVM-FFI library. Pass "
                f"library_path to vecadd or set {_ENV_LIBRARY}. Checked: "
                f"{candidate_text}."
            )

    cache_key = str(resolved)
    module = _MODULE_CACHE.get(cache_key)
    if module is None:
        module = tvm_ffi.load_module(cache_key)
        _MODULE_CACHE[cache_key] = module
    return module


def vecadd(
    out: Any,
    a: Any,
    b: Any,
    library_path: str | os.PathLike[str] | None = None,
) -> Any:
    """Add ``a`` and ``b`` into preallocated ``out`` and return ``out``."""
    module = load_library(library_path)

    try:
        func = module["vecadd"]
    except (KeyError, TypeError, AttributeError):
        try:
            func = getattr(module, "vecadd")
        except AttributeError as error:
            raise RuntimeError(
                "Loaded vecadd library does not export 'vecadd'. Rebuild it with "
                "the TVM_FFI_DLL_EXPORT_TYPED_FUNC exports."
            ) from error

    tensors = []
    for value in (out, a, b):
        module_name = type(value).__module__
        if module_name == "tvm_ffi" or module_name.startswith("tvm_ffi."):
            tensors.append(value)
        elif hasattr(value, "__dlpack__"):
            tensors.append(tvm_ffi.from_dlpack(value, require_contiguous=True))
        else:
            tensors.append(value)

    func(*tensors)
    return out


__all__ = ["load_library", "vecadd"]
