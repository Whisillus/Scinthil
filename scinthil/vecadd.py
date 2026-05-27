"""Python wrapper for the TVM-FFI CUDA vecadd kernel."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import tvm_ffi

from .jit import build_source_module


_ENV_LIBRARY = "SCINTHIL_VECADD_LIBRARY"
_CANDIDATE_NAMES = (
    "libscinthil_vecadd.so",
    "scinthil_vecadd.so",
)
_MODULE_CACHE: dict[str, Any] = {}


def _default_source_path() -> Path:
    return Path(__file__).resolve().parent / "cuda" / "vecadd" / "vecadd.cu"


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


def build_jit(
    *,
    source_path: str | os.PathLike[str] | None = None,
    extra_cuda_cflags: Sequence[str] | None = None,
    build_directory: str | os.PathLike[str] | None = None,
    backend: str = "cuda",
    rebuild: bool = False,
) -> Path:
    """Build the vecadd TVM-FFI source and return its shared-library path."""
    return build_source_module(
        name="scinthil_vecadd",
        sources=[source_path or _default_source_path()],
        extra_cuda_cflags=extra_cuda_cflags,
        build_directory=build_directory,
        backend=backend,
        rebuild=rebuild,
    )


def load_jit(
    *,
    source_path: str | os.PathLike[str] | None = None,
    extra_cuda_cflags: Sequence[str] | None = None,
    build_directory: str | os.PathLike[str] | None = None,
    backend: str = "cuda",
    rebuild: bool = False,
) -> Any:
    """Build if needed, then load and cache the vecadd TVM-FFI source module."""
    binary_path = build_jit(
        source_path=source_path,
        extra_cuda_cflags=extra_cuda_cflags,
        build_directory=build_directory,
        backend=backend,
        rebuild=rebuild,
    )
    cache_key = str(binary_path)
    module = None if rebuild else _MODULE_CACHE.get(cache_key)
    if module is None:
        module = tvm_ffi.load_module(cache_key)
        _MODULE_CACHE[cache_key] = module
    return module


def vecadd(
    out: Any,
    a: Any,
    b: Any,
    library_path: str | os.PathLike[str] | None = None,
    *,
    jit: bool = False,
    jit_source_path: str | os.PathLike[str] | None = None,
    jit_extra_cuda_cflags: Sequence[str] | None = None,
    jit_build_directory: str | os.PathLike[str] | None = None,
    jit_backend: str = "cuda",
    jit_rebuild: bool = False,
) -> Any:
    """Add ``a`` and ``b`` into preallocated ``out`` and return ``out``."""
    if jit and library_path is not None:
        raise ValueError("Pass either library_path or jit=True for vecadd, not both.")

    module = (
        load_jit(
            source_path=jit_source_path,
            extra_cuda_cflags=jit_extra_cuda_cflags,
            build_directory=jit_build_directory,
            backend=jit_backend,
            rebuild=jit_rebuild,
        )
        if jit
        else load_library(library_path)
    )

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


__all__ = ["build_jit", "load_jit", "load_library", "vecadd"]
