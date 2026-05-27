"""Reusable TVM-FFI source JIT loading helpers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Sequence


_ENV_BUILD_DIR = "SCINTHIL_JIT_BUILD_DIR"
_BUILD_CACHE: dict[tuple[str, tuple[str, ...], tuple[str, ...], str, str, str | None], Path] = {}
_SOURCE_MODULE_CACHE: dict[str, Any] = {}


def _resolve_source_paths(
    sources: Sequence[str | os.PathLike[str]],
) -> tuple[Path, ...]:
    resolved_sources: list[Path] = []
    for source in sources:
        resolved = Path(source).expanduser().resolve()
        if not resolved.is_file():
            raise RuntimeError(f"JIT source file not found: {resolved}")
        resolved_sources.append(resolved)
    if not resolved_sources:
        raise ValueError("At least one JIT source file is required.")
    return tuple(resolved_sources)


def _default_build_directory() -> Path:
    if env_path := os.environ.get(_ENV_BUILD_DIR):
        return Path(env_path).expanduser().resolve()
    return (Path.home() / ".cache" / "scinthil" / "jit").resolve()


def _source_digest(source: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_identity(
    *,
    name: str,
    sources: tuple[Path, ...],
    cuda_flags: tuple[str, ...],
    backend: str,
    output_identity: str | None,
) -> str:
    digest = hashlib.sha256()
    for value in (name, backend, output_identity or ""):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    for source in sources:
        digest.update(str(source).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_source_digest(source).encode("ascii"))
        digest.update(b"\0")
    for flag in cuda_flags:
        digest.update(flag.encode("utf-8"))
        digest.update(b"\0")
    safe_name = "".join(character if character.isalnum() or character in "-_" else "_" for character in name)
    return f"{safe_name}-{digest.hexdigest()[:16]}"


def build_source_module(
    *,
    name: str,
    sources: Sequence[str | os.PathLike[str]],
    extra_cuda_cflags: Sequence[str] | None = None,
    build_directory: str | os.PathLike[str] | None = None,
    backend: str = "cuda",
    rebuild: bool = False,
    output_identity: str | None = None,
) -> Path:
    """Build a TVM-FFI C++/CUDA source module and return its shared-library path."""
    resolved_sources = _resolve_source_paths(sources)
    cuda_flags = tuple(extra_cuda_cflags or ())
    resolved_build_directory = (
        Path(build_directory).expanduser().resolve() if build_directory is not None else _default_build_directory()
    )
    identity = _build_identity(
        name=name,
        sources=resolved_sources,
        cuda_flags=cuda_flags,
        backend=backend,
        output_identity=output_identity,
    )
    output_path = resolved_build_directory / f"{identity}.so"
    cache_key = (
        name,
        tuple(str(source) for source in resolved_sources),
        cuda_flags,
        str(resolved_build_directory),
        backend,
        output_identity,
    )

    cached_path = _BUILD_CACHE.get(cache_key)
    if not rebuild and cached_path == output_path and output_path.is_file():
        return output_path
    if not rebuild and output_path.is_file():
        _BUILD_CACHE[cache_key] = output_path
        return output_path

    import tvm_ffi.cpp

    resolved_build_directory.mkdir(parents=True, exist_ok=True)
    built_path = tvm_ffi.cpp.build(
        name=identity,
        sources=[str(source) for source in resolved_sources],
        extra_cuda_cflags=list(cuda_flags),
        build_directory=str(resolved_build_directory),
        backend=backend,
        output=output_path.name,
    )
    if built_path is not None:
        output_path = Path(built_path).expanduser().resolve()
    _BUILD_CACHE[cache_key] = output_path
    return output_path


def load_source_module(
    *,
    name: str,
    sources: Sequence[str | os.PathLike[str]],
    extra_cuda_cflags: Sequence[str] | None = None,
    build_directory: str | os.PathLike[str] | None = None,
    backend: str = "cuda",
    rebuild: bool = False,
    output_identity: str | None = None,
) -> Any:
    """Build if needed, load a TVM-FFI source module binary, and cache it in-process."""
    binary_path = build_source_module(
        name=name,
        sources=sources,
        extra_cuda_cflags=extra_cuda_cflags,
        build_directory=build_directory,
        backend=backend,
        rebuild=rebuild,
        output_identity=output_identity,
    )

    cache_key = str(binary_path)
    module = None if rebuild else _SOURCE_MODULE_CACHE.get(cache_key)
    if module is None:
        import tvm_ffi

        module = tvm_ffi.load_module(cache_key)
        _SOURCE_MODULE_CACHE[cache_key] = module
    return module


__all__ = ["build_source_module", "load_source_module"]
