# Repository Guidelines

## Project Structure & Module Organization

`scinthil/` contains the Python package and CuTe DSL kernels, grouped by kernel family (`flash_attention/`, `grouped_gemm/`) and GPU architecture (`sm120/`). Native CUDA/C++ headers live in `csrc/`. Put correctness tests in `tests/python/` or `tests/cpp/`, and profiling or benchmarking programs in the matching `profile/python/` or `profile/cpp/` tree. Shared CMake helpers belong in `cmake/`. `3rdparty/cutlass/` is a Git submodule; avoid modifying vendored files in feature changes.

## Build, Test, and Development Commands

- `git submodule update --init --recursive` initializes CUTLASS after cloning.
- `python -m pip install -e '.[dev]'` installs Scinthil and its lint/test tools in editable mode.
- `cmake -S . -B build -DSCINTHIL_BUILD_PROFILES=OFF` configures native headers and tests; CUDA Toolkit and GTest are required.
- `cmake --build build --target scinthil_test_binary -j` builds the native binary test.
- `python -m pytest tests/python -m 'not gpu and not compile_only'` runs host-safe Python tests.
- `pre-commit run --all-files` applies the full formatting, linting, and type-checking suite.

SM120 GPU tests and profiles require compatible NVIDIA hardware. Do not treat successful compilation as runtime validation; record the GPU model, CUDA version, and command when reporting kernel results.

## Coding Style & Naming Conventions

Python targets 3.10+, uses four-space indentation, a 120-character line limit, Ruff formatting/linting, and mypy. Use `snake_case` for modules and functions, `CapWords` for classes, and architecture suffixes such as `_sm120` for specialized code. CUDA/C++ follows the repository `.clang-format` (Google base, 120 columns, left-aligned pointers). Keep public APIs separate from kernel implementations and reuse helpers from `scinthil/testing/` and `csrc/utils/`.

## Testing Guidelines

Name Python tests `test_*.py` and test functions `test_*`; use pytest parameterization for shape, dtype, and layout matrices. Apply the registered `gpu`, `sm120`, and `compile_only` markers accurately. Native tests use GoogleTest and live beside reusable testbed headers. Add focused correctness coverage for every behavior change; no numeric coverage threshold is configured.

## Commit & Pull Request Guidelines

Recent commits use short, imperative subjects such as `Add SM120 grouped GEMM scheduler` and `Refine ...`. Keep commits focused and avoid committing build products, PTX, CUBIN, SASS, caches, or benchmark logs. Pull requests should explain the change, affected architectures, validation commands, and any untested hardware path. Include before/after correctness or performance data for kernel changes and link the relevant issue when one exists.
