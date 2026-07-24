# Repository Guidelines

## Repository Layout

`scinthil/` contains the Python package and CuTe DSL kernels. Kernel families are organized by operation and GPU
architecture, for example `flash_attention/sm120/` and `grouped_gemm/sm120/`. Native CUDA/C++ headers live in
`csrc/`. Correctness tests belong in `tests/python/` or `tests/cpp/`; benchmarks and compile/profile programs belong
under the matching `profile/` tree. Shared CMake helpers live in `cmake/`. Treat `3rdparty/cutlass/` as vendored code.

Machine-specific command placement and toolchain constraints belong in the untracked `DEVELOPMENT.md`. Read it when
present, but keep those local details out of this guide.

## Build, Test, and Development Commands

- `git submodule update --init --recursive`: initialize CUTLASS after cloning.
- `python -m pip install -e '.[dev]'`: install Scinthil and development tools in editable mode.
- `cmake -S . -B build -DSCINTHIL_BUILD_PROFILES=OFF`: configure native tests.
- `cmake --build build --target scinthil_test_binary -j`: build the C++ test binary.
- `python -m pytest tests/python -m 'not gpu and not compile_only'`: run host-safe Python tests.
- `pre-commit run --all-files`: run formatting, linting, and type checks.

## Kernel Development

Keep public launch APIs separate from kernel implementations and reuse existing scheduler, layout, and testing helpers.
Architecture-specific modules and symbols use suffixes such as `_sm120`. Compilation proves only that a kernel is
well-formed for its target; runtime correctness and performance require compatible hardware. Do not commit generated
PTX, CUBIN, SASS, compiler dumps, caches, or benchmark logs.

## Coding Style & Naming Conventions

Python targets 3.10+, uses four-space indentation and a 120-character limit, and is checked by Ruff and mypy. Use
`snake_case` for modules and functions and `CapWords` for classes. CUDA/C++ follows the repository `.clang-format`
(Google base, 120 columns, left-aligned pointers). Add comments only where layout or synchronization intent is unclear.

## Testing Guidelines

Name Python tests and functions `test_*.py` and `test_*`. Parameterize shape, dtype, and layout matrices. Apply the
`gpu`, `sm120`, and `compile_only` markers accurately. Native tests use GoogleTest. Add focused coverage for behavior
changes; no numeric coverage threshold is configured.

## Commit & Pull Request Guidelines

Use short, imperative commit subjects such as `Add SM120 grouped GEMM scheduler`. Keep commits focused. Pull requests
should describe the change, affected architectures, validation commands, and untested hardware paths. Include
before/after correctness or performance data for kernel changes and link relevant issues.
