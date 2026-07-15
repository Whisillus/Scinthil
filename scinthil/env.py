import os
from pathlib import Path


def get_cutedsl_env() -> dict[str, str]:
    keep = os.environ.setdefault("SCINTHIL_CUTE_DSL_KEEP", os.environ.get("CUTE_DSL_KEEP", "all"))
    arch = os.environ.setdefault("SCINTHIL_CUTE_DSL_ARCH", os.environ.get("CUTE_DSL_ARCH", "sm_120a"))
    cutedsl_env = {
        "CUTE_DSL_KEEP": keep,
        "CUTE_DSL_ARCH": arch,
    }
    os.environ.update(cutedsl_env)
    return cutedsl_env


def get_cutedsl_dump_dir(kernel_family: str, arch: str) -> Path:
    default_base_dir = Path.home() / ".cache" / "scinthil" / "cutedsl" / "dump"
    base_dir = Path(os.environ.setdefault("SCINTHIL_CUTE_DSL_DUMP_BASE_DIR", str(default_base_dir))).expanduser()
    os.environ["SCINTHIL_CUTE_DSL_DUMP_BASE_DIR"] = str(base_dir)
    dump_dir = base_dir / kernel_family / arch
    os.environ["CUTE_DSL_DUMP_DIR"] = str(dump_dir)
    return dump_dir


def get_sass_dump_dir(kernel_family: str, arch: str) -> Path:
    default_base_dir = Path(
        os.environ.get(
            "SCINTHIL_CUTE_DSL_DUMP_BASE_DIR",
            Path.home() / ".cache" / "scinthil" / "cutedsl" / "dump",
        )
    )
    base_dir = Path(os.environ.setdefault("SCINTHIL_CUTE_DSL_SASS_DUMP_BASE_DIR", str(default_base_dir))).expanduser()
    os.environ["SCINTHIL_CUTE_DSL_SASS_DUMP_BASE_DIR"] = str(base_dir)
    sass_dump_dir = base_dir / kernel_family / arch
    sass_dump_dir.mkdir(parents=True, exist_ok=True)
    return sass_dump_dir
