import os
import re
import shutil
from pathlib import Path

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

ROOT = Path(__file__).parent.resolve()
# Default bundle target (legacy): installs under top-level 'livekit'
TARGET = ROOT / "vyvoj25_framework"

NS_MODE_FLAG = "ns"  # CLI arg to enable namespaced bundling


def _read_version_from_pyproject() -> str:
    pyproject = ROOT / "pyproject.toml"
    if not pyproject.exists():
        return "0.0.0"
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    return data.get("project", {}).get("version", "0.0.0")


def copy_agents(base_pkg_dir: Path):
    default_src = ROOT / "livekit-agents" / "livekit" / "agents"
    agents_dst = base_pkg_dir / "agents"
    agents_src = default_src
    if not default_src.exists():
        # Search for nested layouts, e.g. monorepo variants that still contain
        # a Python package at some/*/livekit/agents
        candidates = []
        la_root = ROOT / "livekit-agents"
        if la_root.exists():
            for p in la_root.rglob("livekit/agents"):
                s = str(p)
                # Skip generated or irrelevant paths
                if any(x in s for x in ("vyvoj25_fork_build", "overlay", "dist", "build")):
                    continue
                candidates.append(p)
        if candidates:
            # Prefer the shortest path (closest to la_root)
            agents_src = sorted(candidates, key=lambda x: len(x.as_posix()))[0]
            print(f"🔍 Using discovered agents source: {agents_src}")
        else:
            raise FileNotFoundError(
                f"Agents source not found at {default_src} and no nested 'livekit/agents' found under {la_root}"
            )
    shutil.copytree(agents_src, agents_dst)


def copy_plugins(base_pkg_dir: Path):
    plugins_root = ROOT / "livekit-plugins"
    # Fallback: when upstream is tracked as a single monorepo submodule at
    # ROOT/livekit-agents, plugins live under livekit-agents/livekit-plugins
    if not plugins_root.exists():
        alt = ROOT / "livekit-agents" / "livekit-plugins"
        if alt.exists():
            plugins_root = alt
        else:
            raise FileNotFoundError(
                f"Plugins source not found: {ROOT / 'livekit-plugins'} or {alt}"
            )

    plugins_dst_root = base_pkg_dir / "plugins"
    plugins_dst_root.mkdir(parents=True, exist_ok=True)

    for plugin_dir in plugins_root.iterdir():
        if not plugin_dir.is_dir():
            continue
        if not plugin_dir.name.startswith("livekit-plugins-"):
            continue
        plugin_name = plugin_dir.name.replace("livekit-plugins-", "")
        src_path = plugin_dir / "livekit" / "plugins" / plugin_name
        if not src_path.exists():
            print(f"⚠️  Skipping {plugin_dir.name}, path not found: {src_path}")
            continue
        dst_path = plugins_dst_root / plugin_name
        shutil.copytree(src_path, dst_path)
        print(f"✅ Copied plugin: {plugin_name}")


def _merge_dir(src: Path, dst: Path) -> None:
    """Recursively copy files from src into dst, overwriting existing files.

    Creates directories as needed. Similar to an rsync overlay.
    """
    if not src.exists():
        return
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        target_dir = dst / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for fname in files:
            shutil.copy2(Path(root) / fname, target_dir / fname)


def apply_overlay(base_pkg_dir: Path) -> None:
    """Apply overlay/ files on top of the copied sources.

    Expected structure under `overlay/` mirrors livekit subpackages:
      - overlay/agents/** -> {base_pkg_dir}/agents/**
      - overlay/plugins/** -> {base_pkg_dir}/plugins/**
      - overlay/<subpkg>/** for other subpackages (llm, voice, etc.)
    """
    overlay_root = ROOT / "overlay"
    if not overlay_root.exists():
        return
    subdirs = [
        "agents",
        "plugins",
        "llm",
        "metrics",
        "stt",
        "tokenize",
        "tts",
        "utils",
        "vad",
        "voice",
        "ipc",
    ]
    for sub in subdirs:
        src = overlay_root / sub
        if src.exists():
            _merge_dir(src, base_pkg_dir / sub)
    print("🧩 Applied overlay on top of upstream sources")


def _rewrite_imports_to_namespace(ns_root: Path, new_top: str = "vyvoj25_fork") -> None:
    """Rewrite absolute imports for LiveKit Agents subpackages to a new namespace.

    Only rewrites imports that start with one of the agents-related subpackages:
    agents, plugins, llm, metrics, stt, tokenize, tts, utils, vad, voice, ipc.
    Core SDK imports like `import livekit` or `from livekit import api, rtc` are
    left untouched so they continue to refer to the official SDK.

    Args:
        ns_root: directory containing the new top-level package (e.g. /.../vyvoj25_framework_ns/vyvoj25_framework)
        new_top: top-level package name to replace 'livekit.<subpkg>' with.
    """
    py_files = list(ns_root.rglob("*.py"))
    if not py_files:
        return
    subpkgs = r"(agents|plugins|llm|metrics|stt|tokenize|tts|utils|vad|voice|ipc)"
    from_pattern_from = re.compile(rf"\bfrom\s+livekit\.{subpkgs}(\.[\w\.]+)?\s+import\s")
    from_pattern_import = re.compile(rf"\bimport\s+livekit\.{subpkgs}(\.[\w\.]+)?\b")
    from_pattern_root = re.compile(rf"\bfrom\s+livekit\s+import\s+{subpkgs}\b")

    for f in py_files:
        text = f.read_text(encoding="utf-8")
        new = text
        new = from_pattern_from.sub(lambda m: f"from {new_top}.{m.group(1)}{m.group(2) or ''} import ", new)
        new = from_pattern_import.sub(lambda m: f"import {new_top}.{m.group(1)}{m.group(2) or ''}", new)
        new = from_pattern_root.sub(lambda m: f"from {new_top} import {m.group(1)}", new)
        if new != text:
            f.write_text(new, encoding="utf-8")


def main():
    # Determine mode
    ns_mode = NS_MODE_FLAG in os.sys.argv[1:]

    # Remove old target
    if TARGET.exists():
        shutil.rmtree(TARGET)

    if ns_mode:
        # Namespaced layout: top-level import will be 'vyvoj25_framework.*'
        ns_target = ROOT / "vyvoj25_fork_build"
        if ns_target.exists():
            shutil.rmtree(ns_target)
        base_pkg_dir = ns_target / "vyvoj25_fork"
        base_pkg_dir.mkdir(parents=True, exist_ok=True)

        version = _read_version_from_pyproject()
        # Write top-level package __init__ with version
        (base_pkg_dir / "__init__.py").write_text(f'__version__ = "{version}"\n')

        # Copy sources into namespaced package
        copy_agents(base_pkg_dir)
        copy_plugins(base_pkg_dir)

        # Apply any user overrides from overlay/
        apply_overlay(base_pkg_dir)

        # Rewrite absolute imports from 'livekit' -> 'vyvoj25_fork'
        _rewrite_imports_to_namespace(base_pkg_dir, new_top="vyvoj25_fork")

        # Create a dedicated pyproject.toml for the namespaced build so it can be
        # built independently via `python -m build vyvoj25_fork_build`.
        py_ns = ns_target / "pyproject.toml"
        py_ns.write_text(
            """
[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"

[project]
name = "vyvoj25-fork"
version = """.strip()
            + f'"{version}"'
            + """
description = "Namespaced fork of LiveKit agents/plugins under 'vyvoj25_fork.*' to co-install with upstream."
readme = "README.md"
requires-python = ">=3.10"
license = { file = "LICENSE" }
authors = [{ name = "vyvoj25" }]
classifiers = [
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: Apache Software License",
  "Operating System :: OS Independent",
]
dependencies = [
  "click~=8.1",
  "livekit>=1.0.12,<2",
  "livekit-api>=1.0.4,<2",
  "livekit-protocol~=1.0",
  "livekit-blingfire~=1.0",
  "protobuf>=3",
  "pyjwt>=2.0",
  "types-protobuf>=4,<5",
  "watchfiles>=1.0",
  "psutil>=7.0",
  "aiohttp~=3.10",
  "typing-extensions>=4.12",
  "sounddevice>=0.5",
  "docstring_parser>=0.16",
  "eval-type-backport",
  "colorama>=0.4.6",
  "av>=14.0.0",
  "numpy>=1.26.0",
  "pydantic>=2.0,<3",
  "nest-asyncio>=1.6.0",
  "opentelemetry-api>=1.34",
  "opentelemetry-sdk>=1.34.1",
  "opentelemetry-exporter-otlp>=1.34.1",
  "prometheus-client>=0.22",
]

[project.urls]
Homepage = "https://github.com/vyvoj25/livekit_agents_framework"

[tool.hatch.build.targets.wheel]
packages = [
  "vyvoj25_fork"
]
include = [
  "vyvoj25_fork/**/resources/**",
  "vyvoj25_fork/agents/debug/index.html"
]

[tool.hatch.build.targets.sdist]
include = ["/vyvoj25_fork"]
"""
        )

        # Copy root README and LICENSE for completeness if present
        for fname in ("README.md", "LICENSE"):
            src = ROOT / fname
            if src.exists():
                shutil.copy2(src, ns_target / fname)

        print("🎉 Namespaced framework bundling completed! (import as 'vyvoj25_fork.*')")
    else:
        # Legacy layout: installs under top-level 'livekit'
        (TARGET / "livekit").mkdir(parents=True, exist_ok=True)
        # sync version from pyproject
        version = _read_version_from_pyproject()
        (TARGET / "__init__.py").write_text(f'__version__ = "{version}"\n')
        # Make livekit a namespace-friendly package to co-exist with upstream
        (TARGET / "livekit" / "__init__.py").write_text(
            "from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n"
        )
        # Copy code
        copy_agents(TARGET / "livekit")
        copy_plugins(TARGET / "livekit")
        # Apply overlay (same layout under overlay/)
        apply_overlay(TARGET / "livekit")
        print("🎉 Framework bundling completed! (import as 'livekit.*')")


if __name__ == "__main__":
    main()
