# Maintenance and Release Guide

A practical checklist for updating to a new upstream LiveKit release and publishing your fork.

## Overview
- Upstream code is tracked via the `livekit-agents/` Git submodule (Python monorepo).
- Your changes live under `overlay/`, mirroring LiveKit subpackages. During bundling, `overlay/` files overwrite upstream files.
- Two build flavors:
  - Legacy (project: `vyvoj25-framework`): installs under `livekit.*` (not safe to co-install with upstream).
  - Namespaced (project: `vyvoj25-fork`): installs under `vyvoj25_fork.*` (safe to co-install with upstream). Recommended.

## Prerequisites
- Python 3.10+
- `.env` with PyPI credentials:
  - `TWINE_USERNAME=__token__`
  - `TWINE_PASSWORD=pypi-...`
- Makefile targets do the heavy lifting.

## One-time notes
- Plugins may exist either in `livekit-plugins/` (top-level) or inside the monorepo at `livekit-agents/livekit-plugins/`.
- The bundler prefers the top-level `livekit-plugins/` if present; otherwise it falls back to the monorepo path automatically.
- ElevenLabs plugin exists in upstream monorepo. Your customizations live in `overlay/plugins/elevenlabs/...`.

## When a new upstream version is out
1) Update/initialize submodule
```bash
make submodules-init       # if first time
make submodules-update     # fetch latest from remotes
make submodules-status
```

2) Checkout the desired upstream tag/commit
```bash
make submodules-checkout TAG=<upstream-tag-or-commit>
make submodules-status
```

3) Inspect overlay contents (what you override)
```bash
make overlay-ls
# Common overrides:
# overlay/plugins/elevenlabs/realtime/realtime_model.py
# overlay/agents/voice/agent_activity.py
# overlay/agents/voice/generation.py
```

4) Rebuild and smoke-test
- Namespaced (co-installable; recommended):
```bash
make build-ns     # bundles + builds vyvoj25_fork*
make check-ns
make test-install-ns
```
- Legacy (drop-in under livekit.*; not co-installable):
```bash
make bundle
make build
make check
make test-install
```

5) Fix any breakages
- If upstream APIs changed, adjust your overlay files accordingly (they are applied post-copy).
- Rebuild until the quick tests pass.

6) Bump version
- Edit `pyproject.toml` and bump `project.version` (this version propagates to both builds):
```toml
[project]
name = "vyvoj25_framework"
version = "X.Y.Z"
```
- Confirm:
```bash
make print-version
```

7) Release
- Namespaced (recommended):
```bash
make release-ns  # uploads vyvoj25-fork to PyPI
```
- Legacy (only if you need `livekit.*` drop-in):
```bash
make release     # uploads vyvoj25-framework to PyPI
```

## Import patterns (consumers)
- Upstream: `from livekit.agents import ...`, `from livekit.plugins.elevenlabs import ...`
- Your namespaced fork: `from vyvoj25_fork.agents import ...`, `from vyvoj25_fork.plugins.elevenlabs import ...`
- Do not mix imports from `livekit.*` and `vyvoj25_fork.*` in the same code path.

## Overlay reference
- Place custom files under `overlay/` mirroring the subpackage structure. Examples:
  - `overlay/plugins/elevenlabs/realtime/realtime_model.py`
  - `overlay/agents/voice/agent_activity.py`
  - `overlay/agents/voice/generation.py`
- The bundler applies overlay before namespacing.
- To add new customizations, create the corresponding path under `overlay/` and rebuild.

## Common pitfalls
- "Where is ElevenLabs plugin?"
  - If top-level `livekit-plugins/` exists and doesn’t include it, bundler will still create it from the overlay.
  - Alternatively, remove/rename top-level `livekit-plugins/` to prefer monorepo plugins.
- Mixed imports across upstream and fork cause confusion. Pick one per app/process.
- Forgetting to bump the version results in upload rejection on PyPI.

## Useful commands (cheat sheet)
```bash
# Submodules
make submodules-init
make submodules-update
make submodules-checkout TAG=v1.2.3
make submodules-status

# Overlay
make overlay-ls

# Build & test (namespaced)
make build-ns
make check-ns
make test-install-ns

# Build & test (legacy)
make bundle
make build
make check
make test-install

# Release
make release-ns   # vyvoj25-fork (namespaced)
make release      # vyvoj25-framework (legacy)

# Misc
make print-version
```

## Directory map
- `livekit-agents/` — upstream monorepo (git submodule)
- `livekit-plugins/` — optional top-level plugins directory (if present, preferred)
- `overlay/` — your customizations
- `vyvoj25_fork_build/` — namespaced build output
- `vyvoj25_framework/` — legacy bundle output

## Notable overlay customizations (context)
- ElevenLabs realtime (`overlay/plugins/elevenlabs/realtime/realtime_model.py`):
  - Conversation integration with a LiveKit audio bridge
  - `auto_tool_reply_generation = False` to keep tool execution in the LiveKit job context
  - Session lifecycle management, resampling to 16 kHz mono, interruption handling

---
If anything breaks after an upstream update, start by rebuilding (build-ns), reading the bundler logs, and reconciling your overlay files with upstream changes. Then re-run quick install tests and publish.
