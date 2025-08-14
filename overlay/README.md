# Overlay

Put your custom modifications here. The directory layout mirrors the LiveKit agents/plugins subpackages.
These files are copied on top of upstream sources during bundling (both legacy and namespaced builds).

- overlay/agents/** -> livekit.agents subpackage
- overlay/plugins/** -> livekit.plugins subpackage
- overlay/voice/**, overlay/llm/**, etc. also supported

Examples for this repo:
- plugins/elevenlabs/realtime/realtime_model.py (custom real-time integration)
- agents/voice/agent_activity.py (custom activity logic)
- agents/voice/generation*.py (custom generation flow)

Workflow:
1. Update submodules to the upstream versions you want.
2. Place or update your modified files under overlay/ with the same relative paths.
3. Run `make bundle` or `make build-ns` (for namespaced). The overlay is applied automatically.

Notes:
- The namespaced build rewrites imports like `from livekit.agents...` to `from vyvoj25_fork.agents...` after applying the overlay.
- Keep your changes focused in overlay/ so updating upstream is easy (no merge conflicts inside submodules).
