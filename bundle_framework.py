import os
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
TARGET = ROOT / "vyvoj25_framework"

def copy_agents():
    agents_src = ROOT / "livekit-agents" / "livekit" / "agents"
    agents_dst = TARGET / "livekit" / "agents"
    if not agents_src.exists():
        raise FileNotFoundError(f"Agents source not found: {agents_src}")
    shutil.copytree(agents_src, agents_dst)

def copy_plugins():
    plugins_root = ROOT / "livekit-plugins"
    plugins_dst_root = TARGET / "livekit" / "plugins"
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

def main():
    # Remove old target
    if TARGET.exists():
        shutil.rmtree(TARGET)
    # Create base structure
    (TARGET / "livekit").mkdir(parents=True, exist_ok=True)
    (TARGET / "__init__.py").write_text('__version__ = "1.0.5"\n')
    (TARGET / "livekit" / "__init__.py").write_text("")
    # Copy code
    copy_agents()
    copy_plugins()
    print("🎉 Framework bundling completed!")

if __name__ == "__main__":
    main()
