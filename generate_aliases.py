import os

root = "vyvoj25_framework/livekit/plugins"
os.makedirs(root, exist_ok=False)

plugins = [
    "anthropic", "assemblyai", "aws", "azure", "baseten", "cartesia", "clova",
    "deepgram", "elevenlabs", "fal", "google", "lmnt", "nltk", "openai",
    "rime", "silero", "speechmatics", "turn-detector", "neuphonic", "playai",
    "groq", "gladia", "resemble", "bey", "bithuman", "speechify", "tavus",
    "hume", "hedra", "spitch", "langchain", "sarvam", "inworld", "mistralai",
    "anam", "simli"
]

for plugin in plugins:
    pkg_name = f"livekit_plugins_{plugin.replace('-', '_')}"
    plugin_dir = os.path.join(root, plugin)
    os.makedirs(plugin_dir, exist_ok=True)
    with open(os.path.join(plugin_dir, "__init__.py"), "w", encoding="utf-8") as f:
        f.write(f"from {pkg_name} import *\n")

# Agents alias
agents_dir = "vyvoj25_framework/livekit/agents"
os.makedirs(agents_dir, exist_ok=True)
with open(os.path.join(agents_dir, "__init__.py"), "w", encoding="utf-8") as f:
    f.write("from livekit_agents import *\n")