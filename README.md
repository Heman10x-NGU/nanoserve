# nanoserve

`nanoserve` is a minimal, readable LLM inference engine for Apple Silicon,
with continuous batching and an OpenAI-compatible streaming server layered on
top. The implementation is being built phase by phase; measured results replace
this note once the benchmark gates pass.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
nanoserve --help
```

The default model is deliberately small:
`mlx-community/Qwen2.5-0.5B-Instruct-4bit`.

