# Isolated LaMa worker

LaMa runs in its own subprocess while sharing the
[model-worker environment](model-workers.md) with segmentation and the other
ML backends.

The worker lazily loads the Big-LaMa TorchScript model and retains it between
requests. Set `LAMA_MODEL` to a local model or `LAMA_MODEL_URL` to an alternate
download. The default remains the established `big-lama.pt` release.

Protocol version 1 uses bounded JSON-lines over stdin/stdout. Images cross the
boundary as PNG files in a per-request temporary directory. Cancellation,
timeout, worker exit, malformed output, or a protocol mismatch terminates the
worker; the next request starts a fresh process.

Run a real end-to-end smoke (the first run downloads the model):

```sh
./venv/bin/python scripts/smoke_lama_worker.py
```
