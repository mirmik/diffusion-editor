# Isolated rembg segmentation worker

Background segmentation runs in its own subprocess while sharing the
[model-worker environment](model-workers.md) with LaMa and the other ML
backends.

The worker lazily creates and retains its ONNX Runtime session. Protocol
version 1 uses bounded JSON-lines over stdin/stdout; RGB input and grayscale
mask output cross the boundary as PNG files. Cancellation, timeout, worker
failure, or malformed output terminates the process, and the next request
starts a fresh worker.

Run a real end-to-end smoke (the first run downloads the U²-Net model):

```sh
./venv/bin/python scripts/smoke_segmentation_worker.py
```
