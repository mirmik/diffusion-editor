# Diffusers/Transformers worker

Diffusion, InstructPix2Pix, Grounding DINO, and SAM 2.1 run in a persistent
subprocess using the shared [model-worker environment](model-workers.md).

Requests and responses use bounded, versioned JSON-lines messages. Images and
masks cross the boundary as PNG files; detections use a bounded JSON manifest
plus optional mask PNGs. A failed operation terminates the suspect process so
the next request starts cleanly.

The deterministic smoke downloads no models:

```sh
./venv/bin/python scripts/smoke_ml_worker.py
```

A real smoke requires a local SDXL checkpoint and may download other model
weights:

```sh
.venv-workers/bin/python scripts/smoke_ml_worker.py \
  --real --sdxl /path/to/model.safetensors
```

CPU, CUDA, and ROCm installation are documented in the shared environment
guide.
