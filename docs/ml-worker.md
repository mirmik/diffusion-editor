# Diffusers/Transformers worker

Diffusion, image editing, Grounding DINO, SAM 2.1, and Depth Anything V2 Small
run in a persistent subprocess using the shared
[model-worker environment](model-workers.md).

`AI -> Create Depth Map` estimates relative monocular depth for the current
composite with `depth-anything/Depth-Anything-V2-Small-hf`. The first run may
download model weights; later runs reuse the Hugging Face cache. The result is
an opaque grayscale layer where white is closer and black is farther. Values
are normalized per image and are not distances in metres.

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

The real smoke selects an available GPU by default. Pass `--device cpu` only
when a CPU run is intentional.

CPU, CUDA, and ROCm installation are documented in the shared environment
guide.
