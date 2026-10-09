# Qwen Image 2.1

Select **Qwen Image 2.1** in either **Text to Image** or **AI Edit**. Existing
2512/2511 defaults and saved settings remain independent. The new profile uses
40 Euler/simple steps and CFG 1.0. Negative prompts take effect above CFG 1.
AI Edit accepts the source and one optional reference image (the current editor
contract), referenced in the prompt as `<image1>` and `<image2>`.

**Preserve transparency** retains native RGBA output, including when applying a
selection mask. It does not force the model to remove a background. For transparent
generation, explicitly describe an RGBA image, its alpha channel, and a transparent
background, as recommended by the [official examples](https://github.com/QwenLM/Qwen-Image-2.1#transparent-image-generation-rgba).
**Opaque RGB** composites the generated RGBA image onto white.
Transparent source/reference layers and embedded PNG files retain their alpha.

Output follows the target layer (generation) or source patch (editing). Inference
rounds up to multiples of 32, aligns the first reference to those dimensions, and
resizes the result back to the requested size. An additional reference retains its
own aspect ratio. VAE tiling is optional; automatic offload is the default, with a
low-VRAM mode available.

## Local runtime

The provider uses Comfy inference libraries in a private, persistent subprocess;
it starts no HTTP server and loads no custom nodes. This reuses the downloaded
BF16 transformer, RGBA VAE and INT8 ConvRot Qwen3-VL encoder without converting
weights or changing the shared Diffusers environment. The child exits on unload,
load failure, timeout, or death of the owning ML worker (including cancellation).
PyTorch SDPA is selected explicitly: the installed xFormers cannot execute masked
attention on the RTX 5090.

Default paths, all editable in the profile:

| Component | Path relative to `~/soft/ComfyUI/` | Environment override |
| --- | --- | --- |
| Runtime checkout | `experiments/qwen-image-2.1/ComfyUI` | `DIFFUSION_EDITOR_QWEN21_ROOT` |
| Runtime Python | `experiments/qwen-image-2.1/venv/bin/python` | `DIFFUSION_EDITOR_QWEN21_PYTHON` |
| Transformer | `models/diffusion_models/qwen_image_2.1_bf16.safetensors` | `DIFFUSION_EDITOR_QWEN21_MODEL` |
| Text encoder | `models/text_encoders/qwen3vl_8b_int8_convrot.safetensors` | `DIFFUSION_EDITOR_QWEN21_TEXT_ENCODER` |
| VAE | `models/vae/qwen_image_2.1_vae_bf16.safetensors` | `DIFFUSION_EDITOR_QWEN21_VAE` |

On another machine, supply a Comfy checkout with native `TextEncodeQwenImage21`
support and its own installed Python environment, plus compatible local weights.
The main older Comfy checkout is insufficient. `setup-workers.sh` continues to
install the shared worker environment; it does not provision this optional runtime.
There are no implicit model downloads. LoRA entries must be local safetensors
compatible with Qwen Image 2.1; older 2511/2512 adapters are not defaults for 2.1.

Verified local runtime: Comfy commit
`f1072eb0350638a3390ddb6afbcaa8c6b237c6fd`, Python 3.10.19,
PyTorch 2.10.0+cu130, RTX 5090 32 GiB. The actual runtime revision and attention
backend are recorded in generation provenance.

## Verification

Import/API check without loading model weights or requiring CUDA:

```sh
./venv/bin/python scripts/smoke_qwen_image21.py --check-only
```

GPU smoke for generation, unload/reload, two-image editing, and RGBA PNG output:

```sh
.venv-workers/bin/python scripts/smoke_qwen_image21.py \
  --steps 40 --size 512 --output /tmp/diffusion-editor-qwen21-verified
```

The output directory contains source/result PNGs and generation provenance JSON.
The smoke uses the editor's real backend and the dedicated runtime subprocess.
Contract tests cover process errors/timeouts, cleanup, reference alpha, frozen
result application, profile settings, and compatibility of the old profiles.
