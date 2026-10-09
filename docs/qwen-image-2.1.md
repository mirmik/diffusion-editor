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

Runtime stderr is streamed to the launching terminal, including the live step
counter, while retaining a temporary diagnostic log. Errors include the log tail;
the subprocess and its output relay are stopped before the log is removed.

## Turbo8

Select **Qwen Image 2.1 Turbo8** in **Text to Image** or **AI Edit**. This separate
profile loads the [Turbo8 adapter](https://huggingface.co/chriswritescode/Turbo8-LoRA-Qwen-Image-2.1)
at strength 1.0 and fixes sampling to 8 steps, CFG 1, Euler/simple. It supports
the same transparent output and reference images as the base profile. Base
settings remain independent. The author reports reduced accuracy for long or
dense text compared with the 40-step model.

The enabled **Turbo8 8-step** entry must remain in the LoRA list at strength 1.0.
To use the model without it, select the ordinary **Qwen Image 2.1** profile.
Additional compatible artistic LoRAs can be added through the same list.

Default adapter path:
`~/soft/ComfyUI/models/loras/turbo8_lora_step2500.safetensors`.
Override it in the LoRA list or through `DIFFUSION_EDITOR_QWEN21_TURBO8_LORA`.
The file is installed locally; other machines need to download it explicitly.
Verified download on 2026-10-09:

- Repository revision: `611a3f86d0cc53d21a1a09d07ea7dc82df3d68a1`.
- Size: 1,359,148,744 bytes.
- SHA256: `a0b7a7634ff2f82ccbe209e16b0ba19b25e2d9e18624d89edc490503b6a75d39`.

The runtime requires all 227 transformer layers to map successfully. It applies
`ModelSamplingFlux(max_shift=0.6935, base_shift=0.5)` using each request's actual
inference dimensions for generation and editing, including after a size change.
This reproduces the adapter's resolution-dependent noise schedule without terminal
stretching. The loaded model is not permanently modified by the sampling patch.
Provenance records the Turbo8 mode, adapter SHA256, mapped layer count, strength,
shift parameters, and requested/inference dimensions.

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

For Turbo8, use `--turbo8 --size 1024 --repeat-size 512` (the step count is fixed
to eight). This also checks changing resolution without reloading the model.

Verified on the local RTX 5090: Turbo8 generation and two-reference editing at
512 and 1024, RGBA alpha spanning 0–255, repeated generation at a different
resolution, unload/reload, and the editor's real ML worker IPC. Provenance reports
shift 0.6935 at 1024 and 0.5387 at 512. The base 40-step profile also passes
generation/editing; both modes show live terminal progress.

The output directory contains source/result PNGs and generation provenance JSON.
The smoke uses the editor's real backend and the dedicated runtime subprocess.
Contract tests cover process errors/timeouts, cleanup, reference alpha, frozen
result application, profile settings, and compatibility of the old profiles.
