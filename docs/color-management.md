# Color management

The editor uses an explicit transfer-function contract. It does not treat
sRGB channel values as light intensities during compositing.

## Pixel contract

| Boundary | RGB | Alpha / masks | Representation |
|---|---|---|---|
| Layer storage and project files | sRGB encoded | linear | straight RGBA8 |
| Public `LayerStack` composite APIs | sRGB encoded | linear | straight RGBA8 |
| CPU compositor cache | linear light | linear | premultiplied float32 |
| GPU layer texture | sRGB texture sampling | linear | straight RGBA8 |
| GPU compositor targets | linear light | linear | premultiplied/straight RGBA16F |
| Native Canvas borrowed texture | linear light | linear | straight RGBA16F |
| PNG/JPEG export and CPU readback | sRGB encoded | linear, if present | straight RGBA8 |

RGB is decoded once when a stored layer enters a compositor. Porter-Duff
source-over and group opacity operate on premultiplied linear RGB. RGB is
encoded once when a linear result crosses back into the public RGBA8 API.
Alpha and masks never receive an sRGB transfer function.

For example, 50% opaque white over opaque black produces linear RGB 0.5,
which encodes to approximately sRGB 188. Blending the encoded bytes directly
would incorrectly produce 128.

## CPU and GPU parity

The CPU reference and GPU compositor implement the same boundaries:

- CPU converts layer RGB with the standard sRGB electro-optical transfer
  function before premultiplication.
- GPU uploads layer images with `TextureEncoding.SRGB`; the sampler performs
  the corresponding decode.
- Both accumulate premultiplied linear values.
- GPU readback and CPU public results encode straight linear RGB to sRGB8.

Intermediate GPU targets use RGBA16F. An RGBA8 linear target loses excessive
precision in dark tones before the final sRGB encoding.

## Current scope

Project files do not yet carry ICC profiles or configurable working spaces.
Imported RGB images are currently interpreted as sRGB, and generated model
outputs are also assumed to be sRGB. Supporting embedded profiles, wide-gamut
working spaces, HDR, and display-profile transforms requires a separate
document-format and color-management extension; those concerns must not be
implemented by silently changing this transfer contract.
