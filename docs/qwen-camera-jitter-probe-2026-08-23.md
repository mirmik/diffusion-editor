# Qwen camera-jitter residual probe — 2026-08-23

## Question

Can a head recover the difference between the requested semantic azimuth and
the camera azimuth actually represented by Qwen's target-token readout?

An initial absolute-angle probe was rejected as non-diagnostic: its training
data always had prompt angle equal to camera angle, so it could decode the
prompt without learning camera error. The corrected experiment introduces the
error during training.

## Corrected supervision contract

- Prompt azimuth: one of the eight production commands, `0, 45, …, 315°`.
- Rendered target azimuth: `prompt + δ`.
- `|δ|` is sampled deterministically in `[3°, 20°]`, with independently chosen
  sign and two samples per prompt/identity.
- Target: signed residual `δ = actual - prompt`, wrapped to `[-180°, 180°)`.
- Exact, non-jittered front/back views from the previous dataset provide Qwen
  conditioning. Jitter targets are never reused as conditioning anchors.

The dataset contains seven training identities (three Quaternius and four OGA)
and two complete held-out identities (Wizard and OGA-09): 112 train and 32
validation targets. Train residuals are balanced (47.3% negative, mean
`0.03°`).

## Probe

- Frozen Qwen Image Edit 2511 Multiple Angles pipeline.
- Production four-step schedule, final transformer invocation only (`step 3`).
- Candidate target-token `img_norm2` readouts: blocks 30, 45, and 59.
- Each 64×64×3072 map is pooled to a 4×4 spatial grid.
- Dual ridge regression predicts `sin(δ), cos(δ)`.
- No prompt text, view ID, ray map, or camera FiLM is given to this baseline.
- Block and ridge are selected only on the two held-out identities.

## Result

| metric | train | held-out validation |
| --- | ---: | ---: |
| selected block / ridge | 30 / 1000 | 30 / 1000 |
| residual MAE | 2.54° | 13.24° |
| Pearson correlation | 0.981 | 0.177 |
| sign accuracy | 100% | 62.5% |

The zero-correction validation baseline is `13.49° MAE`. The probe improves it
by only `0.25°`; a paired bootstrap 95% interval for the improvement is
`[-2.82°, +3.21°]`. This is not evidence of useful transfer.

Per identity, OGA-09 reaches `12.04° MAE` with negative correlation `-0.189`;
Wizard reaches `14.44° MAE` with correlation `0.181`. Predictions are dominated
by identity bias: mostly negative for OGA-09 and mostly positive for Wizard.

Applying the same probe to saved real-generation rings produces the same
pattern. Wizard corrections average `0.39°` signed (`2.99°` absolute), while
OGA-09 corrections average `-14.07°` and are negative for every requested
view. They must not be used as camera calibration.

## Conclusion

The corrected low-capacity linear probe fails. It memorizes training identities
but does not recover a transferable camera residual from the pooled current
readout. This does **not** prove that Qwen contains no usable camera signal: the
probe does not explicitly receive the known nominal camera and cannot model a
nonlinear comparison between nominal orientation and spatial target evidence.

A meaningful next gate is a small spatial camera head conditioned explicitly
on nominal camera (`sin/cos` through FiLM or an equivalent interaction). It
must use the same whole-identity split and pass prompt-only/zero-readout
ablations. It should be rejected unless the gain over zero correction has a
positive held-out confidence interval, positive correlation on both held-out
identities, and stable sign accuracy. Real Qwen corrections should only be
inspected after this synthetic gate passes.

## Reproduction and artifacts

- Config: `data/qwen-camera-jitter-kill-test-2026-08-23.json`
- Renderer: `scripts/render-rain-canonical-dataset.py --azimuth-pair PROMPT:CAMERA`
- Extractor: `scripts/extract-qwen-canonical-features.py --schedule-step 3`
- Probe: `scripts/probe-qwen-camera-readout.py --target residual --pool-grid 4`
- Dataset/features/report root:
  `/home/mirmik/mnt-nvme/canonical-experiments/camera-jitter-2026-08-23`
- Machine-readable report:
  `/home/mirmik/mnt-nvme/canonical-experiments/camera-jitter-2026-08-23/probe-grid4/report.json`
- Selected probe artifact:
  `/home/mirmik/mnt-nvme/canonical-experiments/camera-jitter-2026-08-23/probe-grid4/selected-probe.npz`
- Verification: `./venv/bin/pytest -q tests/test_qwen_camera_readout_probe.py tests/test_canonical_feature_extractor.py tests/test_canonical_identity_runner.py tests/test_canonical_feature_loader.py` (`13 passed`).
