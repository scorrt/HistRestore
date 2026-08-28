# Protocol and Parameter Notes

This note records implementation details for the synthetic `compound-proxy` subset and the MixedDoc utility coefficients used in the released results.

## Compound-Proxy Synthetic Degradation

The `compound-proxy` subset is generated from clean historical page images with `scripts/generate_compound_proxy.py`. The script writes paired clean/degraded pages and a `metadata.jsonl` manifest containing the sampled operators and parameters for each generated page.

Historical-537 contains 120 compound-proxy samples derived from 60 unique clean Jung source pages, with two degraded variants per clean page in the released grouped protocol. Source-page grouping keeps variants from the same clean page in the same split.

Default generation settings:

- `variants=3`
- `max_side=1600`
- `seed=20260805`
- `max_images=0`, meaning all discovered source images are used unless explicitly limited

For each page variant, the script samples one to three degradation operators:

- number of operators: `{1, 2, 3}` with probabilities `{0.30, 0.50, 0.20}`;
- candidate operators: perspective distortion, shadow, blur, low-contrast yellowing, and bleed-through/stain;
- operator order: sampled randomly without replacement and applied sequentially;
- operator severity: independently sampled from `Uniform(0.25, 0.90)`;
- JPEG compression: applied after the sampled operators with probability `0.35`, using quality sampled from `[35, 75]`.

Operator details are implemented in `scripts/generate_compound_proxy.py` and summarized in `configs/compound_proxy_generation_config.json`.

## MixedDoc Utility Coefficients

The final MixedDoc Protocol-B result reported as `HistRestore + MMDIR = 27.6673 dB` comes from:

- result table: `results/mixedoc/main_sota_table_mixeddoc.csv`;
- split: stable-hash 60/20/20 over 1837 MixedDoc pages, with 377 held-out test pages;
- candidate bank: `input_degraded`, `classical_shadow_0.40`, `docres_deshadow`, `docres_input_blend_0.90`, `docres_input_blend_0.80`, `docres_classical040_blend_0.90`, `region_controller`, and `mmdir_official`.

The locked utility parameters for this protocol are:

```text
lambda_SSIM = 8.0
lambda_R    = 0.55
lambda_H    = 0.15
```

The training utility is applied relative to the `classical_shadow_0.40` base candidate:

```text
U_train(y_k) =
  PSNR(y_k) - PSNR(base)
  + lambda_SSIM * [SSIM(y_k) - SSIM(base)]
  - lambda_R * max(0, Risk(y_k) - Risk(base))
  - lambda_H * max(0, PSNR(base) - PSNR(y_k)).
```

An earlier lightweight MixedDoc ablation used `ssim_weight=18.0`; that coefficient belongs to the preliminary no-reference selector protocol and is not the parameter used for the final Protocol-B result.
