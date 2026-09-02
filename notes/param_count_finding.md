# Table 9 parameter count discrepancy

Checkpoint: CVPR_2024_weights / model_lgd_grasp_anything++ (2.1 GB)

total 573.63M | trainable 213.08M | frozen 360.54M

Per-module:
  conv1-3, convt1-3, {pos,cos,sin,width}_output : ~0.07M   <- the actual LGD trunk
  y_flatten                                      : 3.75M
  lang_model (CLIP)                              : 151.28M (frozen)
  albef                                          : 418.54M (trainable)

Paper Table 9 claims LGD = 5.18M, ~= trunk + y_flatten only.
Excludes ALBEF (418M, trainable, does vision encoding + cross-modal fusion).

Table 9 argues LGD is the efficiency-optimal baseline vs CLIP-Fusion (13.51M).
That claim rests on omitting the largest trainable component.

TO VERIFY: supplementary E.1 says linguistic baselines also use ALBEF for fusion.
If their counts also exclude it, relative comparison partially holds but absolute
numbers are wrong for every row. Needs baseline configs to settle.
