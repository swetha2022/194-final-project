# Per-tensor weight-change norm variance: LoRA vs Full FT, AdamW vs Muon

Each cell is the **sample variance** (ddof=1) of the per-tensor norm across all tensors in that category for one finetuned run. Bold cells mark the lower AdamW-vs-Muon variance within each finetune type (LoRA columns compared to each other, Full FT columns compared to each other).  
`-` = no row present for that (pretraining optimizer, finetune variant). `NaN` = norm undefined for every tensor in the group (e.g. `rms->rms` for 1-D LayerNorm weights).

## Adam pretraining — $\mathrm{Var}\bigl(\|\Delta W\|_\infty\bigr)$

| Tensor type | AdamW LoRA | Muon LoRA | AdamW Full FT | Muon Full FT |
|---|---:|---:|---:|---:|
| Attention Q proj | **1.149e-05** | 1.318e-04 | **1.885e-06** | 9.295e-05 |
| Attention KV proj | **1.225e-04** | 2.446e-04 | **9.324e-07** | 9.507e-05 |
| Attention O proj | 1.787e-04 | **9.794e-06** | **3.792e-06** | 4.785e-06 |
| MLP / MoE experts | **1.578e-05** | 4.705e-05 | **4.022e-06** | 9.414e-06 |
| LayerNorm / RMSNorm | 0 | 0 | **3.158e-06** | 8.424e-06 |

## Adam pretraining — $\mathrm{Var}\bigl(\|\Delta W\|_{\mathrm{RMS}\rightarrow\mathrm{RMS}}\bigr)$

| Tensor type | AdamW LoRA | Muon LoRA | AdamW Full FT | Muon Full FT |
|---|---:|---:|---:|---:|
| Attention Q proj | 0.1497 | **0.04088** | 0.03232 | **3.225e-04** |
| Attention KV proj | 0.6714 | **0.1189** | 0.1369 | **0.02768** |
| Attention O proj | 25.9 | **0.1689** | 0.03163 | **4.413e-04** |
| MLP / MoE experts | 0.3312 | **0.03222** | **0.06276** | 0.07066 |
| LayerNorm / RMSNorm | NaN | NaN | NaN | NaN |

## Muon pretraining — $\mathrm{Var}\bigl(\|\Delta W\|_\infty\bigr)$

| Tensor type | AdamW LoRA | Muon LoRA | AdamW Full FT | Muon Full FT |
|---|---:|---:|---:|---:|
| Attention Q proj | **7.921e-06** | 4.947e-05 | **1.751e-06** | 3.629e-05 |
| Attention KV proj | **5.942e-05** | 2.959e-04 | **2.337e-06** | 5.086e-05 |
| Attention O proj | 6.136e-06 | **3.486e-06** | **1.561e-06** | 6.295e-06 |
| MLP / MoE experts | **1.142e-05** | 4.810e-05 | **5.190e-06** | 6.565e-06 |
| LayerNorm / RMSNorm | 0 | 0 | **8.770e-07** | 1.011e-06 |

## Muon pretraining — $\mathrm{Var}\bigl(\|\Delta W\|_{\mathrm{RMS}\rightarrow\mathrm{RMS}}\bigr)$

| Tensor type | AdamW LoRA | Muon LoRA | AdamW Full FT | Muon Full FT |
|---|---:|---:|---:|---:|
| Attention Q proj | 0.2841 | **0.06155** | 0.02111 | **2.313e-04** |
| Attention KV proj | 0.1796 | **0.149** | 0.1808 | **0.01474** |
| Attention O proj | **0.08503** | 0.1283 | 0.04684 | **6.139e-04** |
| MLP / MoE experts | 0.5401 | **0.03868** | **0.09063** | 0.1008 |
| LayerNorm / RMSNorm | NaN | NaN | NaN | NaN |

