"""Export SpatialVLA translation-token table on CPU (processor only; no model weights, no inference).
Reproduces scripts/cross_domain_temporal/spatialvla_core.translation_table and adds unclipped xyz / clip flags."""
import os, sys, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from transformers import AutoProcessor
CKPT = "/root/VLA/spatialvla/spatialvla-4b-224-sft-fractal"; UNNORM = "fractal20220817_data/0.1.0"
p = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
tt = p.action_tokenizer.translation_tokenizer
ids = np.arange(tt.token_start_idx, tt.token_end_idx + 1)
norm = tt.decode_token_ids_to_actions(ids)
k = ids - tt.token_start_idx
dt, dp, dr = k // tt.NP, (k % tt.NP) // tt.num_r_bins, k % tt.num_r_bins
th = 0.5 * (tt.theta_bins[dt] + tt.theta_bins[dt + 1]); ph = 0.5 * (tt.phi_bins[dp] + tt.phi_bins[dp + 1]); r = 0.5 * (tt.r_bins[dr] + tt.r_bins[dr + 1])
raw = np.stack(tt.spherical_to_cartesian(th, ph, r), 1)
st = p.statistics[UNNORM]["action"]
lo, hi = np.array(st["q01"])[:3], np.array(st["q99"])[:3]
mask = np.array(st.get("mask", np.ones(len(st["q01"]))), dtype=bool)[:3]
phys = np.where(mask, 0.5 * (norm + 1) * (hi - lo) + lo, norm)
out = sys.argv[1]
np.savez(out, ids=ids, norm=norm, raw=raw, phys=phys, bins=np.stack([dt, dp, dr], 1), lo=lo, hi=hi, mask=mask)
print("tokens", len(ids), "clipped(raw outside [-1,1])", int((np.abs(raw) > 1 + 1e-9).any(1).sum()), "mask", mask, "lo", lo, "hi", hi)
