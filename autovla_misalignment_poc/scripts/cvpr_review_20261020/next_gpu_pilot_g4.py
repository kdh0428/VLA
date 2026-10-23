#!/usr/bin/env python
"""G4 / addendum-1 GPU pilot entry point (PREPARED, NOT RUN). Default: refuses to run.
  --mode token_edit_gain   : prefix-preserving top-k token edits at a runtime step (G4 plan)
  --mode post_cot_matched  : force the 'complex' stem, freeze the natural reasoning prefix across conditions, then apply the
                             B2 matched-prefix cells to the action span (addendum item 1)
Usage (only after approval): CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python next_gpu_pilot_g4.py --mode ... \
    --manifest <json> --out <dir> --enable-new-inference
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["token_edit_gain", "post_cot_matched"], required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--enable-new-inference", action="store_true", default=False)
    a = ap.parse_args()
    if not a.enable_new_inference:
        sys.exit("enable_new_inference=false: this pilot was prepared but not approved; nothing was run.")
    # Implementation outline (uses scripts/_planner.Planner and the B2 cell harness):
    #  1. load manifest (scene tokens, split, seeds); verify input hashes
    #  2. token_edit_gain: keep plan; k_hat = argmax entropy steps 1..5; top-4 alternatives at k_hat; regenerate suffix
    #     post_cot_matched: stem '<think>\nThis is a complex scenario' forced; reasoning decoded once (seed fixed) and frozen;
    #                       B2 cells GT_REF/GT_NSRC/NAT_REF/NAT_NSRC applied to action positions with identical reasoning prefix
    #  3. write records.jsonl with prefix sha, RNG keys, per-step logits summaries; no outcome is read during generation
    raise NotImplementedError("deliberately not implemented beyond the outline until the pilot is approved")


if __name__ == "__main__":
    main()
