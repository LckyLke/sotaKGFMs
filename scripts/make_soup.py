"""Checkpoint soup: average sibling checkpoints into one model (host-side).

Every INCITE run peaked mid-training and decayed at constant lr; the saved
best checkpoints of the four floor-descendants (phase-1 best, and the
2.1b / 2.2 / 2.3 lever bests, all warm-started FROM the phase-1 best) are
fine-tunes of one parent -- the classic validity condition for model soups
(Wortsman et al., 2022). This script averages the tensors every donor
shares with the plain floor architecture and saves a floor-config
checkpoint; lever-specific tensors (walk_module.*, support readout,
relation head extras) are dropped, so the soup evaluates with
configs/incite_phase1.yaml.

Runs on the HOST (torch not needed -- pure tensor averaging via torch? No:
host has no torch). So: run INSIDE the container via docker_run. Usage:
  python scripts/make_soup.py OUT.pth DONOR1.pth DONOR2.pth [...]
The first donor defines the reference key set: keys present in ALL donors
are averaged; keys missing from any donor are taken from the first donor
verbatim (and listed). The step/dev metadata records the recipe.
"""
import sys

import torch


def main():
    out, donors = sys.argv[1], sys.argv[2:]
    assert len(donors) >= 2, "a soup needs at least two donors"
    states = [torch.load(p, map_location="cpu", weights_only=False) for p in donors]
    models = [s["model"] for s in states]
    ref = models[0]
    shared = set(ref)
    for m in models[1:]:
        shared &= set(m)
    averaged, kept = {}, []
    for k in ref:
        if k in shared:
            averaged[k] = torch.stack([m[k].float() for m in models]).mean(0).to(ref[k].dtype)
        else:
            averaged[k] = ref[k]
            kept.append(k)
    torch.save({"model": averaged,
                "step": -1,
                "soup": {"donors": donors, "averaged": len(shared),
                         "kept_from_first": kept},
                "seed": states[0].get("seed")}, out)
    print("soup: %d averaged, %d kept from %s -> %s"
          % (len(shared), len(kept), donors[0], out))
    for k in kept:
        print("  kept:", k)


if __name__ == "__main__":
    main()
