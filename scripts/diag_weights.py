"""Check that the IndicF5 weights actually land in the model.

model.py does `load_state_dict(state_dict, strict=False, assign=True)`, so a key
mismatch is silent: every DiT tensor would stay randomly initialised and the
sampler would emit NaN. This reports the real key overlap and looks for NaN in
the checkpoint itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import s4_voice  # noqa: E402


def main() -> int:
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    p = hf_hub_download(s4_voice.INDICF5, filename="model.safetensors")
    sd = load_file(p)
    keys = list(sd)
    print(f"checkpoint: {len(keys)} tensors")
    print("first keys:", keys[:4])
    print("last keys :", keys[-4:])
    nan = [k for k, v in sd.items() if torch.isnan(v).any()]
    print(f"tensors containing NaN: {len(nan)} {nan[:5]}")
    total = sum(v.numel() for v in sd.values())
    print(f"total params: {total / 1e6:.1f}M  dtype: {next(iter(sd.values())).dtype}")

    model = s4_voice._load_model("cuda")
    res = model.ema_model.load_state_dict(sd, strict=False, assign=True)
    print(f"missing_keys   : {len(res.missing_keys)} {res.missing_keys[:6]}")
    print(f"unexpected_keys: {len(res.unexpected_keys)} {res.unexpected_keys[:6]}")

    ema = model.ema_model.state_dict()
    print(f"model tensors: {len(ema)}")
    nan_m = [k for k, v in ema.items() if torch.isnan(v).any()]
    print(f"model tensors containing NaN: {len(nan_m)} {nan_m[:5]}")

    # do the loaded weights match what the checkpoint wanted to give?
    shared = [k for k in ema if k in sd and ema[k].shape == sd[k].shape]
    if shared:
        k = shared[len(shared) // 2]
        same = bool(torch.equal(ema[k].float().cpu(), sd[k].float().cpu()))
        print(f"spot-check {k}: identical={same}")
    print(f"overlapping tensors: {len(shared)} of {len(ema)} model / {len(sd)} ckpt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())