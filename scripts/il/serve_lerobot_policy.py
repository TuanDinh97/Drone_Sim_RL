"""Serve a LeRobot policy (SmolVLA, pi0, ACT, Diffusion Policy, ...) trained on a quadsim LeRobot export over the openpi
policy-server protocol, so that quadsim evaluates it in closed loop like any other controller.

Runs in the LeRobot environment (``bash scripts/il/setup_lerobot_env.sh --policies``); the simulator connects with the
remote adapter:

    third_party/lerobot-venv/bin/python scripts/il/serve_lerobot_policy.py \\
        --policy logs/lerobot/smolvla_langnav/checkpoints/last/pretrained_model --dataset datasets/lerobot/langnav_rgb_v2
    QUADSIM_REMOTE_BATCHED=1 QUADSIM_REMOTE_HORIZON=10 python scripts/rl/evaluate.py --headless --enable_cameras \\
        --task Quadsim-LangNav-RGB-v0 --num_envs 16 --controllers policy --algo vla \\
        --checkpoint quadsim.il.remote:make_adapter

Requests (one observation or a batch): ``observation/policy`` - the quadsim policy observation, from which
``observation.state`` is rebuilt with the export's column selection (``state_columns`` in quadsim_export.json; exports
without it use ``observation/state``) -, ``observation/image`` (uint8 HWC, fed to the dataset's camera feature) and
``prompt`` (the instruction, LeRobot's ``task``). Reply: ``actions`` (N, H, A), the policy's action chunks after its
own post-processing (un-normalization), i.e. in the dataset's action space: normalized quadsim actions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "source", "quadsim"))
from quadsim.il.remote import make_server  # noqa: E402  (numpy / torch / msgpack / websockets only)


def lerobot_policy_fn(policy_path: str, dataset_dir: str, device: str = "cuda", amp: bool = False,
                      max_batch: int = 0, prompt: str | None = None):
    """``(policy_fn, metadata)`` for :func:`quadsim.il.remote.make_server`; ``max_batch`` > 0 splits large requests
    (bounds the GPU memory of image models); ``prompt`` replaces every instruction (language ablation)."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    rep = json.load(open(os.path.join(dataset_dir, "quadsim_export.json")))
    cols = rep.get("state_columns")
    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(policy_path, config=cfg).to(device).eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=policy_path,
                                         preprocessor_overrides={"device_processor": {"device": device}})
    info = json.load(open(os.path.join(dataset_dir, "meta", "info.json")))
    # the dataset's camera keys; a policy fine-tuned with --rename_map renames them in its own pre-processor
    image_keys = [k for k, f in info["features"].items() if f.get("dtype") in ("image", "video")]
    autocast = torch.autocast(device_type="cuda") if amp and device.startswith("cuda") else nullcontext()

    @torch.inference_mode()
    def infer(batch: dict) -> torch.Tensor:
        with autocast:
            batch = pre(batch)
            policy.reset()  # stateless server: every request is a fresh observation
            chunk = policy.predict_action_chunk(batch)
            if chunk.dim() == 2:  # policies without chunks: (N, A)
                chunk = chunk[:, None]
            return post(chunk).float().cpu()

    def policy_fn(obs: dict) -> dict:
        if cols is not None:
            state = np.array(obs["observation/policy"], dtype=np.float32)
            single = state.ndim == 1
            state = np.atleast_2d(state)[:, cols]
        else:
            state = np.array(obs["observation/state"], dtype=np.float32)
            single = state.ndim == 1
            state = np.atleast_2d(state)
        batch = {"observation.state": torch.from_numpy(state)}
        if image_keys:
            img = torch.from_numpy(np.array(obs["observation/image"]))
            img = (img[None] if single else img).permute(0, 3, 1, 2).float() / 255.0  # (N, 3, H, W) in [0, 1]
            batch.update({k: img for k in image_keys})
        task = obs.get("prompt", "") if prompt is None else prompt
        batch["task"] = [task] * len(state) if isinstance(task, str) else list(task)
        n, step = len(state), max_batch if max_batch > 0 else len(state)
        a = torch.cat([infer({k: v[i: i + step] for k, v in batch.items()}) for i in range(0, n, step)]).numpy()
        return {"actions": a[0] if single else a}

    meta = {"policy": f"lerobot-{cfg.type}", "checkpoint": os.path.abspath(policy_path),
            "dataset": os.path.abspath(dataset_dir), "action_dim": int(cfg.output_features["action"].shape[0]),
            "chunk_size": int(getattr(cfg, "chunk_size", 1) or 1),
            "n_action_steps": int(getattr(cfg, "n_action_steps", 1) or 1), "image_keys": image_keys,
            "state_columns": cols, "prompt_override": prompt}
    return policy_fn, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--policy", required=True, help="LeRobot pretrained_model directory (or Hugging Face repo id)")
    ap.add_argument("--dataset", required=True, help="the quadsim LeRobot export the policy was trained on")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", help="CUDA autocast for inference")
    ap.add_argument("--max_batch", type=int, default=0, help="largest batch per forward pass (0: whole request)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the sampling noise of generative policies")
    ap.add_argument("--prompt", default=None, help="replace every instruction (e.g. '' to test language reliance)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    torch.manual_seed(a.seed)  # flow-matching / diffusion policies sample noise: reproducible request sequences
    fn, meta = lerobot_policy_fn(a.policy, a.dataset, a.device, a.amp, a.max_batch, a.prompt)
    meta["seed"] = a.seed
    print(f"[serve] {meta['policy']} on {a.device}, ws://{a.host}:{a.port}, metadata {json.dumps(meta)}", flush=True)
    make_server(fn, meta, a.host, a.port).serve_forever()


if __name__ == "__main__":
    main()
