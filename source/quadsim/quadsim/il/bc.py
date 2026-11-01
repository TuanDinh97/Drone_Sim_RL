"""Behaviour cloning (and DAgger) policies for state, image and language-conditioned inputs.

:class:`BCPolicy` splits the flat ``policy`` observation according to the dataset's ``obs_layout``: blocks named
``tokens`` are instruction token ids (embedded by a :class:`~quadsim.il.language.BagOfWordsEncoder`), all other blocks
are normalized continuous features. An optional camera image goes through a small CNN whose last feature map is
FiLM-modulated by the instruction embedding (language-conditioned visual features, as in RT-1-style policies). The
fused features feed an MLP with a ``tanh`` output (normalized action).

The policy is registered as algorithm ``bc`` so ``scripts/rl/evaluate.py --algo bc --checkpoint <model.pt>``
evaluates it in closed loop under exactly the same conditions as RL policies and classical controllers.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from quadsim.registry import ALGORITHMS
from quadsim.rl.algorithms import Algorithm

from .language import BagOfWordsEncoder, PretrainedTextEncoder


class FiLMCNN(nn.Module):
    def __init__(self, in_ch: int, cond_dim: int, out_dim: int = 128):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, 2, 2), nn.ELU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ELU(),
            nn.Conv2d(64, 64, 3, 2, 1))
        self.film = nn.Linear(cond_dim, 128) if cond_dim else None
        self.head = nn.Sequential(nn.ELU(), nn.AdaptiveAvgPool2d((3, 4)), nn.Flatten(), nn.Linear(64 * 12, out_dim),
                                  nn.ELU())
        self.out_dim = out_dim

    def forward(self, img, cond=None):
        h = self.convs(img)
        if self.film is not None and cond is not None:
            gamma, beta = self.film(cond).chunk(2, -1)
            h = h * (1 + gamma[..., None, None]) + beta[..., None, None]
        return self.head(h)


class BCPolicy(nn.Module):
    def __init__(self, obs_layout, act_dim: int, vocab_size: int = 0, image_shape=None, hidden=(256, 256),
                 text_dim: int = 32, use_images: bool = True, drop_blocks=(), text_emb_dim: int = 0,
                 text_encoder: str = ""):
        super().__init__()
        self.text_emb_dim, self.text_encoder = int(text_emb_dim), text_encoder
        if self.text_emb_dim:  # instruction given as a (frozen, pretrained) text embedding instead of token ids
            drop_blocks = tuple(drop_blocks) + ("tokens",)
        self.layout = [(str(n), int(d)) for n, d in obs_layout]
        self.drop_blocks = tuple(drop_blocks)
        off, self.cont_idx, self.tok_idx = 0, [], []
        for n, d in self.layout:
            idx = list(range(off, off + d))
            if n == "tokens" and n not in self.drop_blocks:
                self.tok_idx += idx
            elif n not in self.drop_blocks:
                self.cont_idx += idx
            off += d
        self.obs_dim = off
        self.register_buffer("cont_index", torch.tensor(self.cont_idx, dtype=torch.long))
        self.register_buffer("tok_index", torch.tensor(self.tok_idx, dtype=torch.long))
        self.register_buffer("mean", torch.zeros(len(self.cont_idx)))
        self.register_buffer("std", torch.ones(len(self.cont_idx)))
        self.text = BagOfWordsEncoder(vocab_size, text_dim, text_dim) if self.tok_idx and vocab_size else None
        self.text_proj = nn.Sequential(nn.Linear(self.text_emb_dim, text_dim), nn.ELU()) if self.text_emb_dim else None
        t_dim = text_dim if (self.text is not None or self.text_proj is not None) else 0
        self.cnn = FiLMCNN(image_shape[0], t_dim) if (image_shape is not None and use_images) else None
        feat = len(self.cont_idx) + t_dim + (self.cnn.out_dim if self.cnn else 0)
        layers, d = [], feat
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ELU()]
            d = h
        layers.append(nn.Linear(d, act_dim))
        self.mlp = nn.Sequential(*layers)
        self.config = {"obs_layout": self.layout, "act_dim": act_dim, "vocab_size": vocab_size,
                       "image_shape": list(image_shape) if image_shape is not None else None,
                       "hidden": list(hidden), "text_dim": text_dim, "use_images": use_images,
                       "drop_blocks": list(drop_blocks), "text_emb_dim": self.text_emb_dim,
                       "text_encoder": text_encoder}

    def set_normalizer(self, obs: torch.Tensor):
        x = obs[:, self.cont_index]
        self.mean.copy_(x.mean(0))
        self.std.copy_(x.std(0).clamp_min(1e-3))

    def forward(self, obs: torch.Tensor, image: torch.Tensor | None = None,
                text_emb: torch.Tensor | None = None) -> torch.Tensor:
        parts = [(obs[:, self.cont_index] - self.mean) / self.std]
        cond = None
        if self.text_proj is not None:
            cond = self.text_proj(text_emb)
            parts.append(cond)
        elif self.text is not None:
            cond = self.text(obs[:, self.tok_index].round().long())
            parts.append(cond)
        if self.cnn is not None:
            img = image.float() / 255.0 if image.dtype == torch.uint8 else image
            parts.append(self.cnn(img, cond))
        return torch.tanh(self.mlp(torch.cat(parts, -1)))

    def save(self, path: str, extra: dict | None = None):
        torch.save({"algo": "bc", "config": self.config, "state_dict": self.state_dict(), **(extra or {})}, path)

    @classmethod
    def load(cls, path: str, device="cpu") -> BCPolicy:
        ck = torch.load(path, map_location=device, weights_only=False)
        c = ck["config"]
        drop = tuple(b for b in c.get("drop_blocks", ()) if not (c.get("text_emb_dim") and b == "tokens"))
        m = cls(c["obs_layout"], c["act_dim"], c["vocab_size"], c["image_shape"], tuple(c["hidden"]), c["text_dim"],
                c["use_images"], drop, c.get("text_emb_dim", 0), c.get("text_encoder", ""))
        m.load_state_dict(ck["state_dict"])
        m.text_cache = ck.get("text_cache", {})
        return m.to(device).eval()


def train_bc(policy: BCPolicy, train, val=None, epochs: int = 30, batch_size: int = 1024, lr: float = 1e-3,
             device="cuda:0", log=print, weight_decay: float = 1e-5) -> dict:
    """Supervised regression of the expert action (MSE), Adam + cosine schedule; returns the loss history.

    Low-dimensional data live on the GPU; camera frames stay in (pinned) host memory and are transferred per batch,
    so datasets larger than the GPU memory train. Instruction embeddings are a small table indexed per transition
    (``train.text_table`` (I, D), ``train.text_idx`` (N,))."""
    policy.to(device).train()
    policy.set_normalizer(train.obs.to(device))
    opt = torch.optim.AdamW(policy.parameters(), lr, weight_decay=weight_decay)
    n = len(train)
    batch_size = min(batch_size, n)  # a dataset smaller than one batch still trains
    steps = epochs * max(1, n // batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    hist = {"train": [], "val": []}
    obs, act = train.obs.to(device), train.action.to(device)
    img = train.image if (train.image is not None and policy.cnn is not None) else None  # stays on the host
    table = train.text_table.to(device) if policy.text_proj is not None else None
    tidx = train.text_idx.to(device) if policy.text_proj is not None else None
    for ep in range(epochs):
        perm = torch.randperm(n)
        perm_d = perm.to(device)
        tot = 0.0
        for i in range(0, n - batch_size + 1, batch_size):
            b = perm_d[i: i + batch_size]
            im = img[perm[i: i + batch_size]].to(device) if img is not None else None
            pred = policy(obs[b], im, table[tidx[b]] if table is not None else None)
            loss = (pred - act[b]).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        hist["train"].append(tot / max(1, n // batch_size))
        if val is not None and len(val):
            hist["val"].append(evaluate_bc_loss(policy, val, device))
        log(f"[bc] epoch {ep + 1}/{epochs} train {hist['train'][-1]:.5f}"
            + (f" val {hist['val'][-1]:.5f}" if hist["val"] else ""))
    policy.eval()
    return hist


@torch.no_grad()
def evaluate_bc_loss(policy, ds, device, batch_size=4096):
    policy.eval()
    tot, n = 0.0, len(ds)
    for i in range(0, n, batch_size):
        o = ds.obs[i: i + batch_size].to(device)
        im = ds.image[i: i + batch_size].to(device) if (ds.image is not None and policy.cnn is not None) else None
        te = ds.text_table[ds.text_idx[i: i + batch_size]].to(device) if policy.text_proj is not None else None
        tot += (policy(o, im, te) - ds.action[i: i + batch_size].to(device)).square().sum().item()
    policy.train()
    return tot / (n * ds.action.shape[1])


@ALGORITHMS.register("bc")
class _BCAlgorithm(Algorithm):
    """Evaluation / export hook for BC checkpoints (training: ``scripts/il/train_bc.py``; state-only BC policies
    export to ONNX, image / pretrained-text ones do not)."""

    name = "bc"

    def default_cfg(self, task):
        raise NotImplementedError("BC policies are trained with scripts/il/train_bc.py (see docs/TUTORIALS.md §9)")

    def load_policy(self, checkpoint, env, task=None):
        u = env.unwrapped
        pol = BCPolicy.load(checkpoint, str(u.device))
        if pol.text_proj is not None:  # pretrained text embedding of the instruction strings (cached, CPU encoder)
            enc = {}

            def embed(texts):
                missing = [t for t in dict.fromkeys(texts) if t not in pol.text_cache]
                if missing:
                    if "e" not in enc:
                        name, _, sub = pol.text_encoder.partition(":")
                        enc["e"] = PretrainedTextEncoder(name, sub or None)
                    for t, e in zip(missing, enc["e"].embed(missing)):
                        pol.text_cache[t] = e
                return torch.stack([pol.text_cache[t] for t in texts]).to(u.device)

            @torch.no_grad()
            def act(obs):
                return pol(obs["policy"], obs.get("image"), embed(list(u.instructions)))

            act.policy = pol
            return act

        @torch.no_grad()
        def act(obs):
            return pol(obs["policy"], obs.get("image"))

        act.policy = pol
        return act
