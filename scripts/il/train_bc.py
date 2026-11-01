"""Train a behaviour-cloning policy on one or more demo files (pure PyTorch, no simulator needed).

Examples:
    python scripts/il/train_bc.py --data datasets/hover_ppo.h5 --name hover_bc
    python scripts/il/train_bc.py --data datasets/langnav_rgb.h5 --name langnav_rgb_bc --images --epochs 40
    python scripts/il/train_bc.py --data d0.h5 d1_dagger.h5 --name langnav_dagger1     # DAgger aggregation
Closed-loop evaluation: python scripts/rl/evaluate.py --headless --task <task> --algo bc \
    --checkpoint logs/il/<name>/model.pt --controllers policy
Writes logs/il/<name>/{model.pt, train.json}.
"""

import argparse
import json
import os
import time

import torch

from quadsim.il.bc import BCPolicy, train_bc
from quadsim.il.dataset import DemoDataset
from quadsim.il.language import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--data", nargs="+", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--images", action="store_true", help="use the camera images (FiLM-CNN encoder)")
ap.add_argument("--drop", nargs="*", default=[], help="observation blocks to ignore (e.g. landmarks)")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch_size", type=int, default=1024)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--text_encoder", default="", help="frozen pretrained text encoder 'hf_repo[:subfolder]' (local cache "
                "only), e.g. stabilityai/sd-turbo:text_encoder (CLIP ViT-H/14 text tower); default: bag of words")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()


class _Concat:
    def __init__(self, parts):
        self.instructions = [p.instructions[i] for p in parts for i in p.episode.tolist()]  # per transition
        self.obs = torch.cat([p.obs for p in parts])
        self.action = torch.cat([p.action for p in parts])
        imgs = [p.image for p in parts]
        if args.images and any(i is None for i in imgs):
            raise ValueError("--images: some --data files have no camera images")
        self.image = torch.cat(imgs) if all(i is not None for i in imgs) else None
        self.obs_layout, self.attrs = parts[0].obs_layout, parts[0].attrs

    def __len__(self):
        return len(self.action)


def main():
    torch.manual_seed(args.seed)
    tr = _Concat([DemoDataset(f, "train", images=args.images, seed=args.seed) for f in args.data])
    va = _Concat([DemoDataset(f, "val", images=args.images, seed=args.seed) for f in args.data])
    vocab = 0
    if any(n == "tokens" for n, _ in tr.obs_layout):
        vocab = Tokenizer.from_dict(json.loads(tr.attrs["tokenizer"])).vocab_size
    img_shape = tuple(tr.image.shape[1:]) if (args.images and tr.image is not None) else None
    emb_dim, cache = 0, {}
    if args.text_encoder:
        from quadsim.il.language import PretrainedTextEncoder

        name, _, sub = args.text_encoder.partition(":")
        enc = PretrainedTextEncoder(name, sub or None)
        texts = sorted(set(tr.instructions) | set(va.instructions))
        cache = dict(zip(texts, enc.embed(texts)))
        emb_dim = enc.out_dim
        table = torch.stack([cache[t] for t in texts])
        index = {t: i for i, t in enumerate(texts)}
        for d in (tr, va):
            d.text_table = table
            d.text_idx = torch.tensor([index[t] for t in d.instructions], dtype=torch.long)
        print(f"[bc] text encoder {args.text_encoder}: {len(texts)} instructions -> {emb_dim}-d embeddings")
    pol = BCPolicy(tr.obs_layout, tr.action.shape[1], vocab, img_shape, tuple(args.hidden), use_images=args.images,
                   drop_blocks=args.drop, text_emb_dim=emb_dim, text_encoder=args.text_encoder)
    out = os.path.join("logs", "il", args.name)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    hist = train_bc(pol, tr, va, args.epochs, args.batch_size, args.lr, args.device)
    pol.save(os.path.join(out, "model.pt"), {"data": args.data, "args": vars(args), "text_cache": cache})
    rep = {"data": args.data, "train_transitions": len(tr), "val_transitions": len(va), "images": args.images,
           "drop": args.drop, "parameters": sum(p.numel() for p in pol.parameters()), "wall_s": time.time() - t0,
           "final_train_mse": hist["train"][-1], "final_val_mse": hist["val"][-1] if hist["val"] else None,
           "history": hist}
    with open(os.path.join(out, "train.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(f"[bc] saved {out}/model.pt  train {rep['final_train_mse']:.5f}  val {rep['final_val_mse']}")


if __name__ == "__main__":
    main()
