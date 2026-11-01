"""Instruction tokenization and lightweight text encoders for language-conditioned policies.

* :class:`Tokenizer` - deterministic word-level vocabulary (fixed at construction, so token ids are stable across
  runs and can be part of an environment observation). Unknown words map to ``<unk>``, sequences are padded with 0.
* :class:`BagOfWordsEncoder` - learned word embeddings, masked mean pooling, MLP (trained with the policy).
* :class:`PretrainedTextEncoder` - optional frozen Hugging Face text tower (``transformers``), loaded with
  ``local_files_only`` by default so that nothing is downloaded implicitly; embeddings of the (few) distinct
  instructions are cached.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn

PAD, UNK = 0, 1
_BASE_WORDS = ("fly", "to", "the", "go", "one", "navigate", "head", "for", "approach", "move", "toward", "reach",
               "please", "pillar", "post", "marker", "column")


class Tokenizer:
    def __init__(self, words, max_tokens: int = 12):
        self.words = ["<pad>", "<unk>"] + [w for w in dict.fromkeys(words) if w not in ("<pad>", "<unk>")]
        self.index = {w: i for i, w in enumerate(self.words)}
        self.max_tokens = max_tokens

    @classmethod
    def default(cls, colors, max_tokens: int = 12):
        """Vocabulary of the LangNav *training* instruction templates plus the colour words."""
        return cls(list(_BASE_WORDS) + list(colors), max_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.words)

    @staticmethod
    def split(text: str) -> list[str]:
        return re.findall(r"[a-z]+", text.lower())

    def encode(self, text: str) -> list[int]:
        ids = [self.index.get(w, UNK) for w in self.split(text)][: self.max_tokens]
        return ids + [PAD] * (self.max_tokens - len(ids))

    def encode_batch(self, texts, device="cpu") -> torch.Tensor:
        return torch.tensor([self.encode(t) for t in texts], dtype=torch.long, device=device)

    def to_dict(self) -> dict:
        return {"words": self.words, "max_tokens": self.max_tokens}

    @classmethod
    def from_dict(cls, d) -> Tokenizer:
        t = cls([], d["max_tokens"])
        t.words = list(d["words"])
        t.index = {w: i for i, w in enumerate(t.words)}
        return t


class BagOfWordsEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 32, out_dim: int = 32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.mlp = nn.Sequential(nn.Linear(dim, out_dim), nn.ELU())
        self.out_dim = out_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.long()
        mask = (tokens > UNK).float()[..., None]  # padding and out-of-vocabulary words carry no information
        e = (self.emb(tokens) * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return self.mlp(e)


class PretrainedTextEncoder(nn.Module):
    """Frozen pretrained text tower with an embedding cache keyed by the instruction string.

    ``PretrainedTextEncoder("sentence-transformers/all-MiniLM-L6-v2")`` or a CLIP text model
    (``subfolder="text_encoder"`` for diffusers repositories). The model must already be in the local Hugging Face
    cache unless ``local_files_only=False`` is passed explicitly."""

    def __init__(self, name: str, subfolder: str | None = None, local_files_only: bool = True, device="cpu"):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        kw = {"local_files_only": local_files_only}
        sub = {"subfolder": subfolder} if subfolder else {}
        tok_sub = {"subfolder": subfolder.replace("text_encoder", "tokenizer")} if subfolder else {}
        self.tok = AutoTokenizer.from_pretrained(name, **kw, **tok_sub)
        self.model = AutoModel.from_pretrained(name, **kw, **sub).to(device).eval()
        self.device = device
        self.cache: dict[str, torch.Tensor] = {}
        self.out_dim = int(getattr(self.model.config, "hidden_size", 0) or self.model.config.projection_dim)

    @torch.no_grad()
    def embed(self, texts: list[str]) -> torch.Tensor:
        new = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if new:
            b = self.tok(new, padding=True, return_tensors="pt").to(self.device)
            h = self.model(**b).last_hidden_state
            m = b["attention_mask"][..., None].float()
            for t, e in zip(new, (h * m).sum(1) / m.sum(1)):
                self.cache[t] = e.float().cpu()
        return torch.stack([self.cache[t] for t in texts])
