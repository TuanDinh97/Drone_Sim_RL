"""Ablation plugin: the bag-of-words instruction encoder *averages unknown words in* (as the UNK embedding) instead
of ignoring them. Evaluates the effect of the masking with the same checkpoint (BC is trained only on training
phrasings, which contain no unknown word, so training is unaffected).

    python scripts/rl/evaluate.py --headless --task Quadsim-LangNav-Heldout-v0 --controllers policy --algo bc \\
        --checkpoint logs/il/langnav_sym_bc_v2/model.pt --plugins scripts/validation/ablations/bow_unknown_words_averaged.py
"""

from quadsim.il import language


def _forward_unknown_averaged(self, tokens):
    tokens = tokens.long()
    mask = (tokens != language.PAD).float()[..., None]  # padding only: unknown words count like known ones
    e = (self.emb(tokens) * mask).sum(1) / mask.sum(1).clamp_min(1.0)
    return self.mlp(e)


language.BagOfWordsEncoder.forward = _forward_unknown_averaged
