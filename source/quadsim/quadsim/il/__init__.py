"""Imitation learning and vision-language-action (VLA) research tools.

  dataset   episode-structured HDF5 demonstrations (state, images, actions, instructions) + loaders
  language  instruction tokenizer, bag-of-words and optional pretrained text encoders
  bc        behaviour cloning / DAgger policies (state, image, language-conditioned; FiLM fusion)
  vla       adapter interface for external VLA / VLM policies
  remote    openpi-compatible policy server / client: evaluate VLA models served from another process or machine

Exports to the LeRobot dataset format: scripts/il/export_lerobot.py (runs in an optional separate environment).
"""
