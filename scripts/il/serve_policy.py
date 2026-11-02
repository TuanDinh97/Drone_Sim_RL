"""Serve a quadsim behaviour-cloning checkpoint over the openpi policy-server protocol (WebSocket + msgpack).

Usage: python scripts/il/serve_policy.py --checkpoint logs/il/<name>/model.pt [--port 8000] [--device cuda:0]
Clients: ``evaluate.py --algo vla --checkpoint quadsim.il.remote:make_adapter`` (``QUADSIM_POLICY_SERVER=host:port``,
``QUADSIM_REMOTE_BATCHED=1`` to send the whole batch per step) or any openpi ``WebsocketClientPolicy`` sending
``observation/policy`` (+ ``observation/image``, ``prompt``). No simulator needed; see ``quadsim.il.remote``.
"""

import argparse

from quadsim.il.remote import bc_policy_fn, make_server

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--host", default="0.0.0.0")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--device", default="cpu")
args = ap.parse_args()

policy_fn, meta = bc_policy_fn(args.checkpoint, args.device)
with make_server(policy_fn, meta, args.host, args.port) as server:
    print(f"[serve] {meta['policy']} {args.checkpoint} on ws://{args.host}:{args.port}", flush=True)
    server.serve_forever()
