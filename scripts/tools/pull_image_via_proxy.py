"""Pull a public container image through the user's HTTP(S) proxy and load it into Docker.

Why: the Docker daemon does not inherit the shell's HTTPS_PROXY, and NGC (nvcr.io) refuses direct
connections from some networks (HTTP 403). This tool talks to the registry v2 API with the shell's
proxy settings, verifies every blob against its sha256 digest, writes a ``docker load`` archive and
loads it. No daemon or system configuration is changed.

Usage: python scripts/tools/pull_image_via_proxy.py nvcr.io/nvidia/isaac-lab:2.3.2 [--workdir DIR] [--keep]
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.request

ACCEPT = ", ".join([
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
])


def get(url, token=None, accept=None):
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if accept:
        req.add_header("Accept", accept)
    return urllib.request.urlopen(req, timeout=120)


def token_for(registry, repo):
    if registry == "nvcr.io":
        url = f"https://nvcr.io/proxy_auth?scope=repository:{repo}:pull"
    else:  # Docker Hub
        url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
    return json.load(get(url))["token"]


def download_blob(base, repo, digest, token, path):
    if os.path.exists(path) and _sha256(path) == digest.split(":")[1]:
        return
    for attempt in range(5):
        try:
            h = hashlib.sha256()
            with get(f"{base}/v2/{repo}/blobs/{digest}", token) as r, open(path + ".part", "wb") as f:
                while chunk := r.read(1 << 22):
                    f.write(chunk)
                    h.update(chunk)
            if h.hexdigest() != digest.split(":")[1]:
                raise ValueError(f"digest mismatch for {digest}")
            os.replace(path + ".part", path)
            return
        except Exception as e:  # retry transient network errors (token may also expire)
            print(f"  retry {attempt + 1} for {digest[:19]}: {e}", flush=True)
            time.sleep(3)
            token = token_for(base.split("//")[1], repo)
    raise RuntimeError(f"failed to download {digest}")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--workdir", default="outputs/image_pull")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    ref = a.image
    registry, rest = ref.split("/", 1) if "." in ref.split("/")[0] else ("registry-1.docker.io", ref)
    repo, tag = rest.rsplit(":", 1)
    base = f"https://{registry}"
    tok = token_for(registry if registry == "nvcr.io" else "docker.io", repo)
    idx = json.load(get(f"{base}/v2/{repo}/manifests/{tag}", tok, ACCEPT))
    if "manifests" in idx:
        digest = next(m["digest"] for m in idx["manifests"]
                      if m["platform"]["architecture"] == "amd64" and m["platform"]["os"] == "linux")
        man = json.load(get(f"{base}/v2/{repo}/manifests/{digest}", tok, ACCEPT))
    else:
        man = idx
    os.makedirs(a.workdir, exist_ok=True)
    total = sum(layer["size"] for layer in man["layers"])
    print(f"[pull] {ref}: {len(man['layers'])} layers, {total / 1e9:.2f} GB", flush=True)
    cfg_digest = man["config"]["digest"]
    cfg_path = os.path.join(a.workdir, cfg_digest.split(":")[1] + ".json")
    download_blob(base, repo, cfg_digest, tok, cfg_path)
    layer_files = []
    done = 0
    t0 = time.time()
    for i, layer in enumerate(man["layers"]):
        p = os.path.join(a.workdir, layer["digest"].split(":")[1] + ".tar.gz")
        download_blob(base, repo, layer["digest"], tok, p)
        done += layer["size"]
        layer_files.append(p)
        print(f"[pull] layer {i + 1}/{len(man['layers'])} ok  {done / 1e9:.2f}/{total / 1e9:.2f} GB "
              f"({done / 1e6 / max(time.time() - t0, 1e-3):.1f} MB/s)", flush=True)
    archive = os.path.join(a.workdir, "image.tar")
    with tarfile.open(archive, "w") as tar:
        tar.add(cfg_path, arcname=os.path.basename(cfg_path))
        names = []
        for p in layer_files:
            name = os.path.basename(p).split(".")[0] + "/layer.tar"
            tar.add(p, arcname=name)
            names.append(name)
        mf = json.dumps([{"Config": os.path.basename(cfg_path), "RepoTags": [ref], "Layers": names}]).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(mf)
        import io

        tar.addfile(info, io.BytesIO(mf))
    print("[pull] docker load ...", flush=True)
    rc = subprocess.call(["docker", "load", "-i", archive])
    if rc == 0 and not a.keep:
        for p in set(layer_files + [cfg_path, archive]):  # identical layers may repeat in a manifest
            if os.path.exists(p):
                os.remove(p)
    sys.exit(rc)


if __name__ == "__main__":
    main()
