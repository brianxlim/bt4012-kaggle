"""LimiX-2 runner.  python run_limix.py <first_day> <context_rows> <seed> <out_csv> [device]

Env: LIMIX_N_PIPE (use only first N of the 32 default ensemble pipelines; default all), FM_CHUNK (query rows per
     predict call, default 4000 - context is re-encoded on every call, so keep chunks large),
     LIMIX_SRC (default C:\\Users\\brixn\\Documents\\BT4012\\limix-src), LIMIX_REPO (default stable-ai/LimiX-2),
     plus the FM_* vars documented in fm_common.py.
LimiX-2 has no hard feature/row cap in code (feature-group encoder), so all 165 features are used by default.
"""
import json
import os
import sys
import tempfile
import typing
from pathlib import Path

import fm_common as fc

first_day, context_rows, seed, out_csv, device = fc.parse_cli(sys.argv)

import numpy as np
import torch

if not hasattr(typing, "override"):  # LimiX imports typing.override (3.12+); venv .pth shim does the same
    import typing_extensions
    typing.override = typing_extensions.override

os.environ.update(RANK="0", WORLD_SIZE="1", MASTER_ADDR="127.0.0.1", MASTER_PORT="29500")
from huggingface_hub import hf_hub_download

import inference.v2_0.predictor as p2

_orig_init = p2.LimiXPredictor.CacheManager.__init__  # upstream hardcodes /mnt/public/... -> would create C:\mnt on Windows
_cache_dir = str(Path(tempfile.gettempdir()) / "limix_cache_unused")
p2.LimiXPredictor.CacheManager.__init__ = lambda self, cache_dir=None: _orig_init(self, cache_dir=_cache_dir)
from inference.predictor import LimiXPredictor

src = Path(os.environ.get("LIMIX_SRC", r"C:\Users\brixn\Documents\BT4012\limix-src"))
cfg = json.loads((src / "config" / "cls_default_noretrieval_v2.json").read_text())
n_pipe = int(os.environ.get("LIMIX_N_PIPE", str(len(cfg["pipelines"]))))
cfg["pipelines"] = cfg["pipelines"][:n_pipe]
chunk = int(os.environ.get("FM_CHUNK", "4000"))
dev = torch.device(device)
fc.log(f"LimiX-2 pipelines={n_pipe} device={dev} torch={torch.__version__} chunk={chunk}")

xc, yc, xt, test_index, feats = fc.load_problem(first_day, context_rows, seed)

ckpt = hf_hub_download(repo_id=os.environ.get("LIMIX_REPO", "stable-ai/LimiX-2"), filename="LimiX-2.ckpt")
clf = LimiXPredictor(device=dev, model_path=ckpt, inference_config=cfg, seed=seed)


def predict_fn(x):
    pr = np.asarray(clf.predict(xc, yc, x, task_type="Classification"))
    return pr[:, 1]  # classes are 0/1 in order -> column 1 = P(illicit)


fc.predict_in_chunks(predict_fn, xt, chunk, out_csv, test_index)
