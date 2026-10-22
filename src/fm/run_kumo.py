"""NVIDIA Kumo Tabular runner.  python run_kumo.py <first_day> <context_rows> <seed> <out_csv> [device]

Env: KUMO_SIZE (small|medium|large, default large), KUMO_ESTIMATORS (default 8), FM_CHUNK (query rows/call, default 1000),
     plus the FM_* vars documented in fm_common.py.
Kumo takes up to 500 numeric columns (recipe SelectColumns(500)), so all 165 features are used by default.
"""
import os
import sys

import fm_common as fc

first_day, context_rows, seed, out_csv, device = fc.parse_cli(sys.argv)

import numpy as np
import pandas as pd
import torch

import sdm

size = os.environ.get("KUMO_SIZE", "large")
n_est = int(os.environ.get("KUMO_ESTIMATORS", "8"))
chunk = int(os.environ.get("FM_CHUNK", "1000"))
dev = torch.device(device)
fc.log(f"Kumo Tabular size={size} estimators={n_est} device={dev} torch={torch.__version__} sdm={sdm.__version__}")

xc, yc, xt, test_index, feats = fc.load_problem(first_day, context_rows, seed)

ctx_df = pd.DataFrame(xc, columns=feats)
ctx_df["target"] = yc
stypes = {c: sdm.Stype.numerical for c in feats}
ctx_tab = sdm.TableTensor.from_pandas(df=ctx_df, stypes={**stypes, "target": sdm.Stype.categorical}, device=dev)
x_ctx = ctx_tab.drop_columns("target")
y_ctx = ctx_tab[:, "target"]

model = sdm.models.KumoTabular(task="classification", size=size, device=dev)
torch.manual_seed(seed)
gen = torch.Generator(device=dev).manual_seed(seed)
amp = torch.amp.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda")


def p_illicit(out):
    names = list(out.columns[sdm.Stype.numerical])  # class order is shuffled per call -> select by name
    return out.numerical.float().cpu().numpy()[:, names.index("1")]


with torch.no_grad(), amp:
    model.fit(x=x_ctx, y=y_ctx, num_estimators=n_est, generator=gen)  # caches context key/values once

    def predict_fn(x):
        df = pd.DataFrame(x, columns=feats)
        tab = sdm.TableTensor.from_pandas(df=df, stypes=stypes, device=dev)
        return p_illicit(model.predict(tab))

    fc.predict_in_chunks(predict_fn, xt, chunk, out_csv, test_index)
