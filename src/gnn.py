"""Graph neural network over the payment graph.

Each transaction's score is built from its own features and from what its
neighbours and their neighbours look like, passed along the payment links
("message passing"). Links are directed, so incoming and outgoing neighbours
are aggregated separately.

The symmetry rule that cost a submission once: on days 36-49 only labelled
transactions have a feature row; the 46,668 others are bare nodes. On days 1-35
every node has features. So here ONLY labelled transactions carry features, on
both sides: train unlabelled rows are blanked too, and every node gets a flag
saying whether it has features. Train and test graphs then look alike.

Neural nets need scaled inputs (trees do not), so features go through a
quantile transform fitted on training rows, which also clamps the test values
that fall outside the training range.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import QuantileTransformer

from data import FEATURES, load_edges, load_test, load_train


def build_graph():
    """Node table (one row per transaction) and directed edge index."""
    train, test, edges = load_train(), load_test(), load_edges()
    labelled = train[train["label"].notna()]
    nodes = pd.Index(pd.unique(pd.concat([edges.txId1, edges.txId2, train.txId, test.txId])))
    frame = pd.DataFrame(index=nodes)
    frame["label"] = np.nan
    frame.loc[labelled.txId, "label"] = labelled["label"].to_numpy()
    frame["day"] = 0
    frame.loc[train.txId, "day"] = train["time_step"].to_numpy()
    frame.loc[test.txId, "day"] = test["time_step"].to_numpy()
    frame["has_features"] = False
    frame.loc[labelled.txId, "has_features"] = True
    frame.loc[test.txId, "has_features"] = True
    feats = pd.concat([labelled.set_index("txId")[FEATURES], test.set_index("txId")[FEATURES]])
    src = nodes.get_indexer(edges.txId1)
    dst = nodes.get_indexer(edges.txId2)
    return frame, feats, np.stack([src, dst])


def normalised_adjacency(src, dst, n):
    """Row-normalised sparse matrix: row i averages over i's neighbours along src->dst."""
    deg = np.bincount(src, minlength=n).astype("float32")
    values = 1.0 / deg[src]
    index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    return torch.sparse_coo_tensor(index, torch.tensor(values), (n, n), check_invariants=False).coalesce().to_sparse_csr()


class GNN(nn.Module):
    def __init__(self, n_in, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.layers = nn.ModuleList(
            nn.ModuleDict({k: nn.Linear(hidden, hidden) for k in ("self", "out", "in")})
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * (layers + 1), 1)

    def forward(self, x, a_out, a_in):
        h = self.encode(x)
        states = [h]
        for layer, norm in zip(self.layers, self.norms):
            m = layer["self"](h) + layer["out"](torch.sparse.mm(a_out, h)) + layer["in"](torch.sparse.mm(a_in, h))
            h = self.dropout(torch.relu(norm(m))) + h
            states.append(h)
        return self.head(torch.cat(states, 1)).squeeze(1)


class GraphData:
    """Everything a fit needs, built once."""

    def __init__(self, days=None):
        """days: keep only nodes on these days. No edge crosses a day, so a subset
        of days is an exact subgraph and makes each epoch proportionally cheaper."""
        frame, feats, (src, dst) = build_graph()
        if days is not None:
            # a bare node (no feature row) takes its day from its neighbours
            day = frame["day"].to_numpy().copy()
            for _ in range(3):
                missing = day == 0
                if not missing.any():
                    break
                fill = np.zeros(len(day))
                np.maximum.at(fill, src, day[dst])
                np.maximum.at(fill, dst, day[src])
                day[missing] = fill[missing]
            frame["day"] = day
            keep = np.isin(day, list(days))
            new_id = np.full(len(frame), -1)
            new_id[keep] = np.arange(keep.sum())
            ok = keep[src] & keep[dst]
            src, dst = new_id[src[ok]], new_id[dst[ok]]
            frame = frame[keep]
        self.frame, self.n = frame, len(frame)
        self.a_out = normalised_adjacency(src, dst, self.n)
        self.a_in = normalised_adjacency(dst, src, self.n)
        self.feats = feats
        self.has = frame["has_features"].to_numpy()
        self.y = torch.tensor(frame["label"].fillna(0).to_numpy(), dtype=torch.float32)
        self.day = frame["day"].to_numpy()
        self.labelled = frame["label"].notna().to_numpy()
        deg_out = np.bincount(src, minlength=self.n)
        deg_in = np.bincount(dst, minlength=self.n)
        self.degree = np.stack([np.log1p(deg_out), np.log1p(deg_in)], 1).astype("float32")

    def inputs(self, fit_mask):
        """Quantile-scaled features (fitted on fit_mask rows), zero where missing, plus flags."""
        qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=10**9)
        fit_ids = self.frame.index[fit_mask]
        qt.fit(self.feats.loc[fit_ids].to_numpy())
        x = np.zeros((self.n, len(FEATURES)), dtype="float32")
        rows = np.flatnonzero(self.has)
        x[rows] = qt.transform(self.feats.loc[self.frame.index[rows]].to_numpy())
        return torch.tensor(np.hstack([x, self.has[:, None].astype("float32"), self.degree]))


def fit_predict(data, train_mask, epochs=200, seed=0, hidden=64, layers=2, dropout=0.3,
                lr=3e-3, weight_decay=1e-4, pos_weight=7.57, eval_mask=None, eval_every=0):
    """Train full-batch on train_mask nodes; return scores for every node."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = data.inputs(train_mask & data.has)
    model = GNN(x.shape[1], hidden, layers, dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    idx = torch.tensor(np.flatnonzero(train_mask))
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(x, data.a_out, data.a_in)
        loss = loss_fn(logits[idx], data.y[idx])
        loss.backward()
        opt.step()
        if eval_every and epoch % eval_every == 0 and eval_mask is not None:
            from sklearn.metrics import roc_auc_score
            model.eval()
            with torch.no_grad():
                s = model(x, data.a_out, data.a_in).numpy()
            history.append((epoch, roc_auc_score(data.y.numpy()[eval_mask], s[eval_mask])))
    model.eval()
    with torch.no_grad():
        scores = model(x, data.a_out, data.a_in).numpy()
    return scores, history
