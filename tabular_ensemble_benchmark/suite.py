"""Load OpenML-CC18 tasks with their official 10-fold cross-validation splits.

Uses the OpenML REST API directly plus sklearn.datasets.fetch_openml, so no
OpenML client is required. Everything is cached under data/.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

API = "https://www.openml.org/api/v1/json"
CC18_STUDY = 99
CACHE = Path(__file__).resolve().parent.parent / "data"

# Inclusion rule fixed before any results were seen: tasks whose datasets fit
# a 4-CPU budget for 10-fold CV with 49 model fits per fold.
MAX_INSTANCES = 12_000
MAX_FEATURES = 300


def _fetch(url: str, attempts: int = 6) -> bytes:
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError):
            if i == attempts - 1:
                raise
            time.sleep(2 ** (i + 1))


def _get_json(url: str, cache_name: str):
    path = CACHE / "api" / cache_name
    if path.exists():
        return json.loads(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(_fetch(url))
    path.write_text(json.dumps(data))
    return data


def cc18_task_ids() -> list[int]:
    study = _get_json(f"{API}/study/{CC18_STUDY}", "study_99.json")["study"]
    return [int(t) for t in study["tasks"]["task_id"]]


def dataset_qualities() -> pd.DataFrame:
    raw = _get_json(f"{API}/data/list/tag/OpenML-CC18/limit/500", "cc18_datalist.json")
    rows = []
    for d in raw["data"]["dataset"]:
        q = {x["name"]: float(x["value"]) for x in d.get("quality", []) if x.get("value") is not None}
        rows.append({
            "data_id": int(d["did"]), "name": d["name"],
            "n_instances": int(q.get("NumberOfInstances", 0)),
            "n_features": int(q.get("NumberOfFeatures", 0)) - 1,  # minus target
            "n_classes": int(q.get("NumberOfClasses", 0)),
            "n_missing": int(q.get("NumberOfMissingValues", 0)),
            "n_categorical": int(q.get("NumberOfSymbolicFeatures", 0)) - 1,
        })
    return pd.DataFrame(rows).set_index("data_id")


@dataclass
class Task:
    task_id: int
    data_id: int
    name: str
    target: str
    splits_url: str


def load_task_meta(task_id: int) -> Task:
    t = _get_json(f"{API}/task/{task_id}", f"task_{task_id}.json")["task"]
    inputs = {i["name"]: i for i in t["input"]}
    ds = inputs["source_data"]["data_set"]
    est = inputs["estimation_procedure"]["estimation_procedure"]
    return Task(task_id, int(ds["data_set_id"]), t["task_name"].split(": ", 1)[1].split(" (")[0],
                ds["target_feature"], est["data_splits_url"])


def selected_tasks() -> pd.DataFrame:
    """Benchmark tasks, sorted by estimated cost (largest first)."""
    q = dataset_qualities()
    rows = []
    for tid in cc18_task_ids():
        meta = load_task_meta(tid)
        info = q.loc[meta.data_id]
        rows.append({"task_id": tid, "data_id": meta.data_id, "name": info["name"],
                     **{k: info[k] for k in ["n_instances", "n_features", "n_classes",
                                             "n_missing", "n_categorical"]}})
    df = pd.DataFrame(rows)
    df["included"] = (df.n_instances <= MAX_INSTANCES) & (df.n_features <= MAX_FEATURES)
    df["cost"] = df.n_instances * np.sqrt(df.n_features) * np.maximum(df.n_classes, 2)
    return df.sort_values("cost", ascending=False).reset_index(drop=True)


def _load_splits(task: Task) -> pd.DataFrame:
    path = CACHE / "splits" / f"{task.task_id}.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        text = _fetch(task.splits_url).decode()
        body = text[text.lower().index("@data") + 5:].strip().splitlines()
        rows = [ln.split(",") for ln in body if ln and not ln.startswith("%")]
        df = pd.DataFrame(rows, columns=["type", "rowid", "repeat", "fold"])
        df[["rowid", "repeat", "fold"]] = df[["rowid", "repeat", "fold"]].astype(int)
        df.to_csv(path, index=False)
    return pd.read_csv(path)


def load_task(task_id: int):
    """Return (X, y, folds, meta); folds is a list of (train_idx, test_idx)."""
    meta = load_task_meta(task_id)
    X, y = fetch_openml(data_id=meta.data_id, target_column=meta.target,
                        return_X_y=True, as_frame=True, parser="auto",
                        data_home=str(CACHE / "openml"))
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    splits = _load_splits(meta)
    splits = splits[splits["repeat"] == 0]
    folds = []
    for f in sorted(splits["fold"].unique()):
        s = splits[splits["fold"] == f]
        folds.append((np.sort(s.loc[s.type == "TRAIN", "rowid"].to_numpy()),
                      np.sort(s.loc[s.type == "TEST", "rowid"].to_numpy())))
    return X, y, folds, meta
