"""Train failure detectors on extracted uncertainty features (no sklearn — numpy only).

Detectors (labels = row error, i.e. NOT row-EM, from <tag>_rows.jsonl 'correct'):
  1. baselines: single signals evaluated directly on test (min/mean logprob,
     self-consistency, semantic entropy, validator flags)
  2. logistic combiner over scalar features from <tag>_features.jsonl
     (L2, lambda chosen by 5-fold CV on train)
  3. hidden-state probe: L2 logistic on last-completion-token hidden states
     (every recorded layer), layer + lambda chosen by the same CV
  4. stack: probe logit (chosen layer) + raw signals -> logistic
  5. conformal abstention on the best detector: threshold calibrated on train so
     error among accepted rows <= target eps (split-conformal style), report test
     coverage + realized error

Usage:
  python eval/train_detector.py --train-tag llama-3.2-1b-h90-holdout \
      --test-tags llama-3.2-1b-h90-test,llama-1b-merged-test
Outputs eval/results/detector_<train-tag>.json
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "eval" / "results"

FEATS = ["mean_logprob", "min_logprob", "first_val_logprob", "slot_lp_min_overall",
         "slot_lp_min_mean", "slot_lp_mean_min", "null_lp_min", "value_lp_min",
         "se_max", "se_mean", "agree_frac", "n_ungrounded", "n_pred_null",
         "n_slots", "utt_len"]


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUC with tie handling. y=1 means error."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sv = s[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    r1 = ranks[y].sum()
    return float((r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0))


class Logistic:
    """L2-regularized logistic regression, full-batch Adam, numpy only."""

    def __init__(self, lam: float = 1.0, iters: int = 1500, lr: float = 0.05):
        self.lam, self.iters, self.lr = lam, iters, lr

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Z = np.concatenate([(X - self.mu) / self.sd, np.ones((len(X), 1))], axis=1)
        n, d = Z.shape
        w = np.zeros(d)
        m = np.zeros(d)
        v = np.zeros(d)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, self.iters + 1):
            z = Z @ w
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            g = Z.T @ (p - y) / n
            g[:-1] += self.lam / n * w[:-1]
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            mh = m / (1 - b1 ** t)
            vh = v / (1 - b2 ** t)
            w -= self.lr * mh / (np.sqrt(vh) + eps)
        self.w = w
        return self

    def logit(self, X):
        X = np.asarray(X, dtype=np.float64)
        Z = np.concatenate([(X - self.mu) / self.sd, np.ones((len(X), 1))], axis=1)
        return Z @ self.w


def load_tag(tag: str):
    feats = {}
    with open(RES / f"{tag}_features.jsonl") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                feats[d["id"]] = d
    y, ids = {}, []
    with open(RES / f"{tag}_rows.jsonl") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                y[d["id"]] = 0 if d.get("correct") else 1
                ids.append(d["id"])
    ids = [i for i in ids if i in feats]
    X = np.array([[float(feats[i].get(k) if feats[i].get(k) is not None else 0.0)
                   for k in FEATS] for i in ids])
    yy = np.array([y[i] for i in ids])
    hidden = None
    npz_p = RES / f"{tag}_hidden.npz"
    if npz_p.exists():
        z = np.load(npz_p)
        hid_ids = list(z["ids"])
        pos = {h: k for k, h in enumerate(hid_ids)}
        sel = [pos[i] for i in ids if i in pos]
        hidden = {k: z[k][sel].astype(np.float32) for k in z.files if k != "ids"}
    return ids, X, yy, hidden, feats


def cv_auc(make_X, y, lams, k=5, seed=0):
    """5-fold CV mean AUC per lambda; returns (best_lambda, {lam: auc})."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, k)
    out = {}
    for lam in lams:
        sc = np.zeros(len(y))
        for f in folds:
            tr = np.setdiff1d(idx, f)
            m = Logistic(lam=lam).fit(make_X(tr), y[tr])
            sc[f] = m.logit(make_X(f))
        out[lam] = auc(y, sc)
    best = max(out, key=out.get)
    return best, out


def coverage_tables(y, s):
    """Escalate top-x% by score -> (frac, errors caught, false alarms)."""
    tab = []
    order = np.argsort(-s)
    n = len(y)
    for frac in (0.1, 0.2, 0.3, 0.4, 0.5):
        m = max(1, int(round(frac * n)))
        top = order[:m]
        caught = int(y[top].sum())
        tab.append({"escalate_frac": round(frac, 2), "n_rows": m,
                    "errors_total": int(y.sum()), "errors_caught": caught,
                    "error_recall": round(caught / max(1, int(y.sum())), 4),
                    "false_alarms": int(m - caught)})
    return tab


def conformal(y_tr, s_tr, y_te, s_te, targets=(0.05, 0.10, 0.15)):
    """Largest threshold tau on train with error|accepted <= target; apply to test."""
    out = []
    for eps in targets:
        tau = -np.inf
        lo, hi = np.min(s_tr), np.max(s_tr)
        for q in np.linspace(lo, hi, 200):
            acc = s_tr <= q
            if acc.sum() == 0:
                continue
            if y_tr[acc].mean() <= eps:
                tau = q
        acc_te = s_te <= tau if np.isfinite(tau) else np.zeros(len(s_te), bool)
        cov = float(acc_te.mean()) if len(s_te) else 0.0
        err = float(y_te[acc_te].mean()) if acc_te.sum() else float("nan")
        out.append({"target_error": eps, "tau": float(tau) if np.isfinite(tau) else None,
                    "test_coverage": round(cov, 4),
                    "test_error_among_accepted": round(err, 4) if err == err else None,
                    "test_accuracy_among_accepted": round(1 - err, 4) if err == err else None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-tag", required=True)
    ap.add_argument("--test-tags", required=True, help="comma-separated")
    ap.add_argument("--lams", default="0.01,0.1,1,10")
    a = ap.parse_args()

    lams = [float(x) for x in a.lams.split(",")]
    ids_tr, X_tr, y_tr, H_tr, feats_tr = load_tag(a.train_tag)
    report = {"train_tag": a.train_tag, "n_train": len(ids_tr),
              "train_error_rate": round(float(y_tr.mean()), 4), "detectors": {}}

    # ---- 1. baselines evaluated on each test tag (no fitting)
    base_cols = {"minus_min_logprob": "min_logprob", "minus_mean_logprob": "mean_logprob",
                 "one_minus_agree": "agree_frac", "se_max": "se_max",
                 "minus_first_val": "first_val_logprob"}

    # ---- 2. logistic combiner over scalar features
    best_lam, cv = cv_auc(lambda ix: X_tr[ix], y_tr, lams)
    comb = Logistic(lam=best_lam).fit(X_tr, y_tr)
    report["detectors"]["combiner"] = {"cv_best_lambda": best_lam, "cv_auc": cv,
                                       "feature_order": FEATS,
                                       "weights": {k: round(float(w), 4) for k, w
                                                   in zip(FEATS, comb.w[:-1])}}

    # ---- 3. hidden-state probe per layer
    probe_cv = {}
    best_layer, best_layer_auc = None, -1
    if H_tr:
        for li, Hmat in sorted(H_tr.items()):
            bl, cvl = cv_auc(lambda ix, Hm=Hmat: Hm[ix], y_tr, lams)
            probe_cv[li] = {"best_lambda": bl, "cv_auc": round(cvl[bl], 4)}
            if cvl[bl] > best_layer_auc:
                best_layer, best_layer_auc, best_layer_lam = li, cvl[bl], bl
        Hp = H_tr[best_layer]
        probe = Logistic(lam=best_layer_lam).fit(Hp, y_tr)
        report["detectors"]["probe"] = {"layer_cv": probe_cv, "chosen_layer": best_layer,
                                        "chosen_lambda": best_layer_lam}
        # ---- 4. stack: probe logit + scalar features
        Xs = np.concatenate([X_tr, probe.logit(Hp)[:, None]], axis=1)
        bsl, cvs = cv_auc(lambda ix: Xs[ix], y_tr, lams)
        stack = Logistic(lam=bsl).fit(Xs, y_tr)
        report["detectors"]["stack"] = {"best_lambda": bsl, "cv_auc": cvs,
                                        "feature_order": FEATS + ["probe_logit"]}

    # train-set scores per detector (needed for conformal threshold calibration)
    train_scores = {"combiner": comb.logit(X_tr)}
    if H_tr and best_layer is not None:
        train_scores["probe"] = probe.logit(H_tr[best_layer])
        train_scores["stack"] = stack.logit(np.concatenate([X_tr, train_scores["probe"][:, None]], axis=1))

    # ---- evaluate everything on test tags
    per_test = {}
    for tag in a.test_tags.split(","):
        ids, X, y, H, _ = load_tag(tag)
        if len(ids) == 0:
            continue
        r = {"n": len(ids), "error_rate": round(float(y.mean()), 4), "auc": {}, "coverage": {}}
        for name, col in base_cols.items():
            v = X[:, FEATS.index(col)]
            s = -v if name.startswith("minus") or name == "minus_first_val" else (1 - v if name.startswith("one_minus") else v)
            if name.startswith("one_minus"):
                s = 1 - v
            r["auc"][name] = round(auc(y, s), 4)
        s_comb = comb.logit(X)
        r["auc"]["combiner"] = round(auc(y, s_comb), 4)
        r["coverage"]["combiner"] = coverage_tables(y, s_comb)
        entry = {"combiner": s_comb}
        if H and best_layer is not None and best_layer in H:
            s_probe = probe.logit(H[best_layer])
            r["auc"]["probe"] = round(auc(y, s_probe), 4)
            r["coverage"]["probe"] = coverage_tables(y, s_probe)
            entry["probe"] = s_probe
            Xste = np.concatenate([X, s_probe[:, None]], axis=1)
            s_stack = stack.logit(Xste)
            r["auc"]["stack"] = round(auc(y, s_stack), 4)
            r["coverage"]["stack"] = coverage_tables(y, s_stack)
            entry["stack"] = s_stack
        best_name = max(r["auc"], key=lambda k: r["auc"][k] if r["auc"][k] == r["auc"][k] else -1)
        r["best_signal"] = best_name
        s_best_train = train_scores.get(best_name)
        if s_best_train is None:  # raw signal column: reuse train feature matrix
            col = base_cols.get(best_name)
            v = X_tr[:, FEATS.index(col)]
            s_best_train = -v if best_name.startswith("minus") else (1 - v if best_name.startswith("one_minus") else v)
        r["conformal"] = conformal(y_tr, s_best_train, y, entry.get(best_name, s_comb))
        per_test[tag] = r

    report["test"] = per_test
    out = RES / f"detector_{a.train_tag}.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"-> {out}")
    for tag, r in per_test.items():
        print(f"{tag}: " + " ".join(f"{k}={v}" for k, v in r["auc"].items()))


if __name__ == "__main__":
    main()
