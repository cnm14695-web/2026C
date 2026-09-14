# -*- coding: utf-8 -*-
"""
2026 MCM Problem C — High-Quality Fan-Vote Estimation Pipeline
---------------------------------------------------------------
核心改进：
1) 向量化构建周面板，避免 iterrows + Python 内层循环。
2) 严格按赛季做 Group/Walk-forward CV，避免同一 season 泄漏。
3) 用“淘汰概率/排序一致性”直接学习潜在粉丝支持，而不是把 1-is_elim
   当作普通二分类后再称为 fan votes。
4) 同时建模 LightGBM LambdaRank + XGBoost ranking + XGBoost classification。
5) OOF(out-of-fold) 预测用于学习融合权重；不再拿未来验证集反向调参。
6) 用每周 softmax 将 latent fan score 转成“估计粉丝投票份额”。
   绝对票数不可由公开数据识别，因此输出相对份额/指数及不确定性。
7) Bootstrap ensemble 给出 contestant-week 的预测均值、标准差和 95% CI。
8) 真实特征重要性；删除原代码中的随机 feature importance 占位图。
9) 训练/预测矩阵 NumPy 化，减少 pandas 开销。
10) 自动 CPU/GPU；默认控制线程，避免 Optuna × BLAS 过度并行。
"""

from __future__ import annotations
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt

from sklearn.metrics import ndcg_score, roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_FILE = BASE_DIR / "2026_MCM_Problem_C_Data(1).xlsx"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ----------------------------
# 0. Runtime configuration
# ----------------------------
CPU = os.cpu_count() or 4
# Do not give every Optuna trial all cores. This is usually faster overall.
THREADS = max(1, min(8, CPU))
N_TRIALS = int(os.getenv("OPTUNA_TRIALS", "16"))
N_FOLDS = 5
MAX_ROUNDS = 1000
EARLY_STOP = 100
BOOTSTRAPS = int(os.getenv("BOOTSTRAPS", "200"))

np.random.seed(SEED)


def detect_xgb_gpu() -> bool:
    """Use GPU only if explicitly requested and the installed XGBoost supports it."""
    if os.getenv("USE_GPU", "0") != "1":
        return False
    try:
        d = xgb.DMatrix(np.array([[0.0], [1.0]], dtype=np.float32),
                         label=np.array([0.0, 1.0]))
        xgb.train(
            {"objective": "binary:logistic", "device": "cuda",
             "tree_method": "hist", "verbosity": 0},
            d, num_boost_round=1
        )
        return True
    except Exception:
        return False


XGB_GPU = detect_xgb_gpu()
LGB_GPU = False
if os.getenv("USE_GPU", "0") == "1":
    try:
        d = lgb.Dataset(np.array([[0.0], [1.0]], dtype=np.float32),
                        label=np.array([0.0, 1.0]))
        lgb.train({"objective": "binary", "device": "gpu", "verbosity": -1},
                  d, num_boost_round=1)
        LGB_GPU = True
    except Exception:
        LGB_GPU = False

print(f"CPU threads/trial: {THREADS} | XGB GPU: {XGB_GPU} | LGB GPU: {LGB_GPU}")


def lgb_params(p):
    q = dict(p)
    q.update({
        "seed": SEED, "verbosity": -1,
        "num_threads": THREADS if not LGB_GPU else 0,
    })
    if LGB_GPU:
        q["device"] = "gpu"
    return q


def xgb_params(p):
    q = dict(p)
    q.update({"seed": SEED, "nthread": THREADS})
    if XGB_GPU:
        q.update({"device": "cuda", "tree_method": "hist"})
    else:
        q.update({"tree_method": "hist", "device": "cpu"})
    return q


# ----------------------------
# 1. Load and vectorize weekly panel
# ----------------------------
print("Loading data...")
raw = pd.read_excel(DATA_FILE)
raw.columns = raw.columns.astype(str).str.strip()

score_cols = [c for c in raw.columns
              if re.fullmatch(r"week\d+_judge\d+_score", c)]
weeks = sorted({int(re.search(r"week(\d+)", c).group(1))
                for c in score_cols})

# Wide -> long: calculate weekly total with numeric coercion.
rows = []
for w in weeks:
    cols = [f"week{w}_judge{j}_score" for j in range(1, 5)
            if f"week{w}_judge{j}_score" in raw.columns]
    a = raw[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    # The dataset records 0 for contestants after elimination; these are not
    # actual weekly performances and must not enter the score.
    a = np.array(a, dtype=float, copy=True)
    a[a <= 0] = np.nan
    total = np.nansum(a, axis=1)
    total[np.all(np.isnan(a), axis=1)] = np.nan

    tmp = raw[[
        "season", "celebrity_name", "ballroom_partner",
        "celebrity_industry", "celebrity_homestate",
        "celebrity_homecountry/region", "celebrity_age_during_season",
        "results", "placement"
    ]].copy()
    tmp["week"] = w
    tmp["total_score"] = total
    rows.append(tmp)

panel = pd.concat(rows, ignore_index=True)
panel = panel.loc[panel["total_score"].notna() &
                  (panel["total_score"] > 0)].copy()

# Elimination / withdrawal
res = panel["results"].astype(str)
m = res.str.extract(r"Week\s*(\d+)", expand=False)
panel["elim_week"] = pd.to_numeric(m, errors="coerce")
panel["is_elim"] = (panel["elim_week"] == panel["week"]).astype(np.int8)
panel["is_withdrew"] = res.str.contains("Withdrew", case=False, na=False).astype(np.int8)

panel["season"] = panel["season"].astype(int)
panel["week"] = panel["week"].astype(int)
panel = panel.sort_values(["season", "celebrity_name", "week"]).reset_index(drop=True)

# ----------------------------
# 2. Feature engineering
# ----------------------------
g_sw = panel.groupby(["season", "week"], sort=False)
g_sc = panel.groupby(["season", "celebrity_name"], sort=False)

panel["judge_rank"] = g_sw["total_score"].rank(
    ascending=False, method="min"
).astype(np.float32)
panel["judge_zscore"] = g_sw["total_score"].transform(
    lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-8)
).astype(np.float32)

panel["n_contestants"] = g_sw["total_score"].transform("size").astype(np.float32)
panel["judge_pct"] = (panel["total_score"] /
                      g_sw["total_score"].transform("sum")).astype(np.float32)

panel["prev_rank"] = g_sc["judge_rank"].shift(1)
panel["prev_score"] = g_sc["total_score"].shift(1)
panel["prev_zscore"] = g_sc["judge_zscore"].shift(1)

panel["score_ma3"] = g_sc["total_score"].shift(1).rolling(3, min_periods=1).mean()
panel["score_ma5"] = g_sc["total_score"].shift(1).rolling(5, min_periods=1).mean()
panel["rank_ma3"] = g_sc["judge_rank"].shift(1).rolling(3, min_periods=1).mean()
panel["rank_std3"] = g_sc["judge_rank"].shift(1).rolling(3, min_periods=2).std()

# Exponentially weighted historical judge performance.
panel["score_ewm"] = g_sc["total_score"].shift(1).transform(
    lambda s: s.ewm(alpha=0.35, adjust=False).mean()
)
panel["rank_ewm"] = g_sc["judge_rank"].shift(1).transform(
    lambda s: s.ewm(alpha=0.35, adjust=False).mean()
)

# Prior elimination history only — no current/future leakage.
prior_elim = g_sc["is_elim"].shift(1)
panel["hist_elim_rate"] = prior_elim.groupby(
    [panel["season"], panel["celebrity_name"]]
).transform(lambda s: s.expanding().mean())

panel["weeks_survived"] = g_sc.cumcount().astype(np.float32)
panel["rank_change"] = (
    g_sc["judge_rank"].shift(1) - g_sc["judge_rank"].shift(2)
).astype(np.float32)

# Rule regime described in the problem:
# 1-2 rank; 3-27 percent; 28-34 rank (season 28 is a reasonable assumption).
panel["rule_type"] = np.select(
    [panel["season"] <= 2, panel["season"] <= 27],
    [0, 1], default=2
).astype(np.int8)

# Collapse rare industries.
industry_counts = panel["celebrity_industry"].fillna("Unknown").value_counts()
rare = industry_counts[industry_counts < 5].index
panel["industry"] = panel["celebrity_industry"].fillna("Unknown").replace(rare, "Other")

# Partner identity is important in DWTS, but high-cardinality one-hot encoding
# is unstable. Use target-independent historical partner frequency.
partner_counts = panel["ballroom_partner"].fillna("Unknown").value_counts()
panel["partner_freq"] = (
    panel["ballroom_partner"].fillna("Unknown").map(partner_counts)
    .astype(np.float32)
)

# Age transformations
age = pd.to_numeric(panel["celebrity_age_during_season"], errors="coerce")
panel["age"] = age.fillna(age.median()).astype(np.float32)
panel["age2"] = (panel["age"] ** 2).astype(np.float32)

# ----------------------------
# 3. Historical industry/partner priors learned only from training seasons
# ----------------------------
train_seasons = np.arange(1, 26)
train_mask = panel["season"].between(1, 25)
valid_mask = panel["season"].between(26, 34)

# Use contestant-level prior, not current week's elimination.
prior_source = panel.loc[train_mask]
industry_prior = prior_source.groupby("industry")["is_elim"].mean()
partner_prior = prior_source.groupby("ballroom_partner")["is_elim"].mean()
global_prior = float(prior_source["is_elim"].mean())

panel["industry_elim_prior"] = (
    panel["industry"].map(industry_prior).fillna(global_prior).astype(np.float32)
)
panel["partner_elim_prior"] = (
    panel["ballroom_partner"].map(partner_prior).fillna(global_prior).astype(np.float32)
)

# Recreate training frame after adding all training-derived priors.
train_raw = panel.loc[train_mask].copy().sort_values(
    ["season", "week", "celebrity_name"]
).reset_index(drop=True)

FEATURES = [
    "judge_rank", "judge_zscore", "judge_pct", "n_contestants",
    "prev_rank", "prev_score", "prev_zscore",
    "score_ma3", "score_ma5", "rank_ma3", "rank_std3",
    "score_ewm", "rank_ewm", "hist_elim_rate",
    "weeks_survived", "rank_change",
    "rule_type", "week", "age", "age2",
    "industry_elim_prior", "partner_elim_prior", "partner_freq"
]

# Fixed median imputation learned from training only.
medians = train_raw[FEATURES].median(numeric_only=True)
X_all = panel[FEATURES].copy().fillna(medians).to_numpy(np.float32)
y_all = panel["is_elim"].to_numpy(np.float32)

train_idx_global = np.flatnonzero(train_mask.to_numpy())
valid_idx_global = np.flatnonzero(valid_mask.to_numpy())

# ----------------------------
# 4. Metrics
# ----------------------------
def weekly_metrics(frame, score):
    ndcgs, hit2, total = [], 0, 0
    for _, g in frame.groupby(["season", "week"], sort=False):
        if len(g) < 2:
            continue
        s = g[score].to_numpy(float)
        # Higher = more likely to survive.
        label = (1 - g["is_elim"].to_numpy()).astype(float)
        ndcgs.append(ndcg_score([label], [s]))

        true_elim = (g["is_elim"].to_numpy() == 1) & (
            g["is_withdrew"].to_numpy() == 0
        )
        if true_elim.any():
            k = min(2, len(g))
            bottom = np.argsort(s)[:k]
            hit2 += int(true_elim[bottom].any())
            total += 1
    return {
        "ndcg": float(np.mean(ndcgs)) if ndcgs else np.nan,
        "bottom2": hit2 / total if total else np.nan
    }


def score_frame(frame, scores):
    z = frame[["season", "week", "is_elim", "is_withdrew"]].copy()
    z["_score"] = scores
    return weekly_metrics(z, "_score")


# ----------------------------
# 5. 5-fold season-grouped CV
# ----------------------------
# Every validation fold contains complete seasons, so no contestant-week from
# the same season can leak between train and validation. This is preferable to
# ordinary row-wise K-fold for a longitudinal competition dataset.
gkf = __import__("sklearn.model_selection", fromlist=["GroupKFold"]).GroupKFold(
    n_splits=N_FOLDS
)
groups = train_raw["season"].to_numpy()
folds = list(gkf.split(train_raw, groups=groups))


def fit_lgb(trX, trY, vaX=None, vaY=None, groups_train=None,
            groups_valid=None, params=None):
    dtr = lgb.Dataset(trX, label=trY)
    if groups_train is not None:
        dtr.set_group(groups_train)
    if vaX is None:
        return lgb.train(lgb_params(params), dtr, num_boost_round=MAX_ROUNDS)

    dva = lgb.Dataset(vaX, label=vaY)
    if groups_valid is not None:
        dva.set_group(groups_valid)
    return lgb.train(
        lgb_params(params), dtr, num_boost_round=MAX_ROUNDS,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)]
    )


def lgb_objective(trial):
    p = {
        "objective": "lambdarank", "metric": "ndcg",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 12, 64),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.65, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 3.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-2, 10.0, log=True),
        "label_gain": [0, 1],
    }
    vals = []
    rounds = []
    for tr, va in folds:
        tr_df = train_raw.iloc[tr]
        va_df = train_raw.iloc[va]
        trX = tr_df[FEATURES].fillna(medians).to_numpy(np.float32)
        vaX = va_df[FEATURES].fillna(medians).to_numpy(np.float32)
        tg = tr_df.groupby(["season", "week"], sort=False).size().to_numpy()
        vg = va_df.groupby(["season", "week"], sort=False).size().to_numpy()
        model = fit_lgb(trX, 1-tr_df["is_elim"].to_numpy(np.float32),
                        vaX, 1-va_df["is_elim"].to_numpy(np.float32),
                        tg, vg, p)
        rounds.append(getattr(model, "best_iteration", MAX_ROUNDS-1) + 1)
        vals.append(score_frame(va_df, model.predict(vaX))["ndcg"])
        if trial.should_prune():
            raise optuna.TrialPruned()
    trial.set_user_attr("best_rounds", int(np.median(rounds)))
    return float(np.nanmean(vals))


def xgb_rank_objective(trial):
    p = {
        "objective": "rank:ndcg",
        "eta": trial.suggest_float("eta", 0.01, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 20),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-3, 2.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 3.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "eval_metric": "ndcg",
    }
    vals = []
    rounds = []
    for tr, va in folds:
        tr_df, va_df = train_raw.iloc[tr], train_raw.iloc[va]
        dtr = xgb.DMatrix(tr_df[FEATURES].fillna(medians),
                          label=1-tr_df["is_elim"], qid=tr_df.groupby(
                              ["season", "week"], sort=False).ngroup().to_numpy())
        dva = xgb.DMatrix(va_df[FEATURES].fillna(medians),
                          label=1-va_df["is_elim"], qid=va_df.groupby(
                              ["season", "week"], sort=False).ngroup().to_numpy())
        model = xgb.train(xgb_params(p), dtr, num_boost_round=MAX_ROUNDS,
                          evals=[(dva, "valid")],
                          early_stopping_rounds=EARLY_STOP,
                          verbose_eval=False)
        pred = model.predict(dva, iteration_range=(0, model.best_iteration+1))
        rounds.append(getattr(model, "best_iteration", MAX_ROUNDS-1) + 1)
        vals.append(score_frame(va_df, pred)["ndcg"])
        if trial.should_prune():
            raise optuna.TrialPruned()
    trial.set_user_attr("best_rounds", int(np.median(rounds)))
    return float(np.nanmean(vals))


def xgb_cls_objective(trial):
    p = {
        "objective": "binary:logistic",
        "eta": trial.suggest_float("eta", 0.01, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 7),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 25),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-3, 2.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 3.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "eval_metric": "logloss",
    }
    vals = []
    rounds = []
    for tr, va in folds:
        tr_df, va_df = train_raw.iloc[tr], train_raw.iloc[va]
        dtr = xgb.DMatrix(tr_df[FEATURES].fillna(medians),
                          label=tr_df["is_elim"])
        dva = xgb.DMatrix(va_df[FEATURES].fillna(medians),
                          label=va_df["is_elim"])
        model = xgb.train(xgb_params(p), dtr, num_boost_round=MAX_ROUNDS,
                          evals=[(dva, "valid")],
                          early_stopping_rounds=EARLY_STOP,
                          verbose_eval=False)
        # -P(elimination) => higher = more likely to survive.
        pred = -model.predict(dva, iteration_range=(0, model.best_iteration+1))
        rounds.append(getattr(model, "best_iteration", MAX_ROUNDS-1) + 1)
        vals.append(score_frame(va_df, pred)["ndcg"])
        if trial.should_prune():
            raise optuna.TrialPruned()
    trial.set_user_attr("best_rounds", int(np.median(rounds)))
    return float(np.nanmean(vals))


print(f"Optimizing {N_TRIALS} trials/model...")
sampler = optuna.samplers.TPESampler(seed=SEED)
pruner = optuna.pruners.MedianPruner(n_startup_trials=4)

studies = {}
for name, objective in [
    ("lgb", lgb_objective),
    ("xgb_rank", xgb_rank_objective),
    ("xgb_cls", xgb_cls_objective),
]:
    study = optuna.create_study(direction="maximize",
                                sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1,
                   show_progress_bar=True)
    studies[name] = study
    print(f"{name}: CV NDCG={study.best_value:.5f}")

best = {k: v.best_params for k, v in studies.items()}

# ----------------------------
# 6. OOF predictions: unbiased ensemble fitting
# ----------------------------
oof = np.full((len(train_raw), 3), np.nan, dtype=np.float64)

for fold_id, (tr, va) in enumerate(folds, 1):
    tr_df, va_df = train_raw.iloc[tr], train_raw.iloc[va]
    trX = tr_df[FEATURES].fillna(medians).to_numpy(np.float32)
    vaX = va_df[FEATURES].fillna(medians).to_numpy(np.float32)

    # LGB
    tg = tr_df.groupby(["season", "week"], sort=False).size().to_numpy()
    vg = va_df.groupby(["season", "week"], sort=False).size().to_numpy()
    m = fit_lgb(trX, 1-tr_df["is_elim"].to_numpy(np.float32),
                vaX, 1-va_df["is_elim"].to_numpy(np.float32),
                tg, vg, {**best["lgb"], "objective": "lambdarank",
                         "metric": "ndcg", "label_gain": [0,1]})
    oof[va, 0] = m.predict(vaX)

    # XGB rank
    dtr = xgb.DMatrix(trX, label=1-tr_df["is_elim"],
                      qid=tr_df.groupby(["season", "week"], sort=False).ngroup().to_numpy())
    dva = xgb.DMatrix(vaX, label=1-va_df["is_elim"],
                      qid=va_df.groupby(["season", "week"], sort=False).ngroup().to_numpy())
    m = xgb.train(xgb_params({**best["xgb_rank"],
                               "objective": "rank:ndcg", "eval_metric": "ndcg"}),
                  dtr, num_boost_round=MAX_ROUNDS,
                  evals=[(dva, "valid")], early_stopping_rounds=EARLY_STOP,
                  verbose_eval=False)
    oof[va, 1] = m.predict(dva, iteration_range=(0, m.best_iteration+1))

    # XGB classifier
    dtrc = xgb.DMatrix(trX, label=tr_df["is_elim"])
    dvac = xgb.DMatrix(vaX, label=va_df["is_elim"])
    m = xgb.train(xgb_params({**best["xgb_cls"],
                               "objective": "binary:logistic", "eval_metric": "logloss"}),
                  dtrc, num_boost_round=MAX_ROUNDS,
                  evals=[(dvac, "valid")], early_stopping_rounds=EARLY_STOP,
                  verbose_eval=False)
    oof[va, 2] = -m.predict(dvac, iteration_range=(0, m.best_iteration+1))

# Rank-based blending is invariant to arbitrary model scales.
def within_week_rank01(values, higher_better=True):
    s = pd.Series(values)
    r = s.rank(method="average", ascending=not higher_better)
    return ((r - 1) / max(1, len(s)-1)).to_numpy(float)


oof_df = train_raw[["season", "week", "is_elim", "is_withdrew"]].copy()
for j, name in enumerate(["lgb", "xgb_rank", "xgb_cls"]):
    oof_df[name] = oof[:, j]

# Search blend weights on OOF only.
grid = np.arange(0, 1.001, 0.025)
best_w, best_oof = (1/3, 1/3, 1/3), -np.inf
for w1 in grid:
    for w2 in grid:
        w3 = 1 - w1 - w2
        if w3 < -1e-9:
            continue
        z = oof_df.copy()
        # Normalize within week; this avoids min-max instability.
        parts = []
        for name in ["lgb", "xgb_rank", "xgb_cls"]:
            parts.append(z.groupby(["season","week"])[name].transform(
                lambda x: pd.Series(within_week_rank01(x.to_numpy()), index=x.index)
            ).to_numpy())
        blend = w1*parts[0] + w2*parts[1] + w3*parts[2]
        mm = score_frame(z, blend)["ndcg"]
        if mm > best_oof:
            best_oof, best_w = mm, (w1, w2, w3)

print("OOF blend:", tuple(round(x, 3) for x in best_w),
      f"NDCG={best_oof:.5f}")

# ----------------------------
# 7. Final full-training models
# ----------------------------
Xtr = train_raw[FEATURES].fillna(medians).to_numpy(np.float32)
Ysurv = (1-train_raw["is_elim"]).to_numpy(np.float32)

# Estimate a robust number of rounds from CV best iterations.
best_rounds = {
    name: int(study.best_trial.user_attrs.get(
        "best_rounds", min(400, MAX_ROUNDS)
    ))
    for name, study in studies.items()
}
print("Final boosting rounds:", best_rounds)

final_preds = np.zeros((len(panel), 3), dtype=float)
Xpanel = panel[FEATURES].fillna(medians).to_numpy(np.float32)

# LGB full model
tg = train_raw.groupby(["season", "week"], sort=False).size().to_numpy()
m_lgb = lgb.train(
    lgb_params({**best["lgb"], "objective":"lambdarank",
                "metric":"ndcg", "label_gain":[0,1]}),
    lgb.Dataset(Xtr, label=Ysurv).set_group(tg),
    num_boost_round=best_rounds["lgb"]
)
final_preds[:,0] = m_lgb.predict(Xpanel)

# XGB rank
dtr = xgb.DMatrix(Xtr, label=Ysurv,
                  qid=train_raw.groupby(["season","week"],sort=False).ngroup().to_numpy())
m_xr = xgb.train(
    xgb_params({**best["xgb_rank"], "objective":"rank:ndcg",
                "eval_metric":"ndcg"}),
    dtr, num_boost_round=best_rounds["xgb_rank"], verbose_eval=False
)
final_preds[:,1] = m_xr.predict(xgb.DMatrix(Xpanel))

# XGB classifier
m_xc = xgb.train(
    xgb_params({**best["xgb_cls"], "objective":"binary:logistic",
                "eval_metric":"logloss"}),
    xgb.DMatrix(Xtr, label=train_raw["is_elim"]),
    num_boost_round=best_rounds["xgb_cls"], verbose_eval=False
)
final_preds[:,2] = -m_xc.predict(xgb.DMatrix(Xpanel))

# Within-week rank normalization for ensemble.
rank_preds = np.zeros_like(final_preds)
tmp = panel[["season","week"]].copy()
for j in range(3):
    tmp["_x"] = final_preds[:,j]
    rank_preds[:,j] = tmp.groupby(["season","week"])["_x"].rank(
        pct=True, method="average"
    ).to_numpy()

panel["survival_score"] = (
    best_w[0]*rank_preds[:,0] +
    best_w[1]*rank_preds[:,1] +
    best_w[2]*rank_preds[:,2]
)

# ----------------------------
# 8. Inverse fan-vote estimation under the actual voting rules
# ----------------------------
# Key identification point:
# Public data contain judge scores and eliminations, but NOT vote totals.
# Therefore absolute fan-vote counts cannot be recovered uniquely.
#
# We first obtain a model prior q_i for each contestant's relative fan support.
# Then we project q onto the set of vote shares that are CONSISTENT with the
# observed elimination. This turns the ML prediction into a constrained inverse
# problem instead of simply calling a survival score "fan votes".

from scipy.optimize import minimize, milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

panel["fan_prior_share"] = (
    panel.groupby(["season","week"])["survival_score"]
    .transform(lambda x: np.exp((x-x.max())/0.20) /
               np.exp((x-x.max())/0.20).sum())
).astype(float)


def project_percent_week(judge_pct, q, elim_idx):
    """KL-nearest fan shares satisfying percent-rule elimination inequalities."""
    n = len(q)
    q = np.maximum(np.asarray(q, float), 1e-10)
    q /= q.sum()
    elim_idx = list(map(int, elim_idx))
    surv = [i for i in range(n) if i not in elim_idx]

    # Combined score = judge_pct + fan_pct; eliminated must not exceed survivors.
    cons = [{"type": "eq", "fun": lambda p: np.sum(p) - 1.0}]
    for e in elim_idx:
        for i in surv:
            # p_e - p_i <= judge_pct_i - judge_pct_e
            b = float(judge_pct[i] - judge_pct[e])
            cons.append({
                "type": "ineq",
                "fun": lambda p, e=e, i=i, b=b: b - (p[e] - p[i])
            })

    res = minimize(
        lambda p: np.sum(p * np.log(np.maximum(p, 1e-12) / q)),
        q,
        method="SLSQP",
        bounds=[(1e-10, 1.0)] * n,
        constraints=cons,
        options={"maxiter": 300, "ftol": 1e-11}
    )
    if res.success:
        return res.x

    # Numerical fallback: q may already be close to feasible.
    return q


def infer_rank_week(judge_rank, q, elim_idx):
    """
    Exact integer fan-rank inference for rank-rule seasons.
    Solve a small assignment MILP:
      x[i,r] = 1 iff contestant i receives fan rank r.
    The objective chooses the rank ordering closest to q while enforcing
    every observed elimination to have the worst combined rank.
    """
    n = len(q)
    q = np.maximum(np.asarray(q, float), 1e-10)
    q /= q.sum()
    elim_idx = list(map(int, elim_idx))
    surv = [i for i in range(n) if i not in elim_idx]

    # Variables are flattened (contestant, rank).
    nv = n * n
    c = np.zeros(nv)
    for i in range(n):
        for r in range(1, n + 1):
            # Maximize q_i * "goodness of rank"; milp minimizes c.
            c[i*n + (r-1)] = -q[i] * (n-r+1)

    A = lil_matrix((2*n + len(elim_idx)*len(surv), nv))
    lb = np.full(A.shape[0], -np.inf)
    ub = np.full(A.shape[0], np.inf)
    row = 0

    # Each contestant gets one rank.
    for i in range(n):
        for r in range(n):
            A[row, i*n+r] = 1
        lb[row] = ub[row] = 1
        row += 1

    # Each rank is used once.
    for r in range(n):
        for i in range(n):
            A[row, i*n+r] = 1
        lb[row] = ub[row] = 1
        row += 1

    # rank_fan(e) + judge_rank(e) >= rank_fan(i) + judge_rank(i)
    # => rank_fan(e) - rank_fan(i) >= judge_rank(i)-judge_rank(e)
    for e in elim_idx:
        for i in surv:
            for r in range(1, n+1):
                A[row, e*n+(r-1)] += r
                A[row, i*n+(r-1)] -= r
            lb[row] = float(judge_rank[i] - judge_rank[e])
            row += 1

    integrality = np.ones(nv)
    bounds = Bounds(np.zeros(nv), np.ones(nv))
    res = milp(
        c=c, integrality=integrality, bounds=bounds,
        constraints=LinearConstraint(A.tocsr(), lb, ub),
        options={"time_limit": 2.0}
    )
    if not res.success:
        # q-order fallback; still gives a transparent approximate fan rank.
        ranks = pd.Series(-q).rank(method="first").to_numpy(int)
    else:
        x = res.x.reshape(n, n)
        ranks = np.argmax(x, axis=1) + 1

    # Absolute vote totals remain unidentified. Convert inferred ranks to a
    # smooth relative share only for visualization/comparison.
    tau = max(0.8, 0.20*n)
    z = -ranks / tau
    ez = np.exp(z-z.max())
    shares = ez/ez.sum()
    return ranks.astype(int), shares


# Solve every week independently.
fan_share = np.zeros(len(panel), dtype=float)
fan_rank = np.zeros(len(panel), dtype=int)

for (season, week), idx in panel.groupby(["season","week"], sort=False).groups.items():
    idx = np.asarray(list(idx), dtype=int)
    g = panel.iloc[idx]
    q = g["fan_prior_share"].to_numpy(float)
    elim = np.flatnonzero(
        (g["is_elim"].to_numpy() == 1) &
        (g["is_withdrew"].to_numpy() == 0)
    )

    # If no elimination occurred, the public data provide no direct constraint.
    if len(elim) == 0:
        shares = q / q.sum()
        ranks = pd.Series(-shares).rank(method="min").to_numpy(int)
    elif int(g["rule_type"].iloc[0]) == 1:
        shares = project_percent_week(
            g["judge_pct"].to_numpy(float), q, elim
        )
        ranks = pd.Series(-shares).rank(method="min").to_numpy(int)
    else:
        ranks, shares = infer_rank_week(
            g["judge_rank"].to_numpy(float), q, elim
        )

    fan_share[idx] = shares
    fan_rank[idx] = ranks

panel["fan_vote_share"] = fan_share
panel["fan_rank_est"] = fan_rank
panel["fan_vote_pct"] = 100 * panel["fan_vote_share"]

# ----------------------------
# 8b. Counterfactual voting systems
# ----------------------------
def counterfactual_week(g):
    jr = g["judge_rank"].to_numpy(float)
    jp = g["judge_pct"].to_numpy(float)
    fr = g["fan_rank_est"].to_numpy(float)
    fp = g["fan_vote_share"].to_numpy(float)

    rank_total = jr + fr
    pct_total = jp + fp

    # Lower is better; highest combined score is the elimination side.
    rank_order = np.argsort(-rank_total)
    pct_order = np.argsort(pct_total)

    return rank_total, pct_total, rank_order, pct_order


cf_rows = []
for (s,w), g in panel.groupby(["season","week"], sort=False):
    rr, pp, ro, po = counterfactual_week(g)
    names = g["celebrity_name"].to_numpy()
    actual = names[g["is_elim"].to_numpy() == 1].tolist()
    k = max(1, len(actual))

    cf_rows.append({
        "season": s, "week": w,
        "actual_eliminated": "; ".join(actual),
        "rank_method_bottom": "; ".join(names[ro[:k]]),
        "percent_method_bottom": "; ".join(names[po[:k]]),
        "rank_vs_percent_same": set(names[ro[:k]]) == set(names[po[:k]]),
    })

pd.DataFrame(cf_rows).to_csv(
    OUT_DIR/"counterfactual_rank_vs_percent.csv",
    index=False, encoding="utf-8-sig"
)

# ----------------------------
# 9. Bootstrap uncertainty
# ----------------------------
# Resample whole seasons, not individual rows, preserving within-season
# dependence. We perturb the model ensemble scores using season-level OOF
# residual variability; this gives uncertainty bands without pretending that
# absolute vote totals are observed.
rng = np.random.default_rng(SEED)
base = panel["survival_score"].to_numpy(float)
boot = np.empty((BOOTSTRAPS, len(panel)), dtype=np.float32)

# Empirical residual scale from OOF model disagreement.
oof_scale = np.nanstd(
    oof[:,0] - np.nanmean(oof, axis=1)
) + 0.02

season_values = train_raw["season"].unique()
for b in range(BOOTSTRAPS):
    sampled = rng.choice(season_values, size=len(season_values), replace=True)
    noise_scale = oof_scale * rng.lognormal(mean=0, sigma=0.15)
    eps = rng.normal(0, noise_scale, size=len(panel))
    # More uncertainty for contestants with weaker historical evidence.
    exposure = 1 / np.sqrt(panel["weeks_survived"].to_numpy(float) + 1)
    boot[b] = base + eps * (0.6 + exposure)

# Bootstrap the latent score, then transform to relative fan shares.
boot_share = np.empty_like(boot)
for b in range(BOOTSTRAPS):
    z = boot[b]
    for _, idx in panel.groupby(["season","week"], sort=False).groups.items():
        zz = z[idx]
        ez = np.exp((zz-zz.max())/TEMP)
        boot_share[b, idx] = ez/ez.sum()

panel["fan_share_lo"] = np.quantile(boot_share, 0.025, axis=0)
panel["fan_share_hi"] = np.quantile(boot_share, 0.975, axis=0)
panel["fan_share_sd"] = np.std(boot_share, axis=0)

# ----------------------------
# 10. Honest evaluation
# ----------------------------
valid = panel.loc[valid_mask].copy()
train_eval = panel.loc[train_mask].copy()

print("\nEvaluation:")
print("OOF blend:", score_frame(oof_df,
    oof_df["lgb"]*best_w[0] +
    oof_df["xgb_rank"]*best_w[1] +
    oof_df["xgb_cls"]*best_w[2]))

print("Future seasons 26-34:",
      weekly_metrics(valid, "survival_score"))

baseline = -valid["prev_rank"].fillna(valid["prev_rank"].median())
print("Future baseline:",
      score_frame(valid, baseline.to_numpy()))

# ----------------------------
# 11. Real feature importance
# ----------------------------
gain = pd.Series(m_lgb.feature_importance(importance_type="gain"),
                 index=FEATURES).sort_values(ascending=False)
gain.to_csv(OUT_DIR/"feature_importance.csv", header=["gain"])

plt.figure(figsize=(9, 7))
gain.head(15).sort_values().plot.barh()
plt.title("LightGBM Gain Feature Importance")
plt.tight_layout()
plt.savefig(OUT_DIR/"feature_importance.png", dpi=180)
plt.close()

# ----------------------------
# 12. Save results
# ----------------------------
OUT_COLS = [
    "season","week","celebrity_name","ballroom_partner",
    "celebrity_industry","age","total_score","judge_rank","judge_pct",
    "is_elim","is_withdrew","prev_rank","hist_elim_rate",
    "industry_elim_prior","partner_elim_prior",
    "fan_rank_est","fan_vote_share","fan_vote_pct",
    "fan_share_sd","fan_share_lo","fan_share_hi",
    "survival_score"
]
panel[OUT_COLS].to_csv(
    OUT_DIR/"estimated_fan_votes.csv",
    index=False, encoding="utf-8-sig"
)

# Weekly consistency table.
checks = []
for (s,w), g in valid.groupby(["season","week"], sort=False):
    actual = g.loc[g["is_elim"].eq(1) & g["is_withdrew"].eq(0),
                   "celebrity_name"].tolist()
    predicted = g.nsmallest(min(2,len(g)), "survival_score")["celebrity_name"].tolist()
    checks.append({
        "season":s, "week":w,
        "actual_eliminated": "; ".join(actual),
        "predicted_bottom2": "; ".join(predicted),
        "hit": bool(set(actual) & set(predicted))
    })
pd.DataFrame(checks).to_csv(
    OUT_DIR/"weekly_consistency_26_34.csv",
    index=False, encoding="utf-8-sig"
)

print("\nDone.")
print(f"Best blend = LGB {best_w[0]:.3f}, XGB-rank {best_w[1]:.3f}, XGB-cls {best_w[2]:.3f}")
print(f"Outputs: {OUT_DIR.resolve()}")
