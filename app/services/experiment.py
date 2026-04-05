import numpy as np
import pandas as pd

# Sklearn metrikas modeļu novērtēšanai: 
# log_loss: mēra prognožu ticamību (jo mazāks, jo labāk) 
# mean_squared_error: Brier score (jo mazāks, jo labāk)
from sklearn.metrics import log_loss, mean_squared_error

# Izslēdz brīdinājumus par matricas reizināšanu
import warnings 
warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)

# Modeļu būvēšana un prognozēšana
from .models import build_logreg_sgd, build_random_forest, build_xgboost_like, fit_and_predict

def run_experiment(df_norm: pd.DataFrame, lottery: str, split_ratio: str = "70_30"):
    # Izpilda eksperimentu ar trim modeļiem (LogReg, RandomForest, XGBoost-like)
    # Izmanto lagged features: prev_vec -> curr_vec
    # Atgriež metrikas un informāciju par treniņu/testu periodiem

    # Nosaka loterijas parametrus
    if lottery == "viking":
        max_num = 48
        k_main = 6
    elif lottery == "euro":
        max_num = 50
        k_main = 5
    else:
        raise ValueError("Nezināms loterijas tips eksperimentam")

    # Sagatavo lagged features (prev_vec un curr_vec)
    df_feat = _prepare_lagged_features(df_norm, max_num=max_num)
    df_feat = df_feat.sort_values("date").reset_index(drop=True)

    n = len(df_feat)
    if n < 10:
        raise ValueError("Nepietiek datu pēc lagged apstrādes (vajag vismaz 10 rindas)")

    # % treniņam, % testam
    split_map = {
        "50_50": 0.50,
        "55_45": 0.55,
        "60_40": 0.60,
        "65_35": 0.65,
        "70_30": 0.70,
        "75_25": 0.75,
        "80_20": 0.80,
    }

    train_ratio = split_map.get(split_ratio, 0.70)

    split_idx = int(n * train_ratio)
    train = df_feat.iloc[:split_idx]
    test = df_feat.iloc[split_idx:]

    # Sagatavo X un Y matricas
    X_train = np.stack(train["prev_vec"].tolist())
    Y_train = np.stack(train["curr_vec"].tolist())
    X_test = np.stack(test["prev_vec"].tolist())
    Y_test = np.stack(test["curr_vec"].tolist())

    # Datumu diapazoni (informatīvi)
    train_date_from = train["date"].min().date().isoformat()
    train_date_to = train["date"].max().date().isoformat()
    test_date_from = test["date"].min().date().isoformat()
    test_date_to = test["date"].max().date().isoformat()

    results = []

    # Modeļu saraksts
    models = [
        ("logreg_sgd", build_logreg_sgd()),
        ("random_forest", build_random_forest()),
        ("xgboost", build_xgboost_like()),
    ]

    # Izpilda katru modeli
    for name, model in models:
        proba = fit_and_predict(model, X_train, Y_train, X_test)

        # Aizsardzība pret 0 un 1 (logloss nevar aprēķināt)
        proba_clipped = np.clip(proba, 1e-6, 1 - 1e-6)

        # Metrikas
        ll = log_loss(Y_test.ravel(), proba_clipped.ravel())
        brier = mean_squared_error(Y_test.ravel(), proba_clipped.ravel())
        hit_k_main = _hit_at_k(Y_test, proba_clipped, k=k_main)
        hit_10 = _hit_at_k(Y_test, proba_clipped, k=10)

        # Rezultātu rinda
        res = {
            "model": name,
            "logloss": float(ll),
            "brier": float(brier),
            "hit_k_main": float(hit_k_main),
            "hit_10": float(hit_10),
            "k_main": int(k_main),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_date_from": train_date_from,
            "train_date_to": train_date_to,
            "test_date_from": test_date_from,
            "test_date_to": test_date_to,
            "split_ratio": split_ratio,
        }
        results.append(res)

    return results

def summarize_experiment_results(results: list[dict]) -> dict | None:
    # Veido kopsavilkumu par labāko rezultātu pašreizējā eksperimenta ietvaros
    # Atgriež:
    # - labāko modeli katrai metrikai
    # - kopējo labāko modeli pēc rangu summas
    # - split informāciju
    # - labākā varianta metriku vērtības

    if not results:
        return None

    # Drošības filtrs: ņem tikai korektas rindas ar visām vajadzīgajām metrikām
    valid_results = []
    for row in results:
        if (
            "model" in row
            and "logloss" in row
            and "brier" in row
            and "hit_k_main" in row
            and "hit_10" in row
        ):
            valid_results.append(row)

    if not valid_results:
        return None

    # Labākie pēc atsevišķām metrikām
    best_logloss = min(valid_results, key=lambda r: r["logloss"])
    best_brier = min(valid_results, key=lambda r: r["brier"])
    best_hit_k_main = max(valid_results, key=lambda r: r["hit_k_main"])
    best_hit_10 = max(valid_results, key=lambda r: r["hit_10"])

    # Rangi katrai metrikai
    ranked_logloss = sorted(valid_results, key=lambda r: r["logloss"])
    ranked_brier = sorted(valid_results, key=lambda r: r["brier"])
    ranked_hit_k_main = sorted(valid_results, key=lambda r: r["hit_k_main"], reverse=True)
    ranked_hit_10 = sorted(valid_results, key=lambda r: r["hit_10"], reverse=True)

    # Summē rangu vietas katram modelim
    rank_sums: dict[str, int] = {}

    for idx, row in enumerate(ranked_logloss, start=1):
        rank_sums[row["model"]] = rank_sums.get(row["model"], 0) + idx

    for idx, row in enumerate(ranked_brier, start=1):
        rank_sums[row["model"]] = rank_sums.get(row["model"], 0) + idx

    for idx, row in enumerate(ranked_hit_k_main, start=1):
        rank_sums[row["model"]] = rank_sums.get(row["model"], 0) + idx

    for idx, row in enumerate(ranked_hit_10, start=1):
        rank_sums[row["model"]] = rank_sums.get(row["model"], 0) + idx

    # Atrod kopējo labāko modeli pēc mazākās rangu summas
    best_overall = min(valid_results, key=lambda r: rank_sums.get(r["model"], 10**9))
    best_model_name = best_overall["model"]

    metric_ranks = {
        "logloss": next(
            idx for idx, row in enumerate(ranked_logloss, start=1)
            if row["model"] == best_model_name
        ),
        "brier": next(
            idx for idx, row in enumerate(ranked_brier, start=1)
            if row["model"] == best_model_name
        ),
        "hit_k_main": next(
            idx for idx, row in enumerate(ranked_hit_k_main, start=1)
            if row["model"] == best_model_name
        ),
        "hit_10": next(
            idx for idx, row in enumerate(ranked_hit_10, start=1)
            if row["model"] == best_model_name
        ),
    }

    summary = {
        "best_by_metric": {
            "logloss": {
                "model": best_logloss["model"],
                "value": float(best_logloss["logloss"]),
            },
            "brier": {
                "model": best_brier["model"],
                "value": float(best_brier["brier"]),
            },
            "hit_k_main": {
                "model": best_hit_k_main["model"],
                "value": float(best_hit_k_main["hit_k_main"]),
            },
            "hit_10": {
                "model": best_hit_10["model"],
                "value": float(best_hit_10["hit_10"]),
            },
        },
        "best_overall": {
            "model": best_overall["model"],
            "rank_sum": int(rank_sums[best_model_name]),
            "metric_ranks": metric_ranks,
            "logloss": float(best_overall["logloss"]),
            "brier": float(best_overall["brier"]),
            "hit_k_main": float(best_overall["hit_k_main"]),
            "hit_10": float(best_overall["hit_10"]),
            "k_main": int(best_overall["k_main"]),
            "split_ratio": best_overall["split_ratio"],
            "train_rows": int(best_overall["train_rows"]),
            "test_rows": int(best_overall["test_rows"]),
            "train_date_from": best_overall["train_date_from"],
            "train_date_to": best_overall["train_date_to"],
            "test_date_from": best_overall["test_date_from"],
            "test_date_to": best_overall["test_date_to"],
        },
    }

    return summary

def _prepare_lagged_features(df_norm: pd.DataFrame, max_num: int) -> pd.DataFrame:
    # Pārvērš katru izlozi par one-hot vektoru (garums = max_num)
    # Izveido prev_vec (iepriekšējā izloze) un curr_vec (pašreizējā izloze)

    rows = []
    for _, row in df_norm.iterrows():
        mains = [row["n1"], row["n2"], row["n3"], row["n4"], row["n5"]]
        if not pd.isna(row.get("n6", pd.NA)):
            mains.append(row["n6"])
        mains_clean = [int(x) for x in mains if not pd.isna(x)]

        vec = np.zeros(max_num, dtype=int)
        for m in mains_clean:
            if 1 <= m <= max_num:
                vec[m - 1] = 1

        rows.append({"date": pd.to_datetime(row["date"]), "vec": vec})

    df_vec = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    prev_vecs = [None] + df_vec["vec"].tolist()[:-1]
    df_vec["prev_vec"] = prev_vecs
    df_vec["curr_vec"] = df_vec["vec"]

    return df_vec.dropna(subset=["prev_vec"]).reset_index(drop=True)

def _hit_at_k(Y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    # Aprēķina hit@k — cik bieži patiesie skaitļi ir starp top-k prognozētajiem

    hits = []
    for y, p in zip(Y_true, proba):
        top_idx = np.argsort(p)[-k:]
        top_set = set(top_idx.tolist())
        true_idx = set(np.where(y == 1)[0].tolist())
        inter = len(true_idx.intersection(top_set))
        hits.append(inter / float(k))
    return float(np.mean(hits))