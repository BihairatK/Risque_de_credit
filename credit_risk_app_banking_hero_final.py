import math
import os

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
GERMAN_CREDIT_CSV = os.path.join(os.path.dirname(__file__), "german_credit.csv")

# 1. Décodage des variables du German Credit Data 
# Colonnes brutes du CSV en nom court, et mapping code en libelle lisible


CODEBOOK = {"Account Balance": {"new_name": "checking_account_status",
                                "map": {1: "< 0 €", 2: "0-200 €", 3: ">= 200 € / salaire domicilie", 4: "Pas de compte courant"},
            },
            "Payment Status of Previous Credit": {"new_name": "credit_history",
                                                  "map": {0: "Aucun credit / tout rembourse",
                                                          1: "Credits (cette banque) rembourses a temps",
                                                          2: "Credits en cours rembourses normalement",
                                                          3: "Retard de paiement par le passe",
                                                          4: "Compte critique / autres credits existants"},
            },
            "Purpose": {"new_name": "purpose",
                        "map": {0: "Voiture neuve", 1: "Voiture occasion", 2: "Ameublement", 3: "Radio/TV",
                                4: "Electromenager", 5: "Reparations", 6: "Education", 7: "Vacances",
                                8: "Formation", 9: "Business", 10: "Autre"},
            },
            "Value Savings/Stocks": {"new_name": "savings_status",
                                     "map": {1: "< 100 €", 2: "100-500 €", 3: "500-1000 €", 
                                             4: ">= 1000 €", 5: "Inconnu / pas d'epargne"},
            },
            "Length of current employment": {"new_name": "employment_since",
                                             "map": {1: "Sans emploi", 2: "< 1 an", 
                                                     3: "1-4 ans", 4: "4-7 ans", 5: ">= 7 ans"},
            },
            "Guarantors": {"new_name": "guarantors",
                           "map": {1: "Aucun", 2: "Co-emprunteur", 3: "Garant"},
            },
            "Most valuable available asset": {"new_name": "property",
                                              "map": {1: "Bien immobilier", 2: "Epargne / assurance-vie", 
                                                      3: "Voiture ou autre bien", 4: "Aucun bien connu"},
            },
            "Concurrent Credits": {"new_name": "other_installment_plans",
                                   "map": {1: "Autre credit en banque", 2: "Autre credit en magasin", 3: "Aucun autre credit"},
            },
            "Type of apartment": {"new_name": "housing",
                                  "map": {1: "Locataire", 2: "Proprietaire", 3: "Loge gratuitement"},
            },
            "Occupation": {"new_name": "job",
                           "map": {1: "Sans emploi / non qualifie non resident", 2: "Non qualifie resident",
                                   3: "Employe qualifie", 4: "Cadre / independant / hautement qualifie"},
            },
}

NUMERIC_RENAME = {"Duration of Credit (month)": "duration_months",
                  "Credit Amount": "credit_amount",
                  "Instalment per cent": "installment_rate_pct_bracket",
                  "Duration in Current address": "residence_since",
                  "Age (years)": "age",
                  "No of Credits at this Bank": "existing_credits_count",
                  "No of dependents": "dependents",
}

EXCLUDED_RAW_COLUMNS = ["Sex & Marital Status", "Foreign Worker", "Telephone"] # données protegees / obsolète

# Milieu de bracket approximatif (en % du revenu disponible) pour la variable "Instalment per cent",
INSTALLMENT_RATE_MIDPOINT = {1: 0.10, 2: 0.20, 3: 0.30, 4: 0.40}

# Variables "comportementales" que l'on conserve dans le modele, meme si leur IV etait faible.
FORCED_BEHAVIORAL_FEATURES = ["credit_history", "checking_account_status", 
                              "existing_credits_count", "savings_status"]

# Multiplicateur applique aux points de scorecard de ces variables (surcouche "expert judgment", pour renforcer le poids des variables comportementales) 
BEHAVIORAL_POINTS_BOOST = 1.3


def load_and_prepare_data(csv_path):
    """Charge le German Credit Data et retourne un DataFrame decode et pret pour le feature engineering.

    Le chargement est volontairement robuste aux colonnes d'index ajoutees par
    Streamlit, Excel ou une sauvegarde intermediaire du DataFrame.
    """
    raw = pd.read_csv(csv_path)

    # Nettoyage des noms de colonnes. Certaines versions/sauvegardes de
    # Streamlit peuvent ajouter une colonne technique du type
    # ``index -- streamlit-generated``. Elle ne fait pas partie du dataset
    # German Credit et ne doit jamais entrer dans le pipeline.
    raw.columns = raw.columns.astype(str).str.strip()
    technical_index_cols = {
        "index -- streamlit-generated",
        "index",
        "Unnamed: 0",
    }
    raw = raw.drop(
        columns=[c for c in raw.columns if c in technical_index_cols],
        errors="ignore",
    )

    # Verification explicite des colonnes necessaires avant de poursuivre.
    required_columns = set(CODEBOOK.keys()) | set(NUMERIC_RENAME.keys()) | {
        "Creditability",
        "Sex & Marital Status",
        "Foreign Worker",
    }
    missing = sorted(required_columns - set(raw.columns))
    if missing:
        raise ValueError(
            "Le fichier german_credit.csv ne contient pas toutes les colonnes "
            "attendues. Colonnes manquantes : " + ", ".join(missing)
        )

    df = pd.DataFrame(index=raw.index)
    df["default"] = 1 - raw["Creditability"]  # Creditability=1 -> bon payeur ; on veut default=1 = mauvais payeur

    for col, spec in CODEBOOK.items():
        df[spec["new_name"]] = raw[col].map(spec["map"]).fillna("Inconnu")

    for col, new_name in NUMERIC_RENAME.items():
        df[new_name] = raw[col]

   
    ## Feature engineering - Capacite de remboursement 
   
    # PTI (Payment-to-Income) : directement disponible via le bracket "Instalment per cent",
    # qui encode par construction la mensualite en % du revenu disponible.
    df["PTI"] = df["installment_rate_pct_bracket"].map(INSTALLMENT_RATE_MIDPOINT)

    # LTI (Loan-to-Income) : le German Credit ne fournit pas de revenu annuel ; on approxime une
    # "intensite d'endettement" a partir du montant du credit rapporte a sa duree et au PTI (plus le
    # PTI est eleve pour un meme montant, plus le revenu implicite est faible -> LTI plus eleve).
    monthly_installment_proxy = df["credit_amount"] / df["duration_months"]
    implied_monthly_income_proxy = (monthly_installment_proxy / df["PTI"]).clip(lower=1)
    df["LTI"] = (df["credit_amount"] / (implied_monthly_income_proxy * 12)).clip(0, 8)

    # DTI (Debt-to-Income) : composite d'endettement global = PTI (mensualite du pret demande)
    # + une majoration si le client a deja d'autres credits en cours (banque/magasin) ou plusieurs
    # credits existants dans cette banque. Explicitement documente comme un PROXY, pas un vrai DTI
    other_debt_flag = (df["other_installment_plans"] != "Aucun autre credit").astype(float)
    df["DTI"] = (df["PTI"] + 0.05 * other_debt_flag
                 + 0.03 * (df["existing_credits_count"] - 1).clip(lower=0)).clip(0, 3)

    # Proxy de garantie pour la LGD : un bien immobilier ou une epargne/assurance-vie, ou un garant,
    # reduisent la perte en cas de defaut.
    df["has_collateral"] = (df["property"].isin(["Bien immobilier", "Epargne / assurance-vie"])
                            | (df["guarantors"] != "Aucun")
    )

    # Conserve les variables protegees/proxy uniquement pour le monitoring de fairness (jamais
    # utilisees comme predicteurs, cf. EXCLUDED_RAW_COLUMNS ci-dessus).
    df["_sex_marital_status_monitoring_only"] = raw["Sex & Marital Status"]
    df["_foreign_worker_monitoring_only"] = raw["Foreign Worker"]

    return df


CANDIDATE_NUMERIC = ["duration_months", "credit_amount", "age", "residence_since",
                     "existing_credits_count", "dependents", "PTI", "LTI", "DTI"]
CANDIDATE_CATEGORICAL = ["checking_account_status", "credit_history", "purpose", "savings_status",
                         "employment_since", "guarantors", "property", "other_installment_plans",
                         "housing", "job"]



# 2. WoE / IV

def compute_woe_iv(data, feature, target="default", bins=5, is_numeric=True):
    d = data[[feature, target]].copy()
    if is_numeric:
        try:
            d["bin"] = pd.qcut(d[feature], bins, duplicates="drop")
        except ValueError:
            d["bin"] = pd.cut(d[feature], bins)
    else:
        d["bin"] = d[feature]
    grp = d.groupby("bin", observed=True)[target].agg(["count", "sum"])
    grp.columns = ["total", "bad"]
    grp["good"] = (grp["total"] - grp["bad"]).replace(0, 0.5)
    grp["bad"] = grp["bad"].replace(0, 0.5)
    tg, tb = grp["good"].sum(), grp["bad"].sum()
    grp["pct_good"], grp["pct_bad"] = grp["good"] / tg, grp["bad"] / tb
    grp["woe"] = np.log(grp["pct_good"] / grp["pct_bad"])
    grp["iv_bin"] = (grp["pct_good"] - grp["pct_bad"]) * grp["woe"]
    return grp, grp["iv_bin"].sum()


def fit_woe_maps(data, features, target="default", bins=5):
    maps, edges = {}, {}
    for f in features:
        is_num = f in CANDIDATE_NUMERIC
        if is_num:
            _, bin_edges = pd.qcut(data[f], bins, retbins=True, duplicates="drop")
            edges[f] = bin_edges
            binned = pd.cut(data[f], bins=bin_edges, include_lowest=True)
        else:
            binned = data[f]
        tmp = pd.DataFrame({"bin": binned, "y": data[target]})
        grp = tmp.groupby("bin", observed=True)["y"].agg(["count", "sum"])
        grp.columns = ["total", "bad"]
        grp["good"] = (grp["total"] - grp["bad"]).replace(0, 0.5)
        grp["bad"] = grp["bad"].replace(0, 0.5)
        tg, tb = grp["good"].sum(), grp["bad"].sum()
        grp["woe"] = np.log((grp["good"] / tg) / (grp["bad"] / tb))
        maps[f] = grp["woe"].to_dict()
    return maps, edges


def apply_woe(data, features, maps, edges):
    out = pd.DataFrame(index=data.index)
    for f in features:
        if f in CANDIDATE_NUMERIC:
            binned = pd.cut(data[f], bins=edges[f], include_lowest=True)
        else:
            binned = data[f]
        default_woe = float(np.mean(list(maps[f].values())))
        mapped = binned.astype(object).map(maps[f]).astype(float)
        out[f + "_woe"] = mapped.fillna(default_woe)
    return out


# 3. Entrainement complet (donnees -> modeles -> calibration -> scorecard -> monitoring)

def ks_statistic(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


def compute_psi(reference, actual, bins=10):
    breakpoints = np.quantile(reference, np.linspace(0, 1, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    ref_counts = np.histogram(reference, bins=breakpoints)[0] / len(reference)
    act_counts = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    ref_counts = np.where(ref_counts == 0, 1e-4, ref_counts)
    act_counts = np.where(act_counts == 0, 1e-4, act_counts)
    return float(np.sum((act_counts - ref_counts) * np.log(act_counts / ref_counts)))


def pd_to_credit_score(pd_value, base_score=650, base_odds=15, pdo=40):
    factor = pdo / math.log(2)
    offset = base_score - factor * math.log(base_odds)
    pd_value = np.clip(pd_value, 1e-6, 1 - 1e-6)
    odds = (1 - pd_value) / pd_value
    return offset + factor * np.log(odds)


@st.cache_resource(show_spinner="Entrainement du modele de risque de credit...")
def train_pipeline(csv_path: str = GERMAN_CREDIT_CSV):
    df = load_and_prepare_data(csv_path)

    iv_results = {}
    for f in CANDIDATE_NUMERIC:
        _, iv = compute_woe_iv(df, f, bins=5, is_numeric=True)
        iv_results[f] = iv
    for f in CANDIDATE_CATEGORICAL:
        _, iv = compute_woe_iv(df, f, is_numeric=False)
        iv_results[f] = iv
    iv_table = pd.Series(iv_results).sort_values(ascending=False).rename("IV").to_frame()

    # Selection par IV (> 0.02), en forcant l'inclusion des variables comportementales cles
    selected = set(iv_table[iv_table["IV"] > 0.02].index) | set(FORCED_BEHAVIORAL_FEATURES)
    selected_features = [f for f in iv_table.index if f in selected]

    # Split stratifie 60% train / 20% validation (calibration) / 20% test (monitoring, hors echantillon).
    train_df, temp_df = train_test_split(df, test_size=0.4, stratify=df["default"], random_state=RANDOM_STATE)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["default"], random_state=RANDOM_STATE)

    woe_maps, woe_edges = fit_woe_maps(train_df, selected_features)
    X_train = apply_woe(train_df, selected_features, woe_maps, woe_edges)
    X_val = apply_woe(val_df, selected_features, woe_maps, woe_edges)
    X_test = apply_woe(test_df, selected_features, woe_maps, woe_edges)
    y_train, y_val, y_test = train_df["default"], val_df["default"], test_df["default"]

    # Modele principal : Logistic Regression + WoE (scorecard, interpretable, standard reglementaire)
    scorecard_model = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)
    scorecard_model.fit(X_train, y_train)

    # Modele ML de reference (comparaison) : Random Forest sur variables brutes encodees
    def build_raw(data):
        num = data[CANDIDATE_NUMERIC].fillna(data[CANDIDATE_NUMERIC].median())
        cat = pd.get_dummies(data[CANDIDATE_CATEGORICAL], drop_first=True)
        return pd.concat([num, cat], axis=1)

    X_train_raw, X_test_raw = build_raw(train_df), build_raw(test_df).reindex(columns=build_raw(train_df).columns, fill_value=0)
    rf_model = RandomForestClassifier(n_estimators=400, max_depth=6, min_samples_leaf=15,
                                      class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    rf_model.fit(X_train_raw, y_train)

    # Recalibrage de Platt (class_weight='balanced' biaise la PD -> recalibrage sur la validation)
    raw_score_val = scorecard_model.decision_function(X_val).reshape(-1, 1)
    platt = LogisticRegression(max_iter=1000)
    platt.fit(raw_score_val, y_val)

    raw_score_test = scorecard_model.decision_function(X_test).reshape(-1, 1)
    pd_test = platt.predict_proba(raw_score_test)[:, 1]
    pd_train = platt.predict_proba(scorecard_model.decision_function(X_train).reshape(-1, 1))[:, 1]

    score_test = pd_to_credit_score(pd_test)
    score_train = pd_to_credit_score(pd_train)

    # Seuils de decision Accepte / Revue / Refus, derives des quantiles du score (conserver une zone de revue humaine plutot qu'une decision 100% automatique)
    refuse_thresh, accept_thresh = np.quantile(score_test, [0.15, 0.75])

    # Evaluation
    auc_test = roc_auc_score(y_test, pd_test)
    auc_train = roc_auc_score(y_train, pd_train)
    metrics = {"AUC_test": auc_test, "Gini_test": 2 * auc_test - 1, "KS_test": ks_statistic(y_test, pd_test),
               "PR_AUC_test": average_precision_score(y_test, pd_test), "Brier_test": brier_score_loss(y_test, pd_test),
               "AUC_train": auc_train, "Gini_train": 2 * auc_train - 1, "KS_train": ks_statistic(y_train, pd_train),
               "Taux_defaut_train": float(y_train.mean()), "Taux_defaut_test": float(y_test.mean()),
               "PSI_train_vs_test": compute_psi(score_train, score_test),
    }
    calib_curve = calibration_curve(y_test, pd_test, n_bins=8, strategy="quantile")

    # Explicabilite : coefficients (avec surcouche "boost" sur les variables comportementales
    # pour la lecture des points de scorecard uniquement, cf. BEHAVIORAL_POINTS_BOOST)
    coef_series = pd.Series(scorecard_model.coef_[0], index=X_train.columns)
    display_coef = coef_series.copy()
    for f in FORCED_BEHAVIORAL_FEATURES:
        col = f + "_woe"
        if col in display_coef.index:
            display_coef[col] = display_coef[col] * BEHAVIORAL_POINTS_BOOST

    perm = permutation_importance(rf_model, X_test_raw, y_test, n_repeats=8,
                                   random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1)
    perm_importance = pd.Series(perm.importances_mean, index=X_test_raw.columns).sort_values(ascending=False)

    # Politique economique du risque : segmentation en bandes de PD, avec
    # une proposition de marge tarifaire et de plafond de credit indicative par bande
    risk_bands = pd.cut(pd_test, bins=[0, 0.05, 0.10, 0.20, 0.35, 1.0],
                         labels=["Tres faible", "Faible", "Modere", "Eleve", "Tres eleve"])
    pricing_policy = pd.DataFrame({"PD": pd_test, "band": risk_bands, "default": y_test.values}).groupby("band", observed=True
                                                                                                ).agg(nb_dossiers=("PD", "count"), PD_moyenne=("PD", "mean"), taux_defaut_observe=("default", "mean"))
    pricing_policy["marge_risque_suggeree_pct"] = (pricing_policy["PD_moyenne"] * 0.45 * 100).round(2)  # PD x LGD indicative x marge
    pricing_policy["plafond_credit_indicatif_DM"] = (10000 / (1 + pricing_policy["PD_moyenne"] * 10)).round(0)

    return {"df": df, "train_df": train_df, "val_df": val_df, "test_df": test_df,
            "iv_table": iv_table, "selected_features": selected_features,
            "woe_maps": woe_maps, "woe_edges": woe_edges,
            "scorecard_model": scorecard_model, "rf_model": rf_model, "platt": platt,
            "pd_test": pd_test, "score_test": score_test, "y_test": y_test,
            "refuse_thresh": float(refuse_thresh), "accept_thresh": float(accept_thresh),
            "metrics": metrics, "calib_curve": calib_curve,
            "coef_series": display_coef, "perm_importance": perm_importance,
            "pricing_policy": pricing_policy,
            "rf_raw_columns": X_train_raw.columns,
    }


# 4. Scoring d'un dossier individuel

def compute_ratios(credit_amount, duration_months, installment_rate_code, other_debt, existing_credits_count):
    pti = INSTALLMENT_RATE_MIDPOINT[installment_rate_code]
    monthly_installment_proxy = credit_amount / duration_months
    implied_income = max(monthly_installment_proxy / pti, 1)
    lti = min(credit_amount / (implied_income * 12), 8)
    dti = min(pti + 0.05 * float(other_debt) + 0.03 * max(existing_credits_count - 1, 0), 3)
    return {"PTI": pti, "LTI": lti, "DTI": dti}


def score_client(raw_inputs: dict, artifacts: dict):
    """raw_inputs doit fournir toutes les variables brutes (memes noms que selected_features,
    plus les champs necessaires au calcul des ratios et de l'EL)."""

    ratios = compute_ratios(raw_inputs["credit_amount"], raw_inputs["duration_months"],
                            raw_inputs["installment_rate_code"], raw_inputs["other_debt_elsewhere"],
                            raw_inputs["existing_credits_count"],
    )
    client = {**raw_inputs, **ratios}
    client_df = pd.DataFrame([client])

    selected_features = artifacts["selected_features"]
    woe_row = apply_woe(client_df, selected_features, artifacts["woe_maps"], artifacts["woe_edges"]).iloc[0]

    rs = float(np.dot(artifacts["scorecard_model"].coef_[0], woe_row.values) + artifacts["scorecard_model"].intercept_[0])
    pd_value = float(artifacts["platt"].predict_proba([[rs]])[:, 1][0])
    score = float(pd_to_credit_score(pd_value))

    if score >= artifacts["accept_thresh"]:
        decision = "ACCEPTE"
    elif score >= artifacts["refuse_thresh"]:
        decision = "REVUE"
    else:
        decision = "REFUSE"

    lgd = 0.35 if raw_inputs["has_collateral"] else 0.65
    ead = raw_inputs["credit_amount"]
    expected_loss = pd_value * lgd * ead

    return {"PD": pd_value, "credit_score": score, "decision": decision,
            "expected_loss": expected_loss, "LGD": lgd, "EAD": ead, **ratios}


# 5. Interface Streamlit

import base64
from pathlib import Path

BANK_BACKGROUND = os.path.join(os.path.dirname(__file__), "bank_background.jpg")


def _get_bank_image_data(image_path):
    """Retourne l'image bancaire en Data URI Base64."""
    if not os.path.exists(image_path):
        return None
    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _hero_image_html(image_path):
    """Construit l'image bancaire plein format du Hero."""
    image_data = _get_bank_image_data(image_path)
    if image_data is None: 
        return ""
    return (
        '<img class="hero-bank-image" '
        f'src="{image_data}" '
        'alt="Architecture bancaire">'
    )


APP_CSS = """
<style>
.block-container { padding-top: 1.25rem; padding-bottom: 3rem; max-width: 1480px; }
body { background: #f4f8fc; }

/* ========================= SIDEBAR ========================= */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #06203e 0%, #071a31 100%); border-right: 1px solid rgba(255,255,255,.08); }
[data-testid="stSidebar"] * { color: #e7eef8; }
[data-testid="stSidebar"] .stCaption { color: #9fb3ca !important; }

/* ========================= HERO ========================= */
.hero {
    position: relative;
    overflow: hidden;
    min-height: 265px;
    margin-bottom: 18px;
    padding: 0;
    border-radius: 24px;
    border: 1px solid rgba(255,255,255,.16);
    background: #062747;
    box-shadow: 0 16px 42px rgba(15,23,42,.16);
}
.hero-bank-image {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center center;
    display: block;
    z-index: 0;
}
.hero-image-overlay {
    position: absolute;
    inset: 0;
    z-index: 1;
    pointer-events: none;
    background: linear-gradient(90deg, rgba(3,27,54,.94) 0%, rgba(3,27,54,.82) 28%, rgba(3,27,54,.48) 55%, rgba(3,27,54,.16) 78%, rgba(3,27,54,.04) 100%);
}
.hero-content {
    position: relative;
    z-index: 3;
    width: 80%;
    min-height: 265px;
    padding: 34px 34px 30px;
    color: white;
    box-sizing: border-box;
}
.hero-kicker { display:inline-block; padding:6px 12px; border-radius:999px; background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.18); color:#dbeafe; font-size:.74rem; font-weight:800; letter-spacing:.10em; text-transform:uppercase; }
.hero-title { font-size:2.65rem; font-weight:850; line-height:1.08; margin:13px 0 8px; text-shadow:0 2px 12px rgba(0,0,0,.18); }
.hero-subtitle { color:#e6eef8; font-size:1rem; line-height:1.55; max-width:760px; text-shadow:0 1px 6px rgba(0,0,0,.18); }
.hero-pills { margin-top:18px; }
.hero-pill { display:inline-block; margin-right:7px; padding:6px 10px; border-radius:999px; background:rgba(255,255,255,.11); border:1px solid rgba(255,255,255,.18); color:#f8fbff; font-size:.72rem; font-weight:700; }

/* ========================= CARDS ========================= */
.kpi-card,.panel-card,.decision-card,.portfolio-card { border:1px solid rgba(148,163,184,.23); border-radius:18px; background:rgba(255,255,255,.96); box-shadow:0 8px 26px rgba(15,23,42,.055); }
.kpi-card { padding:15px 17px; min-height:108px; }
.kpi-label { font-size:.72rem; font-weight:800; text-transform:uppercase; letter-spacing:.08em; color:#64748b; }
.kpi-value { font-size:1.72rem; font-weight:850; color:#0b2a50; margin-top:5px; }
.kpi-help { font-size:.74rem; color:#64748b; margin-top:3px; }
.kpi-icon { float:right; width:36px; height:36px; border-radius:50%; display:grid; place-items:center; background:#e8f1ff; font-size:1rem; }
.panel-title { font-size:1rem; font-weight:800; color:#0b2a50; }
.panel-subtitle { font-size:.78rem; color:#64748b; margin-top:2px; }

/* ========================= DECISION ========================= */
.decision-card { min-width: 0; width: 100%; max-width: 100%; overflow: hidden; box-sizing: border-box;}
.decision-grid {display: grid; grid-template-columns: minmax(0, 1.05fr) minmax(0, .95fr); gap: 18px; align-items: center; width: 100%;}
.decision-grid > div {min-width: 0; max-width: 100%; box-sizing: border-box;}
.decision-kicker { font-size:.72rem; font-weight:800; color:#64748b; text-transform:uppercase; letter-spacing:.08em; }
.decision-value { font-size:2.05rem; font-weight:900; margin-top:4px; }
.decision-description { color:#475569; font-size:.86rem; line-height:1.45; }
.status-dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:7px; }
.status-accept { background:#16a34a; } .status-review { background:#f59e0b; } .status-refuse { background:#ef4444; }

/* ========================= SCORE GAUGE ========================= */
.score-gauge { margin-top:16px; }
.score-track { height:13px; border-radius:999px; background:linear-gradient(90deg,#ef4444 0%,#f59e0b 42%,#22c55e 70%,#0f9f88 100%); position:relative; }
.score-marker { position:absolute; top:-5px; width:4px; height:23px; border-radius:4px; background:#08264b; box-shadow:0 0 0 3px rgba(255,255,255,.92); }
.score-labels { display:flex; justify-content:space-between; color:#64748b; font-size:.68rem; margin-top:5px; }

/* ========================= MINI METRICS ========================= */
.mini-metric { border:1px solid #e2e8f0; border-radius:13px; padding:11px 12px; background:#f8fafc; }
.mini-label { font-size:.68rem; color:#64748b; }
.mini-value { font-size:1.1rem; font-weight:850; color:#0b2a50; margin-top:3px; }

/* ========================= SIDEBAR CARDS ========================= */
.sidebar-brand { font-size:1.22rem; font-weight:850; color:white; }
.sidebar-sub { color:#9fb3ca; font-size:.74rem; }
.sidebar-section { margin-top:20px; color:#8ea8c3; font-size:.68rem; text-transform:uppercase; letter-spacing:.1em; font-weight:800; }
.sidebar-card { border:1px solid rgba(255,255,255,.10); border-radius:13px; padding:13px; background:rgba(255,255,255,.055); margin:9px 0; }
.sidebar-value { color:#fff; font-weight:750; font-size:.84rem; }
.sidebar-note { color:#9fb3ca; font-size:.70rem; line-height:1.45; }

/* ========================= FORMS / TABLES ========================= */
div[data-testid="stForm"] { border:1px solid rgba(148,163,184,.24); border-radius:18px; padding:22px; background:rgba(255,255,255,.96); box-shadow:0 8px 26px rgba(15,23,42,.045); }
button[kind="primaryFormSubmit"] { border-radius:10px; font-weight:800; }
[data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; }

/* ========================= TABS ========================= */
.stTabs [data-baseweb="tab-list"] { gap:8px; }
.stTabs [data-baseweb="tab"] { border-radius:10px; padding:8px 13px; font-weight:700; }

/* ========================= FOOTER ========================= */
.footer-note { text-align:center; color:#94a3b8; font-size:.72rem; padding:25px 0 4px; }

@media (max-width:900px) {
    .hero { min-height: 300px; }
    .hero-content { width:100%; min-height:300px; padding:28px 24px; background:linear-gradient(90deg,rgba(3,27,54,.93),rgba(3,27,54,.62)); }
    .hero-title { font-size:2rem; }
}
</style>
"""

def kpi_card(label, value, help_text="", icon=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def score_gauge(score):
    pct = float(np.clip((score - 300) / 550 * 100, 0, 100))

    return f"""
    <div class="score-gauge"
         style="width:100%; max-width:100%; box-sizing:border-box;">

        <div style="
            font-size:.72rem;
            color:#64748b;
            font-weight:700;
            margin-bottom:7px;
        ">
            Score de crédit
        </div>

        <div class="score-track"
             style="
                position:relative;
                width:100%;
                height:13px;
                box-sizing:border-box;
             ">

            <div class="score-marker"
                 style="
                    position:absolute;
                    left:{pct:.1f}%;
                    top:-5px;
                 ">
            </div>

        </div>

        <div class="score-labels"
             style="
                width:100%;
                display:flex;
                justify-content:space-between;
                font-size:.68rem;
                color:#64748b;
                margin-top:5px;
                box-sizing:border-box;
             ">
            <span>300</span>
            <span>500</span>
            <span>650</span>
            <span>850</span>
        </div>

    </div>
    """


def decision_card(result):
    decision = result["decision"]

    cfg = {
        "ACCEPTE": (
            "#16a34a",
            "Décision favorable",
            "Le dossier dépasse le seuil d’acceptation automatique."
        ),
        "REVUE": (
            "#f59e0b",
            "Revue humaine",
            "Le dossier se situe dans la zone intermédiaire."
        ),
        "REFUSE": (
            "#ef4444",
            "Décision défavorable",
            "Le score est inférieur au seuil de refus."
        ),
    }

    status_color, title, description = cfg[decision]

    html = f"""
    <div style="
        width:100%;
        max-width:100%;
        box-sizing:border-box;
        overflow:hidden;

        border:1px solid #e2e8f0;
        border-radius:18px;
        background:#ffffff;

        padding:22px;
        margin:0;
    ">

        <div style="
            display:grid;
            grid-template-columns:minmax(0, 1.05fr) minmax(220px, .95fr);
            gap:18px;
            align-items:center;
            width:100%;
        ">

            <!-- =========================
                 COLONNE GAUCHE
                 ========================= -->

            <div style="
                min-width:0;
                width:100%;
                box-sizing:border-box;
            ">

                <div style="
                    font-size:.72rem;
                    font-weight:800;
                    color:#64748b;
                    text-transform:uppercase;
                    letter-spacing:.08em;
                    margin-bottom:8px;
                ">
                    Décision du scorecard
                </div>

                <div style="
                    font-size:2.05rem;
                    font-weight:900;
                    color:#0b2a50;
                    line-height:1.1;
                    margin-bottom:10px;
                ">
                    <span style="
                        display:inline-block;
                        width:10px;
                        height:10px;
                        border-radius:50%;
                        background:{status_color};
                        margin-right:8px;
                        vertical-align:middle;
                    "></span>
                    {decision}
                </div>

                <div style="
                    color:#475569;
                    font-size:.86rem;
                    line-height:1.5;
                    margin-bottom:18px;
                ">
                    <b>{title}</b> — {description}
                </div>

                {score_gauge(result["credit_score"])}

            </div>


            <!-- =========================
                 COLONNE DROITE
                 ========================= -->

            <div style="
                min-width:0;
                width:100%;
                box-sizing:border-box;
            ">

                <div style="
                    width:100%;
                    box-sizing:border-box;

                    border:1px solid #e2e8f0;
                    border-radius:13px;
                    padding:12px;

                    background:#f8fafc;
                ">
                    <div style="
                        font-size:.68rem;
                        color:#64748b;
                        font-weight:700;
                    ">
                        PROBABILITY OF DEFAULT
                    </div>

                    <div style="
                        font-size:1.1rem;
                        font-weight:850;
                        color:#0b2a50;
                        margin-top:3px;
                    ">
                        {result["PD"]:.2%}
                    </div>
                </div>


                <div style="height:9px;"></div>


                <div style="
                    width:100%;
                    box-sizing:border-box;

                    border:1px solid #e2e8f0;
                    border-radius:13px;
                    padding:12px;

                    background:#f8fafc;
                ">
                    <div style="
                        font-size:.68rem;
                        color:#64748b;
                        font-weight:700;
                    ">
                        EXPECTED LOSS
                    </div>

                    <div style="
                        font-size:1.1rem;
                        font-weight:850;
                        color:#0b2a50;
                        margin-top:3px;
                    ">
                        {result["expected_loss"]:,.0f} €
                    </div>
                </div>


                <div style="height:9px;"></div>


                <div style="
                    width:100%;
                    box-sizing:border-box;

                    border:1px solid #e2e8f0;
                    border-radius:13px;
                    padding:12px;

                    background:#f8fafc;
                ">
                    <div style="
                        font-size:.68rem;
                        color:#64748b;
                        font-weight:700;
                    ">
                        CREDIT SCORE
                    </div>

                    <div style="
                        font-size:1.1rem;
                        font-weight:850;
                        color:#0b2a50;
                        margin-top:3px;
                    ">
                        {result["credit_score"]:.0f}
                    </div>
                </div>

            </div>

        </div>

    </div>
    """

    st.html(html)



def render_sidebar(artifacts):
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">🏦 Credit Risk</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-sub">Risk Analytics Platform</div>', unsafe_allow_html=True)
        st.divider()

        st.markdown('<div class="sidebar-section">Architecture</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-card"><div class="sidebar-value">Régression Logistique + WoE</div><div class="sidebar-note">Scorecard interprétable · Platt Scaling</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-card"><div class="sidebar-value">Random Forest</div><div class="sidebar-note">Benchmark machine learning</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">Cadre de décision</div>', unsafe_allow_html=True)
        for item in ["✓ Acceptation automatique", "✓ Zone de revue humaine", "✓ Refus automatique"]:
            st.markdown(f'<div class="sidebar-note" style="margin:7px 0;">{item}</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">Contrôles</div>', unsafe_allow_html=True)
        for item in ["AUC / Gini / KS", "PR-AUC / Brier", "Calibration PD", "PSI", "IV / WoE", "Expected Loss"]:
            st.markdown(f'<div class="sidebar-note" style="margin:6px 0;">• {item}</div>', unsafe_allow_html=True)

        st.divider()
        st.caption(f"{len(artifacts['df']):,} dossiers · German Credit Data")
        st.caption("Prototype analytique · usage démonstratif")


def render_portfolio(artifacts):
    scores = artifacts["score_test"]
    decisions = np.select(
        [scores >= artifacts["accept_thresh"], scores >= artifacts["refuse_thresh"]],
        ["ACCEPTE", "REVUE"],
        default="REFUSE",
    )
    counts = pd.Series(decisions).value_counts().reindex(["ACCEPTE", "REVUE", "REFUSE"], fill_value=0)
    total = int(counts.sum())
    default_rate = artifacts["metrics"]["Taux_defaut_test"]
    pd_mean = float(np.mean(artifacts["pd_test"]))

    st.markdown('<div class="panel-card" style="padding:18px;">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">👥 Portefeuille test</div><div class="panel-subtitle">Vue synthétique sur l’échantillon hors entraînement</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.markdown(f'<div class="mini-metric" style="margin-top:12px;"><div class="mini-label">TAUX DE DÉFAUT</div><div class="mini-value">{default_rate:.1%}</div></div>', unsafe_allow_html=True)
    with b:
        st.markdown(f'<div class="mini-metric" style="margin-top:12px;"><div class="mini-label">PD MOYENNE</div><div class="mini-value">{pd_mean:.1%}</div></div>', unsafe_allow_html=True)

    dist = pd.DataFrame({"Dossiers": counts})
    st.markdown("<div style='margin-top:15px;font-size:.76rem;font-weight:800;color:#334155;'>RÉPARTITION DES DÉCISIONS</div>", unsafe_allow_html=True)
    st.bar_chart(dist, height=180)
    st.caption(f"{total:,} dossiers dans l’échantillon test")
    st.markdown('</div>', unsafe_allow_html=True)


def render_app():
    st.set_page_config(
        page_title="Credit Risk Assessment",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(APP_CSS, unsafe_allow_html=True)

    if not os.path.exists(GERMAN_CREDIT_CSV):
        st.error(
            f"Fichier de données introuvable : `{GERMAN_CREDIT_CSV}`. "
            "Placez `german_credit.csv` à côté de ce script."
        )
        st.stop()

    artifacts = train_pipeline(GERMAN_CREDIT_CSV)
    render_sidebar(artifacts)
    m = artifacts["metrics"]

    hero_image = _hero_image_html(BANK_BACKGROUND)

    st.markdown(
        f"""
        <div class="hero">
            {hero_image}
            <div class="hero-image-overlay"></div>
            <div class="hero-content">
                <div class="hero-kicker">Risk Analytics · Credit Scoring</div>
                <div class="hero-title">Credit Risk Assessment</div>
                <div class="hero-subtitle">Analyse, scoring et décision du risque de crédit à partir d’un scorecard combinant <b>WoE / IV</b>, régression logistique, recalibrage de <b>Platt</b> et <b>Expected Loss</b>.</div>
                <div class="hero-pills">
                    <span class="hero-pill">▣ Credit Scoring</span>
                    <span class="hero-pill">◉ PD</span>
                    <span class="hero-pill">◈ Expected Loss</span>
                    <span class="hero-pill">◌ Monitoring</span>
                    <span class="hero-pill">◍ WoE / IV</span>
                    <span class="hero-pill">◌ Calibration</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================= TOP KPIs =========================
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("AUC", f"{m['AUC_test']:.3f}", "Performance globale du modèle", "↗")
    with c2: kpi_card("GINI", f"{m['Gini_test']:.3f}", "Pouvoir discriminant", "▥")
    with c3: kpi_card("KS", f"{m['KS_test']:.3f}", "Séparation Good / Bad", "⌖")
    with c4: kpi_card("PSI", f"{m['PSI_train_vs_test']:.3f}", "Stabilité train vs test", "◔")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    tab_form, tab_monitoring, tab_explain, tab_model = st.tabs(
        ["📝 Évaluer un dossier", "📊 Monitoring", "🔎 Explicabilité", "🧩 Model Card"]
    )

    # ========================= EVALUATION =========================
    with tab_form:
        left, center, right = st.columns([1.0, 1.35, .92], gap="medium")

        with left:
            st.markdown('<div class="panel-card" style="padding:20px;">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">📝 Évaluer un dossier</div><div class="panel-subtitle">Renseignez les informations du client pour obtenir une évaluation du risque.</div>', unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            with st.form("credit_form"):
                st.markdown("**👤 Informations personnelles**")
                age = st.number_input("Âge", 18, 90, 35)
                dependents = st.number_input("Personnes à charge", 1, 3, 1)

                st.markdown("**💼 Situation professionnelle**")
                employment_since = st.selectbox("Ancienneté professionnelle", list(CODEBOOK["Length of current employment"]["map"].values()))
                job = st.selectbox("Emploi", list(CODEBOOK["Occupation"]["map"].values()))

                st.markdown("**💳 Situation financière**")
                checking_account_status = st.selectbox("Statut du compte courant", list(CODEBOOK["Account Balance"]["map"].values()))
                credit_history = st.selectbox("Historique de crédit", list(CODEBOOK["Payment Status of Previous Credit"]["map"].values()))
                savings_status = st.selectbox("Épargne / valeurs mobilières", list(CODEBOOK["Value Savings/Stocks"]["map"].values()))
                existing_credits_count = st.number_input("Crédits existants dans cette banque", 1, 6, 1)

                st.markdown("**🏠 Garanties & logement**")
                guarantors = st.selectbox("Garant / co-emprunteur", list(CODEBOOK["Guarantors"]["map"].values()))
                property_ = st.selectbox("Bien le plus valorisable", list(CODEBOOK["Most valuable available asset"]["map"].values()))
                housing = st.selectbox("Logement", list(CODEBOOK["Type of apartment"]["map"].values()))
                residence_since = st.number_input("Ancienneté à l’adresse actuelle", 1, 4, 2)

                st.markdown("**💰 Demande de crédit**")
                credit_amount = st.number_input("Montant du crédit (€)", 250, 20000, 3000, step=100)
                duration_months = st.number_input("Durée (mois)", 4, 72, 24)
                purpose = st.selectbox("Objet du crédit", list(CODEBOOK["Purpose"]["map"].values()))
                installment_rate_label = st.select_slider("Mensualité en % du revenu disponible", options=[1,2,3,4], format_func=lambda c: {1:"< 20%",2:"20–25%",3:"25–35%",4:"≥ 35%"}[c], value=2)
                other_installment_plans = st.selectbox("Autres crédits en cours", list(CODEBOOK["Concurrent Credits"]["map"].values()))

                submitted = st.form_submit_button("⚡ Évaluer le dossier", width="stretch", type="primary")
            st.markdown('</div>', unsafe_allow_html=True)

        result = st.session_state.get("last_result")

        if submitted:
            other_debt_elsewhere = other_installment_plans != "Aucun autre crédit"
            has_collateral = (property_ in ["Bien immobilier", "Épargne / assurance-vie"] or guarantors != "Aucun")
            raw_inputs = dict(
                age=age, checking_account_status=checking_account_status, credit_history=credit_history,
                savings_status=savings_status, employment_since=employment_since, job=job,
                existing_credits_count=existing_credits_count, dependents=dependents,
                residence_since=residence_since, guarantors=guarantors, property=property_, housing=housing,
                other_installment_plans=other_installment_plans, credit_amount=credit_amount,
                duration_months=duration_months, purpose=purpose, installment_rate_code=installment_rate_label,
                other_debt_elsewhere=other_debt_elsewhere, has_collateral=has_collateral,
            )
            result = score_client(raw_inputs, artifacts)
            st.session_state["last_result"] = result

        with center:
            st.markdown('<div class="panel-title" style="margin-bottom:8px;">💼 Résultat de l’évaluation</div>', unsafe_allow_html=True)
            if result:
                decision_card(result)
                r1, r2, r3 = st.columns(3)
                vals = [("PTI", f"{result['PTI']:.1%}", "Capacité de remboursement"),
                        ("LTI", f"{result['LTI']:.2f}", "Proxy"),
                        ("DTI", f"{result['DTI']:.1%}", "Proxy"),
                ]
                for col, (lab, val, help_) in zip([r1,r2,r3,], vals):
                    with col: kpi_card(lab, val, help_)

                if result["decision"] == "REVUE":
                    st.info("🧑‍💼 **Revue humaine recommandée** : le dossier se situe dans la zone intermédiaire.")
                elif result["decision"] == "ACCEPTE":
                    st.success("✓ **Dossier accepté automatiquement** selon les seuils du prototype.")
                else:
                    st.error("✕ **Dossier refusé automatiquement** selon les seuils du prototype.")

                st.markdown("#### Pourquoi cette décision ?")
                factors = pd.Series(artifacts["coef_series"]).sort_values()
                low = factors.head(3)
                high = factors.tail(3).sort_values(ascending=False)
                exp1, exp2 = st.columns(2)
                with exp1:
                    st.markdown("**Facteurs à surveiller**")
                    for idx, value in low.items(): st.caption(f"• {idx.replace('_woe','')}: {value:+.3f}")
                with exp2:
                    st.markdown("**Facteurs favorables dans le scorecard**")
                    for idx, value in high.items(): st.caption(f"• {idx.replace('_woe','')}: {value:+.3f}")
            else:
                st.markdown('<div class="panel-card" style="padding:55px 25px;text-align:center;"><div style="font-size:2.2rem;">🏦</div><div class="panel-title" style="margin-top:10px;">Aucun dossier évalué</div><div class="panel-subtitle">Complétez le formulaire à gauche puis lancez l’évaluation.</div></div>', unsafe_allow_html=True)

        with right:
            render_portfolio(artifacts)
            st.markdown('<div class="panel-card" style="padding:18px;margin-top:14px;">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">💡 Cadre analytique</div>', unsafe_allow_html=True)
            st.caption("Le scorecard transforme la PD calibrée en score de crédit, puis applique une zone de revue humaine avant la décision finale.")
            st.caption("Expected Loss = PD × LGD × EAD")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="footer-note">German Credit Data · Prototype Risk Analytics · Les décisions et hypothèses doivent être validées avant tout usage opérationnel.</div>', unsafe_allow_html=True)

    # ========================= MONITORING =========================
    with tab_monitoring:
        st.markdown("### 📊 Monitoring du modèle")
        c1,c2,c3,c4,c5 = st.columns(5)
        for col, label, val, help_ in [
            (c1,"AUC",m['AUC_test'],"Pouvoir discriminant"),(c2,"GINI",m['Gini_test'],"Indice dérivé de l’AUC"),
            (c3,"KS",m['KS_test'],"Séparation Good / Bad"),(c4,"PR-AUC",m['PR_AUC_test'],"Performance sur la classe défaut"),(c5,"BRIER",m['Brier_test'],"Qualité probabiliste")]:
            with col: kpi_card(label, f"{val:.3f}", help_)

        left,right = st.columns(2)
        with left:
            st.markdown("#### 📐 Calibration de la PD")
            frac_pos, mean_pred = artifacts["calib_curve"]
            calib_df = pd.DataFrame({"PD prédite":mean_pred,"Taux de défaut observé":frac_pos})
            st.line_chart(calib_df.set_index("PD prédite"), height=300)
        with right:
            st.markdown("#### 📈 Distribution des scores")
            # Streamlit peut rencontrer un problème avec une Series issue de
            # value_counts() lorsque l'index est utilisé implicitement comme axe X.
            # On construit donc explicitement un DataFrame avec deux colonnes.
            score_series = pd.Series(artifacts["score_test"], name="Credit Score").dropna()
            score_bins = [300, 400, 500, 600, 700, 800, 900]
            score_labels = ["300–399", "400–499", "500–599", "600–699", "700–799", "800–899"]
            score_distribution = (
                pd.cut(
                    score_series,
                    bins=score_bins,
                    labels=score_labels,
                    right=False,
                    include_lowest=True,
                )
                .value_counts()
                .reindex(score_labels, fill_value=0)
                .rename_axis("Tranche de score")
                .reset_index(name="Nombre de dossiers")
            )
            st.bar_chart(
                score_distribution,
                x="Tranche de score",
                y="Nombre de dossiers",
                height=300,
            )

        st.markdown("#### 💶 Politique économique par bande de PD")
        pricing_display = artifacts["pricing_policy"].copy()
        pricing_display["PD_moyenne"] = pricing_display["PD_moyenne"].map(lambda x:f"{x:.2%}")
        pricing_display["taux_defaut_observe"] = pricing_display["taux_defaut_observe"].map(lambda x:f"{x:.2%}")
        pricing_display["marge_risque_suggeree_pct"] = pricing_display["marge_risque_suggeree_pct"].map(lambda x:f"{x:.2f}%")
        pricing_display["plafond_credit_indicatif_DM"] = pricing_display["plafond_credit_indicatif_DM"].map(lambda x:f"{x:,.0f}")
        st.dataframe(pricing_display, width="stretch")

        with st.expander("🛡️ Feuille de route de monitoring production"):
            st.markdown("""
            - Suivre AUC / Gini, KS et calibration de la PD.
            - Contrôler le taux de défaut observé et les migrations de score.
            - Surveiller PSI et stabilité des variables d’entrée.
            - Déclencher des alertes en cas de dérive significative.
            - Réaliser des backtests par cohorte et comparer prédictions / réalisations.
            """)
        with st.expander("⚠️ Limites du prototype"):
            st.markdown("""
            - LGD et EAD sont simplifiées : LGD binaire selon garantie et EAD = montant du crédit.
            - LTI / DTI sont des proxys car le German Credit ne fournit pas de revenu brut.
            - Une validation croisée temporelle, des stress tests et des modèles LGD/EAD/CCF dédiés seraient nécessaires pour un usage réglementaire.
            """)

    # ========================= EXPLAINABILITY =========================
    with tab_explain:
        st.markdown("### 🔎 Explicabilité du modèle")
        left,right = st.columns(2)
        with left:
            st.markdown("#### Scorecard — coefficients WoE")
            st.bar_chart(artifacts["coef_series"].sort_values())
        with right:
            st.markdown("#### Benchmark — permutation importance")
            st.bar_chart(artifacts["perm_importance"].head(12))
        st.markdown("#### Information Value")
        iv_display = artifacts["iv_table"].copy()
        iv_display["Force indicative"] = pd.cut(iv_display["IV"], bins=[-np.inf,.02,.10,.30,np.inf], labels=["Faible / seuil","Modérée","Forte","Très forte"])
        st.dataframe(iv_display, width="stretch")
        st.info(f"Les variables comportementales clés sont affichées avec une surcouche d’importance métier de ×{BEHAVIORAL_POINTS_BOOST} dans la lecture des points de scorecard. Cette surcouche ne modifie pas l’entraînement du modèle.")

    # ========================= MODEL CARD =========================
    with tab_model:
        st.markdown("### 🧩 Model Card")
        a,b = st.columns(2)
        with a:
            st.markdown("#### 🎯 Objectif")
            st.write("Évaluer le risque de défaut d’un dossier de crédit et transformer la probabilité de défaut en score, décision et Expected Loss.")
            st.markdown("#### 🧠 Modèle principal")
            st.markdown("- Régression logistique\n- WoE / IV\n- Variables comportementales clés forcées\n- Platt scaling pour recalibrer la PD")
            st.markdown("#### 🔬 Benchmark")
            st.write("Random Forest utilisé comme référence ML pour comparer la capacité prédictive et l’importance des variables.")
        with b:
            st.markdown("#### 📊 Données")
            st.markdown(f"- Dataset : **German Credit Data**\n- Dossiers : **{len(artifacts['df']):,}**\n- Split : **60 % train / 20 % validation / 20 % test**\n- Cible : **default**")
            st.markdown("#### ⚙️ Cadre de décision")
            st.markdown(f"- Acceptation : score ≥ **{artifacts['accept_thresh']:.1f}**\n- Revue : entre les deux seuils\n- Refus : score < **{artifacts['refuse_thresh']:.1f}**")
            st.markdown("#### 💰 Expected Loss")
            st.code("Expected Loss = PD × LGD × EAD", language="text")
        st.divider()
        st.warning("Ce dashboard est un prototype analytique construit à des fins de démonstration. Les hypothèses de LGD/EAD, les proxys LTI/DTI et les seuils de décision doivent être validés avant toute utilisation opérationnelle ou réglementaire.")


if __name__ == "__main__":
    render_app()

# Lancement : streamlit run "credit_risk_app_banking.py"
