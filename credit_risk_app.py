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
                                "map": {1: "< 0 DM", 2: "0-200 DM", 3: ">= 200 DM / salaire domicilie", 4: "Pas de compte courant"},
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
                                     "map": {1: "< 100 DM", 2: "100-500 DM", 3: "500-1000 DM", 
                                             4: ">= 1000 DM", 5: "Inconnu / pas d'epargne"},
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

EXCLUDED_RAW_COLUMNS = ["Sex & Marital Status", "Foreign Worker", "Telephone"] # car variables protegées / obsolètes

# Milieu de bracket approximatif (en % du revenu disponible) pour la variable "Instalment per cent",
INSTALLMENT_RATE_MIDPOINT = {1: 0.10, 2: 0.20, 3: 0.30, 4: 0.40}

# Variables "comportementales" que l'on conserve dans le modele, meme si leur IV etait faible.
FORCED_BEHAVIORAL_FEATURES = ["credit_history", "checking_account_status", 
                              "existing_credits_count", "savings_status"]

# Multiplicateur applique aux points de scorecard de ces variables (surcouche "expert judgment", pour renforcer le poids des variables comportementales) 
BEHAVIORAL_POINTS_BOOST = 1.3


def load_and_prepare_data(csv_path) :
    
    raw = pd.read_csv(csv_path)

    df = pd.DataFrame(index=raw.index)
    df["default"] = 1 - raw["Creditability"]  # Creditability=1 -> bon payeur ; on veut default=1 -> mauvais payeur

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

    # Conserve les variables protegees/proxy uniquement pour le monitoring de fairness (jamais utilisees comme predicteurs)
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
def train_pipeline(csv_path):
    df = load_and_prepare_data(csv_path)

    iv_results = {}
    for f in CANDIDATE_NUMERIC:
        _, iv = compute_woe_iv(df, f, bins=5, is_numeric=True)
        iv_results[f] = iv
    for f in CANDIDATE_CATEGORICAL:
        _, iv = compute_woe_iv(df, f, is_numeric=False)
        iv_results[f] = iv
    iv_table = pd.Series(iv_results).sort_values(ascending=False).rename("IV").to_frame()

    # Selection par IV (> 0.02), en forcant l'inclusion des variables comportementales clés
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


def score_client(raw_inputs, artifacts):

    """raw_inputs doit fournir toutes les variables brutes necessaires au calcul des ratios et a l'application du WoE"""

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

def render_app():
    st.set_page_config(page_title="Credit Risk Assessment", page_icon="\U0001F4B3", layout="wide")
    st.title("\U0001F4B3 Credit Risk Assessment")
    st.caption(" **Scorecard** : Logistic Regression + WoE entrainé sur le German Credit Data (1000 dossiers réels), calibre par recalibrage de Platt.") 
    st.caption(" **Décision** : Accepte / Revue / Refus + Tarification par Expected Loss.")

    if not os.path.exists(GERMAN_CREDIT_CSV):
        st.error(f"Fichier de donnees introuvable : `{GERMAN_CREDIT_CSV}`.\n\n"
                 "Placez le CSV German Credit (colonnes Statlog standard) a côté de ce script sous ce nom, "
                 "ou modifiez la constante `GERMAN_CREDIT_CSV` en tete de fichier."
        )
        st.stop()

    artifacts = train_pipeline(GERMAN_CREDIT_CSV)

    tab_form, tab_monitoring, tab_explain = st.tabs(["\U0001F4DD Evaluer un dossier", "\U0001F4CA Monitoring", "\U0001F50E Explicabilite"])

    with tab_form:
        with st.form("credit_form"):
            st.subheader("Profil du client")
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", 18, 90, 35)
                checking_account_status = st.selectbox("Statut du compte courant", list(CODEBOOK["Account Balance"]["map"].values()))
                credit_history = st.selectbox("Historique de credit", list(CODEBOOK["Payment Status of Previous Credit"]["map"].values()))
                savings_status = st.selectbox("Epargne / valeurs mobilieres", list(CODEBOOK["Value Savings/Stocks"]["map"].values()))
                employment_since = st.selectbox("Anciennete professionnelle", list(CODEBOOK["Length of current employment"]["map"].values()))
                job = st.selectbox("Emploi", list(CODEBOOK["Occupation"]["map"].values()))
            with col2:
                existing_credits_count = st.number_input("Nombre de credits existants (cette banque)", 1, 6, 1)
                dependents = st.number_input("Personnes a charge", 1, 3, 1)
                residence_since = st.number_input("Anciennete a l'adresse actuelle (annees)", 1, 4, 2)
                guarantors = st.selectbox("Garant / co-emprunteur", list(CODEBOOK["Guarantors"]["map"].values()))
                property_ = st.selectbox("Bien le plus valorisable", list(CODEBOOK["Most valuable available asset"]["map"].values()))
                housing = st.selectbox("Logement", list(CODEBOOK["Type of apartment"]["map"].values()))
                other_installment_plans = st.selectbox("Autres credits en cours", list(CODEBOOK["Concurrent Credits"]["map"].values()))

            st.subheader("Le prêt demande")
            col3, col4 = st.columns(2)
            with col3:
                credit_amount = st.number_input("Montant du crédit (DM)", 250, 20000, 3000, step=100)
                duration_months = st.number_input("Durée (mois)", 4, 72, 24)
            with col4:
                purpose = st.selectbox("Objet du crédit", list(CODEBOOK["Purpose"]["map"].values()))
                installment_rate_label = st.select_slider("Mensualité en % du revenu disponible",
                                                          options=[1, 2, 3, 4],
                                                          format_func=lambda c: {1: "< 20%", 2: "20-25%", 3: "25-35%", 4: ">= 35%"}[c],
                                                          value=2,
                )

            submitted = st.form_submit_button("Evaluer le dossier", width='stretch')

        if submitted:
            other_debt_elsewhere = other_installment_plans != "Aucun autre credit"
            has_collateral = (property_ in ["Bien immobilier", "Epargne / assurance-vie"]) or (guarantors != "Aucun")

            raw_inputs = dict(age=age, checking_account_status=checking_account_status, credit_history=credit_history,
                              savings_status=savings_status, employment_since=employment_since, job=job,
                              existing_credits_count=existing_credits_count, dependents=dependents,
                              residence_since=residence_since, guarantors=guarantors, property=property_,
                              housing=housing, other_installment_plans=other_installment_plans,
                              credit_amount=credit_amount, duration_months=duration_months, purpose=purpose,
                              installment_rate_code=installment_rate_label,
                              other_debt_elsewhere=other_debt_elsewhere, has_collateral=has_collateral,
            )
            result = score_client(raw_inputs, artifacts)

            st.divider()
            st.subheader("Resultat de l'evaluation")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Credit Score", f"{result['credit_score']:.0f}")
            c2.metric("Probability of Default", f"{result['PD']:.2%}")
            c3.metric("Decision", result["decision"])
            c4.metric("Expected Loss", f"{result['expected_loss']:,.0f} DM")

            color = {"ACCEPTE": "green", "REVUE": "orange", "REFUSE": "red"}[result["decision"]]
            st.markdown(f"**Decision automatique : :{color}[{result['decision']}]**")
            if result["decision"] == "REVUE":
                st.info("Ce dossier se situe dans la zone intermediaire : il est transmis a un analyste "
                        "plutot que de faire l'objet d'une decision 100% automatisée.")

            with st.expander("Details du calcul"):
                st.write(f"- PTI (mensualite / revenu) : {result['PTI']:.1%}")
                st.write(f"- LTI (credit / revenu annuel, proxy) : {result['LTI']:.2f}")
                st.write(f"- DTI (endettement global, proxy) : {result['DTI']:.1%}")
                st.write(f"- LGD retenue : {result['LGD']:.0%} (selon garantie/bien/garant)")
                st.write(f"- EAD retenue : {result['EAD']:,.0f} DM")
                st.caption("LTI/DTI sont des proxys documentes : le German Credit ne fournit pas de revenu "
                           "brut. A valider sur des donnees reelles avant tout usage en production ")

    with tab_monitoring:
        st.subheader("Suivi de la performance du modele (test hold-out)")
        m = artifacts["metrics"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("AUC", f"{m['AUC_test']:.3f}", f"{m['AUC_test'] - m['AUC_train']:+.3f} vs train")
        c2.metric("Gini", f"{m['Gini_test']:.3f}")
        c3.metric("KS", f"{m['KS_test']:.3f}")
        c4.metric("PR-AUC", f"{m['PR_AUC_test']:.3f}")
        c5.metric("Brier", f"{m['Brier_test']:.3f}")

        c6, c7 = st.columns(2)
        c6.metric("Taux de defaut observe (test)", f"{m['Taux_defaut_test']:.1%}",
                   f"{m['Taux_defaut_test'] - m['Taux_defaut_train']:+.1%} vs train")
        psi_val = m["PSI_train_vs_test"]
        psi_status = "stable" if psi_val < 0.10 else ("a surveiller" if psi_val < 0.25 else "derive significative")
        c7.metric("PSI (train vs test)", f"{psi_val:.3f}", psi_status)

        st.markdown("##### Courbe de calibration de la PD")
        frac_pos, mean_pred = artifacts["calib_curve"]
        calib_df = pd.DataFrame({"PD predite (moyenne par bin)": mean_pred, "Taux de defaut observe": frac_pos})
        st.line_chart(calib_df.set_index("PD predite (moyenne par bin)"))

        st.markdown("##### Politique economique par bande de PD")
        st.caption("La PD alimente directement la **tarification**, le **plafond de credit indicatif** et le "
                   " **pilotage du portefeuille**, pas seulement le classement des dossiers.")
        st.dataframe(artifacts["pricing_policy"], width='stretch')

        st.markdown("##### Feuille de route monitoring en production")
        st.markdown(
            "- Suivi periodique : AUC/Gini, KS, calibration de la PD, taux de defaut observe, PSI, "
            "taux d'acceptation, stabilite des variables d'entree.\n"
            "- Alertes automatiques si PSI > 0.25, degradation d'AUC > 0.05, ou derive du taux "
            "d'acceptation.\n"
            "- Backtesting periodique de la PD par cohorte et analyse des migrations de score."
        )
        with st.expander("Limites assumees de ce prototype"):
             st.markdown(
                "- **LGD** et **EAD** sont simplifiees (LGD binaire selon garantie, EAD = montant du "
                "credit) : un vrai deploiement necessite des modeles LGD/EAD/CCF dedies, tenant compte "
                "des couts de recouvrement et de l'evolution de l'exposition.\n"
                "- **DTI/LTI** sont des proxys : le German Credit Data ne fournit pas de revenu brut.\n"
                "- Comparaison avec XGBoost/LightGBM, validation croisee temporelle, tests de stress et "
                "documentation de validation independante restent a faire pour un usage reglementaire."
            )

    with tab_explain:
        st.subheader("Contribution des variables au score")
        st.caption("Coefficients de la regression logistique (variables WoE). Les variables comportementales "
                   f"(historique de credit, compte courant, epargne, nombre de credits) sont affichees avec la "
                   f"surcouche d'importance metier (x{BEHAVIORAL_POINTS_BOOST}) appliquee a l'affichage des "
                   f"points de scorecard."
        )
        st.bar_chart(artifacts["coef_series"].sort_values())

        st.subheader("Importance par permutation (Random Forest, benchmark ML)")
        st.bar_chart(artifacts["perm_importance"].head(12))

        st.subheader("Information Value par variable")
        st.dataframe(artifacts["iv_table"], width='stretch')



if __name__ == "__main__":
    render_app()




# Lancement : streamlit run "chemin.py"