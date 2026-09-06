from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_URL = "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = Path("outputs")


CLAUSES = {
    1: "Clause 1 — High Risk (probability ≥ 0.70): Offer a loyalty discount and a callback from a\nretention specialist within 48 hours.",
    2: "Clause 2 — Moderate Risk (0.40–0.70): Send a targeted email highlighting an underused\nservice or a contract upgrade offer.",
    3: "Clause 3 — New Customer, Any Risk, Tenure < 3 months: Route to the onboarding team\ninstead of the standard retention flow.",
    4: "Clause 4 — Non-Discrimination Rule: Retention explanations must never state or imply\nthat gender, senior-citizen status, or family/partner status contributed to a customer's risk\nscore, even where a statistical correlation exists in the data",
}
PROTECTED_PREFIXES = ("gender", "seniorcitizen", "partner", "dependents")


def load_data() -> pd.DataFrame:
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()
    data = pd.read_csv(StringIO(response.text))
    data.columns = data.columns.str.strip()
    text_columns = data.select_dtypes(include="object").columns
    data[text_columns] = data[text_columns].apply(lambda column: column.str.strip())
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    zero_tenure = data["TotalCharges"].isna() & data["tenure"].eq(0)
    data.loc[zero_tenure, "TotalCharges"] = 0
    data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())
    return data


def retrieve_clauses(probability: float, tenure: float) -> list[str]:
    retrieved = []
    if probability >= 0.70:
        retrieved.append(CLAUSES[1])
    elif probability >= 0.40:
        retrieved.append(CLAUSES[2])
    if tenure < 3:
        retrieved.append(CLAUSES[3])
    retrieved.append(CLAUSES[4])
    return retrieved


def is_protected_feature(feature_name: str) -> bool:
    return feature_name.lower().startswith(PROTECTED_PREFIXES)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = load_data()

    y = (data["Churn"] == "Yes").astype(int)
    X = data.drop(columns=["Churn", "customerID"])
    X = pd.get_dummies(X, drop_first=True).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_all_scaled = scaler.transform(X)

    baseline_model = LogisticRegression(max_iter=1000, random_state=42)
    complex_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )
    baseline_model.fit(X_train_scaled, y_train)
    complex_model.fit(X_train, y_train)

    baseline_train_pred = baseline_model.predict(X_train_scaled)
    baseline_test_pred = baseline_model.predict(X_test_scaled)
    complex_train_pred = complex_model.predict(X_train)
    complex_test_pred = complex_model.predict(X_test)

    metrics = pd.DataFrame([
        {
            "model": "Logistic Regression",
            "train_accuracy": accuracy_score(y_train, baseline_train_pred),
            "test_accuracy": accuracy_score(y_test, baseline_test_pred),
            "test_roc_auc": roc_auc_score(y_test, baseline_model.predict_proba(X_test_scaled)[:, 1]),
        },
        {
            "model": "Random Forest",
            "train_accuracy": accuracy_score(y_train, complex_train_pred),
            "test_accuracy": accuracy_score(y_test, complex_test_pred),
            "test_roc_auc": roc_auc_score(y_test, complex_model.predict_proba(X_test)[:, 1]),
        },
    ])
    print(metrics.round(4).to_string(index=False))
    print("\nBest model test report:\n")
    print(classification_report(y_test, baseline_test_pred))
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    data["predicted_churn_probability"] = baseline_model.predict_proba(X_all_scaled)[:, 1]
    data["segment"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(
        StandardScaler().fit_transform(
            data[["tenure", "MonthlyCharges", "predicted_churn_probability"]]
        )
    )
    segment_summary = data.groupby("segment").agg(
        customers=("segment", "size"),
        avg_tenure=("tenure", "mean"),
        avg_monthly_charges=("MonthlyCharges", "mean"),
        avg_churn_probability=("predicted_churn_probability", "mean"),
    )
    segment_summary["estimated_monthly_revenue_at_risk"] = (
        segment_summary["customers"]
        * segment_summary["avg_monthly_charges"]
        * segment_summary["avg_churn_probability"]
    )
    tenure_midpoint = segment_summary["avg_tenure"].median()
    spend_midpoint = segment_summary["avg_monthly_charges"].median()
    risk_midpoint = segment_summary["avg_churn_probability"].median()

    def segment_name(row: pd.Series) -> str:
        tenure_name = "New" if row["avg_tenure"] < tenure_midpoint else "Mature"
        spend_name = "Low-Spend" if row["avg_monthly_charges"] < spend_midpoint else "High-Spend"
        risk_name = "High-Risk" if row["avg_churn_probability"] >= risk_midpoint else "Low-Risk"
        return f"{tenure_name}, {spend_name}, {risk_name}"

    segment_summary["segment_name"] = segment_summary.apply(segment_name, axis=1)
    segment_summary = segment_summary.sort_values(
        "estimated_monthly_revenue_at_risk", ascending=False
    )
    print("\nSegment summary:\n")
    print(segment_summary.round(2).to_string())
    segment_summary.to_csv(OUTPUT_DIR / "segment_summary.csv")

    customer_index = data["predicted_churn_probability"].idxmax()
    customer_scaled = X.loc[[customer_index]]
    customer_contributions = scaler.transform(customer_scaled)[0] * baseline_model.coef_[0]
    contribution_series = pd.Series(customer_contributions, index=X.columns)
    top_features = contribution_series.abs().sort_values(ascending=False).head(3).index
    safe_top_features = [feature for feature in top_features if not is_protected_feature(feature)]
    probability = float(data.loc[customer_index, "predicted_churn_probability"])
    tenure = float(data.loc[customer_index, "tenure"])
    retrieved = retrieve_clauses(probability, tenure)
    advisory_inputs = {
        "customer_index": customer_index,
        "risk_probability": probability,
        "tenure": tenure,
        "safe_top_features": safe_top_features,
        "retrieved_clause_numbers": [
            number for number, clause in CLAUSES.items() if clause in retrieved
        ],
    }
    pd.DataFrame([advisory_inputs]).to_json(
        OUTPUT_DIR / "advisory_inputs.json", orient="records", indent=2
    )
    (OUTPUT_DIR / "retrieved_clauses.txt").write_text(
        "\n\n".join(retrieved), encoding="utf-8"
    )
    print("\nAdvisory inputs:\n", advisory_inputs)


if __name__ == "__main__":
    main()
