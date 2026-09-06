from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.metrics import accuracy_score


PROTECTED_PREFIXES = (
    "gender",
    "seniorcitizen",
    "partner",
    "dependents",
)


def clean_total_charges(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    cleaned["TotalCharges"] = pd.to_numeric(
        cleaned["TotalCharges"].astype(str).str.strip(),
        errors="coerce",
    )
    zero_tenure = cleaned["TotalCharges"].isna() & cleaned["tenure"].eq(0)
    cleaned.loc[zero_tenure, "TotalCharges"] = 0
    return cleaned


def accuracy_comparison(
    model_name: str,
    train_actual: pd.Series,
    train_predicted: Iterable[int],
    test_actual: pd.Series,
    test_predicted: Iterable[int],
) -> dict[str, float | str]:
    return {
        "model": model_name,
        "train_accuracy": accuracy_score(train_actual, train_predicted),
        "test_accuracy": accuracy_score(test_actual, test_predicted),
    }


def estimated_revenue_at_risk(
    customers: pd.Series,
    average_monthly_charges: pd.Series,
    average_churn_probability: pd.Series,
) -> pd.Series:
    return customers * average_monthly_charges * average_churn_probability


def retrieve_clauses(probability: float, tenure: float) -> list[str]:
    clause_1 = (
        "Clause 1 — High Risk (probability ≥ 0.70): Offer a loyalty discount and a callback from a\n"
        "retention specialist within 48 hours."
    )
    clause_2 = (
        "Clause 2 — Moderate Risk (0.40–0.70): Send a targeted email highlighting an underused\n"
        "service or a contract upgrade offer."
    )
    clause_3 = (
        "Clause 3 — New Customer, Any Risk, Tenure < 3 months: Route to the onboarding team\n"
        "instead of the standard retention flow."
    )
    clause_4 = (
        "Clause 4 — Non-Discrimination Rule: Retention explanations must never state or imply\n"
        "that gender, senior-citizen status, or family/partner status contributed to a customer's risk\n"
        "score, even where a statistical correlation exists in the data"
    )

    clauses = []
    if probability >= 0.70:
        clauses.append(clause_1)
    elif probability >= 0.40:
        clauses.append(clause_2)
    if tenure < 3:
        clauses.append(clause_3)
    clauses.append(clause_4)
    return clauses


def is_protected_feature(feature_name: str) -> bool:
    normalized = feature_name.lower()
    return normalized.startswith(PROTECTED_PREFIXES)


def safe_prompt_inputs(
    probability: float,
    tenure: float,
    top_features: list[dict[str, float | str]],
) -> dict[str, object]:
    safe_features = [
        item for item in top_features
        if not is_protected_feature(str(item["feature"]))
    ]
    return {
        "risk_probability": probability,
        "tenure": tenure,
        "top_features": safe_features,
        "retrieved_clauses": retrieve_clauses(probability, tenure),
    }


if __name__ == "__main__":
    print(retrieve_clauses(0.75, 2))
