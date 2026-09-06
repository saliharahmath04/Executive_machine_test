from __future__ import annotations

from io import StringIO

import pandas as pd
import requests


DATA_URL = "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def churn_rate(frame: pd.DataFrame) -> float:
    return (frame["Churn"] == "Yes").mean()


response = requests.get(DATA_URL, timeout=30)
response.raise_for_status()
data = pd.read_csv(StringIO(response.text))
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data["ChurnBinary"] = (data["Churn"] == "Yes").astype(int)

print(f"Rows: {len(data):,}")
print(f"Overall churn rate: {churn_rate(data):.2%}")

contract_summary = (
    data.groupby("Contract", sort=False)
    .agg(customers=("customerID", "size"), churn_rate=("ChurnBinary", "mean"))
)
contract_summary["churn_rate"] = contract_summary["churn_rate"].map(lambda value: f"{value:.2%}")
print("\nChurn by Contract:")
print(contract_summary.to_string())

internet_summary = (
    data.groupby("InternetService", sort=False)
    .agg(customers=("customerID", "size"), churn_rate=("ChurnBinary", "mean"))
)
internet_summary["churn_rate"] = internet_summary["churn_rate"].map(lambda value: f"{value:.2%}")
print("\nChurn by InternetService:")
print(internet_summary.to_string())

print(f"\nPearson correlation (tenure, ChurnBinary): {data['tenure'].corr(data['ChurnBinary']):.3f}")
tenure_summary = data.groupby("Churn", sort=False)["tenure"].agg(["count", "mean", "median"])
print("\nTenure by Churn:")
print(tenure_summary.to_string(float_format=lambda value: f"{value:.1f}"))