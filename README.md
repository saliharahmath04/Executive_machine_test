# Telco Customer Churn Analysis

This project contains the local VS Code version of the churn analysis, modelling, customer segmentation, and retention-playbook retrieval work.

## Files

- `telco_churn_analysis.py`: Main local analysis script.
- `task1_churn_analysis.py`: Part 1 KPI analysis.
- `part4_governance_code.py`: Reusable governance and prompt-safety helpers.
- `part4_reflections.rtf`: Word-compatible reflection and board memo.
- `filepath.txt`: Original dataset URL.
- `requirements.txt`: Python dependencies.

The script downloads the Telco Customer Churn CSV from the URL in the code, cleans `TotalCharges`, compares Logistic Regression with Random Forest, creates four segments, calculates revenue at risk, and retrieves the applicable retention clauses.

## Run in VS Code

Open this folder in VS Code, then open a PowerShell terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python .\telco_churn_analysis.py
```

The script creates an `outputs` folder containing model metrics, the segment summary, advisory inputs, and retrieved clauses.

## GitHub

From this folder:

```powershell
git init
git add .
git commit -m "Add telco churn analysis and governance artifacts"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Replace the repository URL with your own GitHub repository URL. Do not commit API keys or customer data. The current Task 6 work does not call an LLM because no API key is available; it records the retrieval inputs and policy clauses instead.
