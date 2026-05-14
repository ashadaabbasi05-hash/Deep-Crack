# DeepCrack – AI Password Analyzer

DeepCrack is a Streamlit demo app that predicts password strength using a Random Forest model. It offers explainability, interactive visualizations, and a humorous roast system.

## Features

- Password strength prediction and score (weak, medium, strong)
- Crack time estimate
- Feature breakdown and visualizations
- Password generator and history log
- Export analysis report (text)

## Quick start

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the app:

```powershell
streamlit run app.py
```

Notes

- The app will train a lightweight RandomForest the first time if a saved `model.pkl` is not present.
- Toggle sound in the sidebar to enable a short tone for strong passwords.

License: MIT
