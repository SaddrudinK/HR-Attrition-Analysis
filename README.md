# HR Attrition Analysis


**What**: Exploratory analysis and attrition prediction using the `HR_comma_sep.csv` dataset. The project contains data cleaning, EDA, business insights, and a baseline classification pipeline (Logistic Regression + Random Forest). Outputs: charts, model metrics, and a saved model.


**Why useful**: Employee attrition is a high-impact problem — reducing churn improves retention, reduces hiring costs, and helps HR plan interventions.


## Key highlights / Resume lines
- Performed HR analytics to identify drivers of employee attrition (satisfaction, workload, tenure).
- Built classification models (Logistic Regression, Random Forest) to predict employee churn; reported precision/recall/F1 and feature importance.
- Packaged results with reproducible scripts and visualizations.


## How to run (quick)
1. Clone repo.
2. Put `HR_comma_sep.csv` into the `data/` folder.
3. (Optional) create virtual environment: `python -m venv venv && source venv/bin/activate` (or Windows equivalent).
4. Install requirements: `pip install -r requirements.txt`.
5. Run analysis: `python scripts/analysis.py --data data/HR_comma_sep.csv --out outputs/`.
6. Check `outputs/figures/` for charts and `outputs/models/` for saved models.


## Files
- `scripts/analysis.py` — full pipeline for EDA, preprocessing, modeling, metrics, and saving outputs.
- `notebooks/exploration.ipynb` — (optional) step-by-step EDA for interactive exploration.


## Data source
Place the provided `HR_comma_sep.csv` into `data/`. If you want to link to Kaggle, mention the source in the README.


## Reproducibility
- The `analysis.py` script checks for the dataset path and creates `outputs/` automatically.
- Random seeds are fixed so results are reproducible.
