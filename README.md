Next-Day Air Quality Prediction for Delhi
Applied Machine Learning | Project P02 | Tensor Tribe
Can historical pollution readings and calendar patterns predict tomorrow's PM2.5 concentration and AQI category better than simply assuming tomorrow will be like today?

The idea
Air-quality warnings are most useful before pollution becomes dangerous. Our project uses daily pollution observations to build two forecasts: regression for tomorrow's PM2.5 concentration (µg/m³), and classification for tomorrow's AQI category (Good, Satisfactory, Moderate, Poor, Very Poor, or Severe).

Our benchmark is persistence: tomorrow's PM2.5 = today's PM2.5. Since pollution levels are strongly related from one day to the next, a more complicated model should only be preferred if it actually improves the forecast.
Meet Tensor Tribe
Member	SAP ID	Areas of contribution / assigned work
Abhishek	590028847	Repository setup, Delhi EDA, preprocessing and baseline/regression work
Atishey	590028784	PM2.5 and AQI EDA, calendar features, regression-vs-persistence evaluation
Deepanshu	590028785	Date continuity, pollutant analysis, lag features, time-series split and modeling
Ishaan	590024546	Missingness analysis, rolling features, initial linear model and evaluation
Sahil Dabra	590023527	Next-day target validation, leakage checks, additional models and evaluation


The table includes both completed contributions and assigned upcoming responsibilities. The Git commit and pull-request history documents completed work.
Data and study area
We use city_day.csv from Air Quality Data in India (2015–2020), compiled from CPCB measurements by Rohan Rao. We focus on Delhi, which has substantial pollution variation and seasonal peaks.

Important fields include Date, PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, AQI, and AQI_Bucket. The raw file stays unchanged in data/raw/.
How the project works
Raw data (city_day.csv)
          |
          v
Delhi filtering + continuous daily dates + missingness audit
          |
          v
Documented missing-data handling
          |
          v
Historical pollutant lags + rolling statistics + calendar features
          |
          v
Tomorrow's PM2.5 and AQI_Bucket targets
          |
          v
Chronological training / test split
          |
          +--> Regression: persistence, linear, ridge, lasso, nonlinear
          |
          +--> Classification: logistic, tree-based models, SVM
          |
          v
Evaluation + error analysis + reproducibility
Preventing data leakage
- Reindex to a continuous daily calendar before creating lags.
- Use historical pollutant lags (t-1, t-2, t-3, t-7).
- Calculate rolling features from windows ending at t-1.
- Tomorrow's calendar date is known in advance and is safe to use.
- Keep tomorrow's target columns out of the input features.
- Split chronologically; fit preprocessing and tune models on training data only.
- Include same-day AQI only when it would genuinely be available at forecast time.
What we found in EDA
- Delhi had 2,007 non-missing PM2.5 readings, with mean 117.20 µg/m³, median 94.62 µg/m³, and maximum 685.36 µg/m³.
- AQI was available for 1,999 of 2,009 Delhi records (only 0.50% missing).
- AQI labels were imbalanced: Poor 27.11%, Very Poor 26.01%, Moderate 25.96%, Severe 11.96%, Satisfactory 7.90%, and Good 1.05%.
- Mean PM2.5 was 118.41 µg/m³ during 1 January–24 March 2020 versus 47.93 µg/m³ during 25 March–30 June 2020. This is a descriptive comparison, not proof that lockdown alone caused the difference; seasonal and weather effects also matter.
Checkpoint 1: initial regression results
The first Linear Regression model was evaluated against persistence on the same 183 test observations.
Model	MAE (µg/m³) ↓	RMSE (µg/m³) ↓	R² ↑
Persistence baseline	19.7454	29.4525	0.7209
Linear Regression	19.6143	28.4335	0.7399


Linear Regression improved MAE by 0.66% and RMSE by 3.46%. The gain is small, especially for MAE: persistence is already a strong baseline. These results are preliminary until the final feature-availability and leakage review is complete.
Evaluation approach
For regression, we report MAE, RMSE and R² for every model alongside persistence, plus predicted-vs-actual plots and errors during sudden pollution spikes.

For classification, we plan confusion matrices, per-class precision/recall, macro-F1 and multiclass one-vs-rest ROC/AUC. Because AQI categories are imbalanced, plain accuracy can hide poor predictions for rare classes.
Repository structure
aml-2026-pNN-gM/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/city_day.csv
│   └── processed/
├── notebooks/
│   ├── 01-eda.ipynb
│   ├── 02-preprocessing.ipynb
│   ├── 03-models.ipynb
│   └── 04-evaluation.ipynb
├── src/
├── results/
│   ├── figures/
│   └── metrics.csv
├── report/report.pdf
└── REPRODUCIBILITY.md
Getting started
1. Clone the repository
git clone https://github.com/ajain347/NextDay-Air-quality-index_predictor.git
cd NextDay-Air-quality-index_predictor
2. Set up Python
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
On Windows PowerShell, use venv\Scripts\Activate.ps1 instead of source venv/bin/activate.

3. Confirm the dataset

Make sure the unmodified city_day.csv exists in data/raw/ (download it from the dataset page if necessary).

4. Run notebooks in order
1. 01-eda.ipynb — exploration, missingness and seasonal patterns.
2. 02-preprocessing.ipynb — historical features and next-day targets.
3. 03-models.ipynb — model training and tuning.
4. 04-evaluation.ipynb — final metrics and figures.
Open notebooks in VS Code or Jupyter using the configured Python environment. See REPRODUCIBILITY.md for exact reproduction steps when finalized. Later-stage notebooks and outputs are still being developed.
How we collaborate
Each member works on a dedicated Git branch, pushes meaningful commits, and opens a pull request for review before merging into main. The repository history shows the project's progress across the semester. The planned milestones include checkpoint-1 and checkpoint-2; the final submission is the repository URL, with a 4–6 page report in report/report.pdf.
Limitations and next steps
Missing observations, rare AQI categories, and sudden winter pollution spikes make this a difficult forecasting problem. Next we will complete regularized and nonlinear models, multiclass evaluation, leakage audits, error analysis, and clean-checkout reproducibility checks.

Our central question remains: Is the ML forecast actually better than simply using today's PM2.5?
