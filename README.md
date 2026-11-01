# Machine Learning Capstone Project: Projects 3 and 4

**Author:** Marzia Binte Iqbal

**Course:** Machine Learning, Chapter 14 Capstone Project

Two end-to-end machine learning projects. Each follows the same pipeline: data acquisition and EDA, preprocessing, a baseline model plus a complex model, hyperparameter tuning, and rigorous evaluation.

| # | Project | Type | Baseline | Complex model | Main metrics |
|---|---------|------|----------|---------------|--------------|
| 3 | Avocado Price Predictor | Regression | Linear Regression (and a Dummy mean model) | Random Forest (RandomizedSearchCV, time-ordered CV) | RMSE, MAE, R² |
| 4 | Heart Disease Diagnostic Tool | Binary classification (small dataset) | Logistic Regression (and a Dummy model) | SVM and Random Forest (RandomizedSearchCV) | Recall, Precision, F1, ROC-AUC, Confusion Matrix, repeated CV |

## Repository layout

```
.
├── project3_avocado_price_predictor.ipynb
├── project4_heart_disease_diagnostic.ipynb
├── requirements.txt
├── README.md
├── data/        # put the CSV files here (not included)
└── figures/     # charts are saved here automatically when you run the notebooks
```

## Datasets (not included, download them yourself)

| Project | Dataset | Expected file name | Required columns |
|---------|---------|--------------------|------------------|
| 3 | "Avocado Prices" (Hass Avocado Board) on Kaggle | `data/avocado.csv` | `Date, AveragePrice, Total Volume, 4046, 4225, 4770, Total Bags, Small Bags, Large Bags, XLarge Bags, type, region` (`year` and an index column are ignored) |
| 4 | "Heart Disease" (UCI / Cleveland) on Kaggle | `data/heart.csv` | `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal` and a label column `target` (or `num`) |

Check the column names after downloading. Each notebook stops with a clear error message if a column is missing. If your file has a different name, change `DATA_PATH` in the Setup cell.

**Heart Disease dataset, two things to check:**
1. Some copies have about 1,000 rows made of the same ~300 patients repeated. The notebook removes duplicate rows before splitting, so the same patient cannot appear in both train and test. Report the number of patients you end up with.
2. Check what the label values mean in your copy. If 1 means *healthy* instead of *disease*, set `INVERT_TARGET = True` in the Setup cell so that 1 always means disease.

## How to run

### Locally
```bash
pip install -r requirements.txt
jupyter notebook
```
Open a notebook and choose **Run All**.

### Google Colab
1. Upload the notebook and the CSV file.
2. Set `DATA_PATH` to the path of the uploaded CSV (right-click the file, **Copy path**), then **Runtime → Run all**.

## Notes on the method

- **Project 3 splits by time, not at random.** The oldest 80% of dates are used for training and the newest 20% for testing, and tuning uses `TimeSeriesSplit`. A random split would let the model see future weeks, which is data leakage.
- **Project 4 puts all preprocessing inside pipelines** (imputer, scaler, one-hot encoder), so they learn from training data only. The decision threshold is chosen from out-of-fold training predictions by maximising F2 (recall counts twice as much as precision), never from the test set. Because the test set is tiny, the final comparison also uses 5x5 repeated stratified cross-validation.
- This is a learning project. The heart disease model is not a medical device.

## Results

Results on the held-out test sets, from the executed notebooks.

| Project | Final model | Key metrics (test set) |
|---------|-------------|------------------------|
| 3 Avocado Price Predictor | Random Forest (tuned with RandomizedSearchCV, time-ordered CV) | RMSE 0.288, MAE 0.213, R² 0.467 (Linear Regression baseline: RMSE 0.290, MAE 0.217, R² 0.460; Dummy mean: RMSE 0.423, R² -0.152) |
| 4 Heart Disease Diagnostic Tool | SVM (tuned), decision threshold 0.158 chosen on out-of-fold training predictions | Recall 0.939 (31 of 33), Precision 0.738, F1 0.827, ROC-AUC 0.917 on 61 test patients (Logistic Regression baseline at threshold 0.5: Recall 0.848, Precision 0.848, ROC-AUC 0.905) |

**Notes**
- Project 3: the untuned Random Forest scored slightly better on the test set (RMSE 0.279) but the tuned one was chosen on cross-validation of the training data. The differences between Linear Regression and the Random Forest models are small (about 0.01 USD RMSE). The model underestimates the high prices of autumn 2017.
- Project 4: the test set is small (61 patients), so one patient moves recall by about 3 percentage points. In 5x5 repeated cross-validation on all 302 patients, the three real models have ROC-AUC 0.913 to 0.915.
- Project 4, label check: in this copy of the data, patients with `target = 1` have a higher maximum heart rate and lower ST depression, which is the opposite of what is expected for heart disease. The label may be reversed. These results were produced with `INVERT_TARGET = False`; verify the label direction against the original UCI data and re-run if needed.
