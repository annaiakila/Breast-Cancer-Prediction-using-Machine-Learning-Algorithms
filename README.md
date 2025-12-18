# BreastCancerApp

 **Breast Cancer Prediction System** — A small Flask web app that uses a trained machine learning model to predict whether a breast mass is benign or malignant from 9 input features.

---

## Table of Contents
1. Project overview 
2. Repo structure 
3. Files analyzed (frontend, backend, CSS, notebook, model) 
4. Data & preprocessing — step-by-step 
5. Model training pipeline (what was done) 
6. How to run the app locally (step-by-step) 
7. Model input ordering & encodings 
8. Known issues, caveats, and suggested fixes 
9. Improvements & next steps 
10. License & contact 

---

## 1) Project overview 
This repository implements a small web application to predict breast cancer diagnosis using classical ML algorithms and a persisted RandomForest model (`Breastcan.pkl`). The app provides a simple form (HTML + CSS) and a Flask backend to accept inputs, run the model, and return a human-readable result.

---

## 2) Repo structure 
- `app.py` — Flask backend, exposes `/` (form) and `/predict` (POST) endpoints.
- `templates/index.html` — Frontend form and page layout.
- `static/css/style.css` — Styles for the UI (responsive layout, colors).
- `breast-cancer-dataset.csv` — Raw dataset used for EDA and model training.
- `Breastcan.ipynb` — Jupyter notebook with EDA, preprocessing, model experiments and serialization.
- `Breastcan.pkl` — Serialized model loaded by `app.py` at runtime.
- `requirements.txt` — minimal Python dependencies used in this project.

---

## 3) Files analyzed — summary (detailed) 
### Frontend — `templates/index.html` 
- Contains a single form with 9 fields.
- Important: the app depends on a **specific order** when sending inputs (see comment inside the file). Fields (expected order):
  1. Year
  2. Age
  3. Menopause (encoded 0/1)
  4. Tumor Size (cm)
  5. Inv-Nodes
  6. Breast (0=Unknown,1=Left,2=Right)
  7. Metastasis (0/1)
  8. Breast Quadrant (0..5)
  9. History (0/1)
- The result is displayed in-page via a `{{ prediction_text }}` template variable.

### Backend — `app.py` 
- Loads `Breastcan.pkl` with `joblib.load()` at module import time.
- Provides `GET /` (serves form) and `POST /predict` endpoints.
- `predict()` builds a list from `request.form.values()` in form order, converts to float array and calls `model.predict()`.
- Error handling: catches exceptions and shows them to the user on the page.
- Note: the endpoint is order-dependent — if the form or HTML ordering changes the predictions will be wrong.

### CSS — `static/css/style.css` 
- Clean, modern styling with gradient background and a responsive grid-based form.
- Grid changes for tablets / mobile via media queries.

### Notebook — `Breastcan.ipynb` 
Key steps observed (in order):
1. Load CSV and perform EDA (plots: counts, histograms, heatmap).
2. Label-encode categorical columns (`Breast`, `Breast Quadrant`, `Diagnosis Result`).
3. Show class imbalance and attempt oversampling (but the notebook oversamples `Breast` column instead of `Diagnosis Result` — see Caveats below).
4. Clean numeric-like columns by stripping non-digit characters (`#`, `>`, etc.) and coerce to numeric.
5. Impute numeric missing values using median.
6. Scale numeric columns with `StandardScaler`.
7. Train/test split (20% test, `random_state=42`).
8. Fit multiple models: LogisticRegression, DecisionTree, RandomForest, SVM, XGBoost, LightGBM.
9. GridSearchCV on RandomForest to find `best_rf` and print best params.
10. Serialize (`joblib.dump(rf_model, 'Breastcan.pkl')`).

NOTE: Several training cells contain mistakes:
- Some models (RandomForest, SVM, XGBoost, LGBM) are trained *on the test set* (`X_test`, `Y_test`) instead of training data — this likely causes overly optimistic evaluation and is a reproducibility issue.
- The serialized target model is `rf_model` (RandomForest instance) that was fit on `X_test`/`Y_test` earlier — possibly not the tuned `best_rf` from GridSearch.
- There are small issues in metric prints where different predictions are passed into confusion matrix / classification report in some cells.

### Model — `Breastcan.pkl` 
- Contains a serialized RandomForest model (loaded by `app.py`).
- The model expects inputs in the same order as the front-end form (see above) and expects numeric/encoded features.
- There is no serialized preprocessing (LabelEncoders or StandardScaler) saved alongside the model in the repo — the app assumes user supplies already-encoded/numeric values.

---

## 4) Data & preprocessing — step-by-step 
1. Inspect `breast-cancer-dataset.csv` — columns: `S/N, Year, Age, Menopause, Tumor Size (cm), Inv-Nodes, Breast, Metastasis, Breast Quadrant, History, Diagnosis Result`.
2. Observed data issues:
   - Some entries contain `#` or trailing spaces (e.g., `Upper outer `) in categorical columns.
   - Some rows have missing / placeholder tokens in numeric columns (e.g., `#` in `Year` or `Inv-Nodes`).
3. Numeric cleanup performed in the notebook: strip non-digit characters with regex and convert to numeric with `pd.to_numeric(..., errors='coerce')`.
4. Missing numeric values imputed with median.
5. Categorical fields were label-encoded with `LabelEncoder()` (note: using `LabelEncoder` directly on mixed categories with trailing spaces can give inconsistent mappings — better to clean categories first and store encoders for inference).
6. StandardScaler was applied to numeric features before modeling.

---

## 5) Model training pipeline (what was done) 
Concise reproducible pipeline (based on `Breastcan.ipynb`):
1. Clean dataset (remove `S/N`, clean non-digit characters in numeric columns).
2. Encode categorical features (LabelEncoder used in the notebook).
3. Impute numeric missing values (median).
4. Scale numeric features (StandardScaler).
5. Split into train/test (80/20).
6. Train multiple classifiers and compare results.
7. Perform GridSearchCV on RandomForest and inspect best estimator.
8. Export RandomForest as `Breastcan.pkl` with `joblib.dump()`.

---

## 6) How to run the app locally (step-by-step) 
1. Clone or copy this folder.
2. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   venv\Scripts\activate    # Windows
   source venv/bin/activate   # macOS / Linux
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Verify `Breastcan.pkl` is present in the project root (it is required by `app.py`).
5. Start the Flask server:

   ```bash
   python app.py
   ```

6. Open your browser at http://127.0.0.1:5000/ and use the form to make predictions.

---

## 7) Model inputs and encodings 
- Inputs (order matters): Year, Age, Menopause, Tumor Size (cm), Inv-Nodes, Breast, Metastasis, Breast Quadrant, History.
- Encodings (from project notes / UI):
  - `Breast`: 0=Unknown, 1=Left, 2=Right
  - `Breast Quadrant`: 0=Unknown, 1=Lower inner, 2=Lower outer, 3=Upper inner, 4=Upper outer, 5=Upper outer (alt)
  - `Diagnosis Result` target values: 0=Benign, 1=Malignant
- Important: The running Flask app expects already-encoded numeric values in the same order. There is no preprocessing wrapper saved with the model. If you retrain, save a pipeline (e.g., `sklearn.pipeline.Pipeline`) that includes preprocessing + model and update the app to load this pipeline to avoid ordering/encoding mistakes.

---

## 8) Known issues, caveats & suggested fixes 
1. Order-dependence: `app.py` uses `request.form.values()` (which relies on the HTML form order). This is fragile. Fix: read `request.form['fieldname']` by names or accept JSON with explicit keys.
2. Data cleaning: dataset contains `#`, trailing spaces, and placeholders — needs normalization before encoding.
3. LabelEncoder usage: `LabelEncoder` should not be used on features across training and inference without saving the fitted encoder(s). Save your encoders or use consistent mapping or one-hot encoding.
4. Major bug in notebook: several classifiers were trained on the *test set* (`X_test`, `Y_test`) and not on training data — this invalidates reported metrics. Fix: ensure `model.fit(X_train, Y_train)` for training and `model.predict(X_test)` for evaluation.
5. Exported model mismatch: `joblib.dump(rf_model, 'Breastcan.pkl')` serializes `rf_model`, which may not be the tuned `best_rf`. Recommendation: export the final, validated model (e.g., `best_rf`) and consider saving a pipeline: `Pipeline([('scaler', StandardScaler()), ('model', best_rf)])`.
6. No model input validation on the Flask endpoint: the app casts to float directly and may break on invalid inputs. Add stricter validation and clearer error messages.

---

## 9) Improvements & next steps 
- Save a preprocessing + model pipeline and load it in `app.py` (ensures consistent preprocessing during inference).
- Persist LabelEncoder mappings or use `sklearn`'s `ColumnTransformer`/`OneHotEncoder` for safe, reproducible encoding.
- Fix training bug(s) so all models are trained on `X_train` and evaluated on `X_test`. Re-run GridSearch and export the best model.
- Add unit and integration tests for the Flask API (validate inputs, test predictions, ensure consistent outputs).
- Add a Dockerfile for reproducible deployment.
- Add CI that re-runs tests and smoke-tests the app.
- Improve the UI: show probability/confidence and input validation hints.

---

## 10) License & contact 
- This repo has no explicit license file — add a LICENSE (MIT or your preferred license) if you intend to share publicly.
- If you'd like, I can:
  - Fix the training cells and re-train a reproducible model that I then save as a pipeline.
  - Update `app.py` to accept named inputs and saved preprocessing artifacts.

---

If you'd like, I can open a PR that implements the two highest-priority fixes: (1) train/evaluate correctly and export a pipeline, and (2) update `app.py` to load and use that pipeline and accept name-based inputs for robustness. 

