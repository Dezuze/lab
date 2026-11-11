# S5 Machine Learning Lab — Experiments 1 to 12

This pack is restricted to the first 12 detailed practicals in the supplied PCCSL508 manual. Each experiment has a compact Python program and the PDF contains expanded, step-by-step algorithms.

## Offline setup
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create and activate a virtual environment:
   - Windows: `python -m venv .venv` then `.venv\\Scripts\\activate`
   - Linux/macOS: `python -m venv .venv` then `source .venv/bin/activate`
4. Install packages: `pip install -r requirements.txt`
5. Run a program, for example: `python 03_ridge_lasso.py`

## Offline datasets
- Fully bundled in scikit-learn and available offline: Diabetes, Breast Cancer, Iris and Digits (`load_*` datasets).
- `fetch_california_housing` and `fetch_20newsgroups` may require one internet download before they are cached. Run them once while connected, then they can generally be used offline.
- Fashion-MNIST and Online Retail are not bundled with scikit-learn. Download them once and save locally before going offline.
- For CSV use, keep the file in this folder and change the filename/column names in the corresponding script. Experiment 2 expects `mpg.csv`; experiment 6 expects `pima.csv`; experiment 9 prefers `online_retail.csv` with the standard Online Retail columns and derives RFM categories (it also accepts a prepared `customers.csv`); experiment 11 expects `winequality-red.csv` with semicolon separator.
- Experiment 8's supplied script uses the built-in Digits dataset as an offline image-classification substitute. It is not identical to Fashion-MNIST. For the exact manual experiment, use cached/downloaded Fashion-MNIST.

## Important manual numbering note
The table of contents on page 2 lists a different title for experiments 10–12 than the detailed write-ups later in the PDF. This pack follows the detailed practical write-ups: 10 Linear SVM, 11 MLP Architecture, 12 K-Means on Digits. Confirm the numbering with your lab instructor if the table-of-contents list is what your class follows.

## CSV helper
`common.py` contains reusable `load_csv(path, target)` and `split_scale(...)` functions for ordinary tabular CSV data. Text and image CSVs need task-specific preprocessing.

## Quick concepts
- `fit()` trains a model; `predict()` produces outputs.
- `fit_transform()` learns a transformation and applies it to training data; `transform()` applies the same learned transformation to test data.
- Fit scalers and vectorizers on training data only to avoid data leakage.
- Experiment 6 fits its median imputer on the training split only, then compares unscaled and standardized inputs.
- Regression: MSE lower is better; R² closer to 1 is generally better.
- Classification: accuracy, precision, recall, F1 and log-loss.
- Clustering: inertia and silhouette score.
- `np.logspace()` creates values evenly spaced on a logarithmic scale, useful for testing regularization strengths.

## Alignment with the manual
The compact programs are intended to complete the stated aims/tasks, not merely demonstrate an estimator. Experiment 1 explicitly compares the Normal Equation and Gradient Descent. Experiment 9 implements the ID3 entropy/information-gain recursion rather than substituting a library tree. Experiment 10 plots the decision boundary, both margins, and support vectors.
