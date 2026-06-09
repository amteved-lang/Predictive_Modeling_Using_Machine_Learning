import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# -----------------------------------
# 1. Create project folders
# -----------------------------------
os.makedirs("charts", exist_ok=True)

# -----------------------------------
# 2. Create sample dataset automatically
# -----------------------------------
file_path = "ml_dataset.csv"

if not os.path.exists(file_path):
    np.random.seed(42)
    n = 200

    df_sample = pd.DataFrame({
        "Age": np.random.randint(20, 60, n),
        "Income": np.random.randint(25000, 150000, n),
        "Experience_Years": np.random.randint(0, 20, n),
        "Hours_Per_Week": np.random.randint(20, 60, n),
        "Savings": np.random.randint(5000, 500000, n),
        "Loan_Amount": np.random.randint(10000, 300000, n),
    })

    # Classification target
    df_sample["Loan_Risk"] = np.where(
        (df_sample["Income"] < 50000) | (df_sample["Savings"] < 30000) | (df_sample["Loan_Amount"] > 200000),
        1,
        0
    )

    # Regression target
    df_sample["Monthly_Payment"] = (
        0.02 * df_sample["Loan_Amount"]
        + 0.0005 * df_sample["Income"]
        - 2 * df_sample["Savings"] / 1000
        + np.random.normal(0, 500, n)
    )

    df_sample.to_csv(file_path, index=False)
    print(f"Sample dataset created: {file_path}")

# -----------------------------------
# 3. Load dataset
# -----------------------------------
df = pd.read_csv(file_path)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())

# -----------------------------------
# 4. Classification: Loan Risk Prediction
# -----------------------------------
print("\n========== CLASSIFICATION MODELS ==========")

X_class = df[["Age", "Income", "Experience_Years", "Hours_Per_Week", "Savings", "Loan_Amount"]]
y_class = df["Loan_Risk"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

dt_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

dt_clf.fit(X_train_c, y_train_c)
rf_clf.fit(X_train_c, y_train_c)

y_pred_dt = dt_clf.predict(X_test_c)
y_pred_rf = rf_clf.predict(X_test_c)

acc_dt = accuracy_score(y_test_c, y_pred_dt)
acc_rf = accuracy_score(y_test_c, y_pred_rf)

print(f"Decision Tree Accuracy: {acc_dt:.4f}")
print(f"Random Forest Accuracy: {acc_rf:.4f}")

print("\nRandom Forest Classification Report:")
print(classification_report(y_test_c, y_pred_rf))

# Confusion Matrix
cm = confusion_matrix(y_test_c, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("charts/confusion_matrix_random_forest.png", dpi=300)
plt.show()

# ROC Curve
y_prob_rf = rf_clf.predict_proba(X_test_c)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_c, y_prob_rf)
roc_auc = roc_auc_score(y_test_c, y_prob_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc:.2f})", color="darkorange")
plt.plot([0, 1], [0, 1], linestyle="--", color="navy")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.tight_layout()
plt.savefig("charts/roc_curve_random_forest.png", dpi=300)
plt.show()

# -----------------------------------
# 5. Regression: Monthly Payment Prediction
# -----------------------------------
print("\n========== REGRESSION MODELS ==========")

X_reg = df[["Age", "Income", "Experience_Years", "Hours_Per_Week", "Savings", "Loan_Amount"]]
y_reg = df["Monthly_Payment"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

lr_reg = LinearRegression()
dt_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

lr_reg.fit(X_train_r, y_train_r)
dt_reg.fit(X_train_r, y_train_r)
rf_reg.fit(X_train_r, y_train_r)

y_pred_lr = lr_reg.predict(X_test_r)
y_pred_dt_reg = dt_reg.predict(X_test_r)
y_pred_rf_reg = rf_reg.predict(X_test_r)

def regression_metrics(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"\n{name}")
    print(f"R2 Score: {r2:.4f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

regression_metrics("Linear Regression", y_test_r, y_pred_lr)
regression_metrics("Decision Tree Regressor", y_test_r, y_pred_dt_reg)
regression_metrics("Random Forest Regressor", y_test_r, y_pred_rf_reg)

# Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test_r, y_pred_rf_reg, alpha=0.7, color="green")
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], "r--")
plt.xlabel("Actual Monthly Payment")
plt.ylabel("Predicted Monthly Payment")
plt.title("Actual vs Predicted - Random Forest Regression")
plt.tight_layout()
plt.savefig("charts/actual_vs_predicted_rf.png", dpi=300)
plt.show()

# Residual Plot
residuals 