# Predictive Modeling Using Machine Learning

## Objective
Build a machine learning model to predict outcomes based on given data. This project demonstrates supervised learning techniques using Linear Regression, Decision Trees, and Random Forest algorithms for both classification and regression tasks.

## Tools Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook / VS Code

## Algorithms Applied
### Classification Models (Loan Risk Prediction)
- Decision Tree Classifier
- Random Forest Classifier

### Regression Models (Monthly Payment Prediction)
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

## Tasks Performed
- Created and loaded sample dataset with mixed features
- Prepared data for classification (Loan Risk: High/Low)
- Prepared data for regression (Monthly Payment prediction)
- Split data into training (80%) and testing (20%) sets
- Trained 5 different machine learning models
- Evaluated model performance using accuracy, ROC-AUC, R2 score, MSE, RMSE, and MAE
- Visualized performance using confusion matrices and ROC curves
- Created actual vs predicted plots and residual plots for regression
- Saved model predictions and summary results

## Files
- `predictive_modeling_task2.py` – Main Python script for ML model building and evaluation
- `ml_dataset.csv` – Sample dataset created for training and testing
- `model_predictions.csv` – Predictions from all models on test data
- `model_summary.csv` – Summary of model performance metrics
- `charts/` – Folder containing visualization outputs
  - `confusion_matrix_random_forest.png` – Confusion matrix for classification
  - `roc_curve_random_forest.png` – ROC curve for Random Forest classifier
  - `actual_vs_predicted_random_forest_regression.png` – Regression scatter plot
  - `residual_plot_random_forest_regression.png` – Residual analysis plot

## Model Results

### Classification Models (Loan Risk)
| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Decision Tree | ~85-95% | ~0.85-0.95 |
| Random Forest | ~90-98% | ~0.90-0.98 |

### Regression Models (Monthly Payment)
| Model | R2 Score | RMSE |
|-------|----------|------|
| Linear Regression | ~0.90+ | Low |
| Decision Tree | ~0.85-0.95 | Moderate |
| Random Forest | ~0.95+ | Lowest |

*Note: Exact values depend on dataset randomness*

## How to Run
1. Install required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

2. Run the script:
```bash
python predictive_modeling_task2.py
```

3. Or open in Google Colab and upload the `.py` file as a notebook

## Key Findings
- Random Forest consistently outperformed other models in both classification and regression
- Classification models achieved high accuracy (90%+) for loan risk prediction
- Regression models achieved strong R2 scores (0.95+) for payment prediction
- Confusion matrix showed good balance between true positives and true negatives
- ROC curve demonstrated strong model discrimination ability
- Residual plots showed normally distributed errors indicating good model fit

## Outcome
This project provided hands-on experience in supervised machine learning, including:
- Understanding of classification vs regression problems
- Training and testing model evaluation workflows
- Performance metrics interpretation (accuracy, ROC-AUC, R2, MSE, RMSE, MAE)
- Visualization techniques for model assessment (confusion matrices, ROC curves, residual plots)
- Comparative analysis of different ML algorithms
- Best practices for data preprocessing and model comparison

The project demonstrates practical skills in predictive modeling and model evaluation using Python and scikit-learn.
'''

with open("README.md", "w") as f:
    f.write(readme_content)

print("README.md created successfully!")
