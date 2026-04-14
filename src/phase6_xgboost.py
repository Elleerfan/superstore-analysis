import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor

df = pd.read_csv("data/superstore_clean.csv")

print("Libraries loaded!")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

X = df.drop(['Discount', 'Order Date', 'Ship Date', 'Product Name'], axis=1)
y = df['Discount']

categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['number']).columns

print(f"Categorical columns: {list(categorical_cols)}")
print(f"Numerical columns: {list(numerical_cols)}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training rows : {X_train.shape[0]}")
print(f"Testing rows  : {X_test.shape[0]}")

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('pca', PCA())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(objective='reg:squarederror', random_state=42))])

param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__learning_rate': [0.05, 0.1, 0.5],
    'model__max_depth': [3, 5, 7],
    'preprocessor__num__pca__n_components': [0.95, 0.99, len(numerical_cols)]
}

grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")


import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))


ax1.scatter(y_test, y_pred, alpha=0.4, color="steelblue", edgecolor="white", linewidth=0.5)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linewidth=2, linestyle="--", label="Perfect Prediction")
ax1.set_title("XGBoost — Actual vs Predicted Discount", fontsize=14, fontweight="bold")
ax1.set_xlabel("Actual Discount", fontsize=12)
ax1.set_ylabel("Predicted Discount", fontsize=12)
ax1.legend()
ax1.text(0.05, 0.92, f"R² = 0.9823", transform=ax1.transAxes, fontsize=12, color="green", fontweight="bold")


models = ["Linear\nRegression", "Neural\nNetwork", "XGBoost"]
scores = [6.79, 17.91, 98.23]
colors = ["#e74c3c", "#f39c12", "#007d7c"]

bars = ax2.bar(models, scores, color=colors, edgecolor="black", linewidth=1, width=0.5)
ax2.set_title("Model Comparison — R² Score (%)", fontsize=14, fontweight="bold")
ax2.set_ylabel("R² Score (%)", fontsize=12)
ax2.set_ylim(0, 110)
for bar, v in zip(bars, scores):
    ax2.text(bar.get_x() + bar.get_width()/2, v + 1.5, f'{v}%', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig("xgboost_results.png")
plt.show()
print("XGBoost charts saved!")