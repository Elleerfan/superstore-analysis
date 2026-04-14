# 📊 Superstore Sales Analysis & Machine Learning Project

This project analyzes the **Superstore dataset** and applies multiple machine learning models to explore business insights and predict **profit** based on various features such as sales, discount, category, and region.

The project follows a complete **data science pipeline**, starting from data cleaning and exploration to model training and evaluation.

---

# 🚀 Project Goals

The main objectives of this project are:

* Clean and prepare the dataset for analysis
* Explore business patterns in sales and profit
* Visualize important trends and relationships
* Train different machine learning models
* Compare model performance
* Identify which model works best for this type of data

---

# 📂 Project Structure

```
project/
│
├── data/
│   ├── Sample - Superstore.csv
│   └── superstore_clean.csv
│
├── charts/
│   ├── category_chart.png
│   ├── region_chart.png
│   ├── heatmap.png
│   ├── discount_vs_profit.png
│   ├── top_products.png
│   ├── segment_sales.png
│   ├── shipmode_pie.png
│   ├── monthly_trends.png
│   ├── yearly_trends.png
│   └── xgboost_results.png
│
├── phase1_data_cleaning.py
├── phase2_data_analysis.py
├── phase3_visualization.py
├── phase4_linear_regression.py
├── phase5_neural_network.py
├── phase6_xgboost_model.py
│
└── README.md
```

---

# 🧹 Phase 1 — Data Cleaning & Feature Engineering

In the first phase, the dataset is prepared for analysis.

Main tasks:

* Load dataset using **Pandas**
* Inspect dataset structure
* Handle missing values
* Convert date columns to datetime
* Create new useful features

New features created:

* **Days to Ship**
* **Profit Margin (%)**
* **Order Year**
* **Order Month**

Output dataset:

```
superstore_clean.csv
```

---

# 📈 Phase 2 — Exploratory Data Analysis (EDA)

This phase explores the dataset to discover important business insights.

Analysis performed:

* Sales and profit by **Category**
* Sales distribution by **Region**
* Monthly sales trends
* Top profitable products
* Worst performing products
* State-level sales analysis

These analyses help understand **where revenue and profit are generated**.

---

# 📊 Phase 3 — Data Visualization

Several visualizations were created using **Matplotlib** and **Seaborn**.

Charts include:

* Sales & Profit by Category
* Sales Distribution by Region
* Profit Heatmap (Region vs Category)
* Discount vs Profit
* Top 10 Profitable Products
* Sales by Segment
* Shipping Mode Distribution
* Monthly Sales and Profit Trends
* Yearly Trends

These visualizations highlight important **business patterns and correlations**.

---

# 🤖 Phase 4 — Linear Regression Model

The first machine learning model used was **Linear Regression**.

Goal:
Predict **Profit** using available features.

Steps:

* Remove outliers using **IQR**
* Encode categorical variables
* Split dataset into training and testing sets
* Apply scaling and polynomial features

Evaluation metrics:

* **Mean Absolute Error (MAE)**
* **R² Score**

Linear Regression performed poorly because the relationship between variables in this dataset is **not purely linear**.

---

# 🧠 Phase 5 — Neural Network Model

A **Neural Network** was built using **TensorFlow / Keras**.

Architecture:

* Dense (128)
* Batch Normalization
* Dropout
* Dense (64)
* Batch Normalization
* Dropout
* Dense (32)
* Output Layer

Training improvements:

* Early stopping
* Learning rate reduction
* Feature scaling

Although the neural network performed better than linear regression, the improvement was still limited.

---

# 🚀 Phase 6 — XGBoost Model

The final and most powerful model used was **XGBoost Regressor**.

Pipeline components:

* StandardScaler
* OneHotEncoder
* PCA (Dimensionality Reduction)
* XGBoost Regressor

Hyperparameter tuning was performed using:

```
GridSearchCV
```

This model achieved the best performance among all models.

---

# 📊 Model Performance Comparison

| Model             | R² Score |
| ----------------- | -------- |
| Linear Regression | ~6.8%    |
| Neural Network    | ~17.9%   |
| XGBoost           | ~98.2%   |

XGBoost clearly outperformed the other models.

---

# 📦 Technologies Used

Python libraries used in this project:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* tensorflow / keras
* xgboost

---

# 📌 Conclusion

This project explored different machine learning approaches to predict **profit** using the Superstore dataset.

Three models were trained and evaluated:

* Linear Regression
* Neural Network
* XGBoost Regressor

Among them, **XGBoost achieved the best performance**, with an R² score close to **98%**, significantly outperforming the other models.

There are several reasons why XGBoost performed better:

First, the relationships between features such as **discount, sales, category, and region** are highly **non-linear**. Linear regression struggles with these types of relationships.

Second, **XGBoost is a tree-based ensemble model**, which allows it to capture complex feature interactions automatically.

Third, XGBoost uses **gradient boosting**, meaning it builds models sequentially and corrects previous errors at each step, resulting in a much more accurate final model.

Finally, structured business datasets like this one are often better handled by **tree-based algorithms**, which can naturally manage categorical patterns, interactions between variables, and irregular distributions.

Overall, this experiment demonstrates that **advanced boosting algorithms such as XGBoost are often the most effective models for structured tabular datasets**.

