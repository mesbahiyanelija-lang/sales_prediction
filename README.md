# 📊 Superstore Sales Prediction: B2C vs. B2B Segment Analysis

This repository contains an end-to-end Machine Learning pipeline designed to predict and analyze sales dynamics for a retail Superstore dataset. The project focuses on splitting market segments into **Consumer (B2C)** and **Corporate (B2B)** to build tailored predictive models, addressing severe data skewness through advanced target transformations.

## 🚀 Key Features & Pipeline

* **Data Cleaning & Parsing:** Handled real-world transactional data, converting dates and resolving multi-categorical variables.
* **Feature Engineering:** Implemented One-Hot Encoding (`pd.get_dummies`) for high-cardinality categorical variables such as `Category`, `Sub_Category`, `Region`, and `Ship_mode`.
* **Segment-Driven Modeling:** Divided the pipeline to evaluate B2C (Consumer) and B2B (Corporate) purchasing behaviors independently.
* **Mathematical Transformation:** Resolved target variable right-skewness using Log Transformation (`np.log1p`), preventing large invoice outliers from biasing the models.

---

## 🛠️ The Challenge: Engineering Around Skewness & Overflows

The target variable (`Sales`) exhibited a heavy right-tail distribution (a massive volume of low-value orders mixed with rare high-value corporate deals up to $17,000+).

During development, a critical pipeline bug was intercepted: training a non-linear model on raw values while applying an inverse log transformation (`np.expm1`) on the backend triggered a `RuntimeWarning: overflow encountered in expm1`. This led to infinite mathematical values (`inf`) that crashed the MAPE evaluation. The bug was successfully debugged by standardizing the target scaling across the `fit` and `predict` phases.

---

## 📈 Model Performance & Comparison

The project benchmarked **Linear Regression** against an ensemble **Random Forest Regressor** across both market segments using Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE):

| Segment | Model | MAE ($) | MAPE (%) | Scatter Plot Evaluation |
| --- | --- | --- | --- | --- |
| **Consumer (B2C)** | Linear Regression + Log | **153.62** | **140.05%** | Highly compressed predictions; struggles with upper-tail variance. |
| **Consumer (B2C)** | Random Forest + Log | 161.84 | 193.04% | Better representation of true variance and higher-end sales distribution. |
| **Corporate (B2B)** | Linear Regression + Log | **229.17** | **155.61%** | Tends to underpredict high-value custom contracts. |
| **Corporate (B2B)** | Random Forest + Log | 237.84 | 173.49% | Effectively captures non-linear category interactions. |

### 💡 Technical Insights & Takeaways

1. **The MAPE Paradox:** While Linear Regression shows a lower overall MAPE, the scatter plots reveal it heavily underpredicts higher-end transactions.
2. **Ensemble Advantage:** Random Forest delivers a much more realistic spread of predictions across the target range, proving that ensemble methods capture complex, non-linear retail interactions far better than linear boundaries, despite penalized percentage errors on lower-bound invoices.
## 🎯 Business Assumptions & Operational Insights

A purely technical model fails if it does not align with business reality. This pipeline bridges the gap between data science and retail operations through three core strategic assumptions:

### 1. Market Segmentation (B2C vs. B2B Dynamics)
* **Strategic Assumption:** Corporate clients (B2B) and individual consumers (B2C) exhibit fundamentally different purchasing behaviors. B2B transactions are less frequent but feature massive volume spikes (e.g., bulk office supply orders up to $17,000), whereas B2C transactions are high-frequency but low-value.
* **Business Impact:** Evaluating these segments independently prevents large B2B contract anomalies from distorting B2C demand forecasts. This directly enables the supply chain team to optimize inventory levels separately for commercial warehouses and retail storefronts.

### 2. Mitigating High-Value Outliers via Log Normalization
* **Strategic Assumption:** Extreme transactional variance (heavy right-skewed data) can trick machine learning algorithms into overestimating routine sales. This overestimation ties up valuable working capital in unnecessary safety stock.
* **Business Impact:** Applying a logarithmic transformation (`np.log1p`) stabilizes the variance. It allows the model to learn the underlying patterns of high-end corporate contracts without letting a single anomalous order artificially inflate the predicted sales baseline for daily products.

### 3. Algorithm Selection Based on Financial Risk Appetite
* **The Analytical Trade-off:** While **Linear Regression** yielded a lower overall MAPE by tightly fitting low-value items, the **Random Forest** model captured the actual market variance and tail-end distribution of high-value contracts far more accurately.
* **Executive Decision:** In a real-world deployment, the final model selection depends on the company's financial priorities:
    * **Deploy Linear Regression** if the business priority is minimizing minor day-to-day operational tracking errors for high-volume, low-margin items.
    * **Deploy Random Forest** if facing a stockout on premium categories or miscalculating a major B2B procurement contract poses a more severe financial and reputational risk to the enterprise.

## 🛠️ Tech Stack

* **Language:** Python 3.12
* **Libraries:** Pandas, NumPy, Scikit-Learn (LinearRegression, RandomForestRegressor), Matplotlib

