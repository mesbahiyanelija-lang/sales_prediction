import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import matplotlib.pyplot as plt

df = pd.read_csv(
    r"C:\Users\mouood system\Downloads\superstore_final_dataset (1).csv\superstore_final_dataset (1).csv", encoding="latin1")

df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors='coerce')
df["Month"] = df["Order_Date"].dt.month
df["Year"] = df["Order_Date"].dt.year

df = df.dropna(subset=["Month", "Year", "Category",
                       "Sub_Category", "Segment", "Region", "Ship_Mode", "Sales"])

X_raw = df[["Month", "Year", "Category",
            "Sub_Category", "Segment", "Region", "Ship_Mode"]]
y = df["Sales"]

X_encoded = pd.get_dummies(X_raw, columns=[
                           "Category", "Sub_Category", "Segment", "Region", "Ship_Mode"], drop_first=True)

X_encoded['Original_Segment'] = df['Segment']

mask_consumer = X_encoded['Original_Segment'] == 'Consumer'

X_consumer = X_encoded[mask_consumer].drop(columns=['Original_Segment'])
y_consumer = y[mask_consumer]

X_corporate = X_encoded[~mask_consumer].drop(columns=['Original_Segment'])
y_corporate = y[~mask_consumer]

print(
    f"Consumer count: {len(X_consumer)} | Corporate count: {len(X_corporate)}")


def train_and_evaluate_rf_log(X, y, model_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    y_train_log = np.log1p(y_train)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train_log)

    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n--- results {model_name} (RF + Log) ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    return y_test, y_pred


y_test_cons, y_pred_cons = train_and_evaluate_rf_log(
    X_consumer, y_consumer, "Consumer Model")
y_test_corp, y_pred_corp = train_and_evaluate_rf_log(
    X_corporate, y_corporate, "Corporate Model")

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test_cons, y_pred_cons, alpha=0.6,
            color='royalblue', edgecolors='k', linewidths=0.5)

plt.plot([y_test_cons.min(), y_test_cons.max()], [
         y_test_cons.min(), y_test_cons.max()], 'r--', lw=2)
plt.title('Consumer Model: Actual vs Predicted Sales', fontsize=12)
plt.xlabel('Actual Sales', fontsize=10)
plt.ylabel('Predicted Sales', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(1, 2, 2)
plt.scatter(y_test_corp, y_pred_corp, alpha=0.6,
            color='darkorange', edgecolors='k', linewidths=0.5)

plt.plot([y_test_corp.min(), y_test_corp.max()], [
         y_test_corp.min(), y_test_corp.max()], 'r--', lw=2)
plt.title('Corporate Model: Actual  vs Predicted Sales', fontsize=12)
plt.xlabel('Actual Sales', fontsize=10)
plt.ylabel('Predicted Sales', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('sales_comparison.png')
