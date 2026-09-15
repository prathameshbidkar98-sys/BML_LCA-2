import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


housing = fetch_california_housing(as_frame=True)

df = housing.frame

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())


print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())


X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("MedHouseVal")


print("\n" + "=" * 60)
print("SIMPLE LINEAR REGRESSION")
print("=" * 60)

X_simple = df[["MedInc"]]

y_simple = df["MedHouseVal"]


X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
    X_simple,
    y_simple,
    test_size=0.2,
    random_state=42
)

print("\nSimple Linear Regression Data Split:")
print("Training samples:", X_train_simple.shape[0])
print("Testing samples :", X_test_simple.shape[0])


simple_model = LinearRegression()


simple_model.fit(X_train_simple, y_train_simple)

print("\nSimple Linear Regression Model trained successfully!")


print("\nSimple Linear Regression Equation:")

print(
    f"MedHouseVal = {simple_model.intercept_:.4f} + "
    f"({simple_model.coef_[0]:.4f} × MedInc)"
)


y_pred_simple = simple_model.predict(X_test_simple)


mae_simple = mean_absolute_error(y_test_simple, y_pred_simple)
mse_simple = mean_squared_error(y_test_simple, y_pred_simple)
rmse_simple = np.sqrt(mse_simple)
r2_simple = r2_score(y_test_simple, y_pred_simple)

print("\nSimple Linear Regression Evaluation:")
print(f"MAE  : {mae_simple:.4f}")
print(f"MSE  : {mse_simple:.4f}")
print(f"RMSE : {rmse_simple:.4f}")
print(f"R²   : {r2_simple:.4f}")


plt.figure(figsize=(8, 5))

plt.scatter(
    X_test_simple["MedInc"],
    y_test_simple,
    alpha=0.5,
    label="Actual Values"
)

sorted_indices = np.argsort(X_test_simple["MedInc"].values)

plt.plot(
    X_test_simple["MedInc"].values[sorted_indices],
    y_pred_simple[sorted_indices],
    linewidth=2,
    label="Regression Line"
)

plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Simple Linear Regression - California Housing")
plt.legend()
plt.grid(True)

plt.show()


print("\n" + "=" * 60)
print("MULTIPLE LINEAR REGRESSION")
print("=" * 60)


X_multiple = df.drop("MedHouseVal", axis=1)
y_multiple = df["MedHouseVal"]


X_train_multiple, X_test_multiple, y_train_multiple, y_test_multiple = train_test_split(
    X_multiple,
    y_multiple,
    test_size=0.2,
    random_state=42
)

print("\nMultiple Linear Regression Data Split:")
print("Training samples:", X_train_multiple.shape[0])
print("Testing samples :", X_test_multiple.shape[0])



multiple_model = LinearRegression()


multiple_model.fit(X_train_multiple, y_train_multiple)

print("\nMultiple Linear Regression Model trained successfully!")


print("\nMultiple Linear Regression Coefficients:")

coefficients = pd.DataFrame({
    "Feature": X_multiple.columns,
    "Coefficient": multiple_model.coef_
})

print(coefficients)

print(f"\nIntercept: {multiple_model.intercept_:.4f}")


y_pred_multiple = multiple_model.predict(X_test_multiple)


mae_multiple = mean_absolute_error(y_test_multiple, y_pred_multiple)
mse_multiple = mean_squared_error(y_test_multiple, y_pred_multiple)
rmse_multiple = np.sqrt(mse_multiple)
r2_multiple = r2_score(y_test_multiple, y_pred_multiple)

print("\nMultiple Linear Regression Evaluation:")
print(f"MAE  : {mae_multiple:.4f}")
print(f"MSE  : {mse_multiple:.4f}")
print(f"RMSE : {rmse_multiple:.4f}")
print(f"R²   : {r2_multiple:.4f}")


plt.figure(figsize=(8, 5))

plt.scatter(
    y_test_multiple,
    y_pred_multiple,
    alpha=0.5
)

min_value = min(y_test_multiple.min(), y_pred_multiple.min())
max_value = max(y_test_multiple.max(), y_pred_multiple.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    linewidth=2
)

plt.xlabel("Actual House Values")
plt.ylabel("Predicted House Values")
plt.title("Multiple Linear Regression - Actual vs Predicted")
plt.grid(True)

plt.show()


print("\nPractical execution completed successfully!")
