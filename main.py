import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn import metrics

# Load data
car_dataset = pd.read_csv(".\\data\\car_data.csv")

# PHASE 1: EDA
plt.figure(figsize=(10, 8))
numerical_cols = car_dataset.select_dtypes(include=[np.number])
sns.heatmap(numerical_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.savefig("1_correlation_heatmap.png")
plt.close()

# PHASE 2: FEATURE ENGINEERING
brand_avg_price = car_dataset.groupby('Car_Name')['Present_Price'].mean()
expensive_threshold = brand_avg_price.median()
car_dataset['Is_Expensive_Brand'] = car_dataset['Car_Name'].apply(
    lambda x: 1 if brand_avg_price[x] > expensive_threshold else 0
)

current_year = 2026
car_dataset['Car_Age'] = current_year - car_dataset['Year']
car_dataset.drop(['Year', 'Car_Name'], axis=1, inplace=True)

encoded_df = pd.get_dummies(car_dataset,
    columns=["Fuel_Type", "Seller_Type", "Transmission"], dtype=int)

X = encoded_df.drop(["Selling_Price"], axis=1)
Y = encoded_df["Selling_Price"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=2)

# PHASE 3: MODEL TRAINING
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)
y_pred_lin = lin_reg.predict(X_test)

lasso_cv = LassoCV(cv=5, random_state=2, max_iter=100000)
lasso_cv.fit(X_train, Y_train)
y_pred_lasso = lasso_cv.predict(X_test)

print("--- Model Performance ---")
print(f"Linear Reg  -> RMSE: {metrics.root_mean_squared_error(Y_test, y_pred_lin):.4f} | R2: {metrics.r2_score(Y_test, y_pred_lin):.4f}")
print(f"Lasso Reg   -> RMSE: {metrics.root_mean_squared_error(Y_test, y_pred_lasso):.4f} | R2: {metrics.r2_score(Y_test, y_pred_lasso):.4f}")

# PHASE 4: RESIDUAL ANALYSIS
residuals_lin   = Y_test - y_pred_lin
residuals_lasso = Y_test - y_pred_lasso

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_pred_lin, y=residuals_lin, color='blue', alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residuals (Error)")
plt.title("Linear Regression: Residual Analysis")

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_pred_lasso, y=residuals_lasso, color='orange', alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residuals (Error)")
plt.title("Lasso Regression: Residual Analysis")
plt.tight_layout()
plt.savefig("2_residual_analysis.png")
plt.close()

# PHASE 5: FEATURE IMPORTANCE
coeff_df = pd.DataFrame({
    'Feature':      X.columns,
    'Linear_Coeff': lin_reg.coef_,
    'Lasso_Coeff':  lasso_cv.coef_
})
coeff_df['Abs_Linear'] = coeff_df['Linear_Coeff'].abs()
coeff_df = coeff_df.sort_values(by='Abs_Linear', ascending=False).drop('Abs_Linear', axis=1)

coeff_df.set_index('Feature').plot(kind='bar', figsize=(14, 7), color=['blue', 'orange'], alpha=0.8)
plt.title("Feature Weights: Linear vs. Lasso")
plt.ylabel("Mathematical Weight (Coefficient)")
plt.xticks(rotation=45, ha='right')
plt.axhline(y=0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig("3_feature_importance.png")
plt.close()