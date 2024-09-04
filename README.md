# Cotton Price Forecasting
<b>My unsupervised machine learning model was able to predict about 72.4% of the variance in cotton prices using only inflation and crop yield data.</b>

This is a simple model that generalizes well, but if I were building for production, I would include additional indicators (e.g. prices of related commodities, textile production volumes, fertilizer prices, satellite images, USDA Reports, drought data, etc.) and use feature importance analysis to optimize.

I used:
* Historical cotton price data from Macrotrends
* Cotton yield per hectare data from the Food and Agriculture Organization of the United Nations
* Inflation data from Federal Reserve Bank Economic Data.

# Results
In the image below, the red dotted line represents the line of perfect prediction. Ideally, the dots should be along this line where actual prices always equal predicted prices.

It is clear that my ML model fits the line of perfect prediction much better than a linear regression model. It outperformed linear regression both in Mean Square Error (MSE) and R-squared. MSE is the average squared difference between the actual and predicted values, so the lower MSE of about 0.011 is better than 0.0386. R-squared indicates the proportion of variance in the dependent variable that can be predicted from the independent variables. Therefore, a value of 0.724 represents 72.4% predicted variance, and 0.011 represents only 1.1% predicted variance.


![Screenshot 2024-07-15 091538](https://github.com/user-attachments/assets/de6b7c5b-bb9a-4778-84b9-664c096d7d4c)

# Model building
1. Imported libraries
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

```

2. Loaded, cleaned, and merged datasets
```
# Load datasets
cotton_prices = pd.read_csv('cotton-prices-historical-chart-data.csv', skiprows=15)
cotton_yield = pd.read_csv('cotton-yield.csv')
inflation = pd.read_csv('Inflation.csv')

# Rename the columns
inflation.columns = ['Date', 'Inflation']
print(inflation.columns)

# Convert the Date column to datetime format
inflation['Date'] = pd.to_datetime(inflation['Date'], errors='coerce')

# Rename the columns if necessary
cotton_prices.columns = ['Date', 'Price']
print(cotton_prices.columns)

# Convert the Date column to datetime format
cotton_prices['Date'] = pd.to_datetime(cotton_prices['Date'], errors='coerce')

# Convert date columns in inflation dataset
inflation['Date'] = pd.to_datetime(inflation['Date'], errors='coerce')

# Extract year from date columns for merging
cotton_prices['Year'] = cotton_prices['Date'].dt.year
inflation['Year'] = inflation['Date'].dt.year

# Rename columns for consistency
cotton_yield.columns = ['Entity', 'Code', 'Year', 'Yield']
cotton_yield = cotton_yield[['Year', 'Yield']]

# Merge cotton prices with inflation data on Year
merged_data_1 = pd.merge(cotton_prices, inflation, on='Year', how='inner')

# Merge the result with cotton yield data
merged_data_final = pd.merge(merged_data_1, cotton_yield, on='Year', how='inner')

# Handle missing values
merged_data_final = merged_data_final.dropna()
```
3. Split the data into training and testing sets
```
# Features and dependent variable
features = merged_data_final[['Inflation', 'Yield']]
target = merged_data_final['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
```
4. Trained a linear regression model for comparison
```
# Train linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Make predictions with the linear model
y_pred_linear = linear_model.predict(X_test)

# Evaluate the linear model
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

# Evaluation metrics for linear model
print(f'Linear Regression Mean Squared Error: {mse_linear}')
print(f'Linear Regression R-squared: {r2_linear}')
```
5. Trained the random forest ML model
```
# Train Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions with the random forest model
y_pred_rf = rf_model.predict(X_test)

# Evaluate the random forest model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Evaluation metrics for random forest model
print(f'Random Forest Regressor Mean Squared Error: {mse_rf}')
print(f'Random Forest Regressor R-squared: {r2_rf}')
```
6. Plot the predictions of both models against actual values
```
plt.figure(figsize=(14, 7))

# Actual vs. predicted values linear regression
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_linear, alpha=0.7, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Linear Regression Predictions')
plt.grid(True)

# Actual vs. predicted values random forest regression
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_rf, alpha=0.7, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Random Forest Regression Predictions')
plt.grid(True)

plt.tight_layout()
plt.show()
```
7. Determined that inflation data was the most influential in the model predictions and that crop yield data wasn't important using decrease in impurity statistics
```
# Feature importances from the random forest model
importances = rf_model.feature_importances_
feature_names = features.columns

# Create a dataframe for visualization
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='green')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances in Random Forest Model')
plt.gca().invert_yaxis()
plt.show()
```
![Screenshot 2024-07-15 093320](https://github.com/user-attachments/assets/2f49fba2-bec5-4c1c-8b75-b0a7d85d3881)

