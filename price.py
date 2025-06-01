import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load and preprocess the dataset
df = pd.read_csv('Housing.csv')

# Define features (independent variables) and target (dependent variable)
features = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom',
            'basement', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']
X = pd.get_dummies(df[features], drop_first=True)  # Convert categorical to numeric
y = df['price']   # Target variable

# Step 2: Split data into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Fit a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Predict and evaluate
y_pred = model.predict(X_test)

# Define MAPE and Accuracy
def mean_absolute_percentage_error(y_true, y_pred):
    return 100 * sum(abs(y_true - y_pred) / y_true) / len(y_true)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
accuracy = 100 - mape

# Print results
print("\nModel Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Accuracy: {accuracy:.2f}%")

# Display metrics in a well-formatted table
metrics = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'R²', 'MAPE (%)', 'Accuracy (%)'],
    'Value': [mae, mse, r2, mape, accuracy]
})

# Format numbers for readability
metrics['Value'] = metrics['Value'].apply(lambda x: f"{x:,.2f}")

print("\nSummary Table:")
print(metrics.to_string(index=False))

# Step 5: Interpret coefficients
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print("\nFeature Coefficients:")
print(coefficients)

# Step 6: Plot actual vs predicted prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.grid(True)
plt.tight_layout()
plt.show()

