# Task 3: Linear Regression

This task demonstrates the implementation and understanding of simple and multiple linear regression using a housing price dataset. The objective is to build a regression model, evaluate its performance, and interpret the results.

---

## Project Overview

Linear regression is a fundamental supervised learning algorithm used to predict a continuous target variable based on one or more features. This project uses the Scikit-learn library to train a regression model on housing data, evaluate its accuracy, and visualize predictions.

---

## Step-by-Step Summary

### Step 1: Load and Preprocess Data
- Load the housing price dataset (`Housing.csv`).
- Select features (both numerical and categorical).
- Convert categorical variables into dummy/indicator variables using one-hot encoding.

### Step 2: Train-Test Split
- Split the dataset into training (80%) and testing (20%) subsets to evaluate model generalization.

### Step 3: Train Linear Regression Model
- Use `sklearn.linear_model.LinearRegression` to fit the model on training data.

### Step 4: Model Evaluation
- Predict house prices on the test set.
- Calculate evaluation metrics:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R² Score (coefficient of determination)
  - Mean Absolute Percentage Error (MAPE)
  - Accuracy (100 - MAPE)
- Display metrics in a well-formatted summary table.

### Step 5: Interpret Model Coefficients
- Display feature coefficients to understand the impact of each predictor on house prices.

### Step 6: Visualization
- Plot actual vs. predicted prices scatter plot.
- Include a reference diagonal line to show perfect prediction.

---

## Dependencies

Ensure the following Python libraries are installed:

- pandas  
- matplotlib  
- scikit-learn  

Install dependencies using pip:

```bash
pip install pandas matplotlib scikit-learn
