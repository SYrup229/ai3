# Synthetic Dataset Polynomial Regression
# This script demonstrates polynomial regression using a synthetic dataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

# Generate a synthetic dataset with a non-linear relationship
np.random.seed(42)
X = np.sort(np.random.rand(100, 1), axis=0)
y = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create models with different polynomial degrees
degrees = [1, 3, 5]
colors = ['r', 'g', 'b']
plt.figure(figsize=(12, 8))

for degree, color in zip(degrees, colors):
    # Create and train the polynomial regression model
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print the results
    print(f"\nPolynomial Degree {degree}:")
    print(f"Mean squared error: {mse:.4f}")
    print(f"R-squared score: {r2:.4f}")
    
    # Plot the results
    X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    plt.scatter(X, y, color='gray', alpha=0.5, label='Data points' if degree == 1 else '')
    plt.plot(X_plot, y_plot, color=color, label=f'Degree {degree}')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Synthetic Dataset: Polynomial Regression')
plt.legend()
plt.show()

# Optional: Print coefficients for the highest degree model
highest_degree_model = model.named_steps['linearregression']
coeffs = highest_degree_model.coef_
intercept = highest_degree_model.intercept_
print(f"\nCoefficients for degree {degrees[-1]} polynomial:")
for i, coef in enumerate(coeffs):
    print(f"  Degree {i}: {coef:.4f}")
print(f"Intercept: {intercept:.4f}")