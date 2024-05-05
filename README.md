# Logistic Regression with Gradient Descent
This repository contains Python code for implementing logistic regression using gradient descent optimization.

# Table of Contents
 - Overview
 - Usage
 - Example
 - Contributing

# Overview
Logistic regression is a fundamental machine learning algorithm used for binary classification tasks. This implementation provides functions for training a logistic regression model using gradient descent optimization. It includes functions for parameter initialization, forward and backward propagation, optimization, prediction, and model evaluation.
# Usage
To use this implementation, follow these steps:

 - Clone the Repository: Clone this repository to your local machine using git clone: https://github.com/mohamedelsayed0001/cat-classifier.
 - Import the Module: Import the logistic_regression module into your Python script.
 - Prepare Data: Prepare your data in the form of feature matrices X_train, X_test and label vectors Y_train, Y_test.
 - Train the Model: Call the model function with your training and testing data to train the logistic regression model.
 - Make Predictions: Use the trained model to make predictions on new data using the predict function.

# Example
Here's an example of how to use the logistic regression model:

```python

import numpy as np
from logistic_regression import model, predict

# Prepare data
X_train = ...
Y_train = ...
X_test = ...
Y_test = ...

# Train the model
results = model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=True)

# Make predictions
Y_pred_test = predict(results['w'], results['b'], X_test)
```
# Contributing
 - Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.
