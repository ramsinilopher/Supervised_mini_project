# Supervised_mini_project

# Breast Cancer Classification using Supervised Learning

## Objective:
The objective of this project is to apply and evaluate various supervised learning techniques on the breast cancer dataset. The goal is to build and compare the performance of five classification algorithms to predict whether a tumor is malignant or benign.

## Dataset:
The dataset used in this project is the Breast Cancer dataset available in the sklearn library. This dataset contains 30 features describing characteristics of the cell nuclei present in breast cancer biopsies. The target variable indicates whether the tumor is malignant (1) or benign (0).

## Key Components:
Loading and Preprocessing:
Loading the dataset: The dataset is loaded from the sklearn.datasets module.
Preprocessing: The dataset is preprocessed by:
Handling missing values (if any).
Scaling features using StandardScaler to normalize the data, ensuring that all features contribute equally to the model performance.
Justification: Feature scaling is necessary as most machine learning algorithms are sensitive to the scale of the input data. It helps improve model convergence and accuracy.

 Classification Algorithm Implementation:
Five classification algorithms were implemented and evaluated:

## 1. Logistic Regression:
A statistical model that uses a logistic function to model a binary dependent variable. It works well for linear decision boundaries.

## 2. Decision Tree Classifier:
A tree-like structure where each internal node represents a feature test, and each leaf node represents a class label. It is interpretable and works well for both linear and non-linear data.

## 3. Random Forest Classifier:
An ensemble learning method that combines multiple decision trees to improve accuracy and reduce overfitting. It is robust and effective for large datasets.

## 4. Support Vector Machine (SVM):
A supervised learning model that finds the hyperplane that best separates the data into different classes. It is effective in high-dimensional spaces.

## 5. k-Nearest Neighbors (k-NN):
A simple, non-parametric algorithm that classifies a data point based on the majority class of its nearest neighbors. It works well for smaller datasets and non-linear decision boundaries.

# Model Comparison:
After training and testing the models, their performance was compared based on accuracy, precision, recall, F1-score, and confusion matrix.
Best performing model: Logistic Regression achieved the highest accuracy and is the most suitable model for this dataset.
Worst performing model: Decision Tree Classifier, while interpretable, had the lowest accuracy due to overfitting and sensitivity to noise.
Results:
Logistic Regression: 98% accuracy
Decision Tree Classifier: 95% accuracy
Random Forest Classifier: 96% accuracy
Support Vector Machine (SVM): 96% accuracy
k-Nearest Neighbors (k-NN): 96% accuracy

## Conclusion:
Logistic Regression outperformed the other algorithms in terms of accuracy, but Random Forest, SVM, and k-NN also showed competitive performance. Decision Tree, while interpretable, performed the worst due to overfitting.
