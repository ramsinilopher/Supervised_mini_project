#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[2]:


breast_cancer_data = sklearn.datasets.load_breast_cancer()
print(breast_cancer_data)


# In[3]:


# Create a DataFrame
df = pd.DataFrame(breast_cancer_data.data, columns=breast_cancer_data.feature_names)
df


# In[4]:


#Add the target column
df["target"] = breast_cancer_data.target
df


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


# Check for missing values
df.isnull().sum()


# In[10]:


#Checking for outliers

#Calculate IQR for each feature

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = (df < lower_bound) | (df > upper_bound)
print("outliers detected : ")
print(outliers.sum())


# In[11]:


df_capped = df.clip(lower = lower_bound, upper = upper_bound, axis =1)
df_capped


# In[12]:


# Check for outliers

Q1 = df_capped.quantile(0.25)
Q3 = df_capped.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = (df_capped < lower_bound) | (df_capped > upper_bound)
print("outliers detected : ")
print(outliers.sum())


# In[13]:


#Feature scaling


# In[14]:


from sklearn.preprocessing import StandardScaler

x = df_capped.drop("target",axis =1)
y = df_capped["target"]

scaler = StandardScaler()

x_scaled = scaler.fit_transform(x)

x_scaled_df = pd.DataFrame(x_scaled, columns= x.columns)

x_scaled_df


# In[15]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Initialize and train the model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(x_scaled, y)

# Predictions
y_pred_log_reg = log_reg.predict(x_scaled)

# Accuracy
log_reg_accuracy = accuracy_score(y, y_pred_log_reg)
print(f"Logistic Regression Accuracy: {log_reg_accuracy:.2f}")


# In[16]:


x_train,x_test,y_train,y_test = train_test_split(x_scaled_df,y,test_size = 0.2, random_state = 42)


# In[17]:


x_train.shape


# In[18]:


y_train.shape


# In[19]:


x_test.shape


# # 1. Logistic Regression
# 

# 
# 
# Description: Logistic Regression is a supervised learning algorithm used for binary classification problems. Despite its name, it is a classification algorithm, not a regression algorithm. It is a statistical model used for binary classification. It predicts the probability of an outcome (e.g., 0 or 1) using a logistic function (sigmoid). It is used when the target variable is categorical, typically for classification tasks.
# 
# How it works:
# 
# Estimates the probability of a binary outcome using a logistic function (sigmoid)
# Transforms linear regression output into probability values between 0 and 1
# Uses a threshold (typically 0.5) to classify: values above are malignant, below are benign
# 
# Why Logistic Regression Might Be Suitable for This Dataset?
# 
# Well-suited for this dataset due to its binary nature (malignant vs benign)
# Provides feature importance coefficients to understand which tumor characteristics are most predictive
# Computationally efficient and performs well with standardized numerical features
# 

# In[20]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[21]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(x_train,y_train)




# In[22]:


#Make test prediction
y_predict = log_reg.predict(x_test)
y_predict


# In[23]:


#Evaluate the model
#Accuracy Score

accuracy = accuracy_score(y_predict,y_test)
print(f"Accuracy score is : {accuracy}")


# Accuracy measures the proportion of correctly classified instances out of the total number of instances.
# Formula:
# Accuracy
# =
# Number of Correct Predictions
# Total Number of Predictions
# Accuracy= Number of Correct Predictions/Total Number of Predictions
# ​
#  
# In this case, 98.25% of the predictions were correct, indicating that the logistic regression model performed very well on this dataset.

# In[24]:


#Classification Report

report = classification_report(y_test,y_predict)
print(f"Classification report is : {report}")


# 1) Precision:
# 
# Precision for a class is the ratio of correctly predicted positive observations to the total predicted positive observations.
# Formula:
# 
# Precision= 
# True Positives/(True Positives+False Positives)
# 
# ​
#  
# Class 0: 98% precision means that when the model predicts class 0, it is correct 98% of the time.
# Class 1: 99% precision means that when the model predicts class 1, it is correct 99% of the time.
# 
# 2) Recall:
# 
# Recall for a class is the ratio of correctly predicted positive observations to all observations in the actual class.
# Formula:
# 
# Recall= 
# True Positives/(True Positives+False Negatives)
# 
# ​
#  
# Class 0: 98% recall means the model correctly identifies 98% of all actual class 0 instances.
# Class 1: 99% recall means the model correctly identifies 99% of all actual class 1 instances.
# 
# 3) F1-Score
# F1-score is the harmonic mean of precision and recall, balancing the two metrics.
# Formula:
# 
# F1-Score=2⋅ (Precision⋅Recall)/(Precision+Recall)
# 
# 
# ​
#  
# Class 0 and 1 both have high F1-scores (~98% and 99%, respectively), indicating a strong balance between precision and recall.
# 

# In[25]:


#Confusion Matrix 

matrix = confusion_matrix(y_test,y_predict)
print(f"Confusion matrix is : \n{matrix}")

A confusion matrix shows the breakdown of predictions for each class.
	               Predicted Class 0	                 Predicted Class 1
Actual Class 0	  42 (True Negatives)	                1 (False Positives)
Actual Class 1	  1 (False Negatives)	                70 (True Positives)

Key Insights:
True Negatives (42):
The model correctly predicted 42 instances of class 0.
False Positives (1):
The model incorrectly predicted 1 instance of class 1 as class 0.
False Negatives (1):
The model incorrectly predicted 1 instance of class 0 as class 1.
True Positives (70):
The model correctly predicted 70 instances of class 1.
# 
# 

# # 2. Decision Tree Classifier

# What is a Decision Tree Classifier?
# A Decision Tree Classifier is a supervised learning algorithm used for classification tasks. It splits the dataset into subsets based on feature values, forming a tree-like structure where:
# 
# Nodes represent features.
# Branches represent decisions or conditions.
# Leaves represent the final output (class labels).
# 
# How it works: 
# 
# The algorithm recursively splits the data into subsets based on the feature that results in the best separation of the classes . This process continues until the data is perfectly separated or a stopping criterion is met.
# 
# Why it’s suitable?
# 
# It’s easy to interpret, handles both numerical and categorical data, and works well with complex, non-linear relationships in the data.

# In[26]:


from sklearn.tree import DecisionTreeClassifier


# In[27]:


decision_tree = DecisionTreeClassifier(random_state = 42)

decision_tree.fit(x_train,y_train)


# In[28]:


#predict y_test

y_predict = decision_tree.predict(x_test)
y_predict


# In[29]:


#Evaluate the model

# Accuracy score
decision_tree_accuracy = accuracy_score(y_test,y_predict)
print(f"Decisiion tree accuracy is : {decision_tree_accuracy}")


# In[30]:


# classification report
decision_tree_report = classification_report(y_test,y_predict)
print(f"Decision tree classification report is : {decision_tree_report}")


# In[31]:


#confusion matrix
decision_tree_matrix = confusion_matrix(y_test,y_predict)
print(f"Decision tree confusion matrix is : \n{decision_tree_matrix}")


# # Model Analysis
# 
# Accuracy:
# The accuracy of the Decision Tree Classifier is 94.7%, meaning the model correctly predicted 94.7% of the test samples.
# 
# Classification Report:
# 
# Precision:
# Class 0 (Benign): 93% of the predictions for benign tumors were correct.
# Class 1 (Malignant): 96% of the predictions for malignant tumors were correct.
# 
# Recall:
# Class 0: 93% of actual benign tumors were correctly identified.
# Class 1: 96% of actual malignant tumors were correctly identified.
# 
# F1-Score:
# Class 0: F1-score is 0.93, reflecting a balance between precision and recall for benign tumors.
# Class 1: F1-score is 0.96, reflecting high performance in identifying malignant tumors.
# 
# The weighted average F1-score is 0.95, indicating good overall performance.
# 
# 
# Confusion Matrix:
# 
# True Positives (TP): 68 (Malignant correctly identified)
# True Negatives (TN): 40 (Benign correctly identified)
# False Positives (FP): 3 (Benign misclassified as malignant)
# False Negatives (FN): 3 (Malignant misclassified as benign)
# 
# The low values for FP and FN indicate the model is effective at distinguishing between benign and malignant cases.
# 
# Insights:
# The Decision Tree Classifier performs well with high precision, recall, and F1-scores for both classes.
# Misclassifications (FP and FN) are minimal, which is crucial in a medical context where false negatives (missing a malignant case) are especially critical.
# 

# # 3. Random Forest Classifier

# Description:
# Random Forest is an ensemble learning method that builds multiple decision trees and combines their predictions to improve accuracy and reduce overfitting. It is widely used for classification and regression tasks.
# 
# How it works?
# 
# Ensemble method that builds multiple decision trees
# Each tree is built using a random subset of features and samples
# Final prediction is based on majority voting from all trees
# 
# Why It’s Suitable for This Dataset?
# 
# Handles Non-Linearity: Random Forest can capture complex patterns in the data, making it suitable for datasets with non-linear relationships like the breast cancer dataset.
# Feature Importance: It ranks features by importance, helping to identify which attributes are most relevant for classification.
# Robustness: It reduces overfitting by averaging the predictions of multiple trees, which is beneficial for small to medium-sized datasets like this one.
# Accuracy: It performs well on datasets with a mix of numerical features, such as the breast cancer dataset, ensuring reliable predictions.

# In[32]:


from sklearn.ensemble import RandomForestClassifier


# In[33]:


RF_classifier = RandomForestClassifier(random_state =20)
RF_classifier.fit(x_train,y_train)


# In[34]:


y_predict_RF = RF_classifier.predict(x_test)
y_predict_RF


# In[35]:


# Evaluate the model
RF_accuracy = accuracy_score(y_test, y_predict_RF)
print(f"Random Forest Classifier Accuracy: {RF_accuracy}")

# Classification Report
RF_classification_report = classification_report(y_test, y_predict_RF)
print(f"\nRandom Forest Classification Report:\n{RF_classification_report}")

# Confusion Matrix
RF_confusion_matrix = confusion_matrix(y_test, y_predict_RF)
print(f"\nRandom Forest Confusion Matrix:\n{RF_confusion_matrix}")


# # Analysis of Random Forest Classifier Results:
# Overall Accuracy:
# 
# 96.49% accuracy indicates that the Random Forest Classifier performs exceptionally well on this dataset, correctly predicting most of the cases.
# 
# Precision:
# 
# For class 0 (negative cases): Precision is 0.98. This means that 98% of the predictions for class 0 were correct.
# For class 1 (positive cases): Precision is 0.96. This means that 96% of the predictions for class 1 were correct.
# Precision is high for both classes, indicating that the model is good at avoiding false positives.
# 
# Recall:
# 
# For class 0: Recall is 0.93. This means that 93% of the actual class 0 cases were correctly identified.
# For class 1: Recall is 0.99. This means that 99% of the actual class 1 cases were correctly identified.
# The recall for class 1 is slightly higher, indicating that the model is better at identifying positive cases.
# 
# F1-Score:
# 
# For class 0: F1-score is 0.95, showing a good balance between precision and recall.
# For class 1: F1-score is 0.97, indicating excellent performance for positive cases.
# The F1-scores reflect that the model performs consistently well across both classes.
# 
# Confusion Matrix:
# 
# True Positives (TP): 70 (class 1 correctly predicted)
# True Negatives (TN): 40 (class 0 correctly predicted)
# False Positives (FP): 3 (class 0 misclassified as class 1)
# False Negatives (FN): 1 (class 1 misclassified as class 0)
# 
# The confusion matrix shows that misclassifications are minimal, with only 4 errors in total (3 FP and 1 FN).

# # 4. Support Vector Machine (SVM):
# Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks. It is particularly effective in high-dimensional spaces and works well with both linearly separable and non-linearly separable data.
# 
# How SVM Works?
# Finding the Hyperplane:
# 
# SVM aims to find the optimal hyperplane that best separates the data into different classes. The hyperplane is a decision boundary, and data points on either side belong to different classes.
# For linearly separable data, the hyperplane is straight. For non-linear data, SVM uses kernels to map data into a higher-dimensional space where it becomes linearly separable.
# 
# Maximizing the Margin:
# 
# SVM maximizes the margin between the hyperplane and the closest data points (called support vectors). A larger margin reduces the risk of misclassification.
# 
# Why SVM is Suitable for This Dataset?
# 
# High Dimensionality:
# 
# The breast cancer dataset has many features (30 in total). SVM performs well in high-dimensional spaces and can efficiently find decision boundaries.
# 
# Class Separation:
# 
# The dataset has distinct patterns for malignant and benign tumors. SVM's ability to find the optimal hyperplane ensures precise separation of these classes.
# 
# Robustness:
# 
# SVM is less prone to overfitting, especially when using kernels and regularization parameters. This makes it suitable for datasets with some noise or overlap between classes.
# 
# Binary Classification:
# 
# The breast cancer dataset involves a binary classification problem (malignant vs. benign), which aligns perfectly with SVM's capabilities.

# In[36]:


from sklearn.svm import SVC


# In[37]:


svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(x_train, y_train)


# In[38]:


# Predictions on the test set
y_pred_svm = svm_model.predict(x_test)
y_pred_svm


# In[39]:


# Calculate accuracy
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {svm_accuracy}")

# Classification report
svm_classification_report = classification_report(y_test, y_pred_svm)
print("\nSVM Classification Report:")
print(svm_classification_report)

# Confusion matrix
svm_confusion_matrix = confusion_matrix(y_test, y_pred_svm)
print("\nSVM Confusion Matrix:")
print(svm_confusion_matrix)


# # Analysis of SVM Results
# 
# 1. Accuracy:
# 96.49% accuracy indicates that the SVM model correctly classifies most of the samples in the test dataset. This is a strong performance and shows the model's effectiveness in distinguishing between the two classes (malignant and benign tumors).
# 
# 2. Classification Report:
# Precision:
# Class 0: 95% of benign predictions were correct.
# Class 1: 97% of malignant predictions were correct.
# 
# Recall:
# Class 0: 95% of actual benign cases were correctly identified.
# Class 1: 97% of actual malignant cases were correctly identified.
# 
# F1-Score:
# High F1-scores for both classes (0.95 and 0.97) indicate the model balances precision and recall effectively.
# 
# The macro average treats both classes equally, while the weighted average accounts for the class imbalance, confirming consistent performance across the dataset.
# 
# 3. Confusion Matrix
# True Negatives (41): Benign cases correctly identified as benign.
# 
# True Positives (69): Malignant cases correctly identified as malignant.
# 
# False Positives (2): Benign cases misclassified as malignant.
# 
# False Negatives (2): Malignant cases misclassified as benign.
# 
# The confusion matrix shows only 4 misclassifications out of 114 samples, reflecting the model's robustness.
# 
# 

# # k-Nearest Neighbors (k-NN)

# Description :
# k-Nearest Neighbors (k-NN) is a simple, non-parametric classification algorithm. It predicts the class of a data point based on the classes of its nearest neighbors in the feature space.
# 
# How Does k-NN Work?
# 
# k-NN does not explicitly "train" a model. Instead, it memorizes the entire training dataset.
# 
# For a new data point, k-NN calculates the distance (commonly Euclidean) between the point and all points in the training dataset.
# It identifies the k closest points (neighbors).
# The class is assigned based on the majority class among these neighbors.
# 
# Key Parameters:
# 
# k: The number of neighbors to consider. A smaller k makes the model sensitive to noise, while a larger k smoothens predictions but might ignore local patterns.
# Distance Metric: Commonly, Euclidean distance is used, but others like Manhattan or Minkowski can also be applied.
# 
# Why Might k-NN Be Suitable for This Dataset?
# 
# No Assumptions About Data:
# k-NN makes no assumptions about the underlying data distribution, which is useful when the relationship between features and target is complex.
# 
# Handles Non-Linear Boundaries:
# It can effectively classify data with non-linear decision boundaries, which might be the case in this dataset.
# 
# Dataset Size:
# k-NN works well with small to moderately sized datasets, as computational cost increases with the number of data points.
# 
# Feature Scaling:
# The dataset has been scaled (e.g., using StandardScaler), which is crucial for k-NN as it relies on distance metrics.

# In[40]:


from sklearn.neighbors import KNeighborsClassifier


# In[41]:


# Choosing k=5 as a starting point
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model on the training data
knn.fit(x_train, y_train)

# Make predictions on the test set
y_pred_knn = knn.predict(x_test)


# In[42]:


# Evaluate the model
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print(f"k-NN Accuracy: {knn_accuracy:.2f}")

# Classification report
knn_report = classification_report(y_test, y_pred_knn)
print(f"\nk-NN Classification Report:\n{knn_report}")

# Confusion matrix
knn_conf_matrix = confusion_matrix(y_test, y_pred_knn)
print(f"\nk-NN Confusion Matrix:\n{knn_conf_matrix}")


# # Analysis of KNN Results
# 
# 1. Accuracy: 0.96
# The overall accuracy of the k-NN model is 96%. This indicates that 96% of the total predictions made by the model are correct. It is a strong performance, showing that the model is capable of correctly classifying most of the test dataset.
# 
# 2. Classification Report
# Precision, Recall, and F1-Score:
# 
# Precision is the ratio of correctly predicted positive observations to the total predicted positives.
# Class 0 (Malignant): Precision is 0.95, meaning that 95% of the instances predicted as malignant (Class 0) are truly malignant.
# Class 1 (Benign): Precision is 0.96, meaning that 96% of the instances predicted as benign (Class 1) are truly benign.
# 
# Recall is the ratio of correctly predicted positive observations to the total actual positives.
# Class 0 (Malignant): Recall is 0.93, meaning that the model correctly identified 93% of all malignant instances in the dataset.
# Class 1 (Benign): Recall is 0.97, meaning that the model correctly identified 97% of all benign instances.
# 
# F1-Score is the harmonic mean of precision and recall, giving a balanced metric.
# Class 0 (Malignant): F1-Score is 0.94, indicating a strong balance between precision and recall for malignant instances.
# Class 1 (Benign): F1-Score is 0.97, showing that the model is performing very well on benign instances.
# 
#  Macro Average:
# Macro average for precision, recall, and F1-score is 0.96. This suggests that the model is performing similarly across both classes (malignant and benign), without favoring one over the other.
# 
#  Weighted Average:
# The weighted average for precision, recall, and F1-score is also 0.96, which is similar to the macro average. This is a good sign as it shows that the class distribution in the dataset is balanced, and the model is not biased toward the more frequent class.
# 
# 3. Confusion Matrix:
# 
# True Positives (TP):
# Class 0 (Malignant): 40 instances correctly classified as malignant.
# Class 1 (Benign): 69 instances correctly classified as benign.
# 
# False Positives (FP):
# Class 0 (Malignant): 3 instances incorrectly classified as benign.
# Class 1 (Benign): 2 instances incorrectly classified as malignant.
# 
# False Negatives (FN):
# Class 0 (Malignant): 2 instances of malignant cancer misclassified as benign.
# Class 1 (Benign): 3 instances of benign cancer misclassified as malignant.
# 

# # Comparing the Performance of Algorithms :
# 
# The five classification algorithms—Logistic Regression, Decision Tree Classifier, Random Forest Classifier, Support Vector Machine (SVM), and k-Nearest Neighbors (k-NN)—were evaluated on the breast cancer dataset. Logistic Regression achieved the highest accuracy of 98%, demonstrating strong performance with high precision, recall, and F1-score for both classes. It is simple, interpretable, and effective for this dataset, though it assumes linearity, which can be limiting in more complex datasets. Random Forest, SVM, and k-NN all performed similarly with accuracies of 96%, offering robust performance, though they are computationally more expensive and less interpretable than Logistic Regression. Decision Tree, while interpretable, had the lowest accuracy at 95%, likely due to overfitting and sensitivity to noise. Overall, Logistic Regression outperformed the other models, but Random Forest, SVM, and k-NN also provided competitive results, each with their own strengths and weaknesses depending on the dataset and computational resources.
# 
# 
# 
# 
# 
# 
# 

# # Best Performing Model:
# Logistic Regression performed the best with the highest accuracy (98%) and the lowest number of misclassifications (2). It also achieved high precision, recall, and F1-scores for both classes, making it the most reliable model for this dataset.
# 
# 

# 
# 
# # Worst Performing Model: 
# Decision Tree Classifier performed the worst with an accuracy of 95% and the highest number of misclassifications (6). While still a strong performer, it is not as reliable as Logistic Regression.

# In[ ]:




