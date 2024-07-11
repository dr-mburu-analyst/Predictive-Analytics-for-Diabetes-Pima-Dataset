# Pima Indians Diabetes Prediction

## Overview

Accurate detection of diabetes is key in early intervention and effective management of the disease. Use of machine learning can help in early diagnosis.
This project presents a comparative analysis of five models including Decision Tree, Logistic Regression, SVM, Random Forest, and 
K-Nearest Neighbors for diabetes detection on the widely used Pima Indians Diabetes Database(PIDD). The projects aims to evaluate the performance of these models using metrics such as 
accuracy, precision, recall, and F1-score.

### Dataset
The dataset (pima-indians-diabetes.csv) contains the following features:

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration after 2 hours in an oral glucose tolerance test
- **Blood Pressure**: Diastolic blood pressure (mm Hg)
- **Skin Thickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **Diabetes Pedigree Function**: Diabetes pedigree function
- **Age**: Age in years
- **Outcome**: Target variable (0 - non-diabetic, 1 - diabetic)
  
### Project Steps

#### Import Libraries
- import pandas as pd
- import numpy as np
- import matplotlib.pyplot as plt
- import seaborn as sns
- from sklearn.model_selection import train_test_split
- from sklearn.preprocessing import StandardScaler
- from sklearn.pipeline import Pipeline
- from sklearn.metrics import classification_report, confusion_matrix
- from sklearn.linear_model import LogisticRegression
- from sklearn.tree import DecisionTreeClassifier
- from sklearn.ensemble import RandomForestClassifier
- from sklearn.neighbors import KNeighborsClassifier
- from sklearn.svm import SVC


#### Load Dataset

- Load the Pima_Indians_Diabetes dataset on VS Code
- Verify the data by printing the first few rows of the dataset
- Run the script
- Check the output in the terminal to see the printed data.

  
#### Preprocess Data

PIDD is a widely used dataset for diabetes prediction. In this project a series of preprocessing steps were implemented before training the models.
These steps included handling missing values, feature scalinf, and splitting the ata into trainung and testing values.

#### Handling Missing Values:

- For this dataset, there were no instances of missing values.
  
#### Data Exploration:

- Explored data distribution and statistics.
- Visualized key features against the diabetes outcome.
  
#### Feature Selection:

- Selected features based on their impact on the diabetes prediction.

###### Split the dataset into training and testing sets
- Split the data into training and testing sets. The process of splitting the dataset was carried out using scikit-learn's(**train_test_split** function)
  specifying test size as **20%**, and setting the random state for reproducibility.
  
##### Normalize the data
- Normalize the features to ensure consistency.
  
#### Train and Evaluate Models:

Trained several machine learning models including Decision Tree, Logistic Regression, SVM, Random Forest, and K-Nearest Neighbors.
Used GridSearchCV for hyperparameter tuning.
Evaluated models using confusion matrices and classification reports.

### Decision Tree

 Steps involved in applying Decision Tree model include:
  
  -**Data Preparation** {Preprocessed the dataset, including handling missing values, feature scaling, and splitting into training and testing sets}
  -**Model Configuration** {Imported the **DecisionTreeClassifier** from **sklearn.tree**; Defined the hyperparameters for the Decision Tree model}
  -**Model Training** {Trained the model on the training dataset through the use of **fit()** function}
  -**Model Predition** { Used the trained model for making predictions on the test set through the **predict()** function
  -**Model Evaluation** {Assessed the models effectiveness through evaluating metrics such as accuracy, precision, recall, and F1-score}

  
### Logistic Regression

- Steps involved in applying Logistic Regression model include:
  
  -- **Data Preparation** {Preprocessed the dataset, including handling missing values, feature scaling, and splitting into training and testing sets}
  -- **Model Configuration** {Imported the **LogisticRegression** from **sklearn.linear_model**}
  -- **Model Training** {Trained the model on the training dataset through the use of **fit()** function}
  -- **Model Predition** { Used the trained model for making predictions on the test set through the **predict()** function
  -- **Model Evaluation** {Assessed the models effectiveness through evaluating metrics such as accuracy, precision, recall, and F1-score}
  
### Support Vector Machine (SVM)

- Steps involved in applying SVM model include:
  
  --**Data Preparation** {Preprocessed the dataset, including handling missing values, feature scaling, and splitting into training and testing sets}
  -- **Model Configuration** {Imported the **SVC(Support Vector Classifier)** from **sklearn.svm**}
  -- **Model Training** {Trained the model on the training dataset through the use of **fit()** function}
  -- **Model Predition** { Used the trained model for making predictions on the test set through the **predict()** function
  --**Model Evaluation** {Assessed the models effectiveness through evaluating metrics such as accuracy, precision, recall, and F1-score}

### Randon Forest

- Steps involved in applying Random Forest model include:
  
  --**Data Preparation** {Preprocessed the dataset, including handling missing values, feature scaling, and splitting into training and testing sets}
  -- **Model Configuration** {Imported the **Random Forest Classifier** from **sklearn.ensemble**; Defined the hyperparameters for the Random Forest model}
  -- **Model Training** {Trained the model on the training dataset through the use of **fit()** function}
  -- **Model Predition** { Employed the trained model for making predictions on the test set through the **predict()** function
  --**Model Evaluation** {Assessed the models effectiveness through evaluating metrics such as accuracy, precision, recall, and F1-score}
  
### K-Nearest Neighbors

- Steps involved in applying K-Nearest Neighbors model include:
  
  -- **Data Preparation** {Preprocessed the dataset, including handling missing values, feature scaling, and splitting into training and testing sets}
  -- **Model Configuration** {Imported the **KNeighborsClassifier** from **sklearn.neighbors**}
  -- **Model Training** {Trained the model on the training dataset through the use of **fit(**) function}
  -- **Model Predition** { Used the trained model for making predictions on the test set through the **predict()** function
  -- **Model Evaluation** {Assessed the models effectiveness through evaluating metrics such as accuracy, precision, recall, and F1-score}

#### Results:

- After thorough exploration and evaluation of various machine learning models on the Pima Indians dataset, both K-Nearest Neighbors (KNN) and 
Logistic Regression demonstrated equal accuracy of 81%.
- However, KNN had a higher F1 score for class 1 (70%), highlighting its effectiveness in identifying diabetic cases.

### Visualizations

- Distribution of outcomes, pregnancies, glucose levels, BMI, and other features by diabetes status.
- Heatmap showing correlations between features.

  ![Count of Diabetic/Non-Diabetic Patients](images/Count_of_Diabetes_Outcome.png)

  
### Conclusion

- This project demonstrates how machine learning can predict diabetes based on health metrics among the Pima Indians. 
- The insights gained can potentially aid in early diagnosis and intervention.
- Features such as glucose levels, BMI, age, and insulin levels strongly correlate with diabetes status.
- Non-diabetic patients tend to be younger, with lower BMI, and smaller triceps thickness, while diabetic individuals are generally older and have higher BMI and triceps thickness.
- Machine learning models trained on this dataset can effectively predict diabetes status, with features like BMI and glucose playing crucial roles in classification.
