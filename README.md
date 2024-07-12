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

#### Key Visualizations Created


  
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
  
  - **Data Preparation** {Preprocessed the dataset, including handling missing values, feature scaling, and splitting into training and testing sets}
  - **Model Configuration** {Imported the **DecisionTreeClassifier** from **sklearn.tree**; Defined the hyperparameters for the Decision Tree model}
  - **Model Training** {Trained the model on the training dataset through the use of **fit()** function}
  - **Model Predition** { Used the trained model for making predictions on the test set through the **predict()** function
  - **Model Evaluation** {Assessed the models effectiveness through evaluating metrics such as accuracy, precision, recall, and F1-score}

  
### Logistic Regression

Steps involved in applying Logistic Regression model include:
  
  - **Data Preparation** {Preprocessed the dataset, including handling missing values, feature scaling, and splitting into training and testing sets}
  - **Model Configuration** {Imported the **LogisticRegression** from **sklearn.linear_model**}
  - **Model Training** {Trained the model on the training dataset through the use of **fit()** function}
  - **Model Predition** { Used the trained model for making predictions on the test set through the **predict()** function
  - **Model Evaluation** {Assessed the models effectiveness through evaluating metrics such as accuracy, precision, recall, and F1-score}
  
### Support Vector Machine (SVM)

 Steps involved in applying SVM model include:
  
  - **Data Preparation** {Preprocessed the dataset, including handling missing values, feature scaling, and splitting into training and testing sets}
  - **Model Configuration** {Imported the **SVC(Support Vector Classifier)** from **sklearn.svm**}
  - **Model Training** {Trained the model on the training dataset through the use of **fit()** function}
  - **Model Predition** { Used the trained model for making predictions on the test set through the **predict()** function
  - **Model Evaluation** {Assessed the models effectiveness through evaluating metrics such as accuracy, precision, recall, and F1-score}

### Randon Forest

 Steps involved in applying Random Forest model include:
  
  - **Data Preparation** {Preprocessed the dataset, including handling missing values, feature scaling, and splitting into training and testing sets}
  - **Model Configuration** {Imported the **Random Forest Classifier** from **sklearn.ensemble**; Defined the hyperparameters for the Random Forest model}
  - **Model Training** {Trained the model on the training dataset through the use of **fit()** function}
  - **Model Predition** { Employed the trained model for making predictions on the test set through the **predict()** function
  - **Model Evaluation** {Assessed the models effectiveness through evaluating metrics such as accuracy, precision, recall, and F1-score}
  
### K-Nearest Neighbors

Steps involved in applying K-Nearest Neighbors model include:
  
  - **Data Preparation** {Preprocessed the dataset, including handling missing values, feature scaling, and splitting into training and testing sets}
  - **Model Configuration** {Imported the **KNeighborsClassifier** from **sklearn.neighbors**}
  - **Model Training** {Trained the model on the training dataset through the use of **fit(**) function}
  - **Model Predition** { Used the trained model for making predictions on the test set through the **predict()** function
  - **Model Evaluation** {Assessed the models effectiveness through evaluating metrics such as accuracy, precision, recall, and F1-score}

#### Results: Performance Metrics for Diabetes Prediction Models

 This project evaluated the effectiveness of each model by assessing metrics such as accuracy, precision,recall, and F1-score. The results are as shown below:  

                                                                    
         
      |Model                            |Accuracy                 |Precision                    |Recall               |F1-Score|       
      |--------------------------------------------------------------------------------------------------------------------------|
      | Decision Tree                   | 0.78                    |0.63                         |0.66                 | 0.65     |
      | Logistic Regression             | 0.81                    | 0.73                        | 0.57                | 0.64     |
      | Support Vector Machine (SVM)    | 0.80                    | 0.71                        | 0.57                | 0.64     |
      | Randon Forest                   | 0.79                    | 0.67                        | 0.64                | 0.65     |
      | K-Nearest Neighbors             | 0.82                    | 0.74                        | 0.66                | 0.70     |
              

  
### Conclusion

- An investigation of the five models demonstrated that K-Nearest Neighbors achieved superior performance for diabetes detection on the Pima Indian Diabetes dataset, with the higherst accuracy of 82%
  along with precision, recall, and F1-score.




