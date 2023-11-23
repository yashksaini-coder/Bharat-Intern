## **Titanic Survival Prediction Analysis - Classification Project**

## Overview

This repository contains a data science project focused on predicting the survival of Titanic passengers using machine learning models. The dataset includes information about passengers, such as their age, sex, class, and other features, with the goal of predicting whether a passenger survived or not.

## Dataset Information:-

The dataset used in this project is sourced from the Titanic dataset. It includes the following columns:-

- **PassengerId:-** Unique identifier for each passenger
- **Survived:-** Binary variable indicating whether the passenger survived (1) or not (0)
- **Pclass:-** Passenger class (1st, 2nd, or 3rd)
- **Name:-** Passenger's name
- **Sex:-** Passenger's gender (0 for male, 1 for female)
- **Age:-** Passenger's age
- **SibSp:-** Number of siblings/spouses aboard
- **Parch:-** Number of parents/children aboard
- **Ticket:-** Ticket number
- **Fare:-** Fare paid for the ticket
- **Cabin:-** Cabin number
- **Embarked:-** Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

The data has been split into two groups:-
- training set (train.csv)
- test set (test.csv)

The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.

The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.

We also include gender_submission.csv, a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.

Variable |	Definition | Key
----------|--------------|-----
survival |	Survival |	0 = No, 1 = Yes
pclass |	Ticket class |	1 = 1st, 2 = 2nd, 3 = 3rd
sex	| Sex	|
Age	| Age in years	|
sibsp |	# of siblings / spouses aboard the Titanic	|
parch |	# of parents / children aboard the Titanic	|
ticket |	Ticket number |	
fare |	Passenger fare	|
cabin |	Cabin number	|
embarked |	Port of Embarkation |	C = Cherbourg, Q = Queenstown, S = Southampton |

## Data Preprocessing

1. **Handling Missing Values:-**
   - Addressed missing values in the 'Age' column by replacing them with the mean age.
   - Filled missing 'Embarked' values with the most common port.
   - Imputed missing 'Fare' values with the median.

2. **Encoding Categorical Variables:-**
   - Converted categorical variables like 'Sex' into numerical format (0 for male, 1 for female).
   - Created binary columns ('Q' and 'S') to represent the 'Embarked' variable.

## Data Visualization

1. **Count Plot:-**
   - Visualized the distribution of the 'Survived' variable using Seaborn's count plot.

2. **Feature Distribution Plot:-**
   - Plotted the distribution of the 'Fare' feature to understand the spread of ticket prices.

3. **Survival Rate by Class:-**
   - Created a bar plot to show the survival rate based on passenger class.

## Model Training

1. **Random Forest Classifier:-**
   - Utilized a Random Forest Classifier with parameters:-
      - Number of Estimators:- 200
      - Criterion:- Gini
      - Max Features:- Auto (Square root of the total number of features)
      - Max Depth:- 8

2. **Train-Test Split:-**
   - Split the dataset into training and testing sets.

3. **Model Evaluation:-**
   - Evaluated the model's performance on the testing set using metrics like accuracy, precision, and recall.

## Prediction

1. **Generated Predictions:-**
   - Used the trained model to predict survival for new data.

## Basic Data Science Terminologies

- **Data Science:-** The interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract insights and knowledge from structured and unstructured data.

- **Data Preprocessing:-** The process of cleaning and transforming raw data into a format suitable for analysis.

- **Data Visualization:-** The graphical representation of data to reveal insights, patterns, and trends.

- **Train-Test Split:-** The practice of splitting a dataset into two subsets:- one for training a model and the other for testing its performance.

- **Machine Learning Models:-** Algorithms and statistical models that enable computers to learn patterns from data and make predictions or decisions without explicit programming.

- **Prediction:-** The process of using a trained model to make forecasts or estimations on new, unseen data.

## Dependencies

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## How to Run

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Jupyter notebook or Python script.




