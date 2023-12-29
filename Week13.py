"""
TEAM MEMBER’S DETAILS
Group Name: Banking Insights Squad
Group: Members: Canmert Demir & Joseph Pang
Names: Canmert Demir-Bank Marketing (Campaign) -- Group Project
Email: canmertdemir2@gmail.com
Country: Turkey
College/Company: Msc Bartin University-Applied Mathematics / Data Glacier
Specialization: Data Science

Name: Joseph Pang-Bank Marketing (Campaign) -- Group Project
Email: joseph302156@gmail.com
Country: United States
College/Company: University of California, Berkeley/ Data Glacier
Specialization: Data Science
Github Repository:
"""

# There are four datasets:
# 1) bank-additional-full.csv with all examples (41188) and 20 inputs,
# ordered by date (from May 2008 to November 2010), very close to the data analyzed in [Moro et al., 2014]
# 2) bank-additional.csv with 10% of the examples (4119), randomly selected from 1), and 20 inputs.

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
#from UsedFunctions import *

df_test = pd.read_csv("C:\\Users\\Can\\PycharmProjects\\pythonProject\\DataGlacierWeek9\\bank-additional.csv",
                      sep=';')  # testdata
df = pd.read_csv("C:\\Users\\Can\\PycharmProjects\\pythonProject\\DataGlacierWeek9\\bank-additional-full.csv",
                 sep=';')  # traindata

df_test = df_test.sort_index(axis=1).sort_index(axis=0)
df = df.sort_index(axis=1).sort_index(axis=0)

# It is hard to convert duration into bussiness metrics, so we decided to drop it.
# Dropping useless variable
df.drop("duration", axis=1, inplace=True)
df_test.drop("duration", axis=1, inplace=True)

# Categorical Variable Analysis
# The distribution of categorical variables isn't balanced across classes.
# The target variable 'y' exhibits an imbalance.
# 'Education' predominantly comprises individuals with university and high school degrees.
# 'Housing' classes are distributed closely, with 'yes' being the more dominant class.
# 'Job' shows three dominant classes: admin, blue-collar, and technician. Other job categories exhibit minimal differences.
# 'Loan' has three classes, with 'no' being the dominant one among all others.
# 'Marital' features three classes, with the majority being married or single. 'Married' stands as the dominant class.
# 'Month' encompasses 10 classes. May, June, July, August, and November hold significant importance.
# 'Poutcome' showcases three classes, with the 'nonexistent' class being the dominant one. This suggests that the majority of people have not attended a campaign before.
# Numerical variables do not distribute normally.
# Assuming df is your DataFrame
# df["euribor3m"] = np.log(df["euribor3m"]) creates inf and nan values.
# df["emp.var.rate"] = np.log(df["emp.var.rate"]) creating nan values.
# df["cons.conf.idx"] = np.log(df["cons.conf.idx"]) does not use because variable values in [-inf, 0]

df["cons.price.idx"] = np.log(df["cons.price.idx"])
df["nr.employed"] = np.log(df["nr.employed"])

df_test["nr.employed"] = np.log(df_test["nr.employed"])
df_test["cons.price.idx"] = np.log(df_test["cons.price.idx"])

# Define the weights
weight_euribor = 0.7
weight_price = 0.01
weight_conf = 0.29

# Calculate the index
df['economic_index'] = ((weight_euribor * df['euribor3m']) +
                        (weight_price * df['cons.price.idx']) + (weight_conf * df['cons.conf.idx']))
df_test['economic_index'] = ((weight_euribor * df_test['euribor3m']) +
                             (weight_price * df_test['cons.price.idx']) + (weight_conf * df_test['cons.conf.idx']))

df['emp_var_euribor3m_sum'] = df['emp.var.rate'] + df['euribor3m']
df_test['emp_var_euribor3m_sum'] = df_test['emp.var.rate'] + df_test['euribor3m']

df['emp.var.rate_euribor3m_mult'] = df['emp.var.rate'] * df['euribor3m']
df_test['emp.var.rate_euribor3m_mult'] = df_test['emp.var.rate'] * df_test['euribor3m']

df['unemployment_change'] = df['emp.var.rate'] * df['nr.employed']
df_test['unemployment_change'] = df_test['emp.var.rate'] * df_test['nr.employed']

# Encoding Stage
columns_to_encode = df.select_dtypes(include='object').columns

for column in columns_to_encode:

    ordinal_encoder = OrdinalEncoder()

    df[column] = ordinal_encoder.fit_transform(df[[column]])
    df_test[column] = ordinal_encoder.transform(df_test[[column]])

df.drop(columns=['contact', 'day_of_week', 'default', 'education',
                 'housing', 'job', 'loan', 'marital', 'month', 'poutcome'], inplace=True)
df_test.drop(columns=['contact', 'day_of_week', 'default', 'education',
                      'housing', 'job', 'loan', 'marital', 'month', 'poutcome'], inplace=True)

#Seperation target variable and other variables.
X = df.drop('y', axis=1)
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50, shuffle=True, stratify=y)

#Define pipeline components
scaler = StandardScaler()
log_reg = LogisticRegression()
lgbm = LGBMClassifier()
catboost = CatBoostClassifier()

# Define the stacking ensemble
estimators = [
    ('logistic', log_reg),
    ('lgbm', lgbm),
    ('catboost', catboost)
]
# Machine learning alg. stacking.
stacking_classifier = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# Construct the pipeline
pipeline = Pipeline([
    ('scaler', scaler),
    ('stacking_classifier', stacking_classifier)
])

# Fit the pipeline
pipeline.fit(X_train, y_train)
# Predict and evaluate
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy}")

def objective(trial):
    lgbm_params = {
        'n_estimators': trial.suggest_int('n_estimators_lgbm', 50, 500),
        'learning_rate': trial.suggest_loguniform('learning_rate_lgbm', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth_lgbm', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves_lgbm', 3, 40),
    }

    catboost_params = {
        'iterations': trial.suggest_int('iterations_catboost', 50, 500),
        'learning_rate': trial.suggest_loguniform('learning_rate_catboost', 0.01, 0.1),
        'depth': trial.suggest_int('depth_catboost', 3, 10),
    }

    log_reg_params = {
        'C': trial.suggest_loguniform('C_logistic', 0.1, 10),
        'solver': trial.suggest_categorical('solver_logistic', ['liblinear', 'lbfgs']),
    }

    # Create models with suggested hyperparameters
    lgbm = LGBMClassifier(**lgbm_params)
    catboost = CatBoostClassifier(**catboost_params)
    log_reg = LogisticRegression(**log_reg_params)

    # Define the stacking ensemble
    estimators = [
        ('logistic', log_reg),
        ('lgbm', lgbm),
        ('catboost', catboost)
    ]

    stacking_classifier = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

    # Construct the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('stacking_classifier', stacking_classifier)
    ])

    # Set up stratified k-fold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=50)

    # Perform cross-validation
    accuracies = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='accuracy')
    accuracy = accuracies.mean()

    return accuracy

# Optimize hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)  # You can increase the number of trials for better optimization
# Get the best parameters
best_params = study.best_params
print(f"Best Parameters: {best_params}")

# Use the best parameters to train the final model
#Best Parameters: {'n_estimators_lgbm': 444, 'learning_rate_lgbm': 0.09206587020378347,
# 'max_depth_lgbm': 7, 'num_leaves_lgbm': 22, 'iterations_catboost': 482,
# 'learning_rate_catboost': 0.011108413559931083, 'depth_catboost': 8, 'C_logistic': 0.4457968436493604, 'solver_logistic': 'liblinear'}

X_train = df_test.drop("y", axis=1)
y_train = df_test["y"]

# Define classifiers with the best hyperparameters obtained
lgbm_best = LGBMClassifier(n_estimators=444, learning_rate=0.09206587020378347, max_depth=7, num_leaves=22)
catboost_best = CatBoostClassifier(iterations=482, learning_rate=0.011108413559931083, depth=8)
log_reg_best = LogisticRegression(C=0.4457968436493604, solver='liblinear', max_iter=1500)

# Create a VotingClassifier
voting_classifier = VotingClassifier(
    estimators=[
        ('lgbm', lgbm_best),
        ('catboost', catboost_best),
        ('logistic', log_reg_best)
    ],
    voting='soft'  # Use 'soft' for probability voting
)

# Fit the VotingClassifier
voting_classifier.fit(X_train, y_train)

# Predict
y_pred = voting_classifier.predict(X_test)
# Probability estimates for the positive class
y_pred_proba = voting_classifier.predict_proba(X_test)[:, 1]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy}")

# Calculate AUC score
auc = roc_auc_score(y_test, y_pred_proba)
print(f"Test AUC Score: {auc}")

