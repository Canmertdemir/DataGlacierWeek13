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
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

df_test = pd.read_csv("C:\\Users\\Can\\PycharmProjects\\pythonProject\\DataGlacierWeek9\\bank-additional.csv",
                      sep=';')  # testdata
df = pd.read_csv("C:\\Users\\Can\\PycharmProjects\\pythonProject\\DataGlacierWeek9\\bank-additional-full.csv",
                 sep=';')  # traindata

df_test = df_test.sort_index(axis=1).sort_index(axis=0)
df = df.sort_index(axis=1).sort_index(axis=0)

# It is hard to convert duration into bussiness metrics, so we decided to drop it.
df.drop("duration", axis=1, inplace=True)  # Dropping useless variable
df_test.drop("duration", axis=1, inplace=True)
"""
# Some of properties of updated dataset bank_additional
def quick_look(dataframe, head=5):
    print("###################### SHAPE ##########################")
    print(dataframe.shape)

    print("########################## Describe #######################")
    print(dataframe.describe().T)

    print("####################### Variable Types ##################")
    print(dataframe.dtypes)

    print("###################### Head ##########################")
    print(dataframe.head(head))

    print("###################### Tail ##########################")
    print(dataframe.tail(head))
    print("###################### NA ##########################")
    print(dataframe.isna())
    print("###################### NUMBER OF NA ##########################")
    print(dataframe.isna().sum())


quick_look(df)
"""


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

def grab_col_names(dataframe):
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    print("Categorical Variables", cat_cols)

    numerical_cols = [col for col in dataframe.columns if dataframe[col].dtypes in [int, float]]
    print("Those variables numerical  variables", numerical_cols)

    numerical_but_categorical_variable = [col for col in dataframe.columns if
                                          dataframe[col].nunique() < 10 and dataframe[col].dtypes in [int, float]]
    print("Those variables numerical but categorical variables", numerical_but_categorical_variable)

    categorical_but_cardianal_variable = [col for col in dataframe.columns if
                                          dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) not in [int,
                                                                                                               float]]
    print("Those variables categorical variables but cardinal variables", categorical_but_cardianal_variable)

    cat_cols = cat_cols + numerical_but_categorical_variable
    cat_cols = [col for col in cat_cols if col not in categorical_but_cardianal_variable]
    print("All categorical variables are:", cat_cols)

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    # Eğer aynı sütun hem numerical_but_categorical_variable hem de categorical_but_cardianal_variable içindeyse, birini çıkaralım.
    for col in num_cols:
        if col in numerical_but_categorical_variable and col in categorical_but_cardianal_variable:
            categorical_but_cardianal_variable.remove(col)

    print("All numerical variables are:", num_cols)

    print(f"Number Of Observation: {dataframe.shape[0]}")
    print(f"Number Of Variable: {dataframe.shape[1]}")
    print(f"Number Of Categorical Variable: {len(cat_cols)}")
    print(f"Number Of Numerical Variable: {len(num_cols)}")
    print(f"Number Of Categorical but Cardinal Variables: {len(categorical_but_cardianal_variable)}")
    print(f"Number Of Numerical but Cardinal Variables: {len(numerical_but_categorical_variable)}")

    return cat_cols, num_cols, categorical_but_cardianal_variable


cat_cols, num_cols, categorical_but_cardianal_variable = grab_col_names(df)

"""
def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    IQR = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * IQR
    low_limit = quartile1 - 1.5 * IQR
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


cat_cols, num_cols, categorical_but_cardianal_variable = grab_col_names(df)

for col in num_cols:
    print(f"{col}: {check_outlier(df, col)}")


def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


for col in num_cols:
    print(col, grab_outliers(df, col))
"""
"""
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


na_cols = missing_values_table(df)
"""
"""
def categoric_count_plt(dataframe, plot=False):
    for col in dataframe[cat_cols].columns:

        if plot:
            plt.figure(figsize=(15, 6))
            sns.countplot(x=col, data=df[cat_cols])
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.show()


categoric_count_plt(df, plot=True)


# %%
# Scatter Plot of Numerical Variables to see outliars.
# Perfect positive correlation exist.
def num_scatter(dataframe, numerical_cols, plot=False):
    print("Outliers Check")

    if plot:
        for col in numerical_cols:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=dataframe, x=col, y=col, palette="viridis")
            plt.xlabel(col)
            plt.ylabel(col)
            plt.title(f"Scatter plot of {col}")
            plt.show(block=True)


cat_cols, num_cols, categorical_but_cardinal_variable = grab_col_names(df)
num_scatter(df, num_cols, plot=True)


# %%
def analyze_skewness_distribution(data):
    skewness_distribution_results = {}

    for column in data.columns:
        if data[column].dtype == int or data[column].dtype == float:
            _, skewness_pvalue = skewtest(data[column].astype(float).values)

            if skewness_pvalue < 0.05:
                skewness = data[column].skew()
                if skewness < 0:
                    distribution = 'Left Skew'
                elif skewness > 0:
                    distribution = 'Right Skew'
                else:
                    distribution = 'Normal'
            else:
                distribution = 'Normal'

            skewness_distribution_results[column] = distribution

            print(f"{column}: {distribution}")

    return skewness_distribution_results


skewness_distribution_results = analyze_skewness_distribution(df)
"""

# Numerical variables do not distribute normally.
# Assuming df is your DataFrame
# df["euribor3m"] = np.log(df["euribor3m"]) creates inf and nan values.
# df["emp.var.rate"] = np.log(df["emp.var.rate"]) creating nan values.
# df["cons.conf.idx"] = np.log(df["cons.conf.idx"]) does not use because variable values in [-inf, 0]

df["cons.price.idx"] = np.log(df["cons.price.idx"])
df["nr.employed"] = np.log(df["nr.employed"])

df_test["nr.employed"] = np.log(df_test["nr.employed"])
df_test["cons.price.idx"] = np.log(df_test["cons.price.idx"])

"""
def num_summary(dataframe, numerical_col, plot=False):
    print("Numerical variables distribution Check")
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


num_summary(df, num_cols, plot=True)
# %%
age_vs_campaign = df.groupby(["age", "campaign"]).size().reset_index(name='count')
age_vs_campaign_target = age_vs_campaign.groupby("age")['count'].max()
max_campaign_age = age_vs_campaign_target[age_vs_campaign_target == age_vs_campaign_target.max()]
print(max_campaign_age)

default_count_by_group = df.groupby(["age", "campaign", "contact", "day_of_week"])[
    'default'].value_counts().reset_index(name='default_count')
print(default_count_by_group)

default_yes = default_count_by_group[default_count_by_group['default'] == 'yes']
default_no = default_count_by_group[default_count_by_group['default'] == 'no']
default_unknown = default_count_by_group[default_count_by_group['default'] == 'unknown']

print(
    default_yes)  # Default yes people value counts shows that tuesday is good day to call. Old campain stragegy is bad.
# Also default
print(default_no)  # Default no people cardinality is 3976 so bank must be worked on those people.
# Changing strategy, give some promotions such as discount.

print(default_unknown)  # Default unknown people cardinality is 2336. Bank should learn they take campaign or not.
# Or if we assume they all do not take campaign before, bank would change campaign stragey for those people.


housing_loan_vs_default_no = df[(df['housing'] == 'yes') & (df['loan'] == 'yes') & (df['default'] == 'no')]
print(housing_loan_vs_default_no)

housing_loan_vs_default_no_y_yes = housing_loan_vs_default_no[housing_loan_vs_default_no['y'] == 'yes']
print(housing_loan_vs_default_no_y_yes)  # 373 people in this group

housing_loan_vs_default_no_y_no = housing_loan_vs_default_no[housing_loan_vs_default_no['y'] == 'no']
print(housing_loan_vs_default_no_y_no)  # 2595 people in this group, so banking strategy group must work on this people.
# For example, they can try same strategy which used on housing_loan_vs_default_no_y_yes people.

housing_loan_vs_default_yes = df[(df['housing'] == 'yes') & (df['loan'] == 'yes') & (df['default'] == 'yes')]
print(housing_loan_vs_default_yes)  # Empty set, these people churned, so strategy is those people must integrate the
# system by using promotion, discount.

housing_loan_vs_default_yes_y_no = housing_loan_vs_default_yes[housing_loan_vs_default_yes['y'] == 'yes']
print(housing_loan_vs_default_yes_y_no)  # Banking campaing strategist must focus housing=yes, loan=yes people
"""

"""
def cor_analiz_cardinals(dataframe, plot=True):
    numeric_data = dataframe.select_dtypes(include=['float64', 'int64'])
    corr = numeric_data.corr()

    if plot:

        sns.set(rc={"figure.figsize": (12, 12)})
        sns.heatmap(corr, cmap="coolwarm", annot=True)
        plt.show()

        correlated_pairs = {}
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > 0.6:  # Checking absolute correlation value
                    pair = corr.columns[i], corr.columns[j]
                    correlated_pairs[pair] = corr.iloc[i, j]

        return correlated_pairs


correlated_variables = cor_analiz_cardinals(df, plot=False)
print(correlated_variables)

# {('cons.price.idx', 'emp.var.rate'): 0.7753341708348437, ('cons.price.idx', 'euribor3m'): 0.6882301070374915, ('emp.var.rate', 'euribor3m'): 0.9722446711516167,
# ('emp.var.rate', 'nr.employed'): 0.9069701012560616, ('euribor3m', 'nr.employed'): 0.9451544313982757}

# 'emp.var.rate' and  euribor3m variables have high correlation between numerical variables.

# Last week,we merged two datasets which calls bank_additional_full and bank_full. There was a 4 different variable between dataset.
# we used to fill null values mode and median, but it affect ML model performance.
# Filling with null values with NaN to get more efficient prediction.
# Now, further process, we will use bank_additional_full.csv which does not contains any null variable.
# Old feature engineering results show that there was not any good feature which support model's prediction ability.
# In this case, we repeat feature engineering part.
# First, we create basic model to see which feature affect model performance.
# We observed that nr.employed(most important feature), cons.conf.idx, month, contact, poutcome, cons.price.idx, pdays

# %%
# Now, we clean data and protect overfitting.
correlated_variables_updated = cor_analiz_cardinals(df)
print(correlated_variables_updated)
"""

# Define the weights
weight_euribor = 0.7
weight_price = 0.01
weight_conf = 0.29

# Calculate the index
df['economic_index'] = (weight_euribor * df['euribor3m']) + (weight_price * df['cons.price.idx']) + (
        weight_conf * df['cons.conf.idx'])
df_test['economic_index'] = (weight_euribor * df_test['euribor3m']) + (weight_price * df_test['cons.price.idx']) + (
        weight_conf * df_test['cons.conf.idx'])

df['emp_var_euribor3m_sum'] = df['emp.var.rate'] + df['euribor3m']
df_test['emp_var_euribor3m_sum'] = df_test['emp.var.rate'] + df_test['euribor3m']

df['emp.var.rate_euribor3m_mult'] = df['emp.var.rate'] * df['euribor3m']
df_test['emp.var.rate_euribor3m_mult'] = df_test['emp.var.rate'] * df_test['euribor3m']

df['unemployment_change'] = df['emp.var.rate'] * df['nr.employed']
df_test['unemployment_change'] = df_test['emp.var.rate'] * df_test['nr.employed']

"""
df['total_campaign_per_contacts'] = df.groupby('campaign')['contact'].transform('count')
df_test['total_campaign_per_contacts'] = df_test.groupby('campaign')['contact'].transform('count')
# %%
age_vs_campaign = df.groupby(["age", "campaign"]).size().reset_index(name='count')
age_vs_campaign_target = age_vs_campaign.groupby("age")['count'].max()
max_campaign_age = age_vs_campaign_target[age_vs_campaign_target == age_vs_campaign_target.max()]
print(max_campaign_age)

plt.figure(figsize=(10, 6))
plt.bar(max_campaign_age.index, max_campaign_age.values)
plt.xlabel('Age')
plt.ylabel('Max Campaign Count')
plt.title('Max Campaign Count per Age Group')
plt.xticks(max_campaign_age.index)  # Set x-axis ticks to display age values
plt.grid(axis='y')  # Add horizontal gridlines for better readability
plt.show()

# %%
default_count_by_group = df.groupby(["age", "campaign", "contact", "day_of_week"])[
    'default'].value_counts().reset_index(name='default_count')
print(default_count_by_group)

default_yes = default_count_by_group[default_count_by_group['default'] == 'yes']
default_no = default_count_by_group[default_count_by_group['default'] == 'no']
default_unknown = default_count_by_group[default_count_by_group['default'] == 'unknown']

print(
    default_yes)  # Default yes people value counts shows that tuesday is good day to call. Old campain stragegy is bad.
# Also default
print(default_no)  # Default no people cardinality is 3976 so bank must be worked on those people.
# Changing strategy, give some promotions such as discount.

print(default_unknown)  # Default unknown people cardinality is 2336. Bank should learn they take campaign or not.
# Or if we assume they all do not take campaign before, bank would change campaign stragey for those people.

plt.figure(figsize=(10, 6))
sns.barplot(data=default_yes, x='day_of_week', y='default_count')
plt.title('Default Yes - Counts across Days of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=default_no, x='day_of_week', y='default_count')
plt.title('Default No - Counts across Days of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=default_unknown, x='day_of_week', y='default_count')
plt.title('Default Unknown - Counts across Days of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Count')
plt.show()

# %%
housing_loan_vs_default_no = df[(df['housing'] == 'yes') & (df['loan'] == 'yes') & (df['default'] == 'no')]
print(housing_loan_vs_default_no)

housing_loan_vs_default_no_y_yes = housing_loan_vs_default_no[housing_loan_vs_default_no['y'] == 'yes']
print(housing_loan_vs_default_no_y_yes)  # 373 people in this group

housing_loan_vs_default_no_y_no = housing_loan_vs_default_no[housing_loan_vs_default_no['y'] == 'no']
print(housing_loan_vs_default_no_y_no)  # 2595 people in this group, so banking strategy group must work on this people.
# For example, they can try same strategy which used on housing_loan_vs_default_no_y_yes people.

housing_loan_vs_default_yes = df[(df['housing'] == 'yes') & (df['loan'] == 'yes') & (df['default'] == 'yes')]
print(housing_loan_vs_default_yes)  # Empty set, these people churned, so strategy is those people must integrate the
# system by using promotion, discount.

housing_loan_vs_default_yes_y_no = housing_loan_vs_default_yes[housing_loan_vs_default_yes['y'] == 'yes']
print(housing_loan_vs_default_yes_y_no)  # Banking campaing strategist must focus housing=yes, loan=yes people

plt.figure(figsize=(8, 5))
sns.countplot(data=housing_loan_vs_default_no, x='y')
plt.title('Counts for Housing & Loan (Yes) and Default (No)')
plt.xlabel('Response (Yes/No)')
plt.ylabel('Count')
plt.show()
"""

# Encoding Stage
columns_to_encode = df.select_dtypes(include='object').columns

ordinal_encoders = {}

for column in columns_to_encode:
    ordinal_encoder = OrdinalEncoder()

    df[column] = ordinal_encoder.fit_transform(df[[column]])

    df_test[column] = ordinal_encoder.fit_transform(df_test[[column]])

df.drop(columns=['contact', 'day_of_week', 'default', 'education', 'housing', 'job', 'loan', 'marital', 'month',
                 'poutcome'], inplace=True)
df_test.drop(columns=['contact', 'day_of_week', 'default', 'education', 'housing', 'job', 'loan', 'marital', 'month',
                      'poutcome'], inplace=True)

X = df.drop('y', axis=1)

y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50, shuffle=True, stratify=y)

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

# Assuming X_train, X_test, y_train, y_test are defined
"""
# Objective function for hyperparameter optimization
def objective(trial):
    lgbm_params = {
        'n_estimators': trial.suggest_int('n_estimators_lgbm', 50, 500),
        'learning_rate': trial.suggest_loguniform('learning_rate_lgbm', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth_lgbm', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves_lgbm', 3, 40),
        # Add more LGBM hyperparameters for optimization
    }

    catboost_params = {
        'iterations': trial.suggest_int('iterations_catboost', 50, 500),
        'learning_rate': trial.suggest_loguniform('learning_rate_catboost', 0.01, 0.1),
        'depth': trial.suggest_int('depth_catboost', 3, 12),
        # CatBoost için diğer hiperparametreleri buraya ekleyin
    }

    log_reg_params = {
        'C': trial.suggest_loguniform('C_logistic', 0.1, 10),
        'solver': trial.suggest_categorical('solver_logistic', ['liblinear', 'lbfgs']),
        # Add more Logistic Regression hyperparameters for optimization
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

    # Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Optimize hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)  # You can increase the number of trials for better optimization

# Get the best parameters
best_params = study.best_params
print(f"Best Parameters: {best_params}")
"""
# Use the best parameters to train the final model
# ... (same code as before using the best_params to instantiate models and pipeline)

# Best Parameters: {'n_estimators_lgbm': 450, 'learning_rate_lgbm': 0.05325980760529078, 'max_depth_lgbm': 6,
# 'num_leaves_lgbm': 33, 'iterations_catboost': 387, 'learning_rate_catboost': 0.030007096670119283, 'depth_catboost': 9,
# 'C_logistic': 5.093614309443904, 'solver_logistic': 'lbfgs'}

from sklearn.ensemble import VotingClassifier

X_train = df_test.drop("y", axis=1)
y_train = df_test["y"]

# Define classifiers with the best hyperparameters obtained
lgbm_best = LGBMClassifier(n_estimators=450, learning_rate=0.05325980760529078, max_depth=6, num_leaves=33)
catboost_best = CatBoostClassifier(iterations=387, learning_rate=0.030007096670119283, depth=9)
log_reg_best = LogisticRegression(C=5.093614309443904, solver='lbfgs', max_iter=1500)

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
y_pred_proba = voting_classifier.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy}")

# Calculate AUC score
auc = roc_auc_score(y_test, y_pred_proba)
print(f"Test AUC Score: {auc}")
"""
estimators = voting_classifier.estimators_
# Logistic reg. does not support feature importance
for clf in estimators:
    # Check if the classifier supports feature importances
    if hasattr(clf, 'feature_importances_'):
        # Access feature importances for classifiers that support it
        print(clf.__class__.__name__)
        print(clf.feature_importances_)
    else:
        print(f"{clf.__class__.__name__} doesn't support feature importances.")


# LGBMClassifier Feature Importances
lgbm_importances = [2043, 835, 189, 184, 26, 1442, 72, 283, 215, 1216, 503, 528, 0]

# CatBoostClassifier Feature Importances
catboost_importances = [24.58400596, 18.18117565, 4.29555275, 4.66507319, 1.89777152, 7.1069525, 3.51773398,
                        5.20804388, 4.54704207, 9.01263334, 6.4312837, 8.65178253, 1.90094893]

# Assuming you have feature names stored in a list called 'feature_names'
feature_names = ['age', 'campaign', 'cons.conf.idx', 'cons.price.idx', 'emp.var.rate',
       'euribor3m', 'nr.employed', 'pdays', 'previous', 'y', 'economic_index',
       'emp_var_euribor3m_sum', 'emp.var.rate_euribor3m_mult',
       'unemployment_change']

# Sort feature importances for LGBMClassifier and CatBoostClassifier in descending order
indices_lgbm = np.argsort(lgbm_importances)[::-1]
sorted_feature_names_lgbm = [feature_names[i] for i in indices_lgbm]
sorted_importances_lgbm = np.array(lgbm_importances)[indices_lgbm]

indices_catboost = np.argsort(catboost_importances)[::-1]
sorted_feature_names_catboost = [feature_names[i] for i in indices_catboost]
sorted_importances_catboost = np.array(catboost_importances)[indices_catboost]

# Finding common features
common_features = set(sorted_feature_names_lgbm).intersection(set(sorted_feature_names_catboost))

# Displaying common features and their importances
common_features_importance_lgbm = [importance for feature, importance in zip(sorted_feature_names_lgbm, sorted_importances_lgbm) if feature in common_features]
common_features_importance_catboost = [importance for feature, importance in zip(sorted_feature_names_catboost, sorted_importances_catboost) if feature in common_features]

# Plotting common feature importances
plt.figure(figsize=(10, 6))
plt.title("Common Feature Importances between LGBMClassifier and CatBoostClassifier")
plt.bar(range(len(common_features)), common_features_importance_lgbm, align="center", alpha=0.5, label='LGBM')
plt.bar(range(len(common_features)), common_features_importance_catboost, align="center", alpha=0.5, label='CatBoost')
plt.xticks(range(len(common_features)), common_features, rotation=90)
plt.xlabel("Features")
plt.ylabel("Feature Importance")
plt.legend()
plt.tight_layout()
plt.show()
"""
