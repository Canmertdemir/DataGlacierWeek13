import numpy as np
import sns
from matplotlib import pyplot as plt
from scipy.stats import skewtest


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


def categoric_count_plt(dataframe, plot=False):
    for col in dataframe[cat_cols].columns:

        if plot:
            plt.figure(figsize=(15, 6))
            sns.countplot(x=col, data=dataframe[cat_cols])
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.show()


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

def missing_values_table(dataframe):
    df_null_values = dataframe.isnull().sum().to_frame().rename(columns={0: 'Count'})
    df_null_values['Percentage_nulls'] = (df_null_values['Count'] / len(dataframe)) * 100
    df_null_values['Percentage_no_nulls'] = 100 - df_null_values['Percentage_nulls']

    n = len(df_null_values.index)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 8))

    bar_width = 0.4
    gap = 0.2

    rects1 = ax.barh(x - gap / 2, df_null_values['Percentage_nulls'], bar_width, label='Null values', color='red')
    rects2 = ax.barh(x + gap / 2, df_null_values['Percentage_no_nulls'], bar_width, label='No null values',
                     color='orange')

    ax.set_title('Null Values and Non-null Values', fontsize=15, fontweight='bold')
    ax.set_xlabel('% Percentage', fontsize=12, fontweight='bold')
    ax.set_yticks(x)
    ax.set_yticklabels(df_null_values.index, fontsize=10, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            width = rect.get_width()
            ax.annotate(f'{width:.2f}%',
                        xy=(width, rect.get_y() + rect.get_height() / 2),
                        xytext=(2, 0),
                        textcoords="offset points",
                        ha='left', va='center', size=10, weight='bold')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.show()


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



"""

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
"""
df['total_campaign_per_contacts'] = df.groupby('campaign')['contact'].transform('count')
df_test['total_campaign_per_contacts'] = df_test.groupby('campaign')['contact'].transform('count')
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

default_count_by_group = df.groupby(["age", "campaign", "contact", "day_of_week"])[
    'default'].value_counts().reset_index(name='default_count')
print(default_count_by_group)

default_yes = default_count_by_group[default_count_by_group['default'] == 'yes']
default_no = default_count_by_group[default_count_by_group['default'] == 'no']
default_unknown = default_count_by_group[default_count_by_group['default'] == 'unknown']

print(default_yes)  # Default yes people value counts shows that tuesday is good day to call. Old campain stragegy is bad.
# Also default
print(default_no)  # Default no people cardinality is 3976 so bank must be worked on those people.
# Changing strategy, give some promotions such as discount.

print(default_unknown)  # Default unknown people cardinality is 2336. Bank should learn they take campaign or not.
# Or if we assume they all do not take campaign before, bank would change campaign strategy for those people.

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

#for col in num_cols:
    #print(f"{col}: {check_outlier(df, col)}")


def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


#for col in num_cols:
    #print(col, grab_outliers(df, col))


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