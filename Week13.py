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

import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from UsedFunctions import data_fix, feature_eng, ordinal_encoder

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.expand_frame_repr', False)

def final_model_pipeline():

    df_test = pd.read_csv("C:\\Users\\Can\\PycharmProjects\\pythonProject"
                          "\\DataGlacierWeek9\\bank-additional.csv", sep=';')
    df = pd.read_csv("C:\\Users\\Can\\PycharmProjects\\pythonProject"
                     "\\DataGlacierWeek9\\bank-additional-full.csv", sep=';')

    data_fix(df)
    data_fix(df_test)

    feature_eng(df)
    feature_eng(df_test)

    ordinal_encoder(df)
    ordinal_encoder(df_test)

    X_train = df_test.drop("y", axis=1)
    y_train = df_test["y"]

    lgbm_best = LGBMClassifier(n_estimators=444, learning_rate=0.09206587020378347, max_depth=7, num_leaves=22)
    catboost_best = CatBoostClassifier(iterations=482, learning_rate=0.011108413559931083, depth=8)
    log_reg_best = LogisticRegression(C=0.4457968436493604, solver='liblinear', max_iter=1500)

    voting_classifier = VotingClassifier(
        estimators=[
            ('lgbm', lgbm_best),
            ('catboost', catboost_best),
            ('logistic', log_reg_best)
        ],
        voting='soft'
    )

    voting_classifier.fit(X_train, y_train)
    y_pred = voting_classifier.predict(X_test)
    y_pred_proba = voting_classifier.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1_score_value = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Test Accuracy: {accuracy}")
    print(f"F1 Score: {f1_score_value}")
    print(f"Test AUC Score: {auc}")

    df_test["y"] = ["No" if val == 0 else "Yes" for val in df_test["y"]]
    target_customer_list = df_test.loc[(df_test["y"] == "Yes", ["age", "campaign"]) & df["default"]]
    target_customer_list.to_csv("Target_Customers_in_Campain.csv")


