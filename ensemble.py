from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd
from rfpimp import *

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

class ensemble:
    def __init__(self, df):
        self.df = df
        self.X = df.drop(columns='is_scammer')
        self.y = df['is_scammer']
        #baseline vectorizer parameters
        tfidf = TfidfVectorizer(
        stop_words='english',
        min_df=3,  # min count for relevant vocabulary
        max_features=5000,  # maximum number of features
        strip_accents='unicode',  # replace all accented unicode char by their corresponding ASCII char
        analyzer='word',  # features made of words
        token_pattern=r'[a-zA-Z]{4,}',  # tokenize only words of 3+ chars
        ngram_range=(1, 1),  # features made of a single tokens
        use_idf=True,  # enable inverse-document-frequency reweighting
        smooth_idf=True,  # prevents zero division for unseen words
        sublinear_tf=False)
        
        #instantiate classifiers for ensemble
        rf = RandomForestClassifier(n_estimators = 100, n_jobs = -1, oob_score = True)
        xgb = XGBClassifier()
        lgb = LGBMClassifier()
        
        #creating ensemble classifier
        self.eclf = VotingClassifier(estimators=[('xgb', xgb), ('lgb', lgb), ('rf', rf)], voting='soft')

        #create pipeline for vectorizing user's comments for Naive Bayes
        self.model = make_pipeline(tfidf, MultinomialNB())

    def split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)
        self.X_train_MNB = self.X_train['comments_new']
        self.X_train = self.X_train.drop(columns=['users','comments','comments_new', 'len_cs'])
        self.X_test_MNB = self.X_test['comments_new']
        self.X_test = self.X_test.drop(columns=['users','comments', 'comments_new', 'len_cs'])

    def fit(self):
        self.eclf.fit(self.X_train, self.y_train)
        self.model.fit(self.X_train_MNB, self.y_train)
    
    def test_predict(self):
        y_pred = self.eclf.predict_proba(self.X_test)[:,1]
        y_pred_MNB = self.model.predict_proba(self.X_test_MNB)[:,1]
        y_final_pred = (y_pred+y_pred_MNB/2)
        return y_final_pred

    def predict(self, X):
        X_MNB = X['comments_new']
        X = X.drop(columns=['users','comments','comments_new', 'len_cs'])
        y_pred = self.eclf.predict_proba(X)[:,1]
        y_pred_MNB = self.model.predict_proba(X_MNB)[:,1]
        y_final_pred = (y_pred+y_pred_MNB/2)
        return y_final_pred
        
if __name__ == "__main__":
    banned = pd.read_csv('data/ban_df.csv', converters={'comments': eval})
    subs = pd.read_csv('data/merged.csv', converters={'comments': eval}, index_col= 'Unnamed: 0')
    banned['is_scammer'] = True
    subs['is_scammer'] = False

    df = pd.concat([banned,subs], ignore_index=True)
    df['comments_new'] = df['comments'].map(lambda x: " ".join(x))

    # predicted = pred >= .5
    # mat = confusion_matrix(e.y_test, predicted)