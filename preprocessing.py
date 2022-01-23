import re
import math
import nltk
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from decimal import Decimal
from nltk import word_tokenize
# from google.colab import files
import matplotlib.pyplot as plt
from IPython.display import display
from nltk.stem import WordNetLemmatizer
from wordcloud import STOPWORDS, WordCloud
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score)

column_names = ['marketplace', 'customer_id', 'review_id', 'product_id',
                'product_parent', 'product_title', 'product_category', 'star_rating',
                'helpful_votes', 'total_votes', 'vine', 'verified_purchase',
                'review_headline', 'review_body', 'review_date']

score_values = str([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,
                    1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                    2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1,
                    4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5,
                    5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9,
                    7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3,
                    8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7,
                    9.8, 9.9, 10.0])

score_values_a = [1, 2, 3, 4, 5]


def three_score(row):
    # onethirdper = np.percentile(row['score'], 100 / 3)
    # twothirdper = np.percentile(row['score'], 100 / 3 * 2)
    if row['score'] <= 6.5:
        return 0
    if row['score'] >= 7.5:
        return 1
    else:
        return np.NAN


def filter_data(df, to_print):
    len_1 = len(df)
    if to_print:
        print("Original amount of rows: {} \n".format(str(len_1)))
    # df = df.drop(df[pd.isnull(df.score)].index)
    df = df[df['score'].notnull()]
    len_2 = len(df)
    if to_print:
        print("Amount of null rows dropped: {} \n".format(str(len_1 - len_2)))
    df = df.drop(df[[x not in score_values for x in df.score]].index)
    if to_print:
        print("Amount of non-number rows dropped: {} \n".format(str(len_2 - len(df))))
        print("Resulting amount of rows: {} \n".format(str(len(df))))
    df = df[df['review'].notnull()]
    # df['score_dec'] = list(map(Decimal, df['score']))
    df['score'] = df['score'].astype(float)
    df['score_5'] = df.apply(lambda row: min(5, math.floor(row['score'] / 2) + 1), axis=1)
    df['score_6575'] = df.apply(lambda row: three_score(row), axis=1)
    df['score_2'] = df.apply(lambda row: 0 if row['score'] <= 7.3 else 1, axis=1)

    df['metal'] = df.apply(lambda row: ('Metal' in row['genre']) if isinstance(row['genre'], str) else False, axis=1)
    df['rap'] = df.apply(lambda row: ('Rap' in row['genre']) if isinstance(row['genre'], str) else False, axis=1)
    df['rock'] = df.apply(lambda row: ('Rock' in row['genre']) if isinstance(row['genre'], str) else False, axis=1)
    df['folk/country'] = df.apply(
        lambda row: ('Folk/Country' in row['genre']) if isinstance(row['genre'], str) else False,
        axis=1)
    df['global'] = df.apply(lambda row: ('Global' in row['genre']) if isinstance(row['genre'], str) else False, axis=1)
    df['pop/R&B'] = df.apply(lambda row: ('Pop/R&B' in row['genre']) if isinstance(row['genre'], str) else False,
                             axis=1)
    df['electronic'] = df.apply(lambda row: ('Electronic' in row['genre']) if isinstance(row['genre'], str) else False,
                                axis=1)
    df['jazz'] = df.apply(lambda row: ('Jazz' in row['genre']) if isinstance(row['genre'], str) else False, axis=1)
    df['experimental'] = df.apply(
        lambda row: ('Experimental' in row['genre']) if isinstance(row['genre'], str) else False,
        axis=1)
    return df

def score_star(row):
    if row['star_rating'] < 3:
        return 0
    if row['star_rating'] > 3:
        return 1
    else:
        return np.NAN

def filter_data_amazon(df, to_print):
    df = df[df['star_rating'].notnull()]
    df = df[df['review_body'].notnull()]
    df['star_rating'] = df['star_rating'].astype(float)
    df['score_2'] = df.apply(lambda row: 0 if row['star_rating'] < 4 else 1, axis=1)
    df['int_star'] = df['star_rating'].astype(int)
    df['score_1245'] = df.apply(lambda row: 0 if row['star_rating'] < 3 else 1, axis=1)
    # print("before", len(df))
    df = df.drop(df[[len(x) < 15 for x in df['review_body']]].index)
    # print("after", len(df))
    return df

def filter_data_amazon2(df, to_print):
    df = df[df['star_rating'].notnull()]
    df = df[df['review_body'].notnull()]
    df['star_rating'] = df['star_rating'].astype(float)
    df['score_1245'] = df.apply(lambda row: 0 if row['star_rating'] < 3 else 1, axis=1)
    df = df.drop(df[[len(x) < 15 for x in df['review_body']]].index)
    return df


def clean_text(string):
    string = re.sub("[^a-zA-Z]", " ", string)
    string = string.lower()
    return string


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

# drop_indices = np.random.choice(df_a[df_a['star_rating'] == 5].index, 600000, replace=False)
# drop_indices1 = np.random.choice(df_a[df_a['star_rating'] == 5].index, 80000, replace=False)
# drop_indices2 = np.random.choice(df_a[df_a['star_rating'] == 4].index, 80000, replace=False)

# df_a3 = df_a3.drop(drop_indices2)
# df2 = df.copy()
# df2 = df2[df2['score'] <= 6.5]
# df2 = df2[df2['score'] >= 7.5]
# df2 = df2[~df2['score'].between(6.5, 7.5, inclusive="neither")]
# print(df2['score_6575'].value_counts().sort_index())
# print(len(df2['score_6575']))
# display(df2)
# df2.to_excel('Pitchfork_Reviews_6575.xlsx', index=False)