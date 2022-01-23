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

import preprocessing
import plotting
import classification

column_names = ['marketplace', 'customer_id', 'review_id', 'product_id',
       'product_parent', 'product_title', 'product_category', 'star_rating',
       'helpful_votes', 'total_votes', 'vine', 'verified_purchase',
       'review_headline', 'review_body', 'review_date']

print("reading excel data...")
data_frame_amazon = pd.read_excel(f'Amazon_Reviews.xlsx')
print("reading excel data finished")
df_a = preprocessing.filter_data_amazon(data_frame_amazon, True)
scores_5_a = df_a['star_rating']
scores_2_a = df_a['score_2']

plotting.show_bar(df_a['star_rating'], ['1', '2', '3', '4', '5'], 'Distribution of Amazon music review scores', "Amazon/amazon_scores_bar_chart_5", 0)
plotting.show_bar(df_a['score_2'], ['1,2,3', '4,5'], 'Distribution of Amazon music review scores', "Amazon/amazon_scores_bar_chart_2", 1)

plotting.plot_review_dist(df_a, "review_body", "Amazon/amazon_review_len_dis")


reviews_a = map(preprocessing.clean_text, df_a['review_body'])
vectorizer = TfidfVectorizer(min_df=10, max_df=0.90, ngram_range=(1, 4), stop_words='english',tokenizer=preprocessing.LemmaTokenizer())

print("Vectorizing...")
review_features_a = vectorizer.fit_transform(reviews_a)
print("Vectorizing finished")
# review_features_2 = vectorizer_2.fit_transform(reviews)

"""#Classification"""
print("training classifier...")
X_train2_a, X_test2_a, y_train2_a, y_test2_a, clf_SGD2_a = classification.create_train_perform_SGD(scores_2_a, 0.3, review_features_a, 2)
X_train5_a, X_test5_a, y_train5_a, y_test5_a, clf_SGD5_a = classification.create_train_perform_SGD(scores_5_a, 0.3, review_features_a, 5)

"""#Evaluation"""
plotting.create_and_print_confusion_matrix(y_test2_a, clf_SGD2_a.predict(X_test2_a), "SGDClassifier", scores_2_a, "Amazon/confusion_matrix_2_amazon")
plotting.create_and_print_confusion_matrix(y_test5_a, clf_SGD5_a.predict(X_test5_a), "SGDClassifier", scores_5_a, "Amazon/confusion_matrix_5_amazon")

"""#Feature Analysis"""
classification.most_informative_feature_for_binary_classification(vectorizer, clf_SGD2_a, 500, "Amazon/informative_features_amazon_2.txt")
classification.most_informative_features_non_binary(vectorizer, clf_SGD5_a, 500, "Amazon/informative_features_amazon_5.txt")