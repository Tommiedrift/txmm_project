import re
import math
import nltk
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from decimal import Decimal
from nltk import word_tokenize
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

import plotting
import preprocessing
import classification

# nltk.download('punkt')
# nltk.download('wordnet')
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

"""#Preprocessing"""
print("reading excel data...")
data_frame_pitchfork = pd.read_excel(f'Pitchfork_Reviews.xlsx')
print("reading excel data finished")

print("preprocessing...")
df = preprocessing.filter_data(data_frame_pitchfork, False)
reviews = map(preprocessing.clean_text, df['review'])
scores_5 = df['score_5']
scores_3 = df['score_3']
scores_2 = df['score_2']
scores_5_a = df_a['star_rating']
scores_2_a = df_a['score_2']
scores_not3 = df_a[[(x == 0 or x == 1) for x in df_a['score_1245']]]['score_1245']
print("preprocessing finished")

"""#Data Visualization"""
np.savetxt("Pitchfork/pitchfork_data_info", plotting.get_info(df, "score"), fmt="%s")
np.savetxt("Amazon/amazon_data_info", plotting.get_info(df_a, "star_rating"), fmt="%s")

"""#Vectorization"""
vectorizer = TfidfVectorizer(min_df=10, max_df=0.90, ngram_range=(1, 4), stop_words='english',
                             tokenizer=preprocessing.LemmaTokenizer())
vectorizer = TfidfVectorizer(min_df=10, max_df=0.90, ngram_range=(1, 4), stop_words='english',
                             tokenizer=preprocessing.LemmaTokenizer())
vectorizer = TfidfVectorizer(min_df=10, max_df=0.90, ngram_range=(1, 4), stop_words='english',
                             tokenizer=preprocessing.LemmaTokenizer())
vectorizer = TfidfVectorizer(min_df=10, max_df=0.90, ngram_range=(1, 4), stop_words='english',
                             tokenizer=preprocessing.LemmaTokenizer())

print("vectorizing...")
review_features = vectorizer.fit_transform(reviews)
print("vectorizing finished")

"""#Classification"""
print("training classifier...")
X_train2, X_test2, y_train2, y_test2, clf_SGD2 = classification.create_train_perform_SGD(scores_2, 0.3, review_features,
                                                                                         2,
                                                                                         "Pitchfork/pitchfork_classification_report_2")
print("training finished")

"""#Evaluation"""
plotting.create_and_print_confusion_matrix(y_test2, clf_SGD2.predict(X_test2), "SGDClassifier", scores_2,
                                           "Pitchfork/pitchfork_confusion_matrix_2", 10)

"""#Feature Analysis"""
classification.most_informative_feature_for_binary_classification(vectorizer, clf_SGD2, 500,
                                                                  "Pitchfork/pitchfork_informative_features_2.txt")
