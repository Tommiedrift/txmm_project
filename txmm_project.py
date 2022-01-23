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
#data_frame_pitchfork = pd.read_excel(f'Pitchfork_Reviews.xlsx')
data_frame_pitchfork2 = pd.read_excel(f'Pitchfork_Reviews_6575.xlsx')
#data_frame_amazon = pd.read_excel(f'Amazon_Dataset_short.xlsx')
data_frame_amazon_1245 = pd.read_excel(f'Amazon_Dataset_1245.xlsx')
print("reading excel data finished")

print("preprocessing...")
#df = preprocessing.filter_data(data_frame_pitchfork, False)
#df_a = preprocessing.filter_data_amazon(data_frame_amazon, False)
df_a2 = preprocessing.filter_data_amazon2(data_frame_amazon_1245, False)
df2 = data_frame_pitchfork2

#reviews = map(preprocessing.clean_text, df['review'])
reviews2 = map(preprocessing.clean_text, df2['review'])
#reviews_a = map(preprocessing.clean_text, df_a['review_body'])
reviews_a2 = map(preprocessing.clean_text, df_a2['review_body'])

#scores_5 = df['score_5']
# for i in scores_5[:3]:
#     print(type(i))
#scores_3 = df['score_3']
#scores_2 = df['score_2']
#scores_5_a = df_a['star_rating']
#scores_2_a = df_a['score_2']
scores_1245 = df_a2['score_1245']
scores_6575 = df2['score_6575']


print("preprocessing finished")

"""#Data Visualization"""
print("visualizing data...")
#np.savetxt("Pitchfork/pitchfork_data_info.txt", plotting.get_info(df, "score"), fmt="%s")
np.savetxt("Pitchfork/pitchfork_data_info_6575.txt", plotting.get_info(df2, "score_6575"), fmt="%s")
#np.savetxt("Amazon/amazon_data_info.txt", plotting.get_info(df_a, "star_rating"), fmt="%s")
np.savetxt("Amazon/amazon_data_info_1245.txt", plotting.get_info(df_a2, "score_1245"), fmt="%s")

means = []
lens = []
# set_genres = df.columns[-9:]
# for col in df.iloc[:, -9:]:
#     means.append(np.mean(df[df[col]]['score']))
#     lens.append(len(df[df[col]]))

# plotting.show_bar2(set_genres, means, 'Average Pitchfork review scores by genre',
#                    "Pitchfork/pitchfork_scores_average_genre", 0)
# plotting.show_bar2(set_genres, lens, 'Distribution of Pitchfork reviews by genre',
#                    "Pitchfork/pitchfork_scores_dist_genre", 1)
# plotting.show_hist(df['score'], [float(x) for x in sorted(set(list(df['score'])))], 11,
#                    'Distribution of Pitchfork review scores', 'Pitchfork/pitchfork_scores_hist_all', 2)
# plotting.show_bar(df['score_5'], ['1', '2', '3', '4', '5'],
#                   'Distribution of Pitchfork review scores',
#                   "Pitchfork/pitchfork_scores_bar_chart_5", 3)
# plotting.show_bar(df['score_2'], ['<=7.3', '7.3<'], 'Distribution of Pitchfork review scores',
#                   "Pitchfork/pitchfork_scores_bar_chart_2", 4)
# # plotting.show_bar(df['score_3'], ['<=6.5', '6.5< <7.5', '7.5<='],
# #                   'Distribution of Pitchfork review scores: Negative, Neutral, Positive',
# #                   "Pitchfork/pitchfork_scores_bar_chart_3", 5)
#
# plotting.show_bar(df_a['int_star'], ['1', '2', '3', '4', '5'], 'Distribution of Amazon music review scores',
#                   "Amazon/amazon_scores_bar_chart_5", 7)
# plotting.show_bar(df_a['score_2'], ['1,2,3', '4,5'], 'Distribution of Amazon music review scores',
#                   "Amazon/amazon_scores_bar_chart_2", 8)
# plotting.show_bar(df_a2['score_1245'], ['1,2', '4,5'], 'Distribution of Amazon music review scores', "Amazon/amazon_scores_bar_chart_not_3", 10)
#
# plotting.show_bar(df2['score_6575'], ['<=6.5', '7.5<='], 'Distribution of Pitchfork review scores', "Pitchfork/pitchfork_scores_bar_chart_6575", 11)
#
# plotting.plot_review_dist(df_a, "review_body", "Amazon/amazon_review_len_dis", 9, 1000, 'Distribution of Amazon review lengths')
# plotting.plot_review_dist(df2, "review", "Pitchfork/pitchfork_review_len_dist", 6, 10000, 'Distribution of Pitchfork review lengths')

print("visualizing finished")

"""#Vectorization"""
print("vectorizing...")
# vectorizer = TfidfVectorizer(min_df=10, max_df=0.90, ngram_range=(1, 4), stop_words='english',
#                              tokenizer=preprocessing.LemmaTokenizer())
vectorizer2 = TfidfVectorizer(min_df=10, max_df=0.90, ngram_range=(1, 4), stop_words='english',
                             tokenizer=preprocessing.LemmaTokenizer())
# vectorizer_a = TfidfVectorizer(min_df=10, max_df=0.90, ngram_range=(1, 4), stop_words='english',
#                                tokenizer=preprocessing.LemmaTokenizer())
vectorizer_a2 = TfidfVectorizer(min_df=10, max_df=0.90, ngram_range=(1, 4), stop_words='english',
                               tokenizer=preprocessing.LemmaTokenizer())

# review_features = vectorizer.fit_transform(reviews)
review_features2 = vectorizer2.fit_transform(reviews2)
# review_features_a = vectorizer_a.fit_transform(reviews_a)
review_features_a2 = vectorizer_a2.fit_transform(reviews_a2)
print("vectorizing finished")

"""#Classification"""
print("training classifier...")
# X_train2, X_test2, y_train2, y_test2, clf_SGD2 = classification.create_train_perform_SGD(scores_2, 0.3, review_features,
#                                                                                          2,
#                                                                                          "Pitchfork/pitchfork_classification_report_2.txt")
X_train22, X_test22, y_train22, y_test22, clf_SGD22 = classification.create_train_perform_SGD(scores_6575, 0.3, review_features2,
                                                                                         2,
                                                                                         "Pitchfork/pitchfork_opt_classification_report_6575.txt")
# X_train5, X_test5, y_train5, y_test5, clf_SGD5 = classification.create_train_perform_SGD(scores_5, 0.3, review_features,
#                                                                                          5,
#                                                                                          "Pitchfork/pitchfork_classification_report_5")
# X_train2_a, X_test2_a, y_train2_a, y_test2_a, clf_SGD2_a = classification.create_train_perform_SGD(scores_2_a, 0.3,
#                                                                                                    review_features_a, 2,
#                                                                                                    "Amazon/amazon_classification_report_2.txt")
# X_train5_a, X_test5_a, y_train5_a, y_test5_a, clf_SGD5_a = classification.create_train_perform_SGD(scores_5_a, 0.3,
#                                                                                                    review_features_a, 5,
#                                                                                                    "Amazon/amazon_classification_report_5")
X_train2_a2, X_test2_a2, y_train2_a2, y_test2_a2, clf_SGD2_a2 = classification.create_train_perform_SGD(scores_1245, 0.3,
                                                                                                   review_features_a2, 2,
                                                                                                   "Amazon/amazon_opt_classification_report_2_3.txt")
print("training finished")

"""#Evaluation"""
# plotting.create_and_print_confusion_matrix(y_test2, clf_SGD2.predict(X_test2), "Pitchfork review score classifier: Confusion matrix", scores_2,
#                                            "Pitchfork/pitchfork_confusion_matrix_2_all", 12)
plotting.create_and_print_confusion_matrix(y_test22, clf_SGD22.predict(X_test22), "Pitchfork review score classifier: Confusion matrix", scores_6575,
                                           "Pitchfork/pitchfork_confusion_matrix_2", 13)
# plotting.create_and_print_confusion_matrix(y_test5, clf_SGD5.predict(X_test5), "SGDClassifier", scores_5,
#                                            "Pitchfork/pitchfork_confusion_matrix_5", 11)
# plotting.create_and_print_confusion_matrix(y_test2_a, clf_SGD2_a.predict(X_test2_a), "Amazon review score classifier: Confusion matrix", scores_2_a,
#                                            "Amazon/amazon_confusion_matrix_2", 14)
plotting.create_and_print_confusion_matrix(y_test2_a2, clf_SGD2_a2.predict(X_test2_a2), "Amazon review score classifier: Confusion matrix", scores_1245,
                                           "Amazon/amazon_confusion_matrix_2_not3", 15)
# plotting.create_and_print_confusion_matrix(y_test5_a, clf_SGD5_a.predict(X_test5_a), "SGDClassifier", scores_5_a,
#                                            "Amazon/amazon_confusion_matrix_5", 13)

"""#Feature Analysis"""
# classification.most_informative_feature_for_binary_classification(vectorizer, clf_SGD2, 500,
#                                                                   "Pitchfork/pitchfork_informative_features_2.txt")
classification.most_informative_feature_for_binary_classification(vectorizer2, clf_SGD22, 500,
                                                                  "Pitchfork/pitchfork_informative_features_6575.txt")
# classification.most_informative_features_non_binary(vectorizer_a, clf_SGD5_a, 500,
#                                                     "Pitchfork/pitchfork_informative_features_5.txt")
# classification.most_informative_feature_for_binary_classification(vectorizer_a, clf_SGD2_a, 500,
#                                                                   "Amazon/amazon_informative_features_amazon_2.txt")
classification.most_informative_feature_for_binary_classification(vectorizer_a2, clf_SGD2_a2, 500,
                                                                  "Amazon/amazon_informative_features_amazon_not3.txt")
# classification.most_informative_features_non_binary(vectorizer_a, clf_SGD5_a, 500,
#                                                     "Amazon/informative_features_amazon_5.txt")

classification.all_features(vectorizer2, clf_SGD22, "Pitchfork/pitchfork_all_features.txt")
classification.all_features(vectorizer_a2, clf_SGD2_a2, "Amazon/amazon_all_features.txt")

"Cross evaluation"
# new_test = []
# new_y = []
# ca = zip(X_test2_a, y_test2_a, vectorizer_a.get_feature_names_out())
#
# print('ok')
# for index, (ax, ay, af) in enumerate(ca):
#     if index % 1000 == 0:
#         print(index)
#     if af in vectorizer.get_feature_names_out():
#         new_test.append(ax)
#         new_y.append(ay)
# for index, (coef, feat) in enumerate(topn_class):
# print('bruh')
# with open("Pitchfork/predict_amazon_cross_eval", "w") as text_file:
#     text_file.write(classification_report(new_y, clf_SGD2.predict(new_test), digits=4))
# with open("Amazon/predict_pitchfork_cross_eval", "w") as text_file:
#     text_file.write(classification_report(y_test2, clf_SGD2_a.predict(X_test2), digits=4))
