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


def show_hist(column, bins, ticks_range, title, save_title, val):
    plt.figure(val)
    sns.set_style("whitegrid")
    sns.histplot(column, bins=bins)
    plt.title(title, fontsize=20)
    plt.xticks(range(0, ticks_range))
    plt.ylabel('count', fontsize=20)
    plt.xlabel('score', fontsize=20)
    plt.savefig(save_title)
    # plt.ion()
    # fig4.show()
    # plt.pause(0.001)


def show_bar(column, values, title, save_title, n):
    # for i in range(1000):
    #     print(df.iloc[i]['score'], df.iloc[i]['5_score'])
    category_counts = column.value_counts().sort_index()
    plt.figure(n)
    plt.bar(values, category_counts)
    plt.title(title, fontsize=20)
    plt.ylabel('count', fontsize=20)
    plt.xlabel('score', fontsize=20)
    plt.savefig(save_title)
    # plt.ion()
    # fig.show()
    # plt.pause(0.001)


def show_bar2(values, labels, title, save_title, val):
    plt.figure(val, figsize=(12, 18), facecolor='white')
    plt.bar(values, labels)
    plt.title(title, fontsize=40)
    # plt.set_xticklabels(labels, rotation = 45)
    plt.xticks(rotation=45, fontsize = 25)
    plt.ylabel('count', fontsize=40)
    plt.xlabel('feature', fontsize=40)
    plt.savefig(save_title)


#def plot_confusion_matrix(val, lenscores, cm, title, cmap=plt.cm.Blues):
    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title(title, fontsize=25)
    # plt.colorbar()
    # tick_marks = np.arange(len(set(lenscores)))
    # plt.xticks(tick_marks, set(lenscores), rotation=45)
    # plt.yticks(tick_marks, set(lenscores))
    # plt.ylabel('True label', fontsize=25)
    # plt.xlabel('Predicted label', fontsize=25)

def create_and_print_confusion_matrix(y_test, predicted, title, lenscores, save_title, val):
    plt.figure(val, figsize=(9, 7), facecolor='white')
    cm = confusion_matrix(y_test, predicted)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=25)
    plt.colorbar()
    tick_marks = np.arange(len(set(lenscores)))
    plt.xticks(tick_marks, set(lenscores), rotation=45)
    plt.yticks(tick_marks, set(lenscores))
    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Predicted label', fontsize=25)
    #plot_confusion_matrix(val, lenscores, cm, title)
    plt.savefig(save_title)
    # plt.ion()
    # fig4.show()
    # plt.pause(0.001)


def plot_review_dist(df, review_l, save_title, val, filter, title):
    plt.figure(val)
    review_len_dist = pd.DataFrame(df[review_l].str.len())
    review_len_dist = review_len_dist[review_len_dist[review_l] < filter]
    review_len_dist.groupby([review_l])
    review_len_dist_plot = review_len_dist.plot(kind='hist', legend=None, bins=50, figsize=(12, 6))
    #review_len_dist_plot.title(title)
    plt.title(title, fontsize = 20)
    review_len_dist_plot.set_xlabel("Review Length", fontsize = 20)
    review_len_dist_plot.set_ylabel("Count", fontsize = 20)
    plt.savefig(save_title)


def get_info(df, score):
    text_lines = ["Average score: {}".format(np.mean(df[score])), "Median score: {}".format(np.median(df[score])),
                  "1/3rd Percentile: {}".format(np.percentile(df[score], 100 / 3)),
                  "2/3rd Percentile: {}".format(np.percentile(df[score], 100 / 3 * 2)),
                  "Total review: {}".format(len(df)), "Value counts: {}".format(df[score].value_counts().sort_index())]
    text_lines = np.vstack(text_lines)
    return text_lines
