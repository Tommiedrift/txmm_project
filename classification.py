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


def most_informative_feature_for_binary_classification(vectorizer, classifier, n, title):
    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names_out()
    topn_class0 = sorted(zip(classifier.coef_[0], feature_names))[:n]
    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[-n:]

    text_lines = []
    for index, (coef, feat) in enumerate(reversed(topn_class1)):
        text_lines.append((class_labels[1], str(index), coef, '\t', feat))
        # print(class_labels[1], index, coef, '\t', feat)
    # print('\n')
    for index, (coef, feat) in enumerate(topn_class0):
        # print(class_labels[0], index, coef, '\t', feat)
        text_lines.append((class_labels[0], str(index), coef, '\t', feat))

    text_lines = np.vstack(text_lines)
    np.savetxt(title, text_lines, delimiter=" ", newline="\n", fmt="%s")


def all_features(vectorizer, classifier, title):
    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names_out()
    topn_class0 = sorted(zip(classifier.coef_[0], feature_names))
    text_lines = []
    # print('\n')
    for index, (coef, feat) in enumerate(topn_class0):
        # print(class_labels[0], index, coef, '\t', feat)
        text_lines.append((class_labels[0], str(index), coef, '\t', feat))

    text_lines = np.vstack(text_lines)
    np.savetxt(title, text_lines, delimiter=" ", newline="\n", fmt="%s")

def get_most_relevant_phrases():
    # Convert features into an array
    feature_array = np.array(vectorizer.get_feature_names())

    # Sort features by weight.
    tfidf_sorting = np.argsort(review_features.toarray()).flatten()[::-1]

    # Get the top 100 most weighted features.
    top_n = feature_array[tfidf_sorting][:100]
    return top_n

def create_train_perform_SGD(scores, test_size, review_features, n, save_title):

    X_train, X_test, y_train, y_test = train_test_split(review_features, scores, stratify=scores, random_state=12,
                                                        test_size=test_size)
    #clf_SGD.fit(X_train, y_train)
    #parameters = [{'loss': ['hinge', 'log', 'perceptron'],
    #               'alpha': 10.0 ** -np.arange(1, 7),
    #               'penalty': ['l1', 'l2', 'elasticnet'],
    #               'n_iter_no_change': [2, 3, 4, 5]}]
    #clf_SGD_refined = GridSearchCV(SGDClassifier(random_state=22), parameters)
    #clf_SGD_refined.fit(X_train, y_train)
    #print(save_title)
    #print(clf_SGD_refined.best_params_)
    #{'alpha': 1e-05, 'loss': 'log', 'n_iter_no_change': 4, 'penalty': 'elasticnet'}
    clf_SGD = SGDClassifier(random_state=22, alpha = 1e-05, loss = 'log', n_iter_no_change = 4, penalty='elasticnet')
    clf_SGD.fit(X_train, y_train)
    with open(save_title, "w") as text_file:
        #text_file.write(classification_report(y_test, clf_SGD.predict(X_test), digits=4))
        text_file.write(classification_report(y_test, clf_SGD.predict(X_test), digits=4))
    return X_train, X_test, y_train, y_test, clf_SGD

def most_informative_features_non_binary(vectorizer, classifier, n, title):
    # class_labels = classifier.classes_
    # """Prints features with the highest coefficient values, per class"""
    # feature_names = vectorizer.get_feature_names_out()
    # text_lines = []
    # for i, class_label in enumerate(class_labels):
    #     top10 = np.argsort(classifier.coef_[i])[-n:]
    #     print("%s: %s" % (class_label,
    #                       " ".join(feature_names[j] for j in top10)))
    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names_out()
    text_lines = []
    for i, c in enumerate(class_labels):
        topn_class = sorted(zip(classifier.coef_[0], feature_names))[:n]
        for index, (coef, feat) in enumerate(topn_class):
            text_lines.append((c, str(index), coef, '\t', feat))
    text_lines = np.vstack(text_lines)
    np.savetxt(title, text_lines, delimiter=" ", newline="\n", fmt="%s")