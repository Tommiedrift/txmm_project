import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

with open("Filtered/A_neg_f.txt") as file:
    A_neg_lines = file.readlines()
    A_neg_lines = [line.rstrip().split() for line in A_neg_lines]

with open("Filtered/A_pos_f.txt") as file:
    A_pos_lines = file.readlines()
    A_pos_lines = [line.rstrip().split() for line in A_pos_lines]

with open("Filtered/P_neg_f.txt") as file:
    P_neg_lines = file.readlines()
    P_neg_lines = [line.rstrip().split() for line in P_neg_lines]

with open("Filtered/P_pos_f.txt") as file:
    P_pos_lines = file.readlines()
    P_pos_lines = [line.rstrip().split() for line in P_pos_lines]

def plot_features(values, labels, values2, labels2, title, save_title, val):
    fig = plt.figure(val, figsize=(4, 12), facecolor='white')
    #fig = plt.figure(val, facecolor='white')
    plt.barh(labels2, values2)

    plt.barh(labels, values)


    plt.title(title, fontsize=10)
    plt.xticks(rotation=0, fontsize = 10)
    #align = 'center'
    plt.xlabel('coefficient', fontsize=10)
    #plt.xlabel('genre', fontsize=10)
    #fig.subplots_adjust(bottom=0.3)
    fig.tight_layout()
    #fig.subplots_adjust(right=-0.2)
    plt.savefig(save_title)

values = [float(x[2]) for x in P_pos_lines[:30]] + [float(x[2]) for x in P_neg_lines[:30]]
val_p = list(reversed([float(x[2]) for x in P_pos_lines[:30]]))
lab_p = list(reversed([x[3] for x in P_pos_lines[:30]]))
val2_p = [float(x[2]) for x in P_neg_lines[:30]]
lab2_p = [x[3] for x in P_neg_lines[:30]]
plot_features(val_p,lab_p,val2_p,lab2_p, "Pitchfork: Informative features", "Filtered/pitchfork_pos_neg_plot_30h", 0)

val_a = list(reversed([float(x[2]) for x in A_pos_lines[:30]]))
lab_a = list(reversed([x[3] for x in A_pos_lines[:30]]))
val2_a = [float(x[2]) for x in A_neg_lines[:30]]
lab2_a = [x[3] for x in A_neg_lines[:30]]
plot_features(val_a,lab_a,val2_a,lab2_a, "Amazon: Informative features", "Filtered/amazon_pos_neg_plot_30h", 1)
