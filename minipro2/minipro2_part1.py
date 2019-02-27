# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import csv

from functools import reduce 

import math

import matplotlib.pyplot as plt

import numpy as np

import warnings

import scipy.spatial as sp



# QUESTION 1



# groups - groups[i] is name of group i + 1

groups = []

with open('D:算法分析与设计/minipro2/p2_data/groups.csv', 'rt') as csvfile:/

    reader = csv.reader(csvfile)

    for row in reader:

        groups.append(row[0])



# labels - labels[i] has list of all articles (0 indexed instead of 1 indexed) in group i + 1

# labels_reversed[i] gives group id for article i + 1

labels = []

labels_reversed = []

curr_group = 1

with open('D:/算法分析与设计/minipro2/p2_data/label.csv', 'rt') as csvfile:

    reader = csv.reader(csvfile)

    articles = []

    index = 0

    for row in reader:

        group_id = int(row[0])

        if group_id != curr_group:

            labels.append(list(articles))

            articles = []

            curr_group = group_id

        articles.append(index)

        index += 1

        labels_reversed.append(group_id)

    labels.append(list(articles))



# articles - articles[i] has dict of (wordId, wordCount) pairs for articleId i + 1

articles = []

curr_article = 1

max_word_id = -1

with open('D:/算法分析与设计/minipro2/p2_data/data50.csv', 'rt') as csvfile:

    reader = csv.reader(csvfile)
    word_counts = {}

    for row in reader:

        article_id = int(row[0])

        if article_id != curr_article:

            articles.append(dict(word_counts))

            word_counts = {}

            curr_article = article_id

        word_id = int(row[1])

        if word_id > max_word_id:

            max_word_id = word_id

        count = row[2]

        word_counts[word_id] = int(count)

    articles.append(dict(word_counts))





def jaccard(x, y, word_ids):

    numerator = 0

    denominator = 0

    for word in word_ids:

        x_count = x.get(word, 0)

        y_count = y.get(word, 0)

        numerator += min(x_count, y_count)

        denominator += max(x_count,y_count)

    return float(numerator)/denominator



def l2sim(x, y, word_ids):

    similarity = 0

    for word in word_ids:

        x_count = x.get(word, 0)

        y_count = y.get(word, 0)

        similarity += math.pow(x_count - y_count, 2)

    similarity = math.sqrt(similarity)

    return -similarity



def cosine(x, y, word_ids):

    numerator = 0

    x_term = 0

    y_term = 0

    for word in word_ids:

        x_count = x.get(word, 0)

        y_count = y.get(word, 0)

        numerator += x_count * y_count

        x_term += math.pow(x_count, 2)

        y_term += math.pow(y_count, 2)

    x_term = math.sqrt(x_term)

    y_term = math.sqrt(y_term)

    return float(numerator)/(x_term * y_term)



def makeHeatMap(data, names, color, outputFileName):

    #to catch "falling back to Agg" warning

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        #code source: http://stackoverflow.com/questions/14391959/heatmap-in-matplotlib-with-pcolor

        fig, ax = plt.subplots()

        #create the map w/ color bar legend

        heatmap = ax.pcolor(data, cmap=color)

        cbar = plt.colorbar(heatmap)



        # put the major ticks at the middle of each cell

        ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)

        ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)



        # want a more natural, table-like display

        ax.invert_yaxis()

        ax.xaxis.tick_top()



        ax.set_xticklabels(names,rotation=90)

        ax.set_yticklabels(names)



        plt.tight_layout()



        plt.savefig(outputFileName, format = 'png')

        plt.close()



def findSimilarity(func, filename):

    num_groups = len(groups)

    matrix = np.zeros((num_groups, num_groups))

    for i in range(num_groups):

        group1 = labels[i]

        for j in range(num_groups):

            group2 = labels[j]

            num_similarities = 0

            total_similarities = 0

            for article_id1 in group1:

                for article_id2 in group2:

                    x = articles[article_id1]

                    y = articles[article_id2]

                    word_ids = reduce(set.union, map(set, map(dict.keys, [x, y])))

                    total_similarities += func(x, y, word_ids)

                    num_similarities += 1

            matrix[i][j] = float(total_similarities)/num_similarities

    makeHeatMap(matrix, groups, plt.cm.Blues, filename)



print("Jaccard Similarity...")

findSimilarity(jaccard, "jaccard.png")

print("Jaccard Done")

print("L2 Similarity...")

findSimilarity(l2sim, "l2sim.png")

print("L2 Done")

print("Cosine Similarity...")

findSimilarity(cosine, "cosine.png")

print("Cosine Done")



