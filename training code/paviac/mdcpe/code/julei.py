# -*- coding: UTF-8 -*-
import random
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# 计算一个样本与数据集中所有样本的欧氏距离的平方
max_iterations = 200
k = 9
varepsilon = 0.00005

def euclidean_distance(one_sample, X):
    one_sample = one_sample.reshape(1, -1)
    X = X.reshape(X.shape[0], -1)
    distances = np.power(np.tile(one_sample, (X.shape[0], 1)) - X, 2).sum(axis=1)
    return distances


def _closest_centroid(sample, centroids):
    distances = euclidean_distance(sample, centroids)
    closest_i = np.argmin(distances)
    return closest_i


def create_clusters(centroids, X):
    n_samples = np.shape(X)[0]
    clusters = [[] for _ in range(k)]
    for sample_i, sample in enumerate(X):
        centroid_i = _closest_centroid(sample, centroids)
        clusters[centroid_i].append(sample_i)
    return clusters


def update_centroids(clusters, X):
    n_features = np.shape(X)[1]
    centroids = np.zeros((k, n_features))
    for i, cluster in enumerate(clusters):
        centroid = np.mean(X[cluster], axis=0)
        centroids[i] = centroid
    return centroids


def get_cluster_labels(clusters, X, true, true_index):
    y_pred = np.zeros(np.shape(X)[0])
    for cluster_i, cluster in enumerate(clusters):
        for sample_i in cluster:
            y_pred[sample_i] = cluster_i
    y_pred = y_pred+1
    correct_pred = np.where(y_pred == true)
    correct_pred = correct_pred[0]
    np.random.shuffle(correct_pred)
    correct_pred = correct_pred.tolist()
    true_index = np.array(true_index)
    final_index = true_index[correct_pred]
    final_index = final_index.tolist()
    return final_index


def predict(X, centroids, true, true_index):

    for _ in range(max_iterations):
        clusters = create_clusters(centroids, X)
        former_centroids = centroids
        centroids = update_centroids(clusters, X)
        diff = centroids - former_centroids
        if diff.any() < varepsilon:
            print("update_end:", _)
            break
        if _ % 5 == 0 and _ != 0:
            print("update_times:", _)
    return get_cluster_labels(clusters, X, true, true_index)



