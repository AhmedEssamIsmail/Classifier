import numpy as np
import math as mt


def initiate_env(samples, no_classes):
    no_samples = samples.shape[0]
    centroids = np.asarray(samples[:no_classes, :])
    samples_per_centroids = np.zeros(no_classes)
    mini_distance = np.zeros(no_samples)
    return centroids, samples_per_centroids, mini_distance


def distance(sample, centroid):
    dims = len(sample)
    dist = 0
    for i in range(dims):
        dist += mt.pow(sample[i] - centroid[i], 2)
    dist = mt.sqrt(dist)
    return dist


def add_sample(centroid, sample):
    for i in range(len(sample)):
        centroid[i] += sample[i]
    return centroid


def centroid_division(centroid, num_of_samples):
    for i in range(len(centroid)):
        centroid[i] /= num_of_samples
    return centroid


def run_k_means(samples, k):
    centroids, samples_per_centroids, mini_distance = initiate_env(samples, k)
    old_centroids = centroids
    o = 1
    while o:
        o += 1
        for i in range(samples.shape[0]):
            mini = 9223372999
            mini_centroid = 0
            for j in range(k):
                d = distance(samples[i], centroids[j])
                if d < mini:
                    mini_distance[i] = j
                    mini = d
        centroids = np.zeros([k, samples.shape[1]])  # reconstruct centroids
        samples_per_centroids = np.zeros(len(samples_per_centroids))
        for i in range(len(mini_distance)):
            centroids[int(mini_distance[i])] = add_sample(centroids[int(mini_distance[i])], samples[i])
            samples_per_centroids[int(mini_distance[i])] += 1
        for i in range(len(samples_per_centroids)):
            centroids[i] = centroid_division(centroids[i], samples_per_centroids[i])
        if np.array_equal(centroids, old_centroids) or o > 100:
            break
    return centroids
