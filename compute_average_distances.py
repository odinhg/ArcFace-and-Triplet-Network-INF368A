import numpy as np
import pandas as pd
from os.path import isfile
from tqdm import tqdm
from configfile import *
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from utilities import compute_average_distances, save_distance_figure

if __name__ == "__main__":
    if not (isfile("embeddings_train.pkl") and isfile("embeddings_test.pkl") and isfile("embeddings_unseen.pkl")):
        exit("Embeddings not found. Please evaluate model first!")
    #Task 4
    df = pd.read_pickle("embeddings_test.pkl")
    classes = [df[df["label_idx"] == i].iloc[:,2:] for i in class_idx]
    avg_euclidean_distances, avg_angular_distances = compute_average_distances(classes)
    print("Average euclidean distances (test dataset):")
    print(avg_euclidean_distances)
    save_distance_figure(avg_euclidean_distances, class_names, "average_euclidean_distances_test.png")
    print("Average angular distances (test dataset):")
    print(avg_angular_distances)
    save_distance_figure(avg_angular_distances, class_names, "average_angular_distances_test.png")
    
    #Task 5
    df = pd.read_pickle("embeddings_unseen.pkl")
    classes = [df[df["label_idx"] == i].iloc[:,2:] for i in class_idx_unseen]
    avg_euclidean_distances, avg_angular_distances = compute_average_distances(classes)
    print("Average euclidean distances (unseen classes):")
    print(avg_euclidean_distances)
    save_distance_figure(avg_euclidean_distances, class_names_unseen, "average_euclidean_distances_unseen.png")
    print("Average angular distances (unseen classes):")
    print(avg_angular_distances)
    save_distance_figure(avg_angular_distances, class_names_unseen, "average_angular_distances_unseen.png")
