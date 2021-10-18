from numpy import load
import argparse
import pandas as pd
import numpy as np
import random
import os


df = pd.read_csv('moritz.csv')
df_X = df.drop(columns={'label', 'OriginalTweet',
                        'CleanOriginalTweet'}, axis=1, inplace=False)
N = 1
k = 5
n_iter = 150
X = df_X.to_numpy()



#Initialize k random centroides. Centroids can not be inizialized on the same point.
def inizialize_centroids(X, k):
    centroids = []  # array with positions of centroids
    temp = []
    i = 0
    while i < k:
        # get an integer between 0 and number of instances
        rand = random.randrange(0, X.shape[0]-1)
        if rand in temp:
            continue
        else:
            # an instance has been chosen as a centroid
            print(f"Points chosen for cluster initialization {rand}")
            # add that instance to temp to not repeat the same position as centroid
            temp.append(rand)
            centroids.append(X[rand])  # add that instance to the centroid
            i = i+1
    return centroids


def assign_cluster(X, centroids, k):
    # Create list of points which are appointed to each cluster
    clusters = [[] for _ in range(k)]

    # Loop through all instances and determine the closest cluster
    for i, j in enumerate(X):
        # search for the closest distance to a centroid for each instance
        closest_centroid = np.argmin(
            np.sqrt(np.sum((j - centroids) ** 2, axis=1)))
        # once we find the closest centroid, append the instance to that centroid's cluster
        clusters[closest_centroid].append(i)
    return clusters


# once all clusters are determinated, we must recalculate the centroids.
def determine_new_centroids(k, clusters, X):
    # numpy.zeros creates an array filled with 0s, with the shape of k.
    centroids = np.zeros((k, X.shape[1]))
    #print(clusters)
    for i, j in enumerate(clusters):  # loops through all clusters
        # calculates the mean distance between the instances and centroid of each cluster
        new_centroid = np.mean(X[j], axis=0)
        #print(new_centroid)
        # for each clusters assigns the new centroid
        centroids[i] = new_centroid
    return centroids


def generate_centroids(X, k):
    global clusters
    global centroids
    """ Esta funcion calcula los centroides y devuelve la asignación de cada función"""
    centroids = inizialize_centroids(X, k)
    for i in range(n_iter):
        clusters = assign_cluster(X, centroids, k)
        prev_centroids = centroids
        centroids = determine_new_centroids(k, clusters, X)

        diff = centroids - prev_centroids

        if not diff.any():
            print(f"Termination criterion satisfied in iteration {i}")
            break

            # Get label predictions
    return centroids


def predict_cluster():
    global clusters
    # create  x=(number of instances) arrays full of 0s
    y_pred = np.zeros(X.shape[0])
    for i, j in enumerate(clusters):
        for k in j:
            y_pred[k] = i

    return y_pred


def check_if_centroids_are_defined():
    if(not os.path.exists('./Centroids.npy')):
        print("Generando centroides")
        generate_centroids(X, k)
        print("Centroides generados y calculados de forma satisfactoria!")
    else:
        print("Los centroides estan generados, si deseas generarlos de nuevo, pon el argumento --f en la ejecucion del script ")
    centroids = load('Centroids.npy')

    return centroids.tolist()





if __name__ == '__main__':
    #centroids= check_if_centroids_are_defined()
    generate_centroids(X, k)
    # En este punto tenemos que hacer las predicciones
    print("Las predicciones para las instancias del CSV son las siguientes")
    list = np.array(predict_cluster())
    np.savetxt("Predictions.txt",list)
    print("Predicciones guardadas en el fichero Predictions, esta en formato float, perdon las molestias")
