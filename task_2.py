import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def calc_dist_euclidean(vec_1, vec_2):
    dist_euclidean = np.linalg.norm(vec_1 - vec_2, axis=1)
    return dist_euclidean

def init_cent(dataset, k=3):
    # set seed to be 100
    np.random.seed(100)

    centroids = dataset[np.random.randint(dataset.shape[0], size=k)]
    return centroids

def k_means(dataset, k=3):
    iterations = 20
    #initiallise centroids
    centroids = init_cent(dataset, k)
    # create v ectors to store the data points and distances
    cluster_assigning = np.zeros(dataset.shape[0], dtype=np.float64)
    distances = np.zeros([dataset.shape[0], k], dtype=np.float64)
    errors = []
    for i in range(iterations):

        # assign all points in dataset to nearest value
        for j, q in enumerate(centroids):
            distances[:, j] = calc_dist_euclidean(q, dataset)
        
        #check class membership of each point by checking nearest point to it
        cluster_assigning = np.argmin(distances, axis=1)

        for c in range(k):
            centroids[c] = np.mean(dataset[cluster_assigning == c], 0)

        error = np.mean([np.mean(pow(i, 2)) for i in distances])
        errors.append(1 / error)    
        # Loop through cluster assigning and distances 
        # if cluster assigning == 1
        # calc_dist_euclidean with 

    return centroids, cluster_assigning, errors

def plot_k_means(dataset, centroids, cluster_assigning):

    group_colors = ['skyblue', 'coral', 'lightgreen']
    colors = [group_colors[j] for j in cluster_assigning]
    fig, ax = plt.subplots(figsize=(4,4))
    ax.scatter(dataset[:,0], dataset[:,4], color=colors, alpha=0.5)
    ax.scatter(centroids[:,0], centroids[:,4], color=['blue', 'darkred', 'green'], marker='o', lw=2)
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')

    plt.show()

def plot_error(errors, iterations):
    
    x = np.linspace(0, iterations, iterations)
    plt.plot(x, errors)
    plt.show()

dataset = np.array(pd.read_csv('bristol_vacation_rentals_2016.csv', delimiter=',').as_matrix())
dataset[dataset == "Entire home/apt"] = 1
dataset[dataset == "Private room"] = 2
dataset[dataset == "Shared room"] = 3
dataset = dataset.astype(float)
dataset = dataset[:, 3:]
centroids, cluster_assigning, errors = k_means(dataset)

#plot_k_means(dataset, centroids, cluster_assigning)
plot_error(errors, 20)