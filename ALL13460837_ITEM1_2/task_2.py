import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def calc_dist_euclidean(vec_1, vec_2):
    # Subract one vector from the other and normalise before returning
    dist_euclidean = np.linalg.norm(vec_1 - vec_2, axis=1)
    return dist_euclidean

def init_cent(dataset, k=3):
    # set seed to be 100
    np.random.seed(100)
    # pick k size values from the dataset
    centroids = dataset[np.random.randint(dataset.shape[0], size=k)]
    return centroids

def k_means(dataset, k=3):
    iterations = 20
    #initialise centroids
    centroids = init_cent(dataset, k)
    # create vectors to store the data points and distances
    cluster_assigning = np.zeros(dataset.shape[0], dtype=np.float64)
    distances = np.zeros([dataset.shape[0], k], dtype=np.float64)
    # Declare error variable for later use
    errors = []
    for i in range(iterations):
        # assign all points in dataset to nearest value
        for j, q in enumerate(centroids):
            # Calcualte the distances for each of the classes using the centroid
            distances[:, j] = calc_dist_euclidean(q, dataset)
        
        #check class membership of each point by checking nearest point to it
        cluster_assigning = np.argmin(distances, axis=1)

        for c in range(k):
            # Reassign centroids based on the new class membership
            centroids[c] = np.mean(dataset[cluster_assigning == c], 0)

        # Calculate error for each of the iterations
        error = np.mean([np.mean(pow(i, 2)) for i in distances])
        errors.append(1 / error)    

    return centroids, cluster_assigning, errors

def plot_k_means(dataset, centroids, cluster_assigning):
    # Iterate through colours and assign collours to the different class membership
    group_colors = ['skyblue', 'coral', 'lightgreen']
    colors = [group_colors[j] for j in cluster_assigning]
    fig, ax = plt.subplots(figsize=(4,4))
    # Output as scatter graphs
    ax.scatter(dataset[:,0], dataset[:,4], color=colors, alpha=0.5)
    ax.scatter(centroids[:,0], centroids[:,4], color=['blue', 'darkred', 'green'], marker='o', lw=2)
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    # Show scatter graphs
    plt.show()

def plot_error(errors, iterations):
    # Generate iteration number of x values to fill graph
    x = np.linspace(1, iterations, iterations)
    # Plot the errors
    plt.plot(x, errors)
    plt.show()

# Read in CSV and clean it
dataset = np.array(pd.read_csv('bristol_vacation_rentals_2016.csv', delimiter=',').as_matrix())
dataset[dataset == "Entire home/apt"] = 1
dataset[dataset == "Private room"] = 2
dataset[dataset == "Shared room"] = 3
dataset = dataset.astype(float)
dataset = dataset[:, 3:]
# Call k means function

errors = []
for i in range(1, 21):
    _, _, error = k_means(dataset, k=i)
    errors.append(min(error))

#plot_k_means(dataset, centroids, cluster_assigning)
plot_error(errors, 20)