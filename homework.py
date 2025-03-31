import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()
dataset = iris.data

# Часть 1: Метод локтя
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(dataset)
    inertia.append(kmeans.inertia_)

plt.plot(K_range, inertia, marker='x')
plt.title('Метод локтя')
plt.xlabel('Количество кластеров (k)')
plt.ylabel('Inertia')
plt.grid()
plt.show()

optimal_k = 3
# Часть 2
def k_means(dataset, k, max_iters=100, tol=1e-4):
    centroids = dataset[np.random.choice(dataset.shape[0], k, replace=False)]
    
    for iteration in range(max_iters):
        distances = np.linalg.norm(dataset[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        visualize_step(dataset, centroids, labels, iteration)
        
        new_centroids = np.array([dataset[labels == i].mean(axis=0) for i in range(k)])
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        
        centroids = new_centroids
    
    return centroids, labels

def visualize_step(dataset, centroids, labels, iteration):
    plt.figure(figsize=(6, 6))
    colors = ['r', 'g', 'b']
    for i in range(len(centroids)):
        cluster_points = dataset[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i}')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200, label='Centroids')
    plt.title(f'Iteration {iteration + 1}')
    plt.legend()
    plt.show()

centroids, labels = k_means(dataset[:, :2], k=optimal_k)


# Часть 3: Все проекции
def plot_all_projections(dataset, labels, k):
    features = dataset.shape[1]
    fig, axes = plt.subplots(features, features, figsize=(15, 15))
    colors = ['r', 'g', 'b']
    
    for i in range(features): # иду по строкам
        for j in range(features): # иду по столбцам
            ax = axes[i, j]
            if i == j:
                ax.hist(dataset[:, i], bins=20, color='gray', alpha=0.7)
            else:
                for cluster in range(k):
                    cluster_points = dataset[labels == cluster]
                    ax.scatter(cluster_points[:, j], cluster_points[:, i], c=colors[cluster], alpha=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.suptitle('Все проекции данных')
    plt.tight_layout()
    plt.show()

plot_all_projections(dataset, labels, optimal_k)