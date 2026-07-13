# Sample Dataset

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Seed for reproducibility
np.random.seed(0)

# Generating a sample dataset with a million customers
num_customers = 500
data = {
    'CustomerID': np.arange(1, num_customers + 1),
    'Annual Income (k$)': np.random.randint(15, 100, num_customers), # Random incomes between 15k and 100k
    'Spending Score (1-100)': np.random.randint(1, 101, num_customers), # Random scores between 1 and 100
    'Age': np.random.randint(18, 70, num_customers), # Random ages between 18 and 70
    'Membership Duration (years)': np.random.randint(1, 21, num_customers) # Random membership duration between 1 and 20 years
}

df_sample = pd.DataFrame(data)
df = df_sample

# Preprocessing
# It's a good practice to scale the features before applying K-means clustering
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['Annual Income (k$)', 'Spending Score (1-100)', 'Age', 'Membership Duration (years)']])

# Applying K-means clustering
# Determine the optimal number of clusters (Elbow method)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Assuming the optimal number of clusters from the Elbow Method, for example, 5
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_features)

# Adding cluster labels to the dataframe
df['Cluster'] = cluster_labels
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
labels = ['Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5']

# Scatter plot of 'Annual Income (k$)' vs 'Spending Score (1-100)' colored by cluster
plt.figure(figsize=(10, 7))
plt.scatter(df['Annual Income (k$)'][df['Cluster'] == 0], df['Spending Score (1-100)'][df['Cluster'] == 0], s=50, c=colors[0], label='Cluster 1')
plt.scatter(df['Annual Income (k$)'][df['Cluster'] == 1], df['Spending Score (1-100)'][df['Cluster'] == 1], s=50, c=colors[1], label='Cluster 2')
plt.scatter(df['Annual Income (k$)'][df['Cluster'] == 2], df['Spending Score (1-100)'][df['Cluster'] == 2], s=50, c=colors[2], label='Cluster 3')
plt.scatter(df['Annual Income (k$)'][df['Cluster'] == 3], df['Spending Score (1-100)'][df['Cluster'] == 3], s=50, c=colors[3], label='Cluster 4')
plt.scatter(df['Annual Income (k$)'][df['Cluster'] == 4], df['Spending Score (1-100)'][df['Cluster'] == 4], s=50, c=colors[4], label='Cluster 5')

# Plotting the centroids
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], s=300, c='yellow', label='Centroids')

plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Count the number of customers in each cluster
cluster_counts = df['Cluster'].value_counts().sort_index()

# Creating the plot
plt.figure(figsize=(10, 6))
plt.bar([labels[i] for i in cluster_counts.index], cluster_counts.values, color=[colors[i] for i in cluster_counts.index])
plt.title('Number of Customers in Each Cluster')
#plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.xticks(cluster_counts.index)
plt.show()