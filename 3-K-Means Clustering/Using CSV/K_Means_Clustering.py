# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 15:12:03 2023

@author: Axa Mehfooz
"""

import pandas as pd
import seaborn as sns 


#STEP1: Read and Understand the data
df = pd.read_csv("D:\\Anum\\Learning ML DL\\K-Means Clustering\\Using CSV\\K-means_clustering_data.csv")
print(df.head())

sns.regplot(x=df['X'], y=df['Y'], fit_reg=False)

#Seeing the data, identify the number of clusters you need
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters= 4, init="k-means++", max_iter = 300 ,n_init = 10, random_state = None, verbose=0)

model = kmeans.fit(df)

predicted_values = kmeans.predict(df)


from matplotlib import pyplot as plt


plt.scatter(df['X'], df['Y'], c = predicted_values, s=50, cmap = 'viridis')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200, c ='black', alpha = 0.5)
