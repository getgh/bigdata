# For easy to work I have also add the visualization part in the same file.
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

spark = SparkSession.builder \
    .appName("K-Means clustering") \
    .master("local[*]") \
    .getOrCreate()

#convert the data to the libsvm format
def parse_line(line):
    values = line.strip().split()
    try:
        #parsing as space-separated format (x y)
        x = float(values[0])
        y = float(values[1])
    except ValueError:
        print(f"invalid line format: {line.strip()}")
    return Vectors.dense([x, y])

try:
    #read and parse the data
    with open('kmeans_input_raw.txt', 'r') as f:
        data_lines = f.readlines()
    
    #convert the raw data using the parse_line function
    data = [(parse_line(line),) for line in data_lines]
    
    #DataFrame
    columns = ["features"]
    df = spark.createDataFrame(data, columns)
    
except Exception as e:
    print(f"Error in the file: {str(e)}")
    spark.stop()
    exit(1)

#display some the schema and sample data
print("Dataset Schema:")
df.printSchema()
print("\nSample Data - first 5 rows:")
df.show(5)

#count total data points
total_points = df.count()
print(f"\nTotal data points: {total_points}")

#applying K-Means clustering with k=2 here
kmeans = KMeans(k=2, seed=42)
model = kmeans.fit(df)

centers = model.clusterCenters()
print("\nCluster centers:")
for i, center in enumerate(centers):
    print(f"Center of cluster {i+1}: [{center[0]:.4f}, {center[1]:.4f}]")

#predictions
predictions = model.transform(df)
print("\nSample Predictions - first 5 rows:")
predictions.show(5)

#checking the clustering results using Silhouette Score
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print(f"\nSilhouette distance: {silhouette}")

#DataFrame to Pandas for visualization
pandas_df = predictions.toPandas()
features_list = pandas_df['features'].tolist()
x_coords = [float(features[0]) for features in features_list]
y_coords = [float(features[1]) for features in features_list]

pandas_df['x'] = x_coords
pandas_df['y'] = y_coords
center_x = [float(center[0]) for center in centers]
center_y = [float(center[1]) for center in centers]

plt.figure(figsize=(10, 8))

# Plot data colors
cluster0 = pandas_df[pandas_df['prediction'] == 0]
cluster1 = pandas_df[pandas_df['prediction'] == 1]

plt.scatter(cluster0['x'], cluster0['y'], c='red', marker='x', s=50, label='Cluster 1')
plt.scatter(cluster1['x'], cluster1['y'], c='blue', marker='o', s=50, label='Cluster 2')

plt.scatter(center_x[0], center_y[0], c='red', marker='^', s=200, 
            edgecolor='black', linewidth=1.5, label='Center of Cluster 1')
plt.scatter(center_x[1], center_y[1], c='blue', marker='s', s=200, 
            edgecolor='black', linewidth=1.5, label='Center of Cluster 2')

plt.title('K-Means Clustering Results (k=2)')
plt.xlabel('Feature 1 (X coordinate)')
plt.ylabel('Feature 2 (Y coordinate)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

#file save
plt.savefig('kmeans_clustering_results.png')

plt.show()

print("\n=== Summary ===")
print(f"Total data points: {total_points}")
print(f"number of clusters: 2")
print("cluster Centers:")
for i, center in enumerate(centers):
    print(f"Center of Cluster {i+1}: [{center[0]:.4f}, {center[1]:.4f}]")
print(f"Silhouette score: {silhouette:.4f}")

pandas_df[['x', 'y', 'prediction']].to_csv('kmeans_results.csv', index=False)

spark.stop()
