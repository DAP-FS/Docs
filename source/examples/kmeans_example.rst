K-Means Clustering: Customer Segmentation
==========================================

Problem Statement
-----------------

A retail company wants to segment its customers based on their annual income and spending behavior to create targeted marketing campaigns.

**Goal:** Identify distinct customer groups using K-Means clustering.

Dataset
-------

We'll use a synthetic customer dataset with:

- **Annual Income (k$):** Customer's yearly income in thousands
- **Spending Score (1-100):** Score assigned based on customer behavior and spending nature

.. code-block:: python

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.cluster import KMeans
   from sklearn.preprocessing import StandardScaler
   from sklearn.metrics import silhouette_score, davies_bouldin_score
   
   # Set style
   sns.set_style('whitegrid')
   plt.rcParams['figure.figsize'] = (12, 6)

Generate Sample Data
--------------------

.. code-block:: python

   # Generate synthetic customer data
   np.random.seed(42)
   
   # Group 1: Low income, Low spending
   group1 = np.random.multivariate_normal([30, 30], [[25, 0], [0, 25]], 50)
   
   # Group 2: High income, Low spending (Careful spenders)
   group2 = np.random.multivariate_normal([70, 35], [[25, 0], [0, 25]], 50)
   
   # Group 3: Low income, High spending (Impulsive buyers)
   group3 = np.random.multivariate_normal([35, 75], [[25, 0], [0, 25]], 50)
   
   # Group 4: High income, High spending (Target customers)
   group4 = np.random.multivariate_normal([75, 75], [[25, 0], [0, 25]], 50)
   
   # Group 5: Medium income, Medium spending
   group5 = np.random.multivariate_normal([50, 50], [[20, 0], [0, 20]], 50)
   
   # Combine all groups
   X = np.vstack([group1, group2, group3, group4, group5])
   
   # Create DataFrame
   df = pd.DataFrame(X, columns=['Annual Income (k$)', 'Spending Score'])
   
   print(df.head(10))
   print(f"\nDataset shape: {df.shape}")
   print(f"\nStatistics:\n{df.describe()}")

Exploratory Data Analysis
--------------------------

.. code-block:: python

   # Visualize the data
   plt.figure(figsize=(14, 5))
   
   # Scatter plot
   plt.subplot(1, 3, 1)
   plt.scatter(df['Annual Income (k$)'], df['Spending Score'], 
               alpha=0.6, s=50)
   plt.xlabel('Annual Income (k$)', fontsize=12)
   plt.ylabel('Spending Score', fontsize=12)
   plt.title('Customer Distribution', fontsize=14, fontweight='bold')
   plt.grid(True, alpha=0.3)
   
   # Income distribution
   plt.subplot(1, 3, 2)
   plt.hist(df['Annual Income (k$)'], bins=20, edgecolor='black', alpha=0.7)
   plt.xlabel('Annual Income (k$)', fontsize=12)
   plt.ylabel('Frequency', fontsize=12)
   plt.title('Income Distribution', fontsize=14, fontweight='bold')
   plt.grid(True, alpha=0.3)
   
   # Spending distribution
   plt.subplot(1, 3, 3)
   plt.hist(df['Spending Score'], bins=20, edgecolor='black', alpha=0.7)
   plt.xlabel('Spending Score', fontsize=12)
   plt.ylabel('Frequency', fontsize=12)
   plt.title('Spending Score Distribution', fontsize=14, fontweight='bold')
   plt.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.show()

Finding Optimal K
-----------------

Elbow Method
~~~~~~~~~~~~

.. code-block:: python

   # Calculate WCSS for different K values
   wcss = []
   K_range = range(1, 11)
   
   for k in K_range:
       kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
       kmeans.fit(df)
       wcss.append(kmeans.inertia_)
   
   # Plot elbow curve
   plt.figure(figsize=(10, 6))
   plt.plot(K_range, wcss, 'bo-', linewidth=2, markersize=10)
   plt.xlabel('Number of Clusters (K)', fontsize=13)
   plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=13)
   plt.title('Elbow Method for Optimal K', fontsize=15, fontweight='bold')
   plt.grid(True, alpha=0.3)
   plt.xticks(K_range)
   
   # Highlight the elbow point (K=5 in this case)
   plt.annotate('Elbow Point', xy=(5, wcss[4]), xytext=(6, wcss[4] + 10000),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, color='red', fontweight='bold')
   
   plt.show()
   
   print(f"WCSS values: {wcss}")

Silhouette Analysis
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate silhouette scores
   silhouette_scores = []
   
   for k in range(2, 11):
       kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
       labels = kmeans.fit_predict(df)
       score = silhouette_score(df, labels)
       silhouette_scores.append(score)
   
   # Plot silhouette scores
   plt.figure(figsize=(10, 6))
   plt.plot(range(2, 11), silhouette_scores, 'go-', linewidth=2, markersize=10)
   plt.xlabel('Number of Clusters (K)', fontsize=13)
   plt.ylabel('Silhouette Score', fontsize=13)
   plt.title('Silhouette Analysis', fontsize=15, fontweight='bold')
   plt.grid(True, alpha=0.3)
   plt.xticks(range(2, 11))
   
   # Highlight best K
   best_k = silhouette_scores.index(max(silhouette_scores)) + 2
   plt.annotate(f'Best K={best_k}', 
                xy=(best_k, max(silhouette_scores)), 
                xytext=(best_k + 1, max(silhouette_scores) - 0.05),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, color='red', fontweight='bold')
   
   plt.show()
   
   print(f"\nSilhouette Scores: {silhouette_scores}")
   print(f"Optimal K (by silhouette): {best_k}")

Apply K-Means Clustering
-------------------------

.. code-block:: python

   # Apply K-Means with optimal K
   optimal_k = 5
   kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
   df['Cluster'] = kmeans.fit_predict(df)
   
   # Get cluster centers
   centers = kmeans.cluster_centers_
   
   print(f"Cluster Centers:\n{centers}")
   print(f"\nCluster Sizes:")
   print(df['Cluster'].value_counts().sort_index())

Visualize Results
-----------------

.. code-block:: python

   # Create visualization
   plt.figure(figsize=(14, 6))
   
   # Plot 1: Clustered data
   plt.subplot(1, 2, 1)
   
   colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
   
   for i in range(optimal_k):
       cluster_data = df[df['Cluster'] == i]
       plt.scatter(cluster_data['Annual Income (k$)'], 
                   cluster_data['Spending Score'],
                   c=colors[i], label=f'Cluster {i}', s=50, alpha=0.6)
   
   # Plot centroids
   plt.scatter(centers[:, 0], centers[:, 1], 
               c='black', s=300, alpha=0.8, marker='X', 
               edgecolors='white', linewidths=2, label='Centroids')
   
   plt.xlabel('Annual Income (k$)', fontsize=12)
   plt.ylabel('Spending Score', fontsize=12)
   plt.title('Customer Segments (K-Means Clustering)', fontsize=14, fontweight='bold')
   plt.legend(loc='upper left')
   plt.grid(True, alpha=0.3)
   
   # Plot 2: Cluster characteristics
   plt.subplot(1, 2, 2)
   
   cluster_stats = df.groupby('Cluster').mean()
   x = np.arange(len(cluster_stats))
   width = 0.35
   
   bars1 = plt.bar(x - width/2, cluster_stats['Annual Income (k$)'], 
                    width, label='Avg Income', alpha=0.8)
   bars2 = plt.bar(x + width/2, cluster_stats['Spending Score'], 
                    width, label='Avg Spending', alpha=0.8)
   
   plt.xlabel('Cluster', fontsize=12)
   plt.ylabel('Average Value', fontsize=12)
   plt.title('Cluster Characteristics', fontsize=14, fontweight='bold')
   plt.xticks(x, [f'C{i}' for i in range(optimal_k)])
   plt.legend()
   plt.grid(True, alpha=0.3, axis='y')
   
   plt.tight_layout()
   plt.show()

Cluster Profiling
-----------------

.. code-block:: python

   # Detailed cluster analysis
   print("="*70)
   print("CLUSTER PROFILING")
   print("="*70)
   
   for i in range(optimal_k):
       cluster_data = df[df['Cluster'] == i]
       print(f"\n📊 Cluster {i} (n={len(cluster_data)}):")
       print(f"   Average Income: ${cluster_data['Annual Income (k$)'].mean():.2f}k")
       print(f"   Average Spending: {cluster_data['Spending Score'].mean():.2f}")
       print(f"   Income Range: ${cluster_data['Annual Income (k$)'].min():.1f}k - ${cluster_data['Annual Income (k$)'].max():.1f}k")
       print(f"   Spending Range: {cluster_data['Spending Score'].min():.1f} - {cluster_data['Spending Score'].max():.1f}")

Business Insights
-----------------

.. code-block:: python

   # Generate marketing recommendations
   def get_segment_name(cluster_id, centers):
       income = centers[cluster_id][0]
       spending = centers[cluster_id][1]
       
       if income < 40 and spending < 40:
           return "Budget Conscious"
       elif income > 60 and spending < 40:
           return "High Earners/Low Spenders"
       elif income < 40 and spending > 60:
           return "Impulsive Buyers"
       elif income > 60 and spending > 60:
           return "Premium Customers"
       else:
           return "Standard Customers"
   
   print("\n" + "="*70)
   print("MARKETING RECOMMENDATIONS")
   print("="*70)
   
   for i in range(optimal_k):
       segment_name = get_segment_name(i, centers)
       cluster_size = len(df[df['Cluster'] == i])
       percentage = (cluster_size / len(df)) * 100
       
       print(f"\n🎯 {segment_name} (Cluster {i}) - {percentage:.1f}% of customers")
       
       if segment_name == "Premium Customers":
           print("   Strategy: VIP programs, exclusive offers, personalized service")
       elif segment_name == "Impulsive Buyers":
           print("   Strategy: Flash sales, limited-time offers, emotional marketing")
       elif segment_name == "High Earners/Low Spenders":
           print("   Strategy: Value proposition, quality emphasis, investment focus")
       elif segment_name == "Budget Conscious":
           print("   Strategy: Discounts, loyalty programs, bundle deals")
       else:
           print("   Strategy: Balanced approach, seasonal promotions")

Model Evaluation
----------------

.. code-block:: python

   # Calculate evaluation metrics
   silhouette_avg = silhouette_score(df[['Annual Income (k$)', 'Spending Score']], 
                                     df['Cluster'])
   davies_bouldin = davies_bouldin_score(df[['Annual Income (k$)', 'Spending Score']], 
                                         df['Cluster'])
   
   print("\n" + "="*70)
   print("MODEL EVALUATION METRICS")
   print("="*70)
   print(f"\nSilhouette Score: {silhouette_avg:.3f}")
   print(f"  → Range: [-1, 1], Higher is better")
   print(f"  → Interpretation: {'Excellent' if silhouette_avg > 0.5 else 'Good' if silhouette_avg > 0.3 else 'Fair'}")
   
   print(f"\nDavies-Bouldin Index: {davies_bouldin:.3f}")
   print(f"  → Lower is better")
   print(f"  → Interpretation: {'Excellent' if davies_bouldin < 1 else 'Good' if davies_bouldin < 1.5 else 'Fair'}")
   
   print(f"\nInertia (WCSS): {kmeans.inertia_:.2f}")
   print(f"Number of iterations: {kmeans.n_iter_}")

Try It Yourself
---------------

.. raw:: html

   <div class="try-this">

**Exercise 1:** Modify the code to include a third feature (e.g., age) and perform 3D clustering.

**Exercise 2:** Experiment with different initialization methods (``init='random'`` vs ``init='k-means++'``).

**Exercise 3:** Apply K-Means to a real dataset like the Iris dataset and compare results.

**Exercise 4:** Implement Mini-Batch K-Means for large datasets and compare performance.

**Challenge:** Create an interactive dashboard using Plotly to explore different K values.

.. raw:: html

   </div>

Complete Code
-------------

.. raw:: html

   <div style="margin: 20px 0;">
   <a href="https://colab.research.google.com/github/USERNAME/unsupervised-ml-course/blob/main/notebooks/kmeans_customer_segmentation.ipynb" 
      class="colab-button" target="_blank">
      📓 Open in Google Colab
   </a>
   </div>

Key Takeaways
-------------

1. **K-Means is effective** for spherical, well-separated clusters
2. **Choosing K** requires multiple methods (Elbow, Silhouette)
3. **Feature scaling** is important for distance-based algorithms
4. **Business context** is crucial for interpreting clusters
5. **Evaluation metrics** help validate clustering quality

Next Steps
----------

- :doc:`hierarchical_example` - Learn when hierarchical clustering is better
- :doc:`dbscan_example` - Handle non-spherical clusters
- :doc:`../labs/lab01_kmeans` - Complete the lab assignment
