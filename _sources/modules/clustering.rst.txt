Clustering Algorithms
=====================

Clustering is the task of grouping a set of objects such that objects in the same group (cluster) are more similar to each other than to those in other groups.

Introduction
------------

**Definition:** Clustering is an unsupervised learning technique that partitions data into groups (clusters) based on similarity.

**Applications:**

- Customer segmentation
- Image segmentation
- Document organization
- Anomaly detection
- Gene sequence analysis

Types of Clustering
-------------------

.. graphviz::

   digraph clustering_types {
       node [shape=box, style="rounded,filled", fillcolor=lightblue];
       
       clustering [label="Clustering\nAlgorithms", fillcolor=lightgreen];
       
       partition [label="Partitioning\n(K-Means, K-Medoids)"];
       hierarchical [label="Hierarchical\n(Agglomerative, Divisive)"];
       density [label="Density-Based\n(DBSCAN, OPTICS)"];
       distribution [label="Distribution-Based\n(GMM, EM)"];
       
       clustering -> partition;
       clustering -> hierarchical;
       clustering -> density;
       clustering -> distribution;
   }

K-Means Clustering
------------------

Concept
~~~~~~~

K-Means is a partitioning algorithm that divides :math:`n` observations into :math:`k` clusters by minimizing within-cluster variance.

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

**Objective Function:**

.. math::

   J = \sum_{i=1}^{k}\sum_{x \in C_i} ||x - \mu_i||^2

where:

- :math:`k` = number of clusters
- :math:`C_i` = set of points in cluster :math:`i`
- :math:`\mu_i = \frac{1}{|C_i|}\sum_{x \in C_i} x` = centroid of cluster :math:`i`
- :math:`||x - \mu_i||^2` = squared Euclidean distance

Algorithm
~~~~~~~~~

.. code-block:: text

   Algorithm: K-Means Clustering
   
   Input: Dataset X = {x₁, x₂, ..., xₙ}, number of clusters k
   Output: Cluster assignments and centroids
   
   1. Initialize k centroids μ₁, μ₂, ..., μₖ randomly
   2. Repeat until convergence:
      a. Assignment Step:
         For each point xᵢ:
            Assign xᵢ to cluster j where j = argmin ||xᵢ - μⱼ||²
      
      b. Update Step:
         For each cluster Cⱼ:
            μⱼ = (1/|Cⱼ|) Σ(x∈Cⱼ) x
   
   3. Return cluster assignments and final centroids

Implementation
~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from sklearn.cluster import KMeans
   from sklearn.datasets import make_blobs
   import matplotlib.pyplot as plt
   import seaborn as sns

   # Generate synthetic data
   X, y_true = make_blobs(
       n_samples=300,
       centers=4,
       cluster_std=0.60,
       random_state=42
   )

   # Apply K-Means
   kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
   y_kmeans = kmeans.fit_predict(X)

   # Visualization
   plt.figure(figsize=(10, 6))
   plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis', alpha=0.6)
   
   # Plot centroids
   centers = kmeans.cluster_centers_
   plt.scatter(centers[:, 0], centers[:, 1], 
               c='red', s=200, alpha=0.8, 
               marker='X', edgecolors='black', linewidths=2)
   
   plt.xlabel('Feature 1')
   plt.ylabel('Feature 2')
   plt.title('K-Means Clustering (k=4)')
   plt.colorbar(label='Cluster')
   plt.grid(True, alpha=0.3)
   plt.show()

   # Cluster statistics
   print(f"Inertia (Within-cluster sum of squares): {kmeans.inertia_:.2f}")
   print(f"Number of iterations: {kmeans.n_iter_}")

Choosing K: Elbow Method
~~~~~~~~~~~~~~~~~~~~~~~~~

The **Elbow Method** helps determine the optimal number of clusters by plotting the within-cluster sum of squares (WCSS) against :math:`k`.

.. code-block:: python

   # Calculate WCSS for different k values
   wcss = []
   K_range = range(1, 11)

   for k in K_range:
       kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
       kmeans.fit(X)
       wcss.append(kmeans.inertia_)

   # Plot elbow curve
   plt.figure(figsize=(10, 6))
   plt.plot(K_range, wcss, 'bo-', linewidth=2, markersize=8)
   plt.xlabel('Number of Clusters (k)', fontsize=12)
   plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
   plt.title('Elbow Method for Optimal k', fontsize=14)
   plt.grid(True, alpha=0.3)
   plt.xticks(K_range)
   plt.show()

**Interpretation:** Look for the "elbow" point where the rate of decrease sharply changes. This indicates the optimal :math:`k`.

Advantages and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Advantages
     - Limitations
   * - Simple and easy to implement
     - Requires pre-specifying :math:`k`
   * - Scales well to large datasets
     - Sensitive to initial centroid placement
   * - Guaranteed to converge
     - Assumes spherical clusters
   * - Fast (linear time complexity)
     - Sensitive to outliers
   * - Works well with compact clusters
     - Cannot handle non-convex shapes

Hierarchical Clustering
------------------------

Concept
~~~~~~~

Hierarchical clustering creates a tree-like structure (dendrogram) of nested clusters without requiring a predefined number of clusters.

Types
~~~~~

**1. Agglomerative (Bottom-Up):**

- Start with each point as a cluster
- Iteratively merge closest clusters
- Continue until one cluster remains

**2. Divisive (Top-Down):**

- Start with all points in one cluster
- Iteratively split clusters
- Continue until each point is its own cluster

Linkage Criteria
~~~~~~~~~~~~~~~~

The linkage criterion determines how distances between clusters are computed:

.. math::

   d(C_i, C_j) = \text{linkage}(\{x : x \in C_i\}, \{y : y \in C_j\})

**Common Linkage Methods:**

1. **Single Linkage (Minimum):**

   .. math::

      d(C_i, C_j) = \min_{x \in C_i, y \in C_j} d(x, y)

2. **Complete Linkage (Maximum):**

   .. math::

      d(C_i, C_j) = \max_{x \in C_i, y \in C_j} d(x, y)

3. **Average Linkage:**

   .. math::

      d(C_i, C_j) = \frac{1}{|C_i| \cdot |C_j|} \sum_{x \in C_i}\sum_{y \in C_j} d(x, y)

4. **Ward's Method:** Minimizes within-cluster variance

   .. math::

      d(C_i, C_j) = \frac{|C_i| \cdot |C_j|}{|C_i| + |C_j|} ||\mu_i - \mu_j||^2

Implementation
~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.cluster import AgglomerativeClustering
   from scipy.cluster.hierarchy import dendrogram, linkage
   import matplotlib.pyplot as plt

   # Generate data
   X, y = make_blobs(n_samples=150, centers=3, random_state=42)

   # Perform hierarchical clustering
   hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
   labels = hierarchical.fit_predict(X)

   # Create dendrogram
   plt.figure(figsize=(12, 6))
   
   # Compute linkage matrix
   Z = linkage(X, method='ward')
   
   # Plot dendrogram
   dendrogram(Z, truncate_mode='lastp', p=20)
   plt.xlabel('Sample Index or (Cluster Size)', fontsize=12)
   plt.ylabel('Distance', fontsize=12)
   plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)', fontsize=14)
   plt.axhline(y=10, c='red', linestyle='--', label='Cut-off')
   plt.legend()
   plt.show()

   # Visualize clusters
   plt.figure(figsize=(10, 6))
   plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
   plt.xlabel('Feature 1')
   plt.ylabel('Feature 2')
   plt.title('Hierarchical Clustering Result')
   plt.colorbar(label='Cluster')
   plt.show()

DBSCAN (Density-Based Clustering)
----------------------------------

Concept
~~~~~~~

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) groups together points that are closely packed and marks points in low-density regions as outliers.

Key Parameters
~~~~~~~~~~~~~~

1. **ε (epsilon):** Maximum distance between two samples to be considered neighbors
2. **MinPts:** Minimum number of points required to form a dense region

Point Classification
~~~~~~~~~~~~~~~~~~~~

- **Core Point:** Has at least MinPts neighbors within ε distance
- **Border Point:** Has fewer than MinPts neighbors but is within ε of a core point
- **Noise/Outlier:** Neither core nor border point

Mathematical Definition
~~~~~~~~~~~~~~~~~~~~~~~

**ε-neighborhood of point :math:`x`:**

.. math::

   N_\epsilon(x) = \{y \in X : d(x, y) \leq \epsilon\}

**Core point condition:**

.. math::

   |N_\epsilon(x)| \geq \text{MinPts}

**Density-reachable:** Point :math:`q` is density-reachable from :math:`p` if there exists a chain of core points connecting them.

Algorithm
~~~~~~~~~

.. code-block:: text

   Algorithm: DBSCAN
   
   Input: Dataset X, ε, MinPts
   Output: Cluster assignments
   
   1. Mark all points as unvisited
   2. For each unvisited point p:
      a. Mark p as visited
      b. Find all neighbors N within ε distance
      c. If |N| < MinPts:
            Mark p as noise
      d. Else:
            Create new cluster C
            Add p to C
            For each point q in N:
               If q is unvisited:
                  Mark q as visited
                  Find neighbors of q
                  If |neighbors| ≥ MinPts:
                     Add neighbors to N
               If q not in any cluster:
                  Add q to C
   
   3. Return cluster assignments

Implementation
~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.cluster import DBSCAN
   from sklearn.preprocessing import StandardScaler
   import numpy as np
   import matplotlib.pyplot as plt

   # Generate data with noise
   from sklearn.datasets import make_moons
   X, y = make_moons(n_samples=300, noise=0.05, random_state=42)

   # Standardize features
   X_scaled = StandardScaler().fit_transform(X)

   # Apply DBSCAN
   dbscan = DBSCAN(eps=0.3, min_samples=5)
   labels = dbscan.fit_predict(X_scaled)

   # Count clusters (excluding noise)
   n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
   n_noise = list(labels).count(-1)

   print(f"Number of clusters: {n_clusters}")
   print(f"Number of noise points: {n_noise}")

   # Visualization
   plt.figure(figsize=(12, 5))

   # Original data
   plt.subplot(1, 2, 1)
   plt.scatter(X[:, 0], X[:, 1], s=50)
   plt.title('Original Data')
   plt.xlabel('Feature 1')
   plt.ylabel('Feature 2')

   # DBSCAN result
   plt.subplot(1, 2, 2)
   
   # Plot clusters
   unique_labels = set(labels)
   colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
   
   for label, color in zip(unique_labels, colors):
       if label == -1:
           # Noise points in black
           color = 'black'
           marker = 'x'
       else:
           marker = 'o'
       
       class_member_mask = (labels == label)
       xy = X[class_member_mask]
       plt.scatter(xy[:, 0], xy[:, 1], c=[color], 
                   s=50, marker=marker, alpha=0.6,
                   label=f'Cluster {label}' if label != -1 else 'Noise')
   
   plt.title(f'DBSCAN Clustering (ε={dbscan.eps}, MinPts={dbscan.min_samples})')
   plt.xlabel('Feature 1')
   plt.ylabel('Feature 2')
   plt.legend()
   plt.tight_layout()
   plt.show()

Advantages and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Advantages
     - Limitations
   * - No need to specify number of clusters
     - Sensitive to ε and MinPts parameters
   * - Handles arbitrary shapes
     - Struggles with varying densities
   * - Robust to outliers
     - Computationally expensive for large datasets
   * - Identifies noise points
     - Curse of dimensionality in high dimensions

Gaussian Mixture Models (GMM)
------------------------------

Concept
~~~~~~~

GMM assumes data is generated from a mixture of several Gaussian distributions with unknown parameters.

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

**Probability density function:**

.. math::

   p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)

where:

- :math:`K` = number of Gaussian components
- :math:`\pi_k` = mixing coefficient for component :math:`k` (:math:`\sum_{k=1}^{K} \pi_k = 1`)
- :math:`\mathcal{N}(x | \mu_k, \Sigma_k)` = multivariate Gaussian distribution
- :math:`\mu_k` = mean vector of component :math:`k`
- :math:`\Sigma_k` = covariance matrix of component :math:`k`

**Multivariate Gaussian:**

.. math::

   \mathcal{N}(x | \mu, \Sigma) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)

EM Algorithm
~~~~~~~~~~~~

GMM parameters are estimated using the **Expectation-Maximization (EM)** algorithm:

**E-Step (Expectation):** Compute responsibilities

.. math::

   \gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)}

**M-Step (Maximization):** Update parameters

.. math::

   \begin{aligned}
   N_k &= \sum_{i=1}^{n} \gamma_{ik} \\
   \mu_k &= \frac{1}{N_k}\sum_{i=1}^{n} \gamma_{ik} x_i \\
   \Sigma_k &= \frac{1}{N_k}\sum_{i=1}^{n} \gamma_{ik}(x_i - \mu_k)(x_i - \mu_k)^T \\
   \pi_k &= \frac{N_k}{n}
   \end{aligned}

Implementation
~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.mixture import GaussianMixture
   import numpy as np
   import matplotlib.pyplot as plt
   from matplotlib.patches import Ellipse

   # Generate data
   X, y = make_blobs(n_samples=300, centers=3, random_state=42)

   # Fit GMM
   gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
   gmm.fit(X)
   labels = gmm.predict(X)

   # Get probabilities
   probs = gmm.predict_proba(X)

   # Visualization
   def draw_ellipse(position, covariance, ax=None, **kwargs):
       """Draw an ellipse representing a 2D Gaussian distribution"""
       ax = ax or plt.gca()
       
       # Convert to principal axes
       if covariance.shape == (2, 2):
           U, s, Vt = np.linalg.svd(covariance)
           angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
           width, height = 2 * np.sqrt(s)
       else:
           angle = 0
           width, height = 2 * np.sqrt(covariance)
       
       # Draw ellipse
       for nsig in range(1, 4):
           ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                                angle=angle, **kwargs))

   plt.figure(figsize=(10, 6))
   plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis', alpha=0.6)

   # Draw ellipses for each component
   for i, (mean, covar) in enumerate(zip(gmm.means_, gmm.covariances_)):
       draw_ellipse(mean, covar, alpha=0.2, edgecolor='red', facecolor='none', linewidth=2)
       plt.scatter(mean[0], mean[1], c='red', s=200, marker='X', edgecolors='black')

   plt.xlabel('Feature 1')
   plt.ylabel('Feature 2')
   plt.title('Gaussian Mixture Model Clustering')
   plt.colorbar(label='Cluster')
   plt.show()

   # Log-likelihood
   print(f"Log-likelihood: {gmm.score(X):.2f}")
   print(f"BIC: {gmm.bic(X):.2f}")
   print(f"AIC: {gmm.aic(X):.2f}")

Comparison of Clustering Algorithms
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 20 25

   * - Algorithm
     - Time Complexity
     - Cluster Shape
     - Handles Outliers
     - Best Use Case
   * - K-Means
     - O(nki)
     - Spherical
     - No
     - Large datasets, spherical clusters
   * - Hierarchical
     - O(n²logn)
     - Any
     - No
     - Small datasets, hierarchical structure
   * - DBSCAN
     - O(n logn)
     - Arbitrary
     - Yes
     - Non-convex shapes, noise present
   * - GMM
     - O(nk²d²)
     - Elliptical
     - Moderate
     - Probabilistic clustering, overlapping clusters

Evaluation Metrics
------------------

Silhouette Score
~~~~~~~~~~~~~~~~

Measures how similar an object is to its own cluster compared to other clusters:

.. math::

   s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}

where:

- :math:`a(i)` = average distance to points in same cluster
- :math:`b(i)` = minimum average distance to points in other clusters
- Range: [-1, 1], higher is better

.. code-block:: python

   from sklearn.metrics import silhouette_score, davies_bouldin_score

   # Calculate metrics
   silhouette = silhouette_score(X, labels)
   davies_bouldin = davies_bouldin_score(X, labels)

   print(f"Silhouette Score: {silhouette:.3f}")
   print(f"Davies-Bouldin Index: {davies_bouldin:.3f}")

Davies-Bouldin Index
~~~~~~~~~~~~~~~~~~~~

Measures average similarity between each cluster and its most similar cluster:

.. math::

   DB = \frac{1}{k}\sum_{i=1}^{k}\max_{j \neq i}\left(\frac{s_i + s_j}{d(c_i, c_j)}\right)

where:

- :math:`s_i` = average distance of points in cluster :math:`i` to centroid
- :math:`d(c_i, c_j)` = distance between centroids
- Lower is better

Try This
--------

.. raw:: html

   <div class="try-this">

**Exercise 1:** Implement K-Means from scratch without using scikit-learn.

**Exercise 2:** Compare K-Means, DBSCAN, and GMM on the Iris dataset.

**Exercise 3:** Use the Elbow method and Silhouette analysis to find optimal :math:`k` for a given dataset.

.. raw:: html

   </div>

Summary
-------

- **Clustering** groups similar data points without labels
- **K-Means** is fast but requires specifying :math:`k` and assumes spherical clusters
- **Hierarchical clustering** creates a tree structure without pre-specifying :math:`k`
- **DBSCAN** handles arbitrary shapes and identifies outliers
- **GMM** provides probabilistic cluster assignments
- Choose the algorithm based on data characteristics and requirements

Further Reading
---------------

- MacKay, D. J. (2003). *Information Theory, Inference and Learning Algorithms*. Chapter 20-22.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Chapter 9.
- Xu, R., & Wunsch, D. (2005). *Survey of clustering algorithms*. IEEE Transactions on neural networks.

