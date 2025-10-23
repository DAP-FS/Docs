Viva Voce Questions
===================

Comprehensive questions for laboratory viva examinations covering all unsupervised learning topics.

General Concepts
----------------

**Q1: What is unsupervised learning? How does it differ from supervised learning?**

**Answer:** Unsupervised learning discovers patterns in data without labeled outputs. Unlike supervised learning (which learns from input-output pairs), unsupervised learning only has input data and must find structure, groupings, or representations on its own.

Key differences:

- No labels/targets in unsupervised learning
- Goal is pattern discovery vs. prediction
- Harder to evaluate (no ground truth)
- Examples: clustering, dimensionality reduction vs. classification, regression

**Q2: What are the main types of unsupervised learning?**

**Answer:**

1. **Clustering:** Grouping similar data points (K-Means, DBSCAN, Hierarchical)
2. **Dimensionality Reduction:** Reducing feature space (PCA, t-SNE, Autoencoders)
3. **Association Rule Mining:** Finding relationships (Apriori, FP-Growth)
4. **Anomaly Detection:** Identifying outliers (Isolation Forest, One-Class SVM)
5. **Density Estimation:** Modeling data distribution (GMM, KDE)

**Q3: Why is unsupervised learning important?**

**Answer:**

- Most real-world data is unlabeled (labeling is expensive)
- Discovers hidden patterns humans might miss
- Preprocessing step for supervised learning
- Exploratory data analysis
- Feature engineering and data compression
- Anomaly detection in security/fraud

Clustering Questions
--------------------

**Q4: Explain the K-Means algorithm step by step.**

**Answer:**

1. **Initialize:** Randomly select K centroids
2. **Assignment Step:** Assign each point to nearest centroid (Euclidean distance)
3. **Update Step:** Recalculate centroids as mean of assigned points
4. **Repeat:** Continue steps 2-3 until convergence (centroids don't move significantly)

Mathematical objective:

.. math::

   \min \sum_{i=1}^{k}\sum_{x \in C_i} ||x - \mu_i||^2

**Q5: What are the limitations of K-Means?**

**Answer:**

1. **Must specify K in advance** - requires domain knowledge or trials
2. **Sensitive to initialization** - different starting points give different results
3. **Assumes spherical clusters** - fails with elongated or irregular shapes
4. **Sensitive to outliers** - outliers pull centroids
5. **Not suitable for categorical data** - requires numerical features
6. **Curse of dimensionality** - performance degrades in high dimensions

**Q6: How do you determine the optimal number of clusters K?**

**Answer:**

**Method 1: Elbow Method**
- Plot WCSS vs K
- Look for "elbow" where decrease slows
- Choose K at elbow point

**Method 2: Silhouette Analysis**
- Calculate silhouette score for each K
- Choose K with highest average silhouette score
- Score range: [-1, 1], higher is better

**Method 3: Gap Statistic**
- Compare WCSS with expected WCSS from random data
- Choose K where gap is maximized

**Method 4: Domain Knowledge**
- Use business/scientific context
- Sometimes K is predetermined

**Q7: What is the difference between K-Means and K-Medoids?**

**Answer:**

.. list-table::
   :header-rows: 1

   * - Aspect
     - K-Means
     - K-Medoids
   * - Centroid
     - Mean of points (may not be actual data point)
     - Actual data point (medoid)
   * - Robustness
     - Sensitive to outliers
     - More robust to outliers
   * - Distance Metric
     - Only Euclidean
     - Any distance metric
   * - Complexity
     - O(nki)
     - O(n²) per iteration
   * - Best for
     - Large datasets, numerical data
     - Mixed data types, outliers present

**Q8: Explain DBSCAN algorithm and its advantages.**

**Answer:**

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**

**Parameters:**
- ε (epsilon): Maximum distance for neighborhood
- MinPts: Minimum points to form dense region

**Point Types:**
1. **Core point:** Has ≥ MinPts neighbors within ε
2. **Border point:** In neighborhood of core point, but < MinPts neighbors
3. **Noise:** Neither core nor border

**Advantages:**

- No need to specify number of clusters
- Handles arbitrary shapes (non-spherical)
- Identifies noise/outliers
- Robust to outliers
- Single pass through data

**Limitations:**

- Sensitive to ε and MinPts
- Struggles with varying densities
- Computationally expensive for large datasets

**Q9: What is hierarchical clustering? Types?**

**Answer:**

Hierarchical clustering creates a tree-like structure (dendrogram) showing nested clusters.

**Types:**

**1. Agglomerative (Bottom-Up):**
- Start: Each point is a cluster
- Iteratively merge closest clusters
- End: All points in one cluster

**2. Divisive (Top-Down):**
- Start: All points in one cluster
- Iteratively split clusters
- End: Each point is its own cluster

**Linkage Criteria:**
- **Single:** :math:`\min d(a,b)`
- **Complete:** :math:`\max d(a,b)`
- **Average:** Mean of all pairwise distances
- **Ward:** Minimizes within-cluster variance

**Q10: What is the Silhouette Score?**

**Answer:**

Silhouette score measures how well each point fits its cluster:

.. math::

   s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}

where:
- :math:`a(i)` = average distance to points in same cluster
- :math:`b(i)` = minimum average distance to points in other clusters

**Interpretation:**
- **1:** Point very well clustered
- **0:** Point on cluster boundary
- **-1:** Point likely in wrong cluster

**Average silhouette score** across all points indicates overall clustering quality.

Dimensionality Reduction Questions
-----------------------------------

**Q11: What is the curse of dimensionality?**

**Answer:**

The curse of dimensionality refers to phenomena that arise when analyzing data in high-dimensional spaces:

**Problems:**

1. **Data sparsity:** Points become increasingly distant
2. **Computational cost:** Exponential increase in complexity
3. **Overfitting:** Models memorize noise
4. **Distance metrics break:** All distances become similar
5. **Visualization impossible:** Can't plot >3 dimensions

**Example:** In high dimensions, nearest and farthest neighbors have similar distances!

**Solutions:**
- Dimensionality reduction (PCA, t-SNE)
- Feature selection
- Regularization

**Q12: Explain Principal Component Analysis (PCA).**

**Answer:**

PCA finds orthogonal directions (principal components) capturing maximum variance.

**Steps:**

1. **Center ** :math:`\bar{X} = X - \mu`
2. **Compute covariance:** :math:`C = \frac{1}{n-1}\bar{X}^T\bar{X}`
3. **Eigenvalue decomposition:** :math:`C = V\Lambda V^T`
4. **Select top-k eigenvectors:** Largest eigenvalues
5. **Project:** :math:`Z = \bar{X}W`

**Key Properties:**

- Linear transformation
- Components are orthogonal
- Unsupervised (no labels needed)
- Explained variance: :math:`\frac{\lambda_i}{\sum \lambda_j}`

**Applications:**
- Data visualization
- Noise reduction
- Feature extraction
- Compression

**Q13: What is the difference between PCA and t-SNE?**

**Answer:**

.. list-table::
   :header-rows: 1

   * - Aspect
     - PCA
     - t-SNE
   * - Type
     - Linear
     - Non-linear
   * - Objective
     - Maximize variance
     - Preserve local structure
   * - Output
     - Any dimension
     - Typically 2-3D (visualization)
   * - Speed
     - Fast O(nd²)
     - Slow O(n²)
   * - Global structure
     - Preserves
     - May distort
   * - Deterministic
     - Yes
     - No (random initialization)
   * - Inverse transform
     - Yes
     - No
   * - Best for
     - General reduction
     - Visualization

**Q14: What are eigenvectors and eigenvalues in PCA context?**

**Answer:**

For covariance matrix :math:`C` and eigenvector :math:`v`:

.. math::

   Cv = \lambda v

**Eigenvector (:math:`v`):**
- Direction of principal component
- Orthogonal to other eigenvectors
- Unit vector

**Eigenvalue (:math:`\lambda`):**
- Variance along that direction
- Larger eigenvalue = more important component
- Sum of all eigenvalues = total variance

**Intuition:** Eigenvectors show directions of maximum spread; eigenvalues show how much spread.

**Q15: Explain autoencoders and their use in dimensionality reduction.**

**Answer:**

Autoencoders are neural networks that learn compressed representations through an encoder-decoder architecture.

**Architecture:**
- **Encoder:** :math:`z = f(Wx + b)` - compresses input
- **Bottleneck:** :math:`z` - low-dimensional representation
- **Decoder:** :math:`\hat{x} = g(W'z + b')` - reconstructs input

**Loss:** Reconstruction error :math:`||x - \hat{x}||^2`

**Advantages:**

- Non-linear transformations
- Learns task-specific features
- Can handle complex patterns
- Flexible architecture

**Types:**

- **Vanilla:** Basic encoder-decoder
- **Denoising:** Learns to remove noise
- **Variational (VAE):** Probabilistic latent space
- **Sparse:** Encourages sparse activations

Association Rules Questions
---------------------------

**Q16: What are association rules? Explain support, confidence, and lift.**

**Answer:**

Association rules find relationships: :math:`X \Rightarrow Y`

**Support:**

.. math::

   \text{support}(X \Rightarrow Y) = \frac{\text{# transactions with } X \cup Y}{\text{total transactions}}

Measures frequency of itemset.

**Confidence:**

.. math::

   \text{confidence}(X \Rightarrow Y) = \frac{\text{support}(X \cup Y)}{\text{support}(X)}

Measures reliability of the rule.

**Lift:**

.. math::

   \text{lift}(X \Rightarrow Y) = \frac{\text{confidence}(X \Rightarrow Y)}{\text{support}(Y)}

Measures strength of association:
- Lift = 1: No association (independent)
- Lift > 1: Positive association
- Lift < 1: Negative association

**Q17: Explain the Apriori algorithm.**

**Answer:**

Apriori finds frequent itemsets using the principle: *"All subsets of a frequent itemset must also be frequent."*

**Algorithm:**

1. **Find frequent 1-itemsets** (scan database)
2. **For k = 2 to max:**
   
   a. **Join:** Generate k-itemsets from (k-1)-itemsets
   b. **Prune:** Remove candidates with infrequent subsets
   c. **Scan:** Count support in database
   d. **Filter:** Keep itemsets with support ≥ min_support

3. **Generate rules** from frequent itemsets

**Example:**

If :math:`\{A, B, C\}` is frequent, then :math:`\{A, B\}`, :math:`\{A, C\}`, :math:`\{B, C\}`, :math:`\{A\}`, :math:`\{B\}`, :math:`\{C\}` must all be frequent.

**Q18: Compare Apriori and FP-Growth.**

**Answer:**

.. list-table::
   :header-rows: 1

   * - Aspect
     - Apriori
     - FP-Growth
   * - Candidate generation
     - Yes (many candidates)
     - No
   * - Database scans
     - Multiple (k+1 for k-itemsets)
     - Two
   * - Data structure
     - None (or hash tree)
     - FP-tree (compact)
   * - Memory
     - Low
     - Higher (tree structure)
   * - Speed
     - Slower (many scans)
     - Faster
   * - Scalability
     - Poor for large datasets
     - Better
   * - Interpretability
     - Easier
     - More complex

Practical Questions
-------------------

**Q19: Your K-Means model gives different results each time. Why?**

**Answer:**

**Reason:** Random initialization of centroids.

**Solutions:**

1. **Set random_state:** Ensures reproducibility
   

