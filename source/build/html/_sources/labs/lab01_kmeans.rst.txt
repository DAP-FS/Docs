Lab 1: K-Means Clustering for Customer Segmentation
====================================================

**Duration:** 2 weeks | **Weightage:** 15% | **Deadline:** [Check LMS]

Objective
---------

Implement and analyze K-Means clustering to segment customers based on purchasing behavior and demographics. Apply the elbow method and silhouette analysis to determine optimal cluster count.

Learning Outcomes
-----------------

By completing this lab, you will be able to:

1. Preprocess and explore real-world customer data
2. Implement K-Means clustering from scratch and using scikit-learn
3. Determine optimal number of clusters using multiple methods
4. Evaluate clustering quality with appropriate metrics
5. Generate actionable business insights from clusters
6. Create professional visualizations

Problem Statement
-----------------

You are a data scientist at an e-commerce company. The marketing team wants to create personalized campaigns for different customer segments. Your task is to:

1. Analyze customer purchase history and demographics
2. Identify distinct customer segments using clustering
3. Profile each segment with descriptive statistics
4. Provide marketing recommendations for each segment

Dataset
-------

**Source:** `Mall Customer Segmentation Data <https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python>`_

**Download:** Available on course LMS or GitHub repository

**Description:**

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Column
     - Type
     - Description
   * - CustomerID
     - int
     - Unique customer identifier
   * - Gender
     - str
     - Customer gender (Male/Female)
   * - Age
     - int
     - Customer age (18-70)
   * - Annual Income (k$)
     - int
     - Annual income in thousands of dollars
   * - Spending Score (1-100)
     - int
     - Score assigned based on customer behavior and spending

**Size:** 200 samples, 5 features

Tasks
-----

Part A: Data Exploration (10 points)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Load the dataset and display first 10 rows
2. Check for missing values and data types
3. Generate descriptive statistics (mean, std, min, max)
4. Create visualizations:
   
   - Histogram of Age distribution
   - Box plot of Annual Income by Gender
   - Scatter plot of Income vs Spending Score
   - Correlation heatmap

5. Write observations about data distribution

Part B: Data Preprocessing (10 points)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Handle missing values (if any)
2. Encode categorical variables (Gender)
3. Select features for clustering (justify your choice)
4. Standardize/normalize features using StandardScaler
5. Explain why scaling is necessary for K-Means

Part C: Implement K-Means from Scratch (20 points)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement the following functions:

.. code-block:: python

   def initialize_centroids(X, k, random_state=42):
       """
       Randomly initialize k centroids from data points.
       
       Parameters
       ----------
       X : ndarray, shape (n_samples, n_features)
       k : int, number of clusters
       random_state : int
       
       Returns
       -------
       centroids : ndarray, shape (k, n_features)
       """
       # Your code here
       pass
   
   def assign_clusters(X, centroids):
       """
       Assign each point to nearest centroid.
       
       Parameters
       ----------
       X : ndarray, shape (n_samples, n_features)
       centroids : ndarray, shape (k, n_features)
       
       Returns
       -------
       labels : ndarray, shape (n_samples,)
       """
       # Your code here
       pass
   
   def update_centroids(X, labels, k):
       """
       Recompute centroids as mean of assigned points.
       
       Parameters
       ----------
       X : ndarray, shape (n_samples, n_features)
       labels : ndarray, shape (n_samples,)
       k : int
       
       Returns
       -------
       centroids : ndarray, shape (k, n_features)
       """
       # Your code here
       pass
   
   def kmeans_from_scratch(X, k, max_iters=100, random_state=42):
       """
       Full K-Means implementation.
       
       Returns
       -------
       centroids : final centroids
       labels : cluster assignments
       inertia : WCSS value
       """
       # Your code here
       pass

**Requirements:**

- Use Euclidean distance
- Implement convergence check (centroids stop moving)
- Return final centroids, labels, and WCSS

Part D: Determine Optimal K (20 points)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Method 1: Elbow Method**

1. Compute WCSS for K = 1 to 10
2. Plot WCSS vs K
3. Identify elbow point
4. Annotate the elbow on the plot

**Method 2: Silhouette Analysis**

1. Calculate silhouette score for K = 2 to 10
2. Plot silhouette scores
3. Identify K with highest score
4. Create silhouette plots for top 3 K values

**Method 3: Gap Statistic (Bonus)**

Implement gap statistic calculation and determine optimal K.

Part E: Apply Scikit-Learn K-Means (15 points)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Use `KMeans` from scikit-learn with optimal K
2. Compare results with your implementation
3. Experiment with different initialization methods:
   
   - `init='random'`
   - `init='k-means++'`

4. Use different `n_init` values (1, 10, 50)
5. Report differences in convergence and final WCSS

Part F: Cluster Analysis (15 points)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the optimal K:

1. **Cluster Profiling:**
   
   - Calculate mean Age, Income, Spending for each cluster
   - Count number of customers per cluster
   - Determine gender distribution per cluster

2. **Visualization:**
   
   - Create scatter plot with colored clusters
   - Plot centroids as stars/crosses
   - Add cluster size annotations
   - Create radar chart for cluster characteristics

3. **Statistical Tests:**
   
   - Perform ANOVA to test if clusters differ significantly
   - Report F-statistic and p-value

Part G: Business Insights (10 points)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Write a 1-page report answering:

1. **Segment Names:** Give descriptive names to each cluster
2. **Characteristics:** Describe typical customer in each segment
3. **Marketing Strategy:** Recommend specific strategies for each segment
4. **ROI Potential:** Which segments offer highest value?
5. **Actionable Steps:** What should the marketing team do next?

**Example:**

.. code-block:: text

   Cluster 0: "Budget Conscious Shoppers"
   - Age: 35-45 years
   - Income: $30-40k
   - Spending: Low (20-35)
   - Strategy: Offer discounts, loyalty rewards, value bundles
   - ROI: Medium (focus on retention through deals)

Deliverables
------------

Submit a ZIP file named `RollNumber_Lab01_KMeans.zip` containing:

1. **Jupyter Notebook** (`Lab01_KMeans.ipynb`)
   
   - All code cells executed
   - Markdown explanations for each section
   - Visualizations embedded

2. **Python Script** (`Lab01_KMeans.py`)
   
   - Clean, modular code
   - All functions properly documented
   - Can be run from command line

3. **PDF Report** (`Lab01_Report.pdf`)
   
   - Business insights and recommendations
   - Key visualizations
   - 2-3 pages maximum

4. **Dataset** (if modified)

Evaluation Criteria
-------------------

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Component
     - Points
     - Details
   * - Data Exploration
     - 10
     - Complete EDA, meaningful visualizations
   * - Preprocessing
     - 10
     - Proper scaling, feature selection
   * - K-Means Implementation
     - 20
     - Correct algorithm, efficient code
   * - Optimal K Determination
     - 20
     - Multiple methods, proper interpretation
   * - Scikit-learn Comparison
     - 15
     - Thorough analysis, parameter tuning
   * - Cluster Analysis
     - 15
     - Insightful profiling, good visualizations
   * - Business Insights
     - 10
     - Actionable recommendations
   * - **Total**
     - **100**
     - 

Bonus Tasks (10 extra points)
------------------------------

1. **3D Clustering (5 points)**
   
   - Add Age as third dimension
   - Create 3D scatter plot with clusters
   - Use Plotly for interactivity

2. **Gap Statistic Implementation (5 points)**
   
   - Implement gap statistic from scratch
   - Compare with elbow and silhouette methods

3. **Mini-Batch K-Means (3 points)**
   
   - Implement Mini-Batch K-Means
   - Compare speed with standard K-Means
   - Plot convergence curves

4. **Dashboard Creation (7 points)**
   
   - Create interactive dashboard using Streamlit or Dash
   - Allow user to select K and see results
   - Deploy online and share link

Starter Code
------------

.. code-block:: python

   # Lab 1: K-Means Clustering - Starter Code
   # Name: [Your Name]
   # Roll Number: [Your Roll Number]
   # Date: [Submission Date]
   
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.cluster import KMeans
   from sklearn.preprocessing import StandardScaler
   from sklearn.metrics import silhouette_score, silhouette_samples
   import warnings
   warnings.filterwarnings('ignore')
   
   # Set style
   sns.set_style('whitegrid')
   plt.rcParams['figure.figsize'] = (12, 6)
   plt.rcParams['font.size'] = 11
   
   # ============================================================
   # PART A: DATA EXPLORATION
   # ============================================================
   
   # Load data
   df = pd.read_csv('Mall_Customers.csv')
   
   # TODO: Complete Part A tasks
   
   # ============================================================
   # PART B: DATA PREPROCESSING
   # ============================================================
   
   # TODO: Preprocessing steps
   
   # ============================================================
   # PART C: K-MEANS FROM SCRATCH
   # ============================================================
   
   def initialize_centroids(X, k, random_state=42):
       """Your implementation"""
       pass
   
   # TODO: Implement remaining functions
   
   # ============================================================
   # PART D: OPTIMAL K
   # ============================================================
   
   # TODO: Elbow method and silhouette analysis
   
   # ============================================================
   # PART E: SCIKIT-LEARN K-MEANS
   # ============================================================
   
   # TODO: Apply scikit-learn KMeans
   
   # ============================================================
   # PART F: CLUSTER ANALYSIS
   # ============================================================
   
   # TODO: Profiling and visualization
   
   # ============================================================
   # PART G: BUSINESS INSIGHTS
   # ============================================================
   
   # TODO: Write insights in markdown cells

Helpful Resources
-----------------

**Documentation:**

- `Scikit-learn K-Means <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_
- `Silhouette Score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html>`_
- `StandardScaler <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_

**Tutorials:**

- :doc:`../examples/kmeans_example` - Similar worked example
- :doc:`../modules/clustering` - Theory reference

**Videos:**

- StatQuest: K-Means Clustering
- 3Blue1Brown: Visualizing Data

FAQs
----

**Q: Can I use additional features from the dataset?**

A: Yes, but justify why each feature improves clustering.

**Q: How do I handle gender encoding?**

A: Use LabelEncoder or one-hot encoding. Explain your choice.

**Q: What if my elbow plot doesn't show a clear elbow?**

A: This is common! Use multiple methods and consider domain knowledge.

**Q: Should I remove outliers?**

A: Discuss outliers in your report. Try clustering with and without them.

**Q: Can I work in groups?**

A: No, individual submission required. Discuss concepts only.

**Q: How do I create a radar chart?**

A: Use matplotlib's polar projection or Plotly's go.Scatterpolar.

Submission Checklist
--------------------

Before submitting, ensure:

☐ All code cells executed successfully

☐ No errors or warnings

☐ Visualizations have titles, labels, legends

☐ Functions have docstrings

☐ Code follows PEP 8 style

☐ Random seeds set for reproducibility

☐ File names follow convention

☐ All deliverables included in ZIP

☐ Report is well-formatted PDF

☐ Submitted before deadline

Good luck! 🎯

