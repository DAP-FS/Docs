Unsupervised Machine Learning
==============================

.. image:: https://img.shields.io/badge/Python-3.8%2B-blue
   :alt: Python 3.8+

.. image:: https://img.shields.io/badge/scikit--learn-1.3%2B-orange
   :alt: scikit-learn

.. image:: https://img.shields.io/badge/Course-Active-green
   :alt: Course Status

Welcome to Unsupervised Machine Learning
-----------------------------------------

**Course Code:** CS4350 / ML5200

**Instructor:** Dr. Ashwini Kumar Mathur

**Department:** Computer Science & Engineering

**Level:** Undergraduate (B.Tech) / Postgraduate (M.Tech, MSc)

**Duration:** One Semester (August - December 2025)

Course Overview
---------------

This course provides a comprehensive introduction to **Unsupervised Machine Learning** techniques, focusing on discovering hidden patterns and structures in unlabeled data. Students will learn theoretical foundations, mathematical formulations, and practical implementations using Python and scikit-learn.

**Key Topics:**

- Clustering Algorithms (K-Means, Hierarchical, DBSCAN, Gaussian Mixture Models)
- Dimensionality Reduction (PCA, t-SNE, Autoencoders)
- Association Rule Mining (Apriori, FP-Growth)
- Anomaly Detection
- Model Evaluation and Validation

Learning Objectives
-------------------

By the end of this course, students will be able to:

1. **Understand** the fundamental concepts and mathematical principles of unsupervised learning
2. **Apply** clustering algorithms to group similar data points
3. **Implement** dimensionality reduction techniques for data visualization and preprocessing
4. **Analyze** association patterns in transactional data
5. **Evaluate** unsupervised models using appropriate metrics
6. **Design** complete ML pipelines for real-world datasets

Mathematical Foundation
-----------------------

Unsupervised learning aims to discover hidden structure in data without labeled examples. The general objective can be formulated as:

.. math::

   \theta^* = \arg\min_{\theta} \mathcal{L}(X; \theta)

where:

- :math:`X = \{x_1, x_2, \ldots, x_n\}` is the unlabeled dataset
- :math:`\theta` represents model parameters
- :math:`\mathcal{L}` is the loss or objective function

**Example: K-Means Objective**

K-Means clustering minimizes the within-cluster sum of squares:

.. math::

   J = \sum_{i=1}^{k}\sum_{x \in C_i} ||x - \mu_i||^2

where:

- :math:`k` is the number of clusters
- :math:`C_i` is the set of points in cluster :math:`i`
- :math:`\mu_i` is the centroid of cluster :math:`i`

Course Structure
----------------

.. toctree::
   :maxdepth: 2
   :caption: Core Modules

   modules/index

.. toctree::
   :maxdepth: 2
   :caption: Practical Examples

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Laboratory Work

   labs/index

.. toctree::
   :maxdepth: 2
   :caption: Datasets

   datasets/index

.. toctree::
   :maxdepth: 2
   :caption: Assessment

   viva/index

.. toctree::
   :maxdepth: 1
   :caption: Resources

   references

Prerequisites
-------------

**Required:**

- Programming proficiency in Python
- Linear Algebra (vectors, matrices, eigenvalues)
- Probability and Statistics
- Data Structures and Algorithms

**Recommended:**

- Prior exposure to supervised machine learning
- Experience with NumPy, Pandas, Matplotlib
- Basic understanding of optimization

Software Requirements
---------------------

Students must have the following installed:

.. code-block:: bash

   # Create virtual environment
   python -m venv ml_env
   source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate

   # Install required packages
   pip install numpy pandas scikit-learn matplotlib seaborn
   pip install jupyter notebook plotly

**Development Environment:**

- Python 3.8 or higher
- Jupyter Notebook / Google Colab
- VS Code or PyCharm (recommended)

Evaluation Scheme
-----------------

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Component
     - Weightage
     - Description
   * - Mid-Semester Exam
     - 25%
     - Theory and mathematical concepts
   * - End-Semester Exam
     - 35%
     - Comprehensive assessment
   * - Laboratory Assignments
     - 25%
     - 6 programming assignments
   * - Course Project
     - 15%
     - Team-based unsupervised learning project

Quick Start Guide
-----------------

**1. Clone Course Repository**

.. code-block:: bash

   git clone https://github.com/USERNAME/unsupervised-ml-course
   cd unsupervised-ml-course

**2. Explore First Example**

.. code-block:: python

   from sklearn.cluster import KMeans
   from sklearn.datasets import make_blobs
   import matplotlib.pyplot as plt

   # Generate sample data
   X, y = make_blobs(n_samples=300, centers=3, random_state=42)

   # Apply K-Means
   kmeans = KMeans(n_clusters=3, random_state=42)
   labels = kmeans.fit_predict(X)

   # Visualize
   plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
   plt.scatter(kmeans.cluster_centers_[:, 0], 
               kmeans.cluster_centers_[:, 1], 
               marker='X', s=200, c='red')
   plt.title('K-Means Clustering')
   plt.show()

**3. Try in Google Colab**

.. raw:: html

   <a href="https://colab.research.google.com/github/USERNAME/unsupervised-ml-course/blob/main/notebooks/kmeans_intro.ipynb" class="colab-button">
     Open in Colab
   </a>

Course Timeline
---------------

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Week
     - Topic
     - Deliverables
   * - 1-2
     - Introduction & K-Means
     - Lab 1: K-Means Implementation
   * - 3-4
     - Hierarchical & DBSCAN
     - Lab 2: Clustering Comparison
   * - 5-6
     - Gaussian Mixture Models
     - Assignment 1
   * - 7
     - Mid-Semester Exam
     - 
   * - 8-9
     - PCA & SVD
     - Lab 3: Dimensionality Reduction
   * - 10-11
     - t-SNE & Autoencoders
     - Lab 4: Visualization Techniques
   * - 12-13
     - Association Rule Mining
     - Lab 5: Market Basket Analysis
   * - 14
     - Anomaly Detection
     - Lab 6: Outlier Detection
   * - 15
     - Project Presentations
     - Final Project Report

Contact Information
-------------------

**Instructor:**

| Dr. Ashwini Kumar Mathur
| Department of Computer Science & Engineering
| Email: ashwini.mathur@university.edu
| Office: CSE Block, Room 304
| Office Hours: Monday & Thursday, 3:00 PM - 5:00 PM

**Teaching Assistants:**

| TBA

**Discussion Forum:**

| Available on course LMS

Additional Resources
--------------------

- 📚 :doc:`Complete Module List <modules/index>`
- 💻 :doc:`Code Examples <examples/index>`
- 📊 :doc:`Datasets Information <datasets/index>`
- 🎯 :doc:`Viva Questions <viva/index>`
- 📖 :doc:`References & Reading Materials <references>`

---

.. note::

   This documentation is continuously updated. Last updated: October 2025
   
   Report issues or suggest improvements via the GitHub repository.

Indices and Search
------------------

* :ref:`genindex`
* :ref:`search`

