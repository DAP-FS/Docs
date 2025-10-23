Core Modules
============

This section contains comprehensive coverage of all unsupervised machine learning algorithms taught in this course.

.. toctree::
   :maxdepth: 2

   clustering
   dimensionality_reduction
   association_analysis

Module Overview
---------------

The course is organized into three major modules:

**Module 1: Clustering** (Weeks 1-6)
  Learn algorithms to group similar data points together, including K-Means, Hierarchical Clustering, DBSCAN, and Gaussian Mixture Models.

**Module 2: Dimensionality Reduction** (Weeks 8-11)
  Explore techniques to reduce feature space while preserving important information, covering PCA, t-SNE, and Autoencoders.

**Module 3: Association Analysis** (Weeks 12-13)
  Discover patterns and relationships in transactional data using Apriori and FP-Growth algorithms.

Learning Path
-------------

.. graphviz::

   digraph learning_path {
       rankdir=LR;
       node [shape=box, style=filled, fillcolor=lightblue];
       
       intro [label="Introduction\nto UML"];
       clustering [label="Clustering\nAlgorithms"];
       dim_red [label="Dimensionality\nReduction"];
       assoc [label="Association\nRules"];
       project [label="Course\nProject"];
       
       intro -> clustering;
       clustering -> dim_red;
       dim_red -> assoc;
       assoc -> project;
       clustering -> project [style=dashed];
       dim_red -> project [style=dashed];
   }

Prerequisites by Module
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Required Knowledge
   * - Clustering
     - Distance metrics, vector operations, basic optimization
   * - Dimensionality Reduction
     - Linear algebra, eigenvalues/eigenvectors, matrix decomposition
   * - Association Analysis
     - Set theory, combinatorics, basic probability

