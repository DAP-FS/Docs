Laboratory Assignments
======================

Hands-on programming assignments to reinforce theoretical concepts through practical implementation.

.. toctree::
   :maxdepth: 2

   lab01_kmeans
   lab02_pca

Lab Schedule
------------

.. list-table::
   :header-rows: 1
   :widths: 10 30 30 15 15

   * - Lab #
     - Title
     - Topics Covered
     - Duration
     - Weightage
   * - 1
     - K-Means Clustering
     - Customer segmentation, elbow method
     - 2 weeks
     - 15%
   * - 2
     - PCA & Visualization
     - Dimensionality reduction, face recognition
     - 2 weeks
     - 15%
   * - 3
     - Hierarchical & DBSCAN
     - Dendrogram analysis, density-based clustering
     - 2 weeks
     - 15%
   * - 4
     - t-SNE & Autoencoders
     - Non-linear reduction, neural networks
     - 2 weeks
     - 20%
   * - 5
     - Association Rules
     - Market basket analysis, Apriori
     - 2 weeks
     - 15%
   * - 6
     - Course Project
     - End-to-end ML pipeline
     - 4 weeks
     - 20%

Submission Guidelines
---------------------

Format Requirements
~~~~~~~~~~~~~~~~~~~

Each lab submission must include:

1. **Jupyter Notebook (.ipynb)**
   
   - Well-documented code with markdown explanations
   - Clear section headers
   - All outputs visible

2. **Python Script (.py)**
   
   - Clean, modular code
   - Proper function documentation
   - PEP 8 style guidelines

3. **Report (PDF)**
   
   - Problem statement
   - Methodology
   - Results and visualizations
   - Analysis and conclusions
   - References

4. **Dataset (if custom)**
   
   - CSV or appropriate format
   - README explaining features

File Naming Convention
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   RollNumber_LabNumber_Name.ext
   
   Examples:
   - 2021CS001_Lab01_KMeans.ipynb
   - 2021CS001_Lab01_KMeans.py
   - 2021CS001_Lab01_Report.pdf

Code Quality Standards
~~~~~~~~~~~~~~~~~~~~~~

**Required:**

- **Comments:** Explain complex logic
- **Docstrings:** For all functions/classes
- **Variable names:** Descriptive, not x, y, z
- **Imports:** Organized at top
- **Error handling:** Try-except blocks where needed
- **Reproducibility:** Set random seeds

**Example:**

.. code-block:: python

   def calculate_wcss(data, k_range, random_state=42):
       """
       Calculate Within-Cluster Sum of Squares for different K values.
       
       Parameters
       ----------
       data : array-like, shape (n_samples, n_features)
           Input data for clustering
       k_range : iterable
           Range of K values to test
       random_state : int, default=42
           Random seed for reproducibility
       
       Returns
       -------
       wcss_values : list
           WCSS for each K value
       
       Examples
       --------
       >>> wcss = calculate_wcss(X, range(1, 11))
       >>> print(wcss)
       [1500.2, 850.3, 420.1, ...]
       """
       wcss_values = []
       
       for k in k_range:
           kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
           kmeans.fit(data)
           wcss_values.append(kmeans.inertia_)
       
       return wcss_values

Evaluation Rubric
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Criteria
     - Points
     - Description
   * - **Correctness**
     - 40
     - Algorithm implementation, results accuracy
   * - **Code Quality**
     - 20
     - Style, documentation, organization
   * - **Analysis**
     - 20
     - Interpretation, insights, conclusions
   * - **Visualization**
     - 10
     - Clear plots, proper labels, aesthetics
   * - **Report**
     - 10
     - Clarity, completeness, formatting

Late Submission Policy
----------------------

- **Within 24 hours:** 10% penalty
- **24-48 hours:** 25% penalty
- **48-72 hours:** 50% penalty
- **After 72 hours:** Not accepted (0 marks)

**Exceptions:** Medical emergencies with documentation

Academic Integrity
------------------

.. warning::

   **Plagiarism is strictly prohibited!**
   
   - Write your own code
   - Cite external resources
   - Discuss concepts, not copy code
   - Use plagiarism detection tools
   
   **Consequences:** Zero marks, disciplinary action

Collaboration Policy
~~~~~~~~~~~~~~~~~~~~

**Allowed:**

- Discussing high-level approaches
- Helping debug general errors
- Sharing publicly available resources

**Not Allowed:**

- Sharing code files
- Copying from online solutions
- Working together on implementation

Resources
---------

**Getting Help:**

- Office hours: Monday & Thursday, 3-5 PM
- Discussion forum on LMS
- TA sessions: Wednesday, 4-6 PM
- Email: ashwini.mathur@university.edu

**Useful Links:**

- `Scikit-learn Documentation <https://scikit-learn.org/>`_
- `Pandas User Guide <https://pandas.pydata.org/docs/user_guide/index.html>`_
- `Matplotlib Gallery <https://matplotlib.org/stable/gallery/index.html>`_
- `Seaborn Tutorial <https://seaborn.pydata.org/tutorial.html>`_

Lab Environment Setup
---------------------

**Option 1: Google Colab (Recommended)**

.. code-block:: python

   # All required packages pre-installed
   # No setup needed!
   # Access at: https://colab.research.google.com/

**Option 2: Local Installation**

.. code-block:: bash

   # Create virtual environment
   python -m venv ml_lab_env
   source ml_lab_env/bin/activate  # On Windows: ml_lab_env\Scripts\activate
   
   # Install packages
   pip install numpy pandas matplotlib seaborn scikit-learn
   pip install jupyter notebook plotly
   
   # Launch Jupyter
   jupyter notebook

**Option 3: Anaconda**

.. code-block:: bash

   # Create conda environment
   conda create -n ml_lab python=3.9
   conda activate ml_lab
   
   # Install packages
   conda install numpy pandas matplotlib seaborn scikit-learn jupyter
   
   # Launch
   jupyter notebook

Tips for Success
----------------

1. **Start Early:** Don't wait until the deadline
2. **Read Instructions:** Carefully review requirements
3. **Test Thoroughly:** Run code multiple times with different parameters
4. **Comment as You Code:** Don't leave documentation for later
5. **Backup Regularly:** Use Git or cloud storage
6. **Ask Questions:** Clarify doubts early
7. **Review Examples:** Study provided example notebooks
8. **Version Control:** Keep track of changes

Common Mistakes to Avoid
------------------------

❌ **Not setting random seeds** → Non-reproducible results

❌ **Forgetting to scale data** → Poor clustering performance

❌ **Not handling missing values** → Runtime errors

❌ **Hardcoding paths** → Code won't run on other machines

❌ **No error handling** → Crashes on edge cases

❌ **Poor variable names** → Unreadable code

❌ **Missing visualizations** → Hard to interpret results

✅ **Follow best practices from examples!**

