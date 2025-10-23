References and Resources
=========================

Comprehensive list of textbooks, papers, courses, and tools for unsupervised machine learning.

Textbooks
---------

Primary References
~~~~~~~~~~~~~~~~~~

1. **Pattern Recognition and Machine Learning** (2006)
   
   - Author: Christopher M. Bishop
   - Publisher: Springer
   - ISBN: 978-0387310732
   - Chapters: 9 (Mixture Models), 12 (PCA), 20 (Variational Inference)
   - `Online Version <https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/>`_

2. **The Elements of Statistical Learning** (2nd Edition, 2009)
   
   - Authors: Hastie, Tibshirani, Friedman
   - Publisher: Springer
   - ISBN: 978-0387848570
   - Chapters: 14 (Unsupervised Learning)
   - `Free PDF <https://hastie.su.domains/ElemStatLearn/>`_

3. **Introduction to Machine Learning** (4th Edition, 2020)
   
   - Author: Ethem Alpaydin
   - Publisher: MIT Press
   - ISBN: 978-0262043793
   - Comprehensive coverage of all ML topics

4. **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow** (3rd Edition, 2022)
   
   - Author: Aurélien Géron
   - Publisher: O'Reilly
   - ISBN: 978-1098125974
   - Chapters: 8 (Dimensionality Reduction), 9 (Unsupervised Learning)
   - Excellent practical guide

Specialized Books
~~~~~~~~~~~~~~~~~

5. **Clustering Algorithms** (2011)
   
   - Author: Xu, Rui & Wunsch, Donald
   - Publisher: Wiley-IEEE
   - ISBN: 978-0470276808
   - Comprehensive clustering survey

6. **Introduction to Data Mining** (2nd Edition, 2018)
   
   - Authors: Tan, Steinbach, Karpatne, Kumar
   - Publisher: Pearson
   - ISBN: 978-0133128901
   - Chapters: 7 (Clustering), 6 (Association Analysis)

7. **Deep Learning** (2016)
   
   - Authors: Goodfellow, Bengio, Courville
   - Publisher: MIT Press
   - ISBN: 978-0262035613
   - Chapter: 14 (Autoencoders)
   - `Free Online <https://www.deeplearningbook.org/>`_

8. **Python Machine Learning** (3rd Edition, 2019)
   
   - Authors: Sebastian Raschka, Vahid Mirjalili
   - Publisher: Packt
   - ISBN: 978-1789955750
   - Practical implementations

Research Papers
---------------

Foundational Papers
~~~~~~~~~~~~~~~~~~~

**Clustering:**

1. MacQueen, J. (1967). *Some methods for classification and analysis of multivariate observations.* 
   5th Berkeley Symposium on Mathematical Statistics and Probability.
   
   - Original K-Means paper

2. Ester, M., et al. (1996). *A density-based algorithm for discovering clusters in large spatial databases with noise.* 
   KDD-96.
   
   - DBSCAN algorithm

3. Kaufman, L., & Rousseeuw, P. J. (1990). *Finding Groups in Data: An Introduction to Cluster Analysis.* 
   Wiley.
   
   - K-Medoids and PAM algorithm

**Dimensionality Reduction:**

4. Jolliffe, I. T. (2002). *Principal Component Analysis.* 
   Springer.
   
   - Comprehensive PCA reference

5. van der Maaten, L., & Hinton, G. (2008). *Visualizing data using t-SNE.* 
   Journal of Machine Learning Research, 9, 2579-2605.
   
   - t-SNE algorithm

6. McInnes, L., et al. (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.* 
   arXiv:1802.03426.
   
   - Modern alternative to t-SNE

**Association Rules:**

7. Agrawal, R., & Srikant, R. (1994). *Fast algorithms for mining association rules.* 
   VLDB, 1215, 487-499.
   
   - Apriori algorithm

8. Han, J., et al. (2000). *Mining frequent patterns without candidate generation.* 
   ACM SIGMOD Record, 29(2), 1-12.
   
   - FP-Growth algorithm

**Autoencoders:**

9. Hinton, G. E., & Salakhutdinov, R. R. (2006). *Reducing the dimensionality of data with neural networks.* 
   Science, 313(5786), 504-507.
   
   - Deep autoencoders

10. Kingma, D. P., & Welling, M. (2013). *Auto-encoding variational bayes.* 
    arXiv:1312.6114.
    
    - Variational Autoencoders (VAE)

Recent Survey Papers
~~~~~~~~~~~~~~~~~~~~

11. Ezugwu, A. E., et al. (2022). *A comprehensive survey of clustering algorithms: State-of-the-art machine learning applications, taxonomy, challenges, and future research prospects.* 
    Engineering Applications of Artificial Intelligence, 110, 104743.

12. Anowar, F., et al. (2021). *Conceptual and empirical comparison of dimensionality reduction algorithms (PCA, KPCA, LDA, MDS, SVD, LLE, ISOMAP, LE, ICA, t-SNE).* 
    Computer Science Review, 40, 100378.

Online Courses
--------------

MOOCs
~~~~~

1. **Machine Learning Specialization** - Andrew Ng (Coursera)
   
   - URL: https://www.coursera.org/specializations/machine-learning-introduction
   - Weeks 8-9: Unsupervised Learning
   - Excellent foundational course

2. **Unsupervised Learning, Recommenders, Reinforcement Learning** - Andrew Ng, DeepLearning.AI
   
   - URL: https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning
   - Dedicated unsupervised learning course

3. **Machine Learning** - Stanford (CS229)
   
   - URL: http://cs229.stanford.edu/
   - Lecture notes and videos available
   - Advanced material

4. **Introduction to Machine Learning** - MIT (6.036)
   
   - URL: https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/
   - Free MIT OpenCourseWare

YouTube Channels
~~~~~~~~~~~~~~~~

- **StatQuest with Josh Starmer**
  
  - Clear explanations of PCA, t-SNE, clustering
  - URL: https://www.youtube.com/c/joshstarmer

- **3Blue1Brown**
  
  - Visual intuitions for linear algebra and PCA
  - URL: https://www.youtube.com/c/3blue1brown

- **Sentdex (Python Programming)**
  
  - Practical ML implementations
  - URL: https://www.youtube.com/c/sentdex

Documentation & Tutorials
--------------------------

Scikit-learn
~~~~~~~~~~~~

- **Official Documentation:** https://scikit-learn.org/stable/
- **User Guide:** https://scikit-learn.org/stable/user_guide.html
- **Examples:** https://scikit-learn.org/stable/auto_examples/

**Key Sections:**

- Clustering: https://scikit-learn.org/stable/modules/clustering.html
- Dimensionality Reduction: https://scikit-learn.org/stable/modules/decomposition.html
- Preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html

TensorFlow & Keras
~~~~~~~~~~~~~~~~~~

- **TensorFlow:** https://www.tensorflow.org/
- **Keras:** https://keras.io/
- **Autoencoders Tutorial:** https://www.tensorflow.org/tutorials/generative/autoencoder

Interactive Tutorials
~~~~~~~~~~~~~~~~~~~~~

- **Kaggle Learn:** https://www.kaggle.com/learn
  
  - Free micro-courses on ML topics

- **Google's Machine Learning Crash Course:** https://developers.google.com/machine-learning/crash-course

- **fast.ai:** https://www.fast.ai/
  
  - Practical deep learning courses

Tools and Libraries
-------------------

Python Libraries
~~~~~~~~~~~~~~~~

**Core ML:**

- **NumPy:** Numerical computing - https://numpy.org/
- **Pandas:** Data manipulation - https://pandas.pydata.org/
- **Scikit-learn:** Machine learning - https://scikit-learn.org/

**Visualization:**

- **Matplotlib:** Plotting library - https://matplotlib.org/
- **Seaborn:** Statistical visualization - https://seaborn.pydata.org/
- **Plotly:** Interactive plots - https://plotly.com/python/

**Deep Learning:**

- **TensorFlow:** https://www.tensorflow.org/
- **PyTorch:** https://pytorch.org/
- **Keras:** https://keras.io/

**Specialized:**

- **mlxtend:** ML extensions (Apriori, FP-Growth) - http://rasbt.github.io/mlxtend/
- **UMAP-learn:** UMAP implementation - https://umap-learn.readthedocs.io/
- **HDBSCAN:** Hierarchical DBSCAN - https://hdbscan.readthedocs.io/

Development Environments
~~~~~~~~~~~~~~~~~~~~~~~~

- **Jupyter Notebook:** Interactive computing - https://jupyter.org/
- **Google Colab:** Free GPU notebooks - https://colab.research.google.com/
- **VS Code:** Editor with Python support - https://code.visualstudio.com/
- **PyCharm:** Python IDE - https://www.jetbrains.com/pycharm/

Datasets and Benchmarks
------------------------

- **UCI ML Repository:** https://archive.ics.uci.edu/ml/
- **Kaggle Datasets:** https://www.kaggle.com/datasets
- **OpenML:** https://www.openml.org/
- **Google Dataset Search:** https://datasetsearch.research.google.com/

Blogs and Articles
------------------

Recommended Blogs
~~~~~~~~~~~~~~~~~

- **Towards Data Science:** https://towardsdatascience.com/
- **Machine Learning Mastery:** https://machinelearningmastery.com/
- **Distill.pub:** https://distill.pub/ (Visual explanations)
- **Scikit-learn Blog:** https://blog.scikit-learn.org/

Specific Articles
~~~~~~~~~~~~~~~~~

- **Understanding K-Means Clustering:** https://stanford.edu/~cpiech/cs221/handouts/kmeans.html
- **PCA Step-by-Step:** https://builtin.com/data-science/step-step-explanation-principal-component-analysis
- **t-SNE Explained:** https://distill.pub/2016/misread-tsne/

Communities and Forums
----------------------

- **Stack Overflow:** https://stackoverflow.com/questions/tagged/machine-learning
- **Cross Validated:** https://stats.stackexchange.com/
- **Reddit - r/MachineLearning:** https://www.reddit.com/r/MachineLearning/
- **Kaggle Forums:** https://www.kaggle.com/discussion
- **GitHub:** Search for ML repositories and discussions

Software and Tools
------------------

Clustering Tools
~~~~~~~~~~~~~~~~

- **Orange Data Mining:** Visual programming - https://orangedatamining.com/
- **KNIME:** Analytics platform - https://www.knime.com/
- **RapidMiner:** Data science platform - https://rapidminer.com/

Visualization Tools
~~~~~~~~~~~~~~~~~~~

- **Tableau:** Business intelligence - https://www.tableau.com/
- **Power BI:** Microsoft's BI tool - https://powerbi.microsoft.com/
- **Gephi:** Network visualization - https://gephi.org/

Version Control
~~~~~~~~~~~~~~~

- **Git:** https://git-scm.com/
- **GitHub:** https://github.com/
- **GitLab:** https://gitlab.com/

Course-Specific Resources
--------------------------

Course Repository
~~~~~~~~~~~~~~~~~

All course materials available at:

.. code-block:: bash

   git clone https://github.com/USERNAME/unsupervised-ml-course

**Contents:**

- Jupyter notebooks
- Datasets
- Code examples
- Assignments
- Projects

Recommended Reading Schedule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Week
     - Topic
     - Reading
   * - 1-2
     - K-Means
     - Bishop Ch. 9.1, ESL Ch. 14.3.6
   * - 3-4
     - Hierarchical & DBSCAN
     - Tan Ch. 7.2-7.3
   * - 5-6
     - Gaussian Mixture Models
     - Bishop Ch. 9.2-9.4
   * - 8-9
     - PCA
     - Bishop Ch. 12.1, ESL Ch. 14.5
   * - 10-11
     - t-SNE & Autoencoders
     - van der Maaten (2008), Goodfellow Ch. 14
   * - 12-13
     - Association Rules
     - Tan Ch. 6, Agrawal (1994)

Study Tips
----------

1. **Read theory first,** then implement
2. **Experiment with code** - modify parameters
3. **Visualize results** - understanding over memorization
4. **Work on projects** - apply to real data
5. **Join communities** - learn from others
6. **Document learning** - write blog posts
7. **Reproduce papers** - implement from scratch

Citation Guidelines
-------------------

When using resources in assignments/projects:

**Books:**

.. code-block:: text

   Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.

**Papers:**

.. code-block:: text

   van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. 
   Journal of Machine Learning Research, 9, 2579-2605.

**Websites:**

.. code-block:: text

   Scikit-learn developers (2023). Clustering. Retrieved from 
   https://scikit-learn.org/stable/modules/clustering.html

Contact Information
-------------------

**Instructor:**

Dr. Ashwini Kumar Mathur  
Department of Computer Science & Engineering  
Email: ashwini.mathur@university.edu  
Office Hours: Monday & Thursday, 3:00 PM - 5:00 PM

**Teaching Assistants:**

TBA

**Course Website:**

https://your-university.edu/courses/unsupervised-ml

---

*Last Updated: October 2025*

*For corrections or additions, please contact the instructor or submit a pull request to the course repository.*

