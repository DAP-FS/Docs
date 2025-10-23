Datasets for Unsupervised Learning
===================================

Curated list of datasets used throughout the course with descriptions, download links, and usage examples.

Built-in Datasets (Scikit-learn)
---------------------------------

Iris Dataset
~~~~~~~~~~~~

**Description:** Classic dataset with measurements of iris flowers from three species.

**Features:** 4 numerical (sepal length/width, petal length/width)

**Samples:** 150 (50 per class)

**Use Cases:** Classification, clustering validation, dimensionality reduction

.. code-block:: python

   from sklearn.datasets import load_iris
   
   iris = load_iris()
   X = iris.data
   y = iris.target
   feature_names = iris.feature_names
   
   print(f"Shape: {X.shape}")
   print(f"Features: {feature_names}")

Digits Dataset
~~~~~~~~~~~~~~

**Description:** Handwritten digits (0-9) represented as 8x8 images.

**Features:** 64 numerical (pixel intensities)

**Samples:** 1,797

**Use Cases:** Image clustering, dimensionality reduction visualization

.. code-block:: python

   from sklearn.datasets import load_digits
   
   digits = load_digits()
   X = digits.data  # Shape: (1797, 64)
   y = digits.target
   images = digits.images  # Shape: (1797, 8, 8)

Make Blobs
~~~~~~~~~~

**Description:** Generate synthetic clustering data with Gaussian blobs.

**Customizable:** Number of samples, features, clusters, cluster std

**Use Cases:** Testing clustering algorithms, creating examples

.. code-block:: python

   from sklearn.datasets import make_blobs
   
   X, y = make_blobs(
       n_samples=300,
       n_features=2,
       centers=4,
       cluster_std=1.0,
       random_state=42
   )

Real-World Datasets
-------------------

Customer Segmentation Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Source:** `Kaggle <https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python>`_

**File:** ``Mall_Customers.csv``

**Features:**

- CustomerID (int)
- Gender (categorical)
- Age (int)
- Annual Income (k$) (int)
- Spending Score (1-100) (int)

**Samples:** 200

**Download:**

.. code-block:: bash

   # Using Kaggle API
   kaggle datasets download -d vjchoudhary7/customer-segmentation-tutorial-in-python

**Usage Example:**

.. code-block:: python

   import pandas as pd
   
   df = pd.read_csv('Mall_Customers.csv')
   print(df.head())
   print(df.describe())

Online Retail Dataset
~~~~~~~~~~~~~~~~~~~~~

**Source:** `UCI ML Repository <https://archive.ics.uci.edu/ml/datasets/Online+Retail>`_

**Description:** Transactional data from UK-based online retail (2010-2011)

**Features:**

- InvoiceNo: Invoice number
- StockCode: Product code
- Description: Product name
- Quantity: Quantity purchased
- InvoiceDate: Date and time
- UnitPrice: Price per unit
- CustomerID: Customer identifier
- Country: Customer country

**Samples:** 541,909 transactions

**Use Cases:** Association rule mining, customer segmentation

**Download:**

.. code-block:: python

   import pandas as pd
   
   url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'
   df = pd.read_excel(url)

MNIST Dataset
~~~~~~~~~~~~~

**Source:** `Keras Datasets <https://keras.io/api/datasets/mnist/>`_

**Description:** Handwritten digits (70,000 images, 28x28 pixels)

**Training:** 60,000 samples

**Testing:** 10,000 samples

**Use Cases:** Dimensionality reduction, clustering, visualization

.. code-block:: python

   from tensorflow import keras
   
   (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
   
   # Flatten for traditional ML
   X_train_flat = X_train.reshape(60000, 784)
   X_test_flat = X_test.reshape(10000, 784)

Fashion-MNIST
~~~~~~~~~~~~~

**Source:** `Keras Datasets <https://keras.io/api/datasets/fashion_mnist/>`_

**Description:** Zalando's article images (10 categories)

**Features:** 784 (28x28 grayscale images)

**Samples:** 70,000

**Categories:** T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

.. code-block:: python

   (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

Wine Dataset
~~~~~~~~~~~~

**Source:** Scikit-learn / UCI

**Description:** Chemical analysis of wines from three cultivars

**Features:** 13 numerical (alcohol, malic acid, ash, etc.)

**Samples:** 178

**Use Cases:** Clustering, dimensionality reduction

.. code-block:: python

   from sklearn.datasets import load_wine
   
   wine = load_wine()
   X = wine.data
   y = wine.target

Olivetti Faces
~~~~~~~~~~~~~~

**Description:** Grayscale face images (40 subjects, 10 images each)

**Features:** 4,096 (64x64 pixels)

**Samples:** 400

**Use Cases:** PCA (eigenfaces), clustering

.. code-block:: python

   from sklearn.datasets import fetch_olivetti_faces
   
   faces = fetch_olivetti_faces(shuffle=True, random_state=42)
   X = faces.data  # Shape: (400, 4096)
   images = faces.images  # Shape: (400, 64, 64)

Dataset Repositories
--------------------

UCI Machine Learning Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**URL:** https://archive.ics.uci.edu/ml/

**Datasets:** 600+ datasets across domains

**Popular for Unsupervised Learning:**

- Iris
- Wine
- Seeds
- Wholesale Customers
- Online Retail

Kaggle Datasets
~~~~~~~~~~~~~~~

**URL:** https://www.kaggle.com/datasets

**Setup:**

.. code-block:: bash

   # Install Kaggle API
   pip install kaggle
   
   # Download dataset
   kaggle datasets download -d <dataset-name>

**Recommended Datasets:**

- Mall Customer Segmentation
- Credit Card Dataset for Clustering
- Marketing Campaign Dataset
- Uber Pickups Dataset

OpenML
~~~~~~

**URL:** https://www.openml.org/

**API Access:**

.. code-block:: python

   from sklearn.datasets import fetch_openml
   
   # Example: Fashion-MNIST
   X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, parser='auto')

Synthetic Data Generation
--------------------------

Make Moons
~~~~~~~~~~

**Use Case:** Non-linear clustering, DBSCAN testing

.. code-block:: python

   from sklearn.datasets import make_moons
   
   X, y = make_moons(n_samples=300, noise=0.1, random_state=42)

Make Circles
~~~~~~~~~~~~

**Use Case:** Testing non-linear algorithms

.. code-block:: python

   from sklearn.datasets import make_circles
   
   X, y = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)

Make Swiss Roll
~~~~~~~~~~~~~~~

**Use Case:** Manifold learning, dimensionality reduction

.. code-block:: python

   from sklearn.datasets import make_swiss_roll
   
   X, color = make_swiss_roll(n_samples=1500, noise=0.0, random_state=42)

Custom Dataset Creation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   
   def generate_custom_clusters(n_samples=1000, n_features=2, n_clusters=3):
       """
       Generate custom clustered data with controlled overlap.
       """
       from sklearn.datasets import make_blobs
       
       X, y = make_blobs(
           n_samples=n_samples,
           n_features=n_features,
           centers=n_clusters,
           cluster_std=[1.0, 1.5, 0.5],  # Different std for each cluster
           random_state=42
       )
       
       # Add noise
       noise = np.random.normal(0, 0.1, X.shape)
       X_noisy = X + noise
       
       return X_noisy, y

Data Preprocessing Utilities
-----------------------------

Standard Preprocessing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.preprocessing import StandardScaler
   from sklearn.impute import SimpleImputer
   from sklearn.pipeline import Pipeline
   
   # Create preprocessing pipeline
   preprocessor = Pipeline([
       ('imputer', SimpleImputer(strategy='mean')),
       ('scaler', StandardScaler())
   ])
   
   X_processed = preprocessor.fit_transform(X)

Handling Missing Values
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import numpy as np
   
   # Check missing values
   print(df.isnull().sum())
   
   # Drop rows with missing values
   df_clean = df.dropna()
   
   # Fill with mean/median/mode
   df['column'].fillna(df['column'].mean(), inplace=True)
   
   # Use imputer
   from sklearn.impute import SimpleImputer
   imputer = SimpleImputer(strategy='median')
   X_imputed = imputer.fit_transform(X)

Feature Scaling
~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
   
   # Z-score normalization (mean=0, std=1)
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   
   # Min-Max scaling (0-1 range)
   scaler = MinMaxScaler()
   X_scaled = scaler.fit_transform(X)
   
   # Robust to outliers
   scaler = RobustScaler()
   X_scaled = scaler.fit_transform(X)

Dataset Loading Template
-------------------------

.. code-block:: python

   """
   Template for loading and preparing datasets
   """
   import pandas as pd
   import numpy as np
   from sklearn.preprocessing import StandardScaler, LabelEncoder
   from sklearn.model_selection import train_test_split
   
   def load_and_prepare_data(filepath, target_column=None, test_size=0.2):
       """
       Load dataset and perform basic preprocessing.
       
       Parameters
       ----------
       filepath : str
           Path to CSV file
       target_column : str, optional
           Name of target column (for supervised tasks)
       test_size : float
           Proportion for test split
       
       Returns
       -------
       X_train, X_test, y_train, y_test (if target_column provided)
       or
       X (if unsupervised)
       """
       # Load data
       df = pd.read_csv(filepath)
       
       print(f"Dataset shape: {df.shape}")
       print(f"Missing values:\n{df.isnull().sum()}")
       
       # Handle missing values
       df = df.dropna()
       
       # Separate features and target
       if target_column:
           X = df.drop(columns=[target_column])
           y = df[target_column]
       else:
           X = df
           y = None
       
       # Encode categorical variables
       categorical_cols = X.select_dtypes(include=['object']).columns
       for col in categorical_cols:
           le = LabelEncoder()
           X[col] = le.fit_transform(X[col])
       
       # Scale features
       scaler = StandardScaler()
       X_scaled = scaler.fit_transform(X)
       
       # Split if target provided
       if y is not None:
           return train_test_split(X_scaled, y, test_size=test_size, random_state=42)
       else:
           return X_scaled
   
   # Usage
   X = load_and_prepare_data('data.csv')
   # or
   X_train, X_test, y_train, y_test = load_and_prepare_data('data.csv', target_column='label')

Course Dataset Repository
--------------------------

All course datasets are available in the GitHub repository:

.. code-block:: bash

   git clone https://github.com/USERNAME/unsupervised-ml-course
   cd unsupervised-ml-course/data/

**Structure:**

.. code-block:: text

   data/
   ├── clustering/
   │   ├── mall_customers.csv
   │   ├── wholesale_customers.csv
   │   └── seeds.csv
   ├── dimensionality_reduction/
   │   ├── mnist_subset.csv
   │   └── faces_dataset.npz
   ├── association_rules/
   │   ├── online_retail.csv
   │   └── groceries.csv
   └── README.md

Summary
-------

- **Built-in datasets:** Quick testing and prototyping
- **Real-world datasets:** Practical applications
- **Synthetic datasets:** Algorithm evaluation
- **Preprocessing:** Essential for good results
- **Always cite sources** when using external data

