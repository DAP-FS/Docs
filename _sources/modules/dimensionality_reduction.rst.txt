Dimensionality Reduction
========================

Dimensionality reduction techniques transform high-dimensional data into lower dimensions while preserving essential information.

Introduction
------------

**Definition:** Dimensionality reduction is the process of reducing the number of features under consideration by obtaining a set of principal variables.

**Why Reduce Dimensionality?**

- **Curse of Dimensionality:** Performance degrades in high dimensions
- **Visualization:** Humans can only visualize 2-3 dimensions
- **Computational Efficiency:** Faster training and prediction
- **Storage:** Less memory required
- **Noise Reduction:** Remove irrelevant features

**Types:**

1. **Feature Selection:** Select subset of original features
2. **Feature Extraction:** Create new features by transformation

.. graphviz::

   digraph dim_reduction {
       rankdir=TB;
       node [shape=box, style="rounded,filled", fillcolor=lightblue];
       
       dr [label="Dimensionality\nReduction", fillcolor=lightgreen];
       
       linear [label="Linear Methods"];
       nonlinear [label="Non-Linear Methods"];
       
       pca [label="PCA"];
       svd [label="SVD"];
       lda [label="LDA"];
       
       tsne [label="t-SNE"];
       umap [label="UMAP"];
       autoencoder [label="Autoencoders"];
       
       dr -> linear;
       dr -> nonlinear;
       
       linear -> pca;
       linear -> svd;
       linear -> lda;
       
       nonlinear -> tsne;
       nonlinear -> umap;
       nonlinear -> autoencoder;
   }

Principal Component Analysis (PCA)
-----------------------------------

Concept
~~~~~~~

PCA finds orthogonal directions (principal components) that capture maximum variance in the data.

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

Given data matrix :math:`X \in \mathbb{R}^{n \times d}`, PCA finds a projection matrix :math:`W \in \mathbb{R}^{d \times k}` where :math:`k < d`.

**Step 1: Center the data**

.. math::

   \bar{X} = X - \mu

where :math:`\mu = \frac{1}{n}\sum_{i=1}^{n} x_i`

**Step 2: Compute covariance matrix**

.. math::

   C = \frac{1}{n-1}\bar{X}^T\bar{X} \in \mathbb{R}^{d \times d}

**Step 3: Eigenvalue decomposition**

.. math::

   C = V\Lambda V^T

where:

- :math:`V` = matrix of eigenvectors (principal components)
- :math:`\Lambda` = diagonal matrix of eigenvalues

**Step 4: Select top k components**

.. math::

   W = [v_1, v_2, \ldots, v_k]

where :math:`\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_k` are the largest eigenvalues.

**Step 5: Project data**

.. math::

   Z = \bar{X}W \in \mathbb{R}^{n \times k}

Explained Variance
~~~~~~~~~~~~~~~~~~

The proportion of variance explained by the :math:`i`-th component:

.. math::

   \text{Explained Variance Ratio}_i = \frac{\lambda_i}{\sum_{j=1}^{d}\lambda_j}

Cumulative explained variance:

.. math::

   \text{Cumulative Variance}_k = \frac{\sum_{i=1}^{k}\lambda_i}{\sum_{j=1}^{d}\lambda_j}

PCA Visualization
~~~~~~~~~~~~~~~~~

.. graphviz::

   digraph pca_process {
       rankdir=LR;
       node [shape=box, style="rounded,filled", fillcolor=lightblue];
       
       original [label="Original Data\n(High Dimensional)"];
       centered [label="Centered Data"];
       cov [label="Covariance\nMatrix"];
       eigen [label="Eigen\nDecomposition"];
       select [label="Select Top-k\nComponents"];
       project [label="Projected Data\n(Low Dimensional)"];
       
       original -> centered [label="Mean Centering"];
       centered -> cov [label="Compute Cov"];
       cov -> eigen [label="Solve"];
       eigen -> select [label="Sort by λ"];
       select -> project [label="Transform"];
   }

Implementation
~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.decomposition import PCA
   from sklearn.preprocessing import StandardScaler
   from sklearn.datasets import load_iris
   import seaborn as sns

   # Load dataset
   iris = load_iris()
   X = iris.data
   y = iris.target
   feature_names = iris.feature_names
   target_names = iris.target_names

   print(f"Original shape: {X.shape}")

   # Standardize features
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)

   # Apply PCA
   pca = PCA(n_components=2)
   X_pca = pca.fit_transform(X_scaled)

   print(f"Reduced shape: {X_pca.shape}")
   print(f"\nExplained variance ratio: {pca.explained_variance_ratio_}")
   print(f"Cumulative variance: {pca.explained_variance_ratio_.sum():.3f}")

   # Visualization
   plt.figure(figsize=(14, 5))

   # Plot 1: Original features (first two)
   plt.subplot(1, 3, 1)
   for i, target_name in enumerate(target_names):
       plt.scatter(X[y == i, 0], X[y == i, 1], 
                   label=target_name, alpha=0.6, s=50)
   plt.xlabel(feature_names[0])
   plt.ylabel(feature_names[1])
   plt.title('Original Features')
   plt.legend()
   plt.grid(True, alpha=0.3)

   # Plot 2: PCA projection
   plt.subplot(1, 3, 2)
   for i, target_name in enumerate(target_names):
       plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], 
                   label=target_name, alpha=0.6, s=50)
   plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
   plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
   plt.title('PCA Projection (2D)')
   plt.legend()
   plt.grid(True, alpha=0.3)

   # Plot 3: Scree plot
   plt.subplot(1, 3, 3)
   pca_full = PCA().fit(X_scaled)
   plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
            np.cumsum(pca_full.explained_variance_ratio_), 
            'bo-', linewidth=2)
   plt.xlabel('Number of Components')
   plt.ylabel('Cumulative Explained Variance')
   plt.title('Scree Plot')
   plt.grid(True, alpha=0.3)
   plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
   plt.legend()

   plt.tight_layout()
   plt.show()

Component Loadings
~~~~~~~~~~~~~~~~~~

Principal component loadings show the contribution of each original feature:

.. code-block:: python

   # Component loadings
   loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

   # Create loading matrix dataframe
   loading_matrix = pd.DataFrame(
       loadings,
       columns=['PC1', 'PC2'],
       index=feature_names
   )

   print("\nPrincipal Component Loadings:")
   print(loading_matrix)

   # Visualize loadings
   plt.figure(figsize=(10, 6))
   sns.heatmap(loading_matrix, annot=True, cmap='coolwarm', 
               center=0, fmt='.3f', linewidths=1)
   plt.title('PCA Component Loadings')
   plt.ylabel('Original Features')
   plt.xlabel('Principal Components')
   plt.show()

PCA from Scratch
~~~~~~~~~~~~~~~~

.. code-block:: python

   def pca_from_scratch(X, n_components=2):
       """
       Implement PCA from scratch
       
       Parameters:
       -----------
       X : array-like, shape (n_samples, n_features)
           Training data
       n_components : int
           Number of components to keep
       
       Returns:
       --------
       X_transformed : array, shape (n_samples, n_components)
           Transformed data
       components : array, shape (n_components, n_features)
           Principal components
       explained_variance_ratio : array, shape (n_components,)
           Variance explained by each component
       """
       # Center the data
       mean = np.mean(X, axis=0)
       X_centered = X - mean
       
       # Compute covariance matrix
       cov_matrix = np.cov(X_centered.T)
       
       # Eigenvalue decomposition
       eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
       
       # Sort eigenvectors by eigenvalues in descending order
       idx = eigenvalues.argsort()[::-1]
       eigenvalues = eigenvalues[idx]
       eigenvectors = eigenvectors[:, idx]
       
       # Select top k eigenvectors
       components = eigenvectors[:, :n_components]
       
       # Project data
       X_transformed = X_centered.dot(components)
       
       # Explained variance ratio
       explained_variance_ratio = eigenvalues[:n_components] / np.sum(eigenvalues)
       
       return X_transformed, components, explained_variance_ratio

   # Test implementation
   X_manual, components, var_ratio = pca_from_scratch(X_scaled, n_components=2)
   
   print("Manual PCA explained variance:", var_ratio)
   print("Sklearn PCA explained variance:", pca.explained_variance_ratio_)

Reconstruction Error
~~~~~~~~~~~~~~~~~~~~

Measure quality of dimensionality reduction:

.. math::

   \text{Reconstruction Error} = ||X - \hat{X}||_F^2

where :math:`\hat{X} = ZW^T + \mu` is the reconstructed data.

.. code-block:: python

   # Reconstruct data
   X_reconstructed = pca.inverse_transform(X_pca)

   # Calculate reconstruction error
   reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
   print(f"Mean Squared Reconstruction Error: {reconstruction_error:.4f}")

t-SNE (t-Distributed Stochastic Neighbor Embedding)
----------------------------------------------------

Concept
~~~~~~~

t-SNE is a non-linear technique particularly well-suited for visualizing high-dimensional data by reducing it to 2 or 3 dimensions.

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Compute pairwise similarities in high-dimensional space**

Joint probability that :math:`x_i` picks :math:`x_j` as neighbor:

.. math::

   p_{ij} = \frac{\exp(-||x_i - x_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i}\exp(-||x_i - x_k||^2 / 2\sigma_i^2)}

Symmetrized:

.. math::

   p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}

**Step 2: Compute similarities in low-dimensional space**

Using Student's t-distribution with 1 degree of freedom:

.. math::

   q_{ij} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k \neq l}(1 + ||y_k - y_l||^2)^{-1}}

**Step 3: Minimize KL divergence**

.. math::

   C = KL(P||Q) = \sum_i\sum_j p_{ij}\log\frac{p_{ij}}{q_{ij}}

Gradient descent is used to minimize :math:`C`.

Key Parameters
~~~~~~~~~~~~~~

1. **Perplexity:** Balance between local and global structure (typical: 5-50)
2. **Learning Rate:** Step size for optimization (typical: 100-1000)
3. **Number of Iterations:** Usually 1000-5000

Implementation
~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.manifold import TSNE
   from sklearn.datasets import load_digits
   import matplotlib.pyplot as plt
   import time

   # Load digits dataset
   digits = load_digits()
   X = digits.data
   y = digits.target

   print(f"Original shape: {X.shape}")

   # Apply t-SNE with different perplexities
   perplexities = [5, 30, 50]
   
   fig, axes = plt.subplots(1, len(perplexities), figsize=(18, 5))

   for idx, perplexity in enumerate(perplexities):
       print(f"\nFitting t-SNE with perplexity={perplexity}...")
       start_time = time.time()
       
       tsne = TSNE(n_components=2, perplexity=perplexity, 
                   random_state=42, n_iter=1000)
       X_tsne = tsne.fit_transform(X)
       
       elapsed = time.time() - start_time
       print(f"Time elapsed: {elapsed:.2f}s")
       
       # Plot
       scatter = axes[idx].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                   c=y, cmap='tab10', s=20, alpha=0.7)
       axes[idx].set_title(f't-SNE (perplexity={perplexity})')
       axes[idx].set_xlabel('t-SNE Component 1')
       axes[idx].set_ylabel('t-SNE Component 2')
       
   plt.colorbar(scatter, ax=axes, label='Digit Class')
   plt.tight_layout()
   plt.show()

PCA vs t-SNE Comparison
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.datasets import load_digits
   import matplotlib.pyplot as plt

   # Load data
   digits = load_digits()
   X = digits.data
   y = digits.target

   # Scale data
   X_scaled = StandardScaler().fit_transform(X)

   # Apply PCA
   pca = PCA(n_components=2)
   X_pca = pca.fit_transform(X_scaled)

   # Apply t-SNE
   tsne = TSNE(n_components=2, perplexity=30, random_state=42)
   X_tsne = tsne.fit_transform(X_scaled)

   # Visualize
   fig, axes = plt.subplots(1, 2, figsize=(16, 6))

   # PCA plot
   scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], 
                              c=y, cmap='tab10', s=30, alpha=0.7)
   axes[0].set_title(f'PCA (Variance: {pca.explained_variance_ratio_.sum():.2%})')
   axes[0].set_xlabel('PC1')
   axes[0].set_ylabel('PC2')
   axes[0].grid(True, alpha=0.3)

   # t-SNE plot
   scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                              c=y, cmap='tab10', s=30, alpha=0.7)
   axes[1].set_title('t-SNE')
   axes[1].set_xlabel('t-SNE Component 1')
   axes[1].set_ylabel('t-SNE Component 2')
   axes[1].grid(True, alpha=0.3)

   plt.colorbar(scatter2, ax=axes, label='Digit Class')
   plt.tight_layout()
   plt.show()

Autoencoders
------------

Concept
~~~~~~~

Autoencoders are neural networks that learn compressed representations of data through an encoder-decoder architecture.

Architecture
~~~~~~~~~~~~

.. graphviz::

   digraph autoencoder {
       rankdir=LR;
       node [shape=box, style="rounded,filled", fillcolor=lightblue];
       
       input [label="Input\nX ∈ ℝᵈ"];
       encoder [label="Encoder\nφ(X)", fillcolor=lightgreen];
       latent [label="Latent Space\nZ ∈ ℝᵏ\n(k << d)", fillcolor=yellow];
       decoder [label="Decoder\nψ(Z)", fillcolor=lightcoral];
       output [label="Reconstruction\nX̂ ∈ ℝᵈ"];
       
       input -> encoder;
       encoder -> latent;
       latent -> decoder;
       decoder -> output;
   }

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

**Encoder:** :math:`z = \phi(x) = \sigma(Wx + b)`

**Decoder:** :math:`\hat{x} = \psi(z) = \sigma(W'z + b')`

**Loss Function (Reconstruction Error):**

.. math::

   \mathcal{L}(x, \hat{x}) = ||x - \hat{x}||^2

**Objective:** Minimize reconstruction error

.. math::

   \min_{\phi, \psi} \sum_{i=1}^{n} ||x_i - \psi(\phi(x_i))||^2

Implementation with TensorFlow/Keras
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from tensorflow import keras
   from tensorflow.keras import layers
   from sklearn.datasets import load_digits
   from sklearn.model_selection import train_test_split

   # Load data
   digits = load_digits()
   X = digits.data / 16.0  # Normalize to [0, 1]
   y = digits.target

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   # Define autoencoder architecture
   input_dim = X_train.shape[1]  # 64
   encoding_dim = 10  # Compressed representation

   # Encoder
   encoder_input = layers.Input(shape=(input_dim,))
   encoded = layers.Dense(32, activation='relu')(encoder_input)
   encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

   # Decoder
   decoded = layers.Dense(32, activation='relu')(encoded)
   decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

   # Autoencoder model
   autoencoder = keras.Model(encoder_input, decoded)

   # Encoder model (for dimensionality reduction)
   encoder = keras.Model(encoder_input, encoded)

   # Compile
   autoencoder.compile(optimizer='adam', loss='mse')

   # Train
   history = autoencoder.fit(
       X_train, X_train,
       epochs=50,
       batch_size=32,
       shuffle=True,
       validation_data=(X_test, X_test),
       verbose=0
   )

   # Plot training history
   plt.figure(figsize=(10, 5))
   plt.plot(history.history['loss'], label='Training Loss')
   plt.plot(history.history['val_loss'], label='Validation Loss')
   plt.xlabel('Epoch')
   plt.ylabel('Reconstruction Loss (MSE)')
   plt.title('Autoencoder Training')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

   # Encode data to latent space
   X_encoded = encoder.predict(X_test)
   print(f"Encoded shape: {X_encoded.shape}")

   # Reconstruct data
   X_reconstructed = autoencoder.predict(X_test)

   # Visualize original vs reconstructed
   n_samples = 10
   plt.figure(figsize=(20, 4))
   
   for i in range(n_samples):
       # Original
       ax = plt.subplot(2, n_samples, i + 1)
       plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
       plt.title(f'Original\n{y_test[i]}')
       plt.axis('off')
       
       # Reconstructed
       ax = plt.subplot(2, n_samples, i + 1 + n_samples)
       plt.imshow(X_reconstructed[i].reshape(8, 8), cmap='gray')
       plt.title('Reconstructed')
       plt.axis('off')
   
   plt.tight_layout()
   plt.show()

   # Calculate reconstruction error
   mse = np.mean((X_test - X_reconstructed) ** 2)
   print(f"Mean Reconstruction Error: {mse:.6f}")

Variational Autoencoders (VAE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VAE learns a probabilistic latent representation:

.. math::

   \mathcal{L}(\theta, \phi; x) = -\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] + KL(q_\phi(z|x)||p(z))

where:

- First term: Reconstruction loss
- Second term: KL divergence (regularization)

Comparison Table
----------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 20 25
   :class: comparison-table

   * - Method
     - Type
     - Time Complexity
     - Preserves
     - Best For
   * - PCA
     - Linear
     - O(nd²)
     - Global structure
     - General purpose, fast
   * - t-SNE
     - Non-linear
     - O(n²)
     - Local structure
     - Visualization only
   * - UMAP
     - Non-linear
     - O(n log n)
     - Both local & global
     - Large datasets
   * - Autoencoders
     - Non-linear
     - Depends on architecture
     - Learned features
     - Complex patterns

Choosing the Right Method
--------------------------

**Use PCA when:**

- Need fast computation
- Linear relationships expected
- Interpretability important
- Need inverse transformation

**Use t-SNE when:**

- Primary goal is visualization
- Non-linear structure exists
- Dataset size moderate (<10,000 samples)
- Only need 2D/3D output

**Use Autoencoders when:**

- Non-linear relationships
- Large datasets
- Need to learn complex features
- Computational resources available

Try This
--------

.. raw:: html

   <div class="try-this">

**Exercise 1:** Apply PCA to the MNIST dataset and determine how many components are needed to retain 95% variance.

**Exercise 2:** Compare t-SNE with different perplexity values (5, 30, 50, 100) on the Fashion-MNIST dataset.

**Exercise 3:** Build a denoising autoencoder that can remove noise from images.

**Exercise 4:** Implement kernel PCA for non-linear dimensionality reduction.

.. raw:: html

   </div>

Practical Tips
--------------

1. **Always standardize/normalize data** before applying dimensionality reduction
2. **For PCA:** Use scree plot to determine number of components
3. **For t-SNE:** 
   
   - Run multiple times with different random seeds
   - Try different perplexity values
   - Increase iterations if not converged

4. **For Autoencoders:**
   
   - Start with shallow networks
   - Use dropout for regularization
   - Monitor reconstruction error

Summary
-------

- **Dimensionality reduction** reduces features while preserving information
- **PCA** finds linear projections maximizing variance
- **t-SNE** excels at visualizing high-dimensional data
- **Autoencoders** learn non-linear compressed representations
- Choice depends on data characteristics and goals

Further Reading
---------------

- Jolliffe, I. T. (2002). *Principal Component Analysis*. Springer.
- van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of machine learning research*.
- Goodfellow, I., et al. (2016). *Deep Learning*. MIT Press. Chapter 14.

