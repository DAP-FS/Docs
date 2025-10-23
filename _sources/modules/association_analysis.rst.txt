Association Analysis
====================

Association analysis discovers interesting relationships and patterns in large transactional datasets.

Introduction
------------

**Definition:** Association rule mining identifies frequent itemsets and derives rules that show relationships between items.

**Applications:**

- Market basket analysis
- Recommendation systems
- Web usage mining
- Medical diagnosis
- Bioinformatics

Key Concepts
------------

Terminology
~~~~~~~~~~~

- **Transaction:** A set of items purchased together
- **Itemset:** A collection of one or more items
- **Support:** Frequency of itemset occurrence
- **Confidence:** Conditional probability of rule
- **Lift:** Strength of association

Example Transaction Database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Supermarket Transactions
   :header-rows: 1
   :widths: 20 80

   * - TID
     - Items
   * - T1
     - {Bread, Milk}
   * - T2
     - {Bread, Diaper, Beer, Eggs}
   * - T3
     - {Milk, Diaper, Beer, Cola}
   * - T4
     - {Bread, Milk, Diaper, Beer}
   * - T5
     - {Bread, Milk, Diaper, Cola}

Association Rules
-----------------

Basic Formulation
~~~~~~~~~~~~~~~~~

An association rule is an implication:

.. math::

   X \Rightarrow Y

where :math:`X, Y \subseteq I` (itemsets), and :math:`X \cap Y = \emptyset`

**Example:** :math:`\{\text{Bread}, \text{Diaper}\} \Rightarrow \{\text{Beer}\}`

*Interpretation:* Customers who buy bread and diapers also tend to buy beer.

Support
~~~~~~~

Support measures how frequently an itemset appears:

.. math::

   \text{support}(X) = \frac{|\{t \in T : X \subseteq t\}|}{|T|}

where:

- :math:`T` = set of all transactions
- :math:`t` = individual transaction

**Example:**

.. math::

   \text{support}(\{\text{Bread}\}) = \frac{4}{5} = 0.8 = 80\%

Confidence
~~~~~~~~~~

Confidence measures the reliability of the rule:

.. math::

   \text{confidence}(X \Rightarrow Y) = \frac{\text{support}(X \cup Y)}{\text{support}(X)}

**Example:**

.. math::

   \text{confidence}(\{\text{Bread}\} \Rightarrow \{\text{Milk}\}) = \frac{3/5}{4/5} = \frac{3}{4} = 75\%

Lift
~~~~

Lift measures how much more likely :math:`Y` is purchased when :math:`X` is purchased:

.. math::

   \text{lift}(X \Rightarrow Y) = \frac{\text{confidence}(X \Rightarrow Y)}{\text{support}(Y)} = \frac{\text{support}(X \cup Y)}{\text{support}(X) \times \text{support}(Y)}

**Interpretation:**

- :math:`\text{lift} = 1`: No association
- :math:`\text{lift} > 1`: Positive association
- :math:`\text{lift} < 1`: Negative association

Apriori Algorithm
-----------------

Concept
~~~~~~~

Apriori uses a "bottom-up" approach to generate frequent itemsets by exploiting the downward closure property.

**Apriori Principle:** If an itemset is frequent, all its subsets must also be frequent.

.. math::

   \text{If } \text{support}(X) < \text{min\_support}, \text{ then } \text{support}(X \cup Y) < \text{min\_support}

Algorithm Steps
~~~~~~~~~~~~~~~

.. code-block:: text

   Algorithm: Apriori
   
   Input: 
     - Transaction database D
     - Minimum support threshold min_sup
   
   Output: 
     - All frequent itemsets
   
   1. Scan database to find frequent 1-itemsets L₁
   
   2. For k = 2 to max_itemset_size:
      a. Generate candidate k-itemsets Cₖ from Lₖ₋₁
      b. Prune Cₖ using Apriori principle
      c. Scan database to count support of candidates in Cₖ
      d. Lₖ = {itemsets in Cₖ with support ≥ min_sup}
   
   3. Return L = ⋃ₖ Lₖ

Candidate Generation
~~~~~~~~~~~~~~~~~~~~

**Join Step:** Combine (k-1)-itemsets to create k-itemsets

**Example:** 

- :math:`L_2 = \{\{A,B\}, \{A,C\}, \{A,D\}, \{B,C\}, \{B,D\}, \{C,D\}\}`
- Join :math:`\{A,B\}` and :math:`\{A,C\}` → :math:`\{A,B,C\}`

**Prune Step:** Remove candidates with infrequent subsets

Implementation
~~~~~~~~~~~~~~

.. code-block:: python

   from mlxtend.frequent_patterns import apriori, association_rules
   from mlxtend.preprocessing import TransactionEncoder
   import pandas as pd

   # Sample transaction data
   transactions = [
       ['Bread', 'Milk'],
       ['Bread', 'Diaper', 'Beer', 'Eggs'],
       ['Milk', 'Diaper', 'Beer', 'Cola'],
       ['Bread', 'Milk', 'Diaper', 'Beer'],
       ['Bread', 'Milk', 'Diaper', 'Cola']
   ]

   # Transform to binary matrix
   te = TransactionEncoder()
   te_ary = te.fit(transactions).transform(transactions)
   df = pd.DataFrame(te_ary, columns=te.columns_)

   print("Binary Transaction Matrix:")
   print(df)

   # Apply Apriori algorithm
   frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
   
   print("\nFrequent Itemsets:")
   print(frequent_itemsets)

   # Generate association rules
   rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
   
   print("\nAssociation Rules:")
   print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

Visualization
~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   import seaborn as sns

   # Larger dataset for better visualization
   from sklearn.datasets import make_classification
   import numpy as np

   # Create synthetic transactions
   np.random.seed(42)
   n_transactions = 1000
   items = ['Apple', 'Banana', 'Orange', 'Milk', 'Bread', 
            'Butter', 'Cheese', 'Yogurt', 'Eggs', 'Coffee']

   # Generate transactions with some patterns
   transactions_large = []
   for _ in range(n_transactions):
       n_items = np.random.randint(2, 6)
       transaction = list(np.random.choice(items, size=n_items, replace=False))
       transactions_large.append(transaction)

   # Add some intentional patterns
   for _ in range(200):
       transactions_large.append(['Milk', 'Bread', 'Butter'])
   for _ in range(150):
       transactions_large.append(['Coffee', 'Milk'])

   # Transform and apply Apriori
   te = TransactionEncoder()
   te_ary = te.fit(transactions_large).transform(transactions_large)
   df_large = pd.DataFrame(te_ary, columns=te.columns_)

   frequent_itemsets_large = apriori(df_large, min_support=0.1, use_colnames=True)
   rules_large = association_rules(frequent_itemsets_large, metric="lift", min_threshold=1.0)

   # Visualization 1: Support vs Confidence
   plt.figure(figsize=(14, 5))

   plt.subplot(1, 2, 1)
   plt.scatter(rules_large['support'], rules_large['confidence'], 
               c=rules_large['lift'], s=100, cmap='viridis', alpha=0.6)
   plt.colorbar(label='Lift')
   plt.xlabel('Support', fontsize=12)
   plt.ylabel('Confidence', fontsize=12)
   plt.title('Support vs Confidence (colored by Lift)', fontsize=14)
   plt.grid(True, alpha=0.3)

   # Visualization 2: Lift distribution
   plt.subplot(1, 2, 2)
   plt.hist(rules_large['lift'], bins=30, edgecolor='black', alpha=0.7)
   plt.xlabel('Lift', fontsize=12)
   plt.ylabel('Frequency', fontsize=12)
   plt.title('Distribution of Lift Values', fontsize=14)
   plt.axvline(x=1, color='red', linestyle='--', label='Lift = 1 (Independence)')
   plt.legend()
   plt.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

   # Print top rules by lift
   print("\nTop 10 Rules by Lift:")
   top_rules = rules_large.sort_values('lift', ascending=False).head(10)
   print(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

FP-Growth Algorithm
-------------------

Concept
~~~~~~~

FP-Growth (Frequent Pattern Growth) is more efficient than Apriori as it:

- Avoids candidate generation
- Uses a compact tree structure (FP-tree)
- Requires only two database scans

FP-Tree Structure
~~~~~~~~~~~~~~~~~

.. graphviz::

   digraph fptree {
       rankdir=TB;
       node [shape=box, style="rounded,filled", fillcolor=lightblue];
       
       root [label="Root\n{}", fillcolor=lightgreen];
       
       bread1 [label="Bread:4"];
       milk1 [label="Milk:3"];
       diaper1 [label="Diaper:3"];
       
       milk2 [label="Milk:1"];
       diaper2 [label="Diaper:1"];
       beer1 [label="Beer:3"];
       
       root -> bread1;
       bread1 -> milk1;
       milk1 -> diaper1;
       diaper1 -> beer1;
       
       bread1 -> diaper2;
       diaper2 -> milk2;
   }

Algorithm Steps
~~~~~~~~~~~~~~~

.. code-block:: text

   Algorithm: FP-Growth
   
   Input:
     - Transaction database D
     - Minimum support threshold min_sup
   
   Output:
     - All frequent itemsets
   
   1. Scan database to find frequent 1-itemsets
   2. Sort items by decreasing support
   3. Build FP-tree:
      For each transaction:
        - Sort items by frequency
        - Insert into FP-tree
   4. Mine FP-tree recursively:
      For each item in header table:
        - Generate conditional pattern base
        - Construct conditional FP-tree
        - Recursively mine conditional FP-tree

Implementation
~~~~~~~~~~~~~~

.. code-block:: python

   from mlxtend.frequent_patterns import fpgrowth
   import time

   # Compare Apriori vs FP-Growth
   print("Comparing Apriori vs FP-Growth...")

   # Apriori
   start_time = time.time()
   frequent_apriori = apriori(df_large, min_support=0.1, use_colnames=True)
   apriori_time = time.time() - start_time
   print(f"\nApriori: {len(frequent_apriori)} itemsets found in {apriori_time:.4f}s")

   # FP-Growth
   start_time = time.time()
   frequent_fpgrowth = fpgrowth(df_large, min_support=0.1, use_colnames=True)
   fpgrowth_time = time.time() - start_time
   print(f"FP-Growth: {len(frequent_fpgrowth)} itemsets found in {fpgrowth_time:.4f}s")

   print(f"\nSpeedup: {apriori_time/fpgrowth_time:.2f}x")

Advanced Metrics
----------------

Conviction
~~~~~~~~~~

Measures dependency and implication strength:

.. math::

   \text{conviction}(X \Rightarrow Y) = \frac{1 - \text{support}(Y)}{1 - \text{confidence}(X \Rightarrow Y)}

**Interpretation:**

- Higher conviction → stronger rule
- Conviction = ∞ → perfect implication

Kulczynski Measure
~~~~~~~~~~~~~~~~~~

Average of two conditional probabilities:

.. math::

   \text{Kulc}(X, Y) = \frac{1}{2}\left(\frac{\text{support}(X \cup Y)}{\text{support}(X)} + \frac{\text{support}(X \cup Y)}{\text{support}(Y)}\right)

Imbalance Ratio
~~~~~~~~~~~~~~~

Measures imbalance in item frequencies:

.. math::

   IR(X, Y) = \frac{|\text{support}(X) - \text{support}(Y)|}{\text{support}(X) + \text{support}(Y) - \text{support}(X \cup Y)}

Real-World Example: Market Basket Analysis
-------------------------------------------

.. code-block:: python

   # Simulate realistic supermarket data
   import random

   # Define product categories and associations
   dairy = ['Milk', 'Butter', 'Cheese', 'Yogurt']
   bakery = ['Bread', 'Croissant', 'Bagel']
   beverages = ['Coffee', 'Tea', 'Juice', 'Soda']
   snacks = ['Chips', 'Cookies', 'Chocolate']
   produce = ['Apple', 'Banana', 'Orange']

   def generate_transaction():
       """Generate a realistic transaction with associated products"""
       transaction = []
       
       # 60% chance of buying from dairy
       if random.random() < 0.6:
           transaction.extend(random.sample(dairy, k=random.randint(1, 2)))
       
       # If dairy bought, 70% chance of buying bakery (association)
       if any(item in transaction for item in dairy):
           if random.random() < 0.7:
               transaction.extend(random.sample(bakery, k=1))
       
       # Independent beverage purchase
       if random.random() < 0.5:
           transaction.extend(random.sample(beverages, k=1))
       
       # Snacks and produce
       if random.random() < 0.3:
           transaction.extend(random.sample(snacks, k=1))
       if random.random() < 0.4:
           transaction.extend(random.sample(produce, k=random.randint(1, 2)))
       
       return list(set(transaction))  # Remove duplicates

   # Generate 5000 transactions
   supermarket_transactions = [generate_transaction() for _ in range(5000)]

   # Apply association analysis
   te = TransactionEncoder()
   te_ary = te.fit(supermarket_transactions).transform(supermarket_transactions)
   df_market = pd.DataFrame(te_ary, columns=te.columns_)

   # Find frequent itemsets
   frequent = fpgrowth(df_market, min_support=0.05, use_colnames=True)
   
   # Generate rules
   rules_market = association_rules(frequent, metric="lift", min_threshold=1.2)
   rules_market = rules_market.sort_values('lift', ascending=False)

   # Display insights
   print("\nTop 10 Product Associations:")
   print(rules_market.head(10)[['antecedents', 'consequents', 'support', 
                                  'confidence', 'lift']])

   # Actionable insights
   print("\n🛒 Marketing Insights:")
   for idx, row in rules_market.head(5).iterrows():
       ant = ', '.join(list(row['antecedents']))
       con = ', '.join(list(row['consequents']))
       print(f"• Customers buying {ant} are {row['lift']:.2f}x more likely to buy {con}")
       print(f"  Confidence: {row['confidence']:.1%}, Support: {row['support']:.1%}\n")

Comparison: Apriori vs FP-Growth
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - Apriori
     - FP-Growth
   * - Candidate Generation
     - Yes (multiple passes)
     - No
   * - Database Scans
     - Multiple
     - Two
   * - Memory Usage
     - Low
     - Higher (FP-tree)
   * - Speed
     - Slower
     - Faster
   * - Best For
     - Small datasets, interpretability
     - Large datasets, efficiency

Try This
--------

.. raw:: html

   <div class="try-this">

**Exercise 1:** Download a real retail dataset (e.g., Online Retail from UCI) and perform market basket analysis.

**Exercise 2:** Implement the Apriori algorithm from scratch in Python.

**Exercise 3:** Compare execution times of Apriori and FP-Growth on datasets of varying sizes.

**Exercise 4:** Use association rules to build a simple recommendation system.

.. raw:: html

   </div>

Practical Applications
----------------------

**1. Retail & E-commerce**

- Product placement optimization
- Cross-selling strategies
- Promotional bundling

**2. Healthcare**

- Drug interaction analysis
- Symptom-disease associations
- Treatment effectiveness patterns

**3. Web Mining**

- Page navigation patterns
- Click-stream analysis
- User behavior modeling

**4. Telecommunications**

- Service package recommendations
- Churn prediction patterns
- Network fault analysis

Common Pitfalls
---------------

1. **Setting thresholds too low:** Generates too many rules
2. **Setting thresholds too high:** Misses important patterns
3. **Ignoring lift:** High confidence doesn't mean strong association
4. **Spurious correlations:** Always validate with domain knowledge
5. **Data sparsity:** Many transactions with few common items

Summary
-------

- **Association analysis** discovers patterns in transactional data
- **Support, confidence, and lift** are key metrics
- **Apriori** uses candidate generation with pruning
- **FP-Growth** is more efficient using a tree structure
- Applications span retail, healthcare, and web analytics

Further Reading
---------------

- Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules. *VLDB*.
- Han, J., et al. (2000). Mining frequent patterns without candidate generation. *SIGMOD*.
- Tan, P. N., et al. (2005). *Introduction to Data Mining*. Chapter 6.

