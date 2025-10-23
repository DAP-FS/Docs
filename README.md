# Unsupervised Machine Learning - Course Documentation

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://USERNAME.github.io/unsupervised-ml-docs/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/USERNAME/unsupervised-ml-docs/workflows/Deploy%20Sphinx%20Documentation/badge.svg)](https://github.com/USERNAME/unsupervised-ml-docs/actions)

> **Course:** CS4350 / ML5200 - Unsupervised Machine Learning  
> **Instructor:** Dr. Ashwini Kumar Mathur  
> **Institution:** Department of Computer Science & Engineering  
> **Semester:** Fall 2025

## 📚 Course Overview

Comprehensive Sphinx-based documentation for an undergraduate/postgraduate course on Unsupervised Machine Learning, covering clustering, dimensionality reduction, and association rule mining with hands-on Python implementations.

**Topics Covered:**

- 🔵 **Clustering:** K-Means, Hierarchical, DBSCAN, Gaussian Mixture Models
- 📉 **Dimensionality Reduction:** PCA, t-SNE, UMAP, Autoencoders
- 🔗 **Association Analysis:** Apriori, FP-Growth algorithms
- 🎯 **Real-World Applications:** Customer segmentation, anomaly detection, recommendation systems

## 🌐 Live Documentation

Visit the live documentation at: **https://USERNAME.github.io/unsupervised-ml-docs/**

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Local Setup

1. **Clone the repository:**


## 📊 Features

### Technical Features

- ✅ **LaTeX Support:** Full mathematical notation via MathJax
- ✅ **Graphviz Diagrams:** Algorithm flowcharts and visualizations
- ✅ **Code Highlighting:** Syntax-highlighted Python examples
- ✅ **Copy Buttons:** One-click code copying
- ✅ **Mobile Responsive:** Optimized for all devices
- ✅ **Dark/Light Mode:** Theme switcher
- ✅ **Search Functionality:** Full-text search
- ✅ **Auto-Deploy:** GitHub Actions CI/CD

### Educational Features

- 📖 **Theory:** Comprehensive coverage with mathematical formulations
- 💻 **Code Examples:** Practical scikit-learn implementations
- 🧪 **Lab Assignments:** Hands-on programming exercises
- 📊 **Visualizations:** Clear plots and diagrams
- 🎯 **Real-World Applications:** Industry use cases
- ❓ **Viva Questions:** Assessment preparation
- 📚 **References:** Curated reading list

## 🎓 Course Materials

### Modules

1. **[Clustering](source/modules/clustering.rst)**
   - K-Means Algorithm
   - Hierarchical Clustering
   - DBSCAN
   - Gaussian Mixture Models

2. **[Dimensionality Reduction](source/modules/dimensionality_reduction.rst)**
   - Principal Component Analysis (PCA)
   - t-SNE
   - Autoencoders

3. **[Association Analysis](source/modules/association_analysis.rst)**
   - Apriori Algorithm
   - FP-Growth
   - Market Basket Analysis

### Lab Assignments

- **Lab 1:** K-Means Customer Segmentation
- **Lab 2:** PCA for Image Compression
- **Lab 3:** Hierarchical Clustering
- **Lab 4:** t-SNE Visualization
- **Lab 5:** Association Rule Mining
- **Lab 6:** Course Project

### Datasets

All datasets available in the [data/](data/) directory:

- Mall Customer Segmentation
- Online Retail Transactions
- MNIST Digits
- Fashion-MNIST
- Iris Dataset

## 🤝 Contributing

We welcome contributions from students and educators!

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch:**

```

**Code Blocks:**
```
.. code-block:: python

   import numpy as np
   X = np.array([[1][2], [3][4]])
```

**Math:**
```
.. math::

   J = \sum_{i=1}^{k}\sum_{x \in C_i} ||x - \mu_i||^2
```

**Links:**
```
:doc:`clustering` - Internal link
`External Link <https://example.com>`_
```

## 🐛 Troubleshooting

### Common Issues

**Issue:** Sphinx build fails with import errors

**Solution:**
```
pip install --upgrade -r requirements.txt
```

---

**Issue:** LaTeX equations not rendering

**Solution:**
Ensure MathJax is configured in `conf.py`:
```
extensions = ['sphinx.ext.mathjax']
```

---

**Issue:** GitHub Pages shows 404

**Solution:**
- Check branch is `gh-pages`
- Ensure `.nojekyll` file exists
- Verify GitHub Pages settings in repo

---

**Issue:** Images not displaying

**Solution:**
Place images in `source/_static/` and reference:
```
.. image:: _static/diagram.png
```

## 📧 Contact

**Instructor:**

- **Name:** Dr. Ashwini Kumar Mathur
- **Email:** ashwini.mathur@university.edu
- **Office:** CSE Block, Room 304
- **Office Hours:** Monday & Thursday, 3:00 PM - 5:00 PM

**Course Website:** https://your-university.edu/courses/unsupervised-ml

**GitHub Issues:** [Report bugs or request features](https://github.com/USERNAME/unsupervised-ml-docs/issues)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Citation

If you use these materials in your teaching or research, please cite:

```
@misc{unsupervised_ml_docs,
  author = {Mathur, Ashwini Kumar},
  title = {Unsupervised Machine Learning Course Documentation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/USERNAME/unsupervised-ml-docs}
}
```

## 🙏 Acknowledgments

- **PyData Sphinx Theme** for beautiful documentation
- **Scikit-learn** community for excellent examples
- **Stanford CS229** and **MIT 6.036** for inspiration
- All contributors and students

## 📚 Additional Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Course GitHub Repository](https://github.com/USERNAME/unsupervised-ml-course)

## 🗓️ Version History

### v1.0.0 (October 2025)
- ✨ Initial release
- 📖 Complete clustering module
- 📉 Dimensionality reduction coverage
- 🔗 Association rules content
- 💻 6 lab assignments
- 📊 Multiple code examples

### Planned Features

- [ ] Interactive Jupyter widgets
- [ ] Video tutorials integration
- [ ] Quiz system
- [ ] Multilingual support
- [ ] Mobile app version

---

## 🌟 Star This Repository

If you find this course documentation helpful, please consider starring the repository! ⭐

---

**Built with ❤️ using Sphinx and Python**

*Last Updated: October 23, 2025*
```

**LICENSE** (MIT License)
```
MIT License

Copyright (c) 2025 Ashwini Kumar Mathur

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**CONTRIBUTING.md**
```markdown
# Contributing to Unsupervised ML Documentation

Thank you for considering contributing to this project! 🎉

## How to Contribute

### Reporting Issues

Found a typo, broken link, or incorrect information?

1. Check if issue already exists
2. Create new issue with clear description
3. Include page/section reference
4. Add screenshots if relevant

### Suggesting Enhancements

Have ideas for improvements?

1. Open an issue with "Enhancement" label
2. Describe your suggestion clearly
3. Explain the benefit
4. Provide examples if possible

### Adding Content

Want to add examples, explanations, or exercises?

**Before Starting:**
- Check existing content for overlap
- Discuss major changes in an issue first
- Follow the existing structure and style

**Content Guidelines:**

1. **Code Examples:**
   - Include complete, runnable code
   - Add comments explaining key steps
   - Test code before submitting
   - Use consistent style (PEP 8)

2. **Mathematical Notation:**
   - Use LaTeX for equations
   - Define all variables
   - Provide intuitive explanations

3. **Visualizations:**
   - Create clear, labeled plots
   - Use consistent color schemes
   - Make plots accessible (consider colorblind users)

4. **Writing Style:**
   - Be clear and concise
   - Use active voice
   - Define technical terms
   - Provide examples

### Pull Request Process

1. **Fork and Clone:**
   ```
   git clone https://github.com/YOUR-USERNAME/unsupervised-ml-docs.git
   ```

2. **Create Branch:**
   ```
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes:**
   - Edit .rst files
   - Add new content
   - Update index if needed

4. **Test Locally:**
   ```
   make clean
   make html
   # Check build/html/index.html
   ```

5. **Commit:**
   ```
   git add .
   git commit -m "Add: Brief description of changes"
   ```

6. **Push:**
   ```
   git push origin feature/your-feature-name
   ```

7. **Open Pull Request:**
   - Provide clear title and description
   - Reference related issues
   - Explain what changes were made and why

### Code Review

All submissions require review. We'll check:

- Content accuracy
- Code quality
- Documentation clarity
- Build success
- Style consistency

### Style Guide

**File Naming:**
- Use lowercase with underscores: `new_example.rst`
- Be descriptive but concise

**Headers:**
```
Main Title (Once per page)
==========================

Section Header
--------------

Subsection Header



