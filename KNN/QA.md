
# Basics of KNN

**1. What is the K-Nearest Neighbors (KNN) algorithm?**
 KNN is a supervised learning algorithm used for both classification and regression. It works by finding the ‘K’ nearest data points (neighbors) to a given input and making predictions based on majority voting (for classification) or averaging (for regression).

**2. Is KNN a classification or regression algorithm?**
It can be both: `KNeighborsClassifier` for classification and `KNeighborsRegressor` for regression.

**3. How does KNN work — explain step by step.**

1. Store the training data (features + labels).
2. For a new point, compute distance to all training points.
3. Find the K nearest neighbors (smallest distances).
4. For classification: majority vote of neighbors (optionally weighted). For regression: take mean (or weighted mean) of neighbors' values.
5. Return the predicted label/value.

**4. What is the role of the parameter ‘K’ in KNN?**
`K` is the number of neighbors consulted to make a prediction; it controls smoothness of predictions (neighborhood size).

**5. How do you choose the optimal value of K?**
Use cross-validation (e.g., k-fold) to evaluate performance for several K values and pick the one minimizing validation error (or maximizing chosen metric). Also prefer odd K in binary classification to avoid ties.

**6. What happens if you choose K = 1 or K = very large?**

* `K = 1`: model has low bias, high variance — overfits noisy points.
* Very large `K` (→ number of training points): predictions become majority class or overall mean — high bias, low variance, may underfit.

**7. Is KNN a parametric or non-parametric algorithm? Why?**
Non-parametric — it makes no strong assumptions about data distribution and doesn’t learn a fixed set of parameters; it stores training data and uses it at prediction.

**8. Why is KNN considered a lazy learner?**
KNN is called a lazy learner because it does not learn any decision boundary or model during training. Instead, it just stores the entire training dataset. The real computation happens only at prediction time, when it calculates distances from the query point to all training samples to find the nearest neighbors.

**9. What do you mean by instance-based learning in KNN?**
Model decisions are based directly on stored instances (examples), not on a learned global function or parameters.

**10. In what situations would you prefer using KNN?**

* When you have a small to medium dataset.
* When decision boundaries are irregular/nonlinear and you want a simple baseline.
* When feature meaning and a good distance metric are clear.
  Avoid for huge datasets or very high-dimensional sparse data.

# Distance Metrics & Working

**11. What distance metrics are used in KNN?**
Common ones: Euclidean (L2), Manhattan (L1), Minkowski, Chebyshev, Cosine similarity (similarity, not distance), Hamming (for categorical/binary), and Gower (mixed types).

**12. When would you use Euclidean distance vs Manhattan distance?**

* Euclidean for continuous features where straight-line distance is meaningful.
* Manhattan (L1) when you want robustness to outliers or when features are grid-like; also useful when differences add linearly.

**13. Can you use Cosine similarity in KNN?**
Yes — especially for high-dimensional sparse data (e.g., text) where orientation matters more than magnitude. You’d rank neighbors by cosine similarity instead of distance.

**14. What is the effect of feature scaling on KNN performance?**
Big effect: features with larger scales dominate distance computation. Without scaling, KNN will be biased toward those features.

**15. Why do we need to normalize or standardize data before using KNN?**
To put features on comparable scales so each feature contributes appropriately to distance. Use Min-Max scaling for bounded ranges or Standardization (z-score) for normal-like features.

**16. How does dimensionality affect the distance calculation in KNN?**
In high dimensions distances concentrate (differences shrink relative to range), making neighbors less meaningful — distances become less discriminative (curse of dimensionality).

# Performance & Complexity

**17. What is the time complexity of KNN during training and prediction?**

* Training: O(1) work to “learn” plus O(n) to store data (practically O(n) memory).
* Prediction (brute force): O(n * d) per query (n = number of training points, d = features). Using efficient structures can reduce this.

**18. Why is KNN slow at prediction time but fast during training?**
Because training only stores data (fast), while prediction computes distances to many stored points (expensive).

**19. How can we improve the efficiency of KNN for large datasets?**

* Use approximate nearest neighbor methods (LSH, ANNOY, FAISS).
* Use tree structures (KD-tree, Ball-tree) for low-to-moderate dimensions.
* Reduce data (sampling, prototypes), dimensionality reduction (PCA, autoencoders), or use indexing libraries.

**20. What data structures can optimize KNN searches (like KD-trees, Ball-trees)?**
KD-tree (good for low dims), Ball-tree (better for varied metrics/high dims), VP-tree, cover trees, and locality-sensitive hashing (LSH) for approximate NN.

**21. How does curse of dimensionality impact KNN performance?**
As dimensionality grows, distances between points become similar, neighbors lose meaning, and KNN accuracy typically degrades unless you reduce dimensions or increase data massively.

**22. What are some disadvantages of using KNN?**

* Slow prediction on large datasets.
* Sensitive to feature scaling and irrelevant/noisy features.
* Sensitive to class imbalance.
* Suffers in high-dimensional and sparse data without preprocessing.

**23. How can you handle large datasets efficiently with KNN?**
Use approximate NN libraries (FAISS, Annoy), dimensionality reduction, sample/prototype selection, or switch to faster algorithms that learn parameters.

# Model Evaluation & Tuning

**24. How do you evaluate the performance of a KNN classifier?**
Use metrics like accuracy, precision/recall, F1-score, ROC-AUC, confusion matrix, and cross-validation to estimate generalization.

**25. Which metrics would you use for KNN regression and classification?**

* Regression: MSE, RMSE, MAE, R².
* Classification: accuracy, precision, recall, F1, ROC-AUC (for probabilistic outputs or scoring).

**26. How do you tune the value of K using cross-validation?**
Try a range of K values (e.g., 1..20 or sqrt(n)) and compute CV score; pick K with best validation metric (or the simplest K within 1-std of best to avoid overfitting).

**27. How do you handle imbalanced datasets in KNN?**

* Use class weights or distance-weighted voting (gives more weight to close neighbors).
* Resample (oversample minority like SMOTE, undersample majority).
* Use appropriate metrics (precision/recall, F1) rather than accuracy.

**28. What’s the impact of noisy data or outliers on KNN performance?**
Outliers can drastically affect predictions because KNN relies on proximity; a noisy neighbor may mislead votes/averages.

**29. How can you improve KNN accuracy on noisy data?**

* Increase K to smooth noise.
* Use distance-weighting to favor nearer points.
* Remove outliers or use robust feature scaling.
* Use ensemble or prototype selection methods.

**30. Would you use feature selection before applying KNN? Why or why not?**
Yes often — KNN is sensitive to irrelevant features. Feature selection or dimensionality reduction (PCA, feature importance filters) helps improve distance relevance and speed.

# KNN in Practice

**31. Give a real-world example where KNN can be effectively used.**
Handwritten digit recognition (small-scale), simple recommendation (user-based), or quick anomaly detection for small datasets.

**32. Can you use KNN for recommendation systems?**
Yes — user-based/item-based collaborative filtering uses nearest neighbors (similar users or items). For large-scale recommender systems, approximate methods are used.

**33. How does KNN differ from algorithms like Logistic Regression or SVM?**
KNN is non-parametric and instance-based (lazy), while Logistic Regression and SVM learn global decision boundaries (parametric or semi-parametric) during training and are usually faster at prediction.

**34. Can KNN be used for multi-class classification problems?**
Yes — take majority vote among K neighbors; multi-class works naturally.

**35. How does KNN handle missing values in the dataset?**
KNN doesn’t handle missing values inherently. Approaches: impute missing values (mean/median or KNN imputation), or use distance measures that ignore missing dimensions.

**36. How would you implement KNN from scratch in Python?**
Short example below.

**37. Which scikit-learn class is used to implement KNN?**
`sklearn.neighbors.KNeighborsClassifier` and `sklearn.neighbors.KNeighborsRegressor`.

**38. What parameters can you tune in scikit-learn’s KNeighborsClassifier?**
`n_neighbors`, `weights` (uniform/distance/callable), `algorithm` (auto, ball_tree, kd_tree, brute), `leaf_size`, `p` (Minkowski power; p=1 L1, p=2 L2), `metric`, `n_jobs`, `metric_params`.

**39. How do you visualize decision boundaries in KNN?**
Plot predictions on a grid of points over two features, color by predicted class, overlay training points. Works best with 2 features (or use PCA to reduce to 2D).

**40. What is the bias-variance tradeoff in the context of KNN?**

* Small K → low bias, high variance (fits training data closely).
* Large K → high bias, low variance (smoother, may underfit). Choose K to balance.

# Advanced / Conceptual

**41. How can weighted KNN improve performance?**
Weight neighbors by inverse distance (e.g., 1 / (distance + ε)) so that closer neighbors have more influence — reduces effect of farther, possibly irrelevant neighbors.

**42. What are the pros and cons of using distance weighting?**

* Pros: better local sensitivity, can reduce misclassification due to far neighbors.
* Cons: noisy close points still harmful; if distances are unreliable due to scaling or correlation, weighting may mislead.

**43. How does KNN regression work?**
Predict the average (or weighted average) of the target values of the K nearest neighbors.

**44. What happens if your dataset has categorical features?**
You can encode categoricals (one-hot, ordinal) and use Hamming distance for pure categorical features. For mixed types, use Gower distance which handles numeric + categorical.

**45. Can KNN handle mixed-type data (numerical + categorical)?**
Yes with appropriate distance like Gower or by encoding categoricals and using a combined distance metric.

**46. How do you interpret KNN in terms of Bayesian decision theory?**
KNN approximates the Bayes classifier nonparametrically: as n → ∞ and K grows slowly, KNN approaches the Bayes optimal decision boundary under certain conditions (it approximates conditional class probabilities by local frequencies).

**47. What are some variants of KNN (like Radius Neighbors, Fuzzy KNN, etc.)?**

* Radius Neighbors: consider all neighbors within radius r.
* Fuzzy KNN: assigns fuzzy membership based on distances.
* Weighted KNN: distance-weighted votes.
* Edited/Condensed Nearest Neighbor: reduce training set.
* Approximate NN: LSH, Annoy, FAISS.

**48. How do you deal with high-dimensional data in KNN?**
Use dimensionality reduction (PCA, t-SNE for visualization, autoencoders), feature selection, or switch to other algorithms better suited for high dims or use approximate NN.

**49. What happens when features are highly correlated?**
Correlated features can overweight certain information; it’s good to remove redundancy (PCA or feature selection) before applying KNN.

**50. How do you perform feature importance analysis with KNN?**
KNN itself doesn’t give feature importance directly. Use wrapper methods (recursive feature elimination with cross-validation), permutation importance, or use models that provide importance and then apply KNN.

# Tricky & Application-Based

**51. Suppose you increase K from 3 to 15 — what effect will it have on bias and variance?**
Bias increases, variance decreases. The decision boundary becomes smoother; model less sensitive to local noise.

**52. If your dataset is highly imbalanced, how will KNN behave?**
KNN tends to predict the majority class because more neighbors belong to it. Minority class predictions can be drowned out unless you use weighting or resampling.

**53. How would you select features for KNN in a dataset with 1000 features?**
Use feature selection: filter methods (chi-square, mutual info), embedded methods (from other models), wrapper methods (RFE), or dimensionality reduction (PCA, autoencoder). Prefer features that help separate classes in distance space.

**54. Why might KNN perform poorly on sparse data (like text)?**
Because in sparse high-dimensional spaces most distances become similar; also magnitude differences and many zeros reduce meaningful nearest neighbors — use cosine similarity or transform data (TF-IDF + cosine) or use specialized models.

**55. How would you use KNN for anomaly detection?**
Compute distance to nearest neighbors; anomalies have large distance to neighbors. Use thresholding on average distance to K nearest neighbors.

**56. What’s the difference between K-means and KNN?**

* KNN: supervised, instance-based; predicts labels using nearest neighbors.
* K-means: unsupervised clustering; partitions data into K clusters by centroids.

**57. Can KNN be used for online learning or streaming data?**
KNN can be adapted: add new points to memory and optionally remove old ones (sliding window). But naive KNN scales poorly for continuous streams; use summarization or prototype maintenance.

**58. Why is KNN sensitive to irrelevant features?**
Irrelevant features add noise to distance calculation, making true neighbors less distinguishable.

**59. What happens if all features are not on the same scale?**
Features with larger numeric ranges dominate distance measures and bias predictions; always scale features.

**60. Can KNN model non-linear decision boundaries?**
Yes — KNN can model highly non-linear boundaries because decisions are local and can carve complex shapes depending on K and training data.

# Short Python: KNN from scratch (classification)

```python
import numpy as np
from collections import Counter

def euclidean(a, b):
    return np.sqrt(np.sum((a - b)**2))

class SimpleKNN:
    def __init__(self, k=3):
        self.k = k
    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
    def predict(self, X_test):
        X_test = np.array(X_test)
        preds = []
        for x in X_test:
            # compute distances
            dists = np.sqrt(np.sum((self.X - x)**2, axis=1))
            idx = np.argsort(dists)[:self.k]
            votes = self.y[idx]
            pred = Counter(votes).most_common(1)[0][0]
            preds.append(pred)
        return np.array(preds)
```

(For regression, replace majority vote with `np.mean(self.y[idx])` or weighted mean.)

# scikit-learn usage (quick)

```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski', p=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

# Practical tips & interview-ready one-liners

* Always scale features before KNN.
* Start with `K = sqrt(n)` as a heuristic, then cross-validate.
* Use KD-tree/Ball-tree for low dims; use approximate methods (FAISS, Annoy) for large-scale or high-dim.
* For text, prefer cosine + TF-IDF over Euclidean.
* To handle mixed types, use Gower distance.
* To speed up, reduce data (condensed nearest neighbor) or apply dimensionality reduction.

---

