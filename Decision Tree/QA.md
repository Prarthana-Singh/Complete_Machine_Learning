
---

## 🌱 **1. What is a Decision Tree?**

**Answer:**
A Decision Tree is a **supervised learning algorithm** used for both **classification and regression** tasks.
It splits data into **branches based on feature conditions**, creating a **tree-like structure** where internal nodes represent **decisions**, branches represent **conditions**, and leaf nodes represent **final outcomes (predictions)**.

---

## ⚙️ **2. How does a Decision Tree work?**

**Answer:**
The algorithm:

1. Selects the **best feature** to split the data using a criterion (e.g., Gini, Entropy).
2. **Recursively splits** subsets into smaller groups.
3. Stops when a **stopping condition** is met (like max depth or pure leaf).
4. The final prediction is based on the **majority class** (for classification) or **mean value** (for regression).

---

## 🧮 **3. What are the main algorithms used to build Decision Trees?**

* **ID3** (Iterative Dichotomiser 3) — uses *Entropy* & *Information Gain*
* **C4.5** — extension of ID3 (handles continuous data, pruning)
* **CART** (Classification and Regression Tree) — uses *Gini Index* and supports regression

Most modern libraries (like scikit-learn) use **CART**.

---

## 📊 **4. What are the splitting criteria in a Decision Tree?**

### For Classification:

* **Gini Index**
* **Entropy / Information Gain**

### For Regression:

* **Mean Squared Error (MSE)**
* **Mean Absolute Error (MAE)**

---

## 🧠 **5. What is Entropy in Decision Trees?**

**Answer:**
Entropy measures the **impurity or disorder** in a dataset.
[
Entropy = -\sum p_i \log_2(p_i)
]
If all samples belong to one class → Entropy = 0 (pure node).
If samples are equally mixed → Entropy = 1 (impure node).

---

## 📈 **6. What is Information Gain (IG)?**

**Answer:**
Information Gain measures **how much uncertainty is reduced** after a split.
[
IG = Entropy(parent) - \sum \frac{n_i}{n} \times Entropy(child_i)
]
Higher IG → better split.
Used in **ID3** and **C4.5** algorithms.

---

## ⚡ **7. What is the Gini Index?**

**Answer:**
Gini Index measures **impurity** based on the probability of misclassification:
[
Gini = 1 - \sum p_i^2
]
If Gini = 0 → pure node.
CART algorithm uses Gini because it’s computationally faster than Entropy.

---

## 🧩 **8. What is the difference between Gini Index and Entropy?**

| Metric                                                       | Range | Formula                  | Interpretation    |
| ------------------------------------------------------------ | ----- | ------------------------ | ----------------- |
| Entropy                                                      | 0–1   | (-p\log_2 p - q\log_2 q) | Measures disorder |
| Gini                                                         | 0–0.5 | (1 - p^2 - q^2)          | Measures impurity |
| 👉 Both give similar results; Gini is faster in computation. |       |                          |                   |

---

## 🧱 **9. What are hyperparameters of a Decision Tree?**

* `max_depth` → maximum levels of the tree
* `min_samples_split` → min samples required to split
* `min_samples_leaf` → min samples required at a leaf node
* `max_features` → number of features considered for split
* `criterion` → (‘gini’, ‘entropy’, or ‘mse’)
* `splitter` → (‘best’, ‘random’)

---

## 🧮 **10. What is pruning in Decision Trees?**

**Answer:**
**Pruning** reduces tree complexity by removing branches that have little predictive power.
It helps **avoid overfitting**.

* **Pre-pruning (early stopping):** limit tree growth (e.g., `max_depth`, `min_samples_split`).
* **Post-pruning:** grow full tree first, then remove unnecessary branches based on validation accuracy or cost-complexity pruning.

---

## 💡 **11. What are advantages of Decision Trees?**

✅ Easy to understand and interpret
✅ Handles both numerical and categorical data
✅ No need for feature scaling
✅ Non-linear relationships handled well
✅ Can handle missing values

---

## ⚠️ **12. What are disadvantages of Decision Trees?**

❌ High risk of overfitting
❌ Unstable — small changes in data can change structure
❌ Greedy approach may not give global optimum
❌ Biased toward features with more levels

---

## 🧠 **13. How does pruning help prevent overfitting?**

**Answer:**
Pruning removes nodes that **do not contribute significantly to accuracy**, reducing model complexity.
It prevents the model from memorizing noise in the training set, thus improving **generalization**.

---

## 🔢 **14. How are continuous variables handled in Decision Trees?**

**Answer:**
The algorithm finds the **best threshold** to split the data.
Example: for feature “Age,” it might split at “Age < 35”.
It evaluates all possible split points and chooses the one with maximum **information gain**.

---

## 🧮 **15. How does a Decision Tree handle categorical variables?**

**Answer:**
For categorical features, it splits based on each category or group of categories — e.g.,
if “Color” ∈ {Red, Blue, Green}, the tree can create branches like:

* Color = Red
* Color = Blue or Green

---

## 🧭 **16. What’s the difference between a classification and regression tree?**

| Type                | Target      | Split Metric   | Output      |
| ------------------- | ----------- | -------------- | ----------- |
| Classification Tree | Categorical | Gini / Entropy | Class label |
| Regression Tree     | Continuous  | MSE / MAE      | Mean value  |

---

## ⚔️ **17. How does a Decision Tree handle missing values?**

**Answer:**

* Can assign samples with missing values to the **most frequent** or **mean** value branch.
* Some implementations (like XGBoost) learn the **best default direction** for missing data automatically.

---

## 🧩 **18. What are the stopping criteria for Decision Tree growth?**

* All samples in a node belong to one class
* No remaining features to split
* Max depth reached
* Information Gain < threshold
* Minimum samples per node reached

---

## 🧮 **19. How is feature importance calculated in Decision Trees?**

**Answer:**
Feature importance is the **total reduction in impurity (Gini/Entropy/MSE)** a feature contributes across all nodes where it’s used.
In scikit-learn:

```python
model.feature_importances_
```

It’s normalized to sum to 1.

---

## 🧩 **20. What are some real-world use cases of Decision Trees?**

🌾 Agriculture — Crop disease prediction
🏦 Finance — Credit risk assessment
🩺 Healthcare — Disease diagnosis
🛒 Retail — Customer segmentation
💬 NLP — Text classification (with categorical features)

---

## ✅ **Quick Summary (For 1-Minute Recall)**

| Concept             | Key Point                                     |
| ------------------- | --------------------------------------------- |
| Model Type          | Supervised (classification/regression)        |
| Core Idea           | Recursive splitting on best features          |
| Metrics             | Gini, Entropy, MSE                            |
| Key Params          | max_depth, min_samples_split, criterion       |
| Overfitting Control | Pruning                                       |
| Advantages          | Simple, interpretable, handles all data types |
| Disadvantages       | Overfitting, unstable                         |
| Foundation For      | Random Forest, XGBoost, CatBoost              |

---

