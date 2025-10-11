
---

##  **Basic Conceptual Questions**

1. **What is Cross-Validation in Machine Learning?**
2. **Why do we use Cross-Validation?**
3. **How is Cross-Validation different from Train-Test Split?**
4. **What problem does Cross-Validation solve?**
5. **What is the main purpose of using Cross-Validation in model evaluation?**
6. **What are the advantages of using Cross-Validation?**
7. **What are the disadvantages of using Cross-Validation?**
8. **When should you use Cross-Validation?**
9. **When is Cross-Validation not suitable?**
10. **What is the difference between Validation Set and Cross-Validation?**

---

##  **Types of Cross-Validation (Intermediate Questions)**

11. **What is K-Fold Cross-Validation?**
12. **How does K-Fold Cross-Validation work?**
13. **How do you choose the value of K in K-Fold Cross-Validation?**
14. **What is Stratified K-Fold Cross-Validation? When is it used?**
15. **What is Leave-One-Out Cross-Validation (LOOCV)?**
16. **What is Leave-P-Out Cross-Validation?**
17. **What is Monte Carlo (Shuffle Split) Cross-Validation?**
18. **What is Time Series (Rolling) Cross-Validation?**
19. **What is Group K-Fold Cross-Validation?**
20. **How does Repeated K-Fold Cross-Validation differ from normal K-Fold?**

---

##  **Technical & Practical Questions**

21. **How do you implement Cross-Validation in scikit-learn?**
22. **What is the purpose of `cross_val_score()` in scikit-learn?**
23. **What parameters can be passed to `KFold()` in sklearn?**
24. **Why do we use `shuffle=True` in K-Fold Cross-Validation?**
25. **How does random_state affect reproducibility in Cross-Validation?**
26. **What’s the difference between `KFold()` and `StratifiedKFold()` in sklearn?**
27. **How can you use Cross-Validation with GridSearchCV or RandomizedSearchCV?**
28. **How does Cross-Validation help in hyperparameter tuning?**
29. **What metric does Cross-Validation return in sklearn by default?**
30. **Can you use Cross-Validation in regression and classification both?**

---

##  **Scenario-Based Questions**

31. **If your model performs well on training folds but poorly on validation folds, what does it indicate?**
32. **If K is too small, what happens to bias and variance?**
33. **If K is too large, what happens to computational cost and bias?**
34. **How would you perform Cross-Validation for a time-series dataset?**
35. **If your dataset is imbalanced, what type of Cross-Validation would you use?**
36. **You have grouped samples (e.g., multiple images from the same user) — how would you split them for CV?**
37. **If you have a very small dataset, which type of Cross-Validation would you prefer?**
38. **Can Cross-Validation cause data leakage? Give an example.**
39. **What steps can you take to prevent data leakage during Cross-Validation?**
40. **You notice high variation in CV scores — what might be the reason?**

---

##  **Mathematical / Statistical Questions**

41. **How is the average performance metric calculated in K-Fold CV?**
42. **Why does Cross-Validation provide a better estimate of model performance than a single split?**
43. **How does K affect bias and variance in model evaluation?**
44. **Can you derive the relationship between bias, variance, and K in Cross-Validation?**
45. **Why does LOOCV have low bias but high variance?**
46. **How does Cross-Validation reduce the risk of overfitting?**
47. **What is the computational complexity of LOOCV compared to K-Fold CV?**
48. **How does Cross-Validation help in model selection?**
49. **What is the difference between internal and external Cross-Validation?**
50. **How do you calculate standard deviation of CV scores and why is it useful?**

---

##  **Cross-Validation in Model Tuning**

51. **What is the role of Cross-Validation in hyperparameter tuning?**
52. **How does GridSearchCV use Cross-Validation internally?**
53. **What is the difference between GridSearchCV and cross_val_score()?**
54. **What is nested Cross-Validation, and why is it used?**
55. **How does nested CV prevent overfitting during model selection?**
56. **When should you use nested Cross-Validation instead of regular CV?**
57. **Why is it wrong to tune hyperparameters on the test set?**
58. **What is the purpose of keeping a final holdout test set after CV?**
59. **Can you combine Cross-Validation and Bootstrapping?**
60. **How can you use Cross-Validation to compare multiple models fairly?**

---

##  **Advanced / Research-Level Questions**

61. **What are the limitations of K-Fold Cross-Validation?**
62. **Why might Cross-Validation give an overly optimistic or pessimistic estimate?**
63. **Can Cross-Validation handle non-IID (independent and identically distributed) data?**
64. **How can Cross-Validation be adapted for large-scale or streaming data?**
65. **What are some faster alternatives to Cross-Validation for large datasets?**
66. **How does Cross-Validation handle missing data?**
67. **How does stratified sampling help improve reliability in Cross-Validation?**
68. **How does Cross-Validation relate to ensemble learning?**
69. **What is the impact of random seed on CV reproducibility?**
70. **Can Cross-Validation be parallelized? How?**

---

##  **Interview Challenge / Real-World Questions**

71. **Explain Cross-Validation to a non-technical person.**
72. **Give a real-world analogy for Cross-Validation.**
73. **If your model gives consistent CV scores across folds, what does that mean?**
74. **If your CV scores vary a lot across folds, what does that mean?**
75. **You ran CV on a dataset and got high accuracy but poor performance on the test set — why?**
76. **How would you perform CV in an NLP or image classification project?**
77. **How does CV work differently in regression and classification?**
78. **If your dataset changes over time (concept drift), how does it affect CV reliability?**
79. **Can you explain the workflow of k-fold CV with an example?**
80. **Would you use CV for model evaluation in production? Why or why not?**

---

##  **Bonus: Integration & Implementation Questions**

81. **What are the key parameters of `GridSearchCV()` and `RandomizedSearchCV()`?**
82. **How can you visualize Cross-Validation results in Python?**
83. **How can you check which fold performed the worst?**
84. **How do you handle different evaluation metrics across folds?**
85. **What is the role of scoring parameter in cross_val_score()?**
86. **If your model is stochastic (e.g., random forest), how does that affect CV results?**
87. **Should you perform feature scaling before or after CV splitting?**
88. **What’s the correct order of preprocessing steps with CV?**
89. **What are common mistakes people make when using Cross-Validation?**
90. **If your model takes too long to train with CV, what can you do to optimize it?**

---

##  **Concept Integration Questions**

91. **How does Cross-Validation relate to the Bias-Variance tradeoff?**
92. **How does Cross-Validation prevent overfitting compared to a simple holdout set?**
93. **Can you use Cross-Validation for unsupervised learning?**
94. **How would you perform Cross-Validation for clustering algorithms?**
95. **Can Cross-Validation be applied in deep learning models? How?**
96. **Why is Cross-Validation important for ensuring model generalization?**
97. **What are some common pitfalls when applying Cross-Validation in real projects?**
98. **How would you handle imbalanced folds during Cross-Validation?**
99. **What role does randomness play in CV reliability?**
100. **What steps do you follow after Cross-Validation before final model deployment?**

---
