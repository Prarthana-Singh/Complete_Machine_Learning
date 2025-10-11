
---

##  **Basic Conceptual Questions**

1. **What is a Train-Test Split?**
2. **Why do we split data into training and testing sets?**
3. **What happens if you don’t use a test set?**
4. **What is the typical ratio used for train-test split?**
5. **Can you explain the purpose of training data and testing data?**
6. **What is data leakage in the context of train-test split?**
7. **Why should the test set not be used during model training?**
8. **What is the difference between validation data and test data?**
9. **What could happen if the same data is used for both training and testing?**
10. **What is the main goal of a test dataset in machine learning?**

---

##  **Intermediate Questions**

11. **How do you choose the right split ratio (e.g., 70/30, 80/20, 90/10)?**
12. **Does the train-test split ratio depend on the dataset size? Explain.**
13. **What is stratified train-test split, and when should it be used?**
14. **Why is stratified sampling important for classification problems?**
15. **How do you perform a train-test split in Python using scikit-learn?**
16. **What parameters are available in `train_test_split()` function in sklearn?**
17. **Why do we use a random seed (`random_state`) during splitting?**
18. **What does `shuffle=True` mean in the train-test split?**
19. **What are the risks of not shuffling data before splitting?**
20. **Should you perform feature scaling before or after the train-test split? Why?**

---

##  **Technical & Practical Questions**

21. **How do you handle imbalanced data during train-test split?**
22. **What is stratified splitting, and how do you implement it in scikit-learn?**
23. **If your dataset is time-series, can you randomly split data? Why or why not?**
24. **How do you split data for time-series problems?**
25. **What is the difference between random split and sequential split?**
26. **How can you ensure the test set represents the overall data distribution?**
27. **If you have limited data, how can you perform a reliable train-test split?**
28. **Why is cross-validation often used in addition to a train-test split?**
29. **What does `test_size=0.2` mean in sklearn’s `train_test_split()`?**
30. **If the dataset has categorical variables, how can you ensure proper split distribution?**

---

##  **Scenario-Based / Analytical Questions**

31. **You trained a model with 95% accuracy on the training set but only 60% on the test set — what might be wrong?**
32. **If your test set accuracy keeps changing drastically with each run, what could be the reason?**
33. **Your dataset contains duplicate entries — how does that affect the train-test split?**
34. **If you perform data preprocessing before splitting, what risk might arise?**
35. **What is the correct order of operations: scaling, encoding, feature selection, or splitting?**
36. **Your test set contains unseen classes not present in training data — what will happen?**
37. **If your dataset is very small, how can you maximize performance estimation reliability?**
38. **What is data leakage and how can it occur during train-test split?**
39. **What steps can you take to prevent data leakage?**
40. **If you accidentally used test data in feature scaling, how will it affect the results?**

---

##  **Cross-Validation Related Questions**

41. **What is cross-validation, and how does it differ from train-test split?**
42. **Why might you prefer k-fold cross-validation over a single train-test split?**
43. **What is the limitation of using only a single train-test split?**
44. **How does stratified k-fold differ from normal k-fold?**
45. **When is it acceptable to use only train-test split without cross-validation?**
46. **If the dataset is very large, why might cross-validation be avoided?**
47. **In a time-series project, what kind of cross-validation method should be used?**
48. **How does cross-validation help in choosing hyperparameters?**
49. **Can you use the test set during cross-validation? Why or why not?**
50. **Why is it important to keep the test set untouched until the final evaluation?**

---

##  **Mathematical / Statistical Questions**

51. **What is the role of randomness in train-test splitting?**
52. **If your model shows high variance between splits, what does it indicate?**
53. **How does sample size affect bias and variance in a train-test split?**
54. **Can a bad split cause your model to look better or worse than it actually is?**
55. **What are the statistical properties a good split should maintain?**
56. **How can overfitting or underfitting be related to train-test splitting?**
57. **If you increase test size, how will it affect bias and variance in performance estimation?**
58. **Can you calculate performance uncertainty due to data split randomness?**
59. **What is the ideal way to split data for reproducible results?**
60. **What is the danger of tuning hyperparameters using test data?**

---

##  **Interview Challenge / Real-World Questions**

61. **Explain train-test split in simple terms to a non-technical person.**
62. **Give a real-world analogy to explain train-test split.**
63. **If you are building a model for predicting customer churn, how would you perform the split?**
64. **How would you split data in a time-series forecasting problem (like stock prices)?**
65. **If the dataset is extremely imbalanced, how would you handle the split to ensure fair evaluation?**
66. **You’re building a recommendation system — what’s the best splitting strategy?**
67. **If your dataset has multiple versions of the same entity, how do you avoid leakage?**
68. **In an image classification task, how do you ensure that similar images don’t appear in both train and test sets?**
69. **How would you perform a reproducible split across multiple experiments?**
70. **How can you ensure the test data remains unbiased when data is updated periodically?**

---

##  **Advanced Questions**

71. **How can you perform a group-based split to avoid related samples in train and test sets?**
72. **What is GroupKFold, and how is it different from StratifiedKFold?**
73. **What is the difference between a validation set and a test set in hyperparameter tuning?**
74. **How can leakage happen during feature engineering even if you’ve split the data correctly?**
75. **If you use cross-validation for model tuning, what is the purpose of keeping a final test set?**
76. **What is nested cross-validation and when should you use it?**
77. **What is an out-of-sample test and why is it important?**
78. **Can you describe a situation where you would not use a random train-test split?**
79. **How does the train-test split differ in supervised vs unsupervised learning?**
80. **If you deploy a model, how do you ensure future data behaves like your test data?**

---
