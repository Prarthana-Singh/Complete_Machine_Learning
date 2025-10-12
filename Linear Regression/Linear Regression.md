
---

##  **Basic-Level Questions**

1. What is Linear Regression?
2. What is the main goal of Linear Regression?
3. What are the assumptions of Linear Regression?
4. What is the difference between **simple** and **multiple linear regression**?
5. Explain the **equation of a simple linear regression line**.
6. What do the coefficients (slope and intercept) represent?
7. What does the **R² (R-squared)** value represent?
8. What does an **R² value of 0.8** mean?
9. Can R² ever be negative?
10. What are the **independent** and **dependent** variables in Linear Regression?

---

##  **Mathematical & Theoretical Questions**

11. Explain the **cost function** used in Linear Regression.
12. How do we minimize the cost function in Linear Regression?
13. What is **Ordinary Least Squares (OLS)**?
14. Derive the equation for estimating the coefficients (β).
15. What is the difference between **OLS** and **Gradient Descent**?
16. What is the **gradient descent algorithm** and how does it work for Linear Regression?
17. What is the role of the **learning rate (α)** in gradient descent?
18. What is the **normal equation** in Linear Regression?
19. Why do we prefer **gradient descent** for large datasets instead of normal equation?
20. What is the **bias term** in Linear Regression?

---

##  **Assumptions and Diagnostics**

21. List all the assumptions of Linear Regression.
22. What happens if the assumptions are violated?
23. What is **multicollinearity**, and how do you detect it?
24. What is the **Variance Inflation Factor (VIF)**, and what does it indicate?
25. What is **heteroscedasticity**, and how can you detect and fix it?
26. What is **autocorrelation** in residuals?
27. How can you check for **normality of residuals**?
28. What is the effect of **outliers** on Linear Regression?
29. How can you handle **outliers** in a regression model?
30. What is **linearity of relationships**, and how can you test it?

---

##  **Interpretation-Based Questions**

31. How do you interpret the regression coefficients?
32. What does the **p-value** indicate in Linear Regression?
33. What is the difference between **R²** and **Adjusted R²**?
34. When should you use **Adjusted R²** instead of R²?
35. What is the **F-statistic** in Linear Regression?
36. What does a **high p-value** mean for a variable?
37. How do you interpret **confidence intervals** of coefficients?
38. What does it mean if the intercept is negative?
39. How do you interpret **standard error** of coefficients?
40. How do you determine if your model is a good fit?

---

##  **Model Building & Practical Implementation**

41. How do you select important features for Linear Regression?
42. How do you handle **categorical variables** in Linear Regression?
43. Can Linear Regression handle **non-linear relationships**?
44. How do you deal with **missing data** before fitting the model?
45. What are the steps to build a Linear Regression model in Python using scikit-learn?
46. How do you interpret the `.coef_` and `.intercept_` attributes in scikit-learn?
47. How do you evaluate a regression model?
48. What are some **regression performance metrics**?
49. How would you improve a poorly performing regression model?
50. What is **feature scaling**, and is it required for Linear Regression?

---

##  **Regularization & Overfitting**

51. What is **overfitting** and **underfitting** in Linear Regression?
52. How can you detect overfitting?
53. What are **Lasso (L1)** and **Ridge (L2)** regularization?
54. What is the difference between **Ridge** and **Lasso** Regression?
55. What is **Elastic Net** regression?
56. How does **regularization** affect coefficients?
57. What happens to coefficients when λ (regularization parameter) increases?
58. When would you prefer Ridge over Lasso and vice versa?
59. How do you tune the **regularization parameter (α or λ)**?
60. What is the **bias-variance tradeoff** in Linear Regression?

---

##  **Advanced & Conceptual Questions**

61. What is **polynomial regression**, and how is it related to Linear Regression?
62. How does Linear Regression handle **interaction terms** between variables?
63. What is the impact of **multicollinearity** on model coefficients?
64. How do you test for **model significance**?
65. What is the difference between **Linear Regression** and **Logistic Regression**?
66. Can you use Linear Regression for **classification problems**?
67. What happens if the data contains **highly correlated features**?
68. How can **Principal Component Regression (PCR)** help in such cases?
69. Why might you prefer **regularized regression** over OLS in real-world data?
70. What is the impact of adding more variables to the model?

---

##  **Tricky & Real-World Questions**

71. What if your Linear Regression model shows high R² but poor prediction on test data?
72. How can you detect if your model is **biased**?
73. What if your model performs well on training but poorly on validation data?
74. How do you handle **non-linearity** in the data while using Linear Regression?
75. How do you identify and remove **irrelevant features**?
76. Can Linear Regression handle **categorical target variables**?
77. How would you interpret a model where coefficients are **very small but statistically significant**?
78. What’s the difference between **OLS residuals** and **standardized residuals**?
79. How would you interpret a **negative R²** value?
80. How would you perform **cross-validation** for Linear Regression?

---

##  **Applied / Scenario-Based Questions**

81. Suppose you are predicting **house prices** — what factors might violate assumptions?
82. What would you do if two features (e.g., area and number of rooms) are highly correlated?
83. How would you interpret a coefficient value of `0.8` for “Years of Experience” in a salary prediction model?
84. Your model has a low R² but performs well on unseen data — what does that mean?
85. What would you do if the residual plot shows a clear pattern?
86. Your dataset contains extreme outliers — how would you deal with them?
87. If the dependent variable is not normally distributed, can you still apply Linear Regression?
88. How do you transform a **non-linear dataset** to make it suitable for Linear Regression?
89. What’s the difference between **deterministic** and **stochastic** models?
90. Can Linear Regression work with **time series data**? If yes, how?

---

##  **Implementation in Python (scikit-learn & statsmodels)**

91. How do you implement Linear Regression using **scikit-learn**?
92. How do you check **p-values** in Python since scikit-learn doesn’t provide them?
93. How do you visualize **residuals** to check assumptions?
94. How can you use **statsmodels** to perform Linear Regression and interpret results?
95. What’s the difference between **statsmodels** and **scikit-learn** Linear Regression outputs?
96. How do you plot **actual vs predicted values**?
97. How do you calculate **RMSE**, **MAE**, and **R²** in Python?
98. How do you check for **multicollinearity** using Python?
99. What preprocessing steps do you perform before fitting a regression model?
100. How do you save and load a trained Linear Regression model?

---

