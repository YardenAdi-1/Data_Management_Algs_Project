# **The Effect of Design Choices in Data Preprocessing on Estimated Treatment Effects: A Case Study on Academic Success**

This project investigates how **data preprocessing design choices** impact the **Estimated Treatment Effect (ETE)** in causal inference. Using a **case study on academic success**, we analyze how different **age thresholds for adulthood** and **definitions of academic success** affect the **Average Treatment Effect (ATE)** estimation.

We use data from the [**Predicting Student Dropout and Academic Success**](https://www.mdpi.com/2306-5729/7/11/146) dataset from the **Polytechnic Institute of Portalegre** and apply multiple causal inference methods to examine the sensitivity of ATE to preprocessing decisions.

---

## **Project Structure**

### **Data Processing and Exploration**
- `preprocessing.py`: Generates processed data files based on different preprocessing configurations.
- `Exploratory Data Analysis.ipynb`: Contains exploratory data analysis.
- `Checking Common Support.ipynb`: Validates the common support assumption to ensure reliable causal estimates.

### **Causal Effect Estimation**
- `s_learner.py`: Implements the S-Learner method.
- `t_learner.py`: Implements the T-Learner method.
- `inverse_probability_weighting.py`: Implements the Inverse Probability Weighting (IPW) method.
- `propensity_score_matching.py`: Implements the Propensity Score Matching (PSM) method.
- `doubly_robust.py`: Implements the Doubly Robust estimation method.
- `utils.py`: Contains utility functions used across different estimation methods.

### **Results and Reporting**
- `Estimating ATE.ipynb`: Notebook combining all estimation methods to compute ATE for different preprocessing choices.
- `results.csv`: Stores the estimated ATE values along with the bootstrap confidence intervals for different configurations.
- `report.tex`: Final project report summarizing findings.

