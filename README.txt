AISC LAB PRACTICAL - 10 SIMPLE PYTHON PROGRAMS
================================================

Each .py file corresponds to one of the 10 experiments in the SPIT CE303 AISC syllabus.

HOW TO RUN
----------
1. Install Python 3.
2. Open a terminal in this folder.
3. Install the common libraries:

   pip install numpy pandas scikit-learn matplotlib

4. Run one experiment, for example:

   python 04_Single_Layer_Perceptron_Loan_Approval.py

EXPERIMENT FILES
----------------
01 - Logic-based Expert System
02 - Intelligent Agents (five agent types)
03 - Activation Functions + ANN (Half Adder and Full Adder)
04 - Single Layer Perceptron (Loan Approval)
05 - Supervised Learning (Logistic Regression + Decision Tree)
06 - Unsupervised Learning (Iris, K-Means + Agglomerative)
07 - Associative Memory (Hopfield Network)
08 - Fuzzy Sets and Fuzzy Relations
09 - Fuzzy Logic Controller (Mamdani)
10 - ANFIS (Vehicle Fuel Consumption)

IMPORTANT NOTES
---------------
- Experiment 1: Your official syllabus specifies Prolog. The requested Python file
  demonstrates the same facts/rules/query idea in Python, but it is NOT a substitute
  for writing actual Prolog if the examiner specifically requires Prolog syntax.
- Experiment 5: Download the IBM HR Analytics CSV specified in your practical and
  place WA_Fn-UseC_-HR-Employee-Attrition.csv in this folder before running.
- Experiment 6: Species is kept out of clustering and used only for ARI/NMI evaluation.
- Experiment 10: This is a compact educational Sugeno ANFIS. It uses fixed Gaussian
  antecedent membership functions and least-squares consequent learning, matching
  the simplified model approach in your submitted notebook.
- Some numerical results can vary slightly by library version or training settings.

These are intentionally short study/practical programs. They prioritize the central
algorithm and readable code over exhaustive report-generation and optional plots.
