# fairness-by-location

====================================================
## Overview
Code for paper: Fairness by "Where": A Statistically-Robust and Model-Agnostic Bi-Level Learning Framework. AAAI 2022.


====================================================
## Explanation of files

* X_train.npy: all training samples extracted from the satellite-based crop monitoring dataset.
* y_train.npy: the corresponding labels for training samples.
* train_id.pickle: training samples' indices for all partitions within each candidate partitioning.
* X_test.npy: all testing samples (not overlapped with training samples).
* y_test.npy: the corresponding labels for testing samples.
* test_id.pickle: training samples' indices for all partitions within each candidate partitioning.
* results: an example model.

## Explanation of the code:

#### Procedures

model_train.py:
1. Training a base model with training data with 300 epochs.
2. Applying stochastic and bi-level training strategies to the base model with 50 epochs.

evaluation.py:

3. Comparing the overall performance and fairness between the base model and the final model.
