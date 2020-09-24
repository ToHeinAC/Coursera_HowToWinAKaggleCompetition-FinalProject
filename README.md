# Coursera_HowToWinAKaggleCompetition-FinalProject
Final Project on the Coursera Course "How To Win A Kaggle Competition"

Title: Tree-based regression prediction solution with bagging and stacking
Description:
My solution of the final project for "How to win a data science competition" Coursera course based on the "Predict Future Sales"-dataset from Kaggle consists of a zip file called final_submission.zip. In this archive you can find:
•	two .ipynb files (data_prep_final.ipynb and eval_tree_final.ipynb) for data preparation and regression evaluation, respectively:
o	data_prep_final.ipynb includes exploratory data analysis (EDA), data cleaning and feature engineering. Here the most interesting part is the data aggregation and feature engineering on the lag quantities. Be aware that not all generated features will be used in the end, also because of limited computational resources.
o	eval_tree_final.ipynb includes the train-validation-test splits of the prepared data, the tree-based regressions (random forest and gradient boosted with XGboost, LightGBM and CatBoost; the models are chosen to be different in order to get some benefit from stacking, see below) as well as bagging and stacking. For the latter, the most interesting part is the random stacking selection of zero level regression predictions for the first level metamodel leading to the final second level regression prediction.
•	5 serialized model files as (intermediate) results for convenience and clarity
o	data_prep_final.pkl is the result of data_prep_final.ipynb and consists of the final data after the cleaning and feature engineering procedure. This file is the main input for the evaluation notebook eval_tree_final.ipynb.
o	trees_valid_matrix.csv is the prediction result of the zero level tree-based regressors on the validation dataset with bagging of three random seeds each. It serves as the training input of the first level random stacking metamodel.
o	trees_test_matrix.csv is the prediction result of the zero level tree-based regressors on the test dataset with bagging of three random seeds each. It serves as the test input of the first level random stacking metamodel.
o	stacktrees_sub1.csv consists of the final prediction of future sales based on the usage of all predictions in form of columns in trees_test_matrix.csv. Here, a linear regression of the first level metamodel is used for the final prediction.
o	stacktrees_sub2.csv consists of the final prediction of future sales based on the usage of some predictions in form of randomly chosen columns in trees_test_matrix.csv. Here, a gradient boosted decision tree regression is used for the final prediction. The final prediction is chosen to be the final prediction of the best model (based on validation rsme) of 1.000 random inputs of the first level metamodel.
Note: The hyperparameters of each regressor type have been optimized beforehand via grid-search and cross-validation. The latter operations are not shown here for simplicity and to keep the submission fairly small.
