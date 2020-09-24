# Coursera_HowToWinAKaggleCompetition-FinalProject
Final Project on the Coursera Course "How To Win A Kaggle Competition" (see: [How to Win a Data Science Competition: Learn from Top Kagglers](https://www.coursera.org/learn/competitive-data-science))


## Solution: Tree-based regression prediction solution with bagging and stacking

My solution of the final project for &quot;How to win a data science competition&quot; Coursera course based on the &quot;Predict Future Sales&quot;-dataset from Kaggle consists of a zip file called final\_submission.zip. In this archive you can find:

- two .ipynb files (**data\_prep\_final.ipynb** and **eval\_tree\_final.ipynb**) for data preparation and regression evaluation, respectively:
  - **data\_prep\_final.ipynb** includes exploratory data analysis (EDA), data cleaning and feature engineering. Here the most interesting part is the data aggregation and feature engineering on the lag quantities. Be aware that not all generated features will be used in the end, also because of limited computational resources.
  - **eval\_tree\_final.ipynb** includes the train-validation-test splits of the prepared data, the tree-based regressions (random forest and gradient boosted with XGboost, LightGBM and CatBoost; the models are chosen to be different in order to get some benefit from stacking, see below) as well as bagging and stacking. For the latter, the most interesting part is the random stacking selection of zero level regression predictions for the first level metamodel leading to the final second level regression prediction.
- 5 serialized model files as (intermediate) results for convenience and clarity:
  - _data\_prep\_final.pkl_ is the result of **data\_prep\_final.ipynb** and consists of the final data after the cleaning and feature engineering procedure. This file is the main input for the evaluation notebook **eval\_tree\_final.ipynb**. Here in this archive it is not included due to file size issues, but can be produced within a few minutes.
  - _trees\_valid\_matrix.csv_ is the prediction result of the zero level tree-based regressors on the validation dataset with bagging of three random seeds each. It serves as the training input of the first level random stacking metamodel.
  - _trees\_test\_matrix.csv_ is the prediction result of the zero level tree-based regressors on the test dataset with bagging of three random seeds each. It serves as the test input of the first level random stacking metamodel.
  - _stacktrees\_sub1.csv_ consists of the final prediction of future sales based on the usage of all predictions in form of columns in _trees\_test\_matrix.csv_. Here, a linear regression of the first level metamodel is used for the final prediction.
  - _stacktrees\_sub2.csv_ consists of the final prediction of future sales based on the usage of some predictions in form of randomly chosen columns in _trees\_test\_matrix.csv_. Here, a gradient boosted decision tree regression is used for the final prediction. The final prediction is chosen to be the final prediction of the best model (based on validation rsme) of 1.000 random inputs of the first level metamodel.

Note: The hyperparameters of each regressor type have been optimized beforehand via grid-search and cross-validation. The latter operations are not shown here for simplicity and to keep the submission fairly small.

**Results:**
Public LB score is 0.8886 for _stacktrees\_sub1.csv_ (with a slightly worse private LB score; slightly overfitting?) and 0.9126 for _stacktrees\_sub2.csv_ (with a slightly better private LB score). This may place me well into the top 1.000 of the nearly 9.000 competitors atm.  

## Useful links provided by course

### Recap of main ML algorithms

- [tensorflow](http://playground.tensorflow.org/#activation=tanh&amp;batchSize=10&amp;dataset=circle&amp;regDataset=reg-plane&amp;learningRate=0.03&amp;regularizationRate=0&amp;noise=0&amp;networkShape=4,2&amp;seed=0.14066&amp;showTestData=false&amp;discretize=false&amp;percTrainData=50&amp;x=true&amp;y=true&amp;xTimesY=false&amp;xSquared=false&amp;ySquared=false&amp;cosX=false&amp;sinX=false&amp;cosY=false&amp;sinY=false&amp;collectStats=false&amp;problem=classification&amp;initZero=false&amp;hideText=false)

- [VolpalWabbit](https://github.com/VowpalWabbit/vowpal\_wabbit)

- [XGBoost](https://github.com/dmlc/xgboost)

- [https://github.com/geffy/tffm](https://github.com/geffy/tffm)

### Software and hardware requirements

**StandCloud Computing:**

- [AWS](https://aws.amazon.com/), [Google Cloud](https://cloud.google.com/), [Microsoft Azure](https://azure.microsoft.com/)

**AWS spot option:**

- [Overview of Spot mechanism](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-instances.html)

- [Spot Setup Guide](http://www.datasciencebowl.com/aws_guide/)

**Stack and packages:**

- [Basic SciPy stack (ipython, numpy, pandas, matplotlib)](https://www.scipy.org/)

- [Jupyter Notebook](http://jupyter.org/)

- [Stand-alone python tSNE package](https://github.com/danielfrg/tsne)

- Libraries to work with sparse CTR-like data: [LibFM](http://www.libfm.org/), [LibFFM](https://www.csie.ntu.edu.tw/~cjlin/libffm/)

- Another tree-based method: RGF ([implemetation](https://github.com/baidu/fast_rgf), [paper](https://arxiv.org/pdf/1109.0887.pdf))

- Python distribution with all-included packages: [Anaconda](https://www.continuum.io/what-is-anaconda)

- [Blog &quot;datas-frame&quot; (contains posts about effective Pandas usage)](https://tomaugspurger.github.io/)

### Feature preprocessing and feature generation

- https://sebastianraschka.com/Articles/2014\_about\_feature\_scaling.html

- https://www.coursera.org/learn/machine-learning

- [https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/](https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/)

### Feature extraction from text and images

**Feature extraction from text**

**Bag of words**

- [Feature extraction from text with Sklearn](http://scikit-learn.org/stable/modules/feature_extraction.html)

- [More examples of using Sklearn](https://andhint.github.io/machine-learning/nlp/Feature-Extraction-From-Text/)

**Word2vec**

- [Tutorial to Word2vec](https://www.tensorflow.org/tutorials/word2vec)

- [Tutorial to word2vec usage](https://rare-technologies.com/word2vec-tutorial/)

- [Text Classification With Word2Vec](http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/)

- [Introduction to Word Embedding Models with Word2Vec](https://taylorwhitten.github.io/blog/word2vec)

NLP Libraries

- [NLTK](http://www.nltk.org/)

- [TextBlob](https://github.com/sloria/TextBlob)

**Feature extraction from images**

**Pretrained models**

- [Using pretrained models in Keras](https://keras.io/applications/)

- [Image classification with a pre-trained deep neural network](https://www.kernix.com/blog/image-classification-with-a-pre-trained-deep-neural-network_p11)

**Finetuning**

- [How to Retrain Inception's Final Layer for New Categories in Tensorflow](https://www.tensorflow.org/tutorials/image_retraining)

- [Fine-tuning Deep Learning Models in Keras](https://flyyufelix.github.io/2016/10/08/fine-tuning-in-keras-part2.html)

### Exploratory Data Analysis (EDA)

**Visualization tools**

- [Seaborn](https://seaborn.pydata.org/)

- [Plotly](https://plot.ly/python/)

- [Bokeh](https://github.com/bokeh/bokeh)

- [ggplot](http://ggplot.yhathq.com/)

- [Graph visualization with NetworkX](https://networkx.github.io/)

**Others**

- [Biclustering algorithms for sorting corrplots](http://scikit-learn.org/stable/auto_examples/bicluster/plot_spectral_biclustering.html)

### Data leakages

- [Perfect score script by Oleg Trott](https://www.kaggle.com/olegtrott/the-perfect-score-script) -- used to probe leaderboard

- [Page about data leakages on Kaggle](https://www.kaggle.com/docs/competitions#leakage)

- [Another page about data leakages on Kaggle](https://www.kaggle.com/dansbecker/data-leakage)

### Metric Optimization

**Classification**

- [Evaluation Metrics for Classification Problems: Quick Examples + References](http://queirozf.com/entries/evaluation-metrics-for-classification-quick-examples-references)

- [Decision Trees: &quot;Gini&quot; vs. &quot;Entropy&quot; criteria](https://www.garysieling.com/blog/sklearn-gini-vs-entropy-criteria)

- [Understanding ROC curves](http://www.navan.name/roc/)

**Ranking**

- [Learning to Rank using Gradient Descent](http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf) -- original paper about pairwise method for AUC optimization

- [Overview of further developments of RankNet](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf)

- [RankLib](https://sourceforge.net/p/lemur/wiki/RankLib/) (implemtations for the 2 papers from above)

- [Learning to Rank Overview](https://wellecks.wordpress.com/2015/01/15/learning-to-rank-overview)

**Clustering**

- [Evaluation metrics for clustering](http://nlp.uned.es/docs/amigo2007a.pdf)

### Hyperparameter tuning

- [Tuning the hyper-parameters of an estimator (sklearn)](http://scikit-learn.org/stable/modules/grid_search.html)

- [Optimizing hyperparameters with hyperopt](http://fastml.com/optimizing-hyperparams-with-hyperopt/)

- [Complete Guide to Parameter Tuning in Gradient Boosting (GBM) in Python](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)

Tips &amp; tricks

- [Far0n's framework for Kaggle competitions &quot;kaggletils&quot;](https://github.com/Far0n/kaggletils)

- [28 Jupyter Notebook tips, tricks and shortcuts](https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/)

### Advanced Features

**Matrix Factorization:**

- [Overview of Matrix Decomposition methods (sklearn)](http://scikit-learn.org/stable/modules/decomposition.html)

**t-SNE:**

- [Multicore t-SNE implementation](https://github.com/DmitryUlyanov/Multicore-TSNE)

- [Comparison of Manifold Learning methods (sklearn)](http://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html)

- [How to Use t-SNE Effectively (distill.pub blog)](https://distill.pub/2016/misread-tsne/)

- [tSNE homepage (Laurens van der Maaten)](https://lvdmaaten.github.io/tsne/)

- [Example: tSNE with different perplexities (sklearn)](http://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html#sphx-glr-auto-examples-manifold-plot-t-sne-perplexity-py)

**Interactions:**

- [Facebook Research&#39;s paper about extracting categorical features from trees](https://research.fb.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/)

- [Example: Feature transformations with ensembles of trees (sklearn)](http://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html)

Ensembling

- [Kaggle ensembling guide at MLWave.com (overview of approaches)](https://mlwave.com/kaggle-ensembling-guide/)

- [StackNet — a computational, scalable and analytical meta modelling framework (by KazAnova)](https://github.com/kaz-Anova/StackNet)

- [Heamy — a set of useful tools for competitive data science (including ensembling)](https://github.com/rushter/heamy)

### Competitions go through

**Past solutions**

- [http://ndres.me/kaggle-past-solutions/](http://ndres.me/kaggle-past-solutions/)

- [https://www.kaggle.com/wiki/PastSolutions](https://www.kaggle.com/wiki/PastSolutions)

- [http://www.chioka.in/kaggle-competition-solutions/](http://www.chioka.in/kaggle-competition-solutions/)

- [https://github.com/ShuaiW/kaggle-classification/](https://github.com/ShuaiW/kaggle-classification/)
