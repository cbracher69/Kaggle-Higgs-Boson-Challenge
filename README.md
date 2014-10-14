Kaggle: Higgs-Boson Challenge
=============================

My solution to the [Higgs Boson Challenge at Kaggle](https://www.kaggle.com/c/higgs-boson)

Christian Bracher

September - October 2014

### Introduction

Discovery of the Higgs boson was announced at CERN in July 2012.  High-energy physicists now are on a quest to measure its characteristics, such as the Higgs decay modes, and determine if it fits the current model of nature.

ATLAS is a particle physics experiment installed at the Large Hadron Collider at CERN that searches for new particles and processes using head-on collisions of protons of extraordinarily high energy. The ATLAS experiment has observed a signal of the Higgs boson decaying into two tau lepton particles, but this process is a small signal buried in "background" noise. 

The goal of the Higgs Boson Machine Learning Challenge is to explore the potential of machine learning methods to improve the discovery significance of the experiment.  Using simulated data with features characterizing events detected by ATLAS, the task is to classify events into "tau tau decay of a Higgs boson" versus "background."

A data set of 250,000 "training" events is given, including their weight *w* (a continuous quantity), and their character (signal or background).  In the simulated set, there is a sharp threshold in *w* separating signals (*w* < 0.05) and background (*w* > 0.05).  The challenge asks to classify 550,000 data points in a test set as signal or background, including a likelihood ranking.  The scoring algorithm takes into account the aggregate weights of both "true positives" (signals correctly identified) and "false positives" (background events mistaken for signals), but none of the "negatives" (data evaluated as background, rightly or wrongly).

Given this compound structure, a classification algorithm can involve regression, classification, or a combination of both.

### My Approach

I took this challenge as a chance to learn more about machine learning, and stayed purposefully away from pre-fabricated or "black box" algorithms.  I wanted to see how far I would get with my own means, and decided to base the method on the simplest of regression algorithms, linear regression.  (The winning algorithms used mostly advanced ensemble methods, like Gradient Boosted Trees, or neural networks, often in combination.)

* Split into Subclasses

Examination of the training data showed that the set contained a mix of four different physical decay mechanisms that involved any number between 0 and 3 "hadronic jets."  Features that described jet properties often required a minimum number of jets present; otherwise they were meaningless and marked as "missing."  After sorting the training set into classes by number of jets (with generally increasing number of relevant features), missing data was confined to a single data column, the predicted mass of the Higgs boson.  As a workaround, I split each jet class further into events with and without predicted mass (which mostly provided background), ending with 8 distinct classes of events, but no missing data.

* Data Preprocessing

Considering the definitions of different columns, and physical symmetries (like rotational symmetry) underlying the data, I found that certain sets of variables were linearly dependent, or irrelevant (e.g., only differences between angles have physical meaning).  Irrelevant features were removed, and linearly dependent sets reduced to basis sets.  Momentum-related features showed a wide range of values, with a skewed distribution toward small values.  They were transformed into a more compact distribution by application of the logarithm.  (The same was true for the attributed weights *w* of events.)  All non-periodic features (i.e., everything but angle differences) were finally normalized to zero mean and unit variance.

* Feature Creation

Preprocessing yielded classes with as little as 10 meaningful features.  In order to expand the flexibility of the linear regression approach, I decided to multiply each column into a number of generated features by "Gaussian splits," defined by the overlap between a feature and a set of shifted Gaussians, in the spirit of kernel methods.  Experiments showed that classifier improvement tapered off with 4- to 8-fold splits of each feature, so I settled on 8-fold splits for the competition.

* Regression and Validation

The eight separate classes of events then were used as training sets for a linear regressor for the logarithm of the weight ln(*w*).  (Weights were distributed over three decades, so some regularization seemed appropriate.)  For validation purposes, I used a shuffle-split algorithm.

Since the competition scoring metric is based on both weights and signal classification, outcomes from regression for the various classes was combined and sorted.  This was used as input for a scoring function that yielded the predicted score as a function of a threshold weight *wt* that differentiated between signal and background.  The threshold weight I found was about *wt* = 0.04, compatible with the actual threshold seen in the training data, for a score in the range of 2.7 ... 2.9 during cross-validation.

* Prediction

The same algorithm was run on the full training set to extract the fit coefficients for each class.  Then, the test set was preprocessed in the same way as the training set, and the regression applied.  The test set was sorted by predicted weight (signal likelihood), and selection of the signal and background labels was based on the threshold weight *wt*.

On the full test set, the algorithm achieved a score slightly above 2.7.  (The best submitted models reached a score of 3.8.)

### Implementation Notes

My code is written in Python, in the form of an iPython notebook.  It uses the standard scientific and machine learning libraries, viz., numpy, scipy, pandas, and scikit-learn, to perform the classification.  It also makes use of matplotlib/pyplot to visualize the results.  All algorithms are in the public domain.

I spent a few days working on the project, and perhaps 1-2 hours of computation time.  The algorithm is fast, and will train the regressor in about 30 seconds on an older Windows laptop (Intel i5 mobile processor, 8 GB RAM - much less would suffice).  On the same machine, a prediction based on the test set is done in about 15 seconds. 
