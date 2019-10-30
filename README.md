# RedditScammerDetector

Flask app for a Reddit scammer detector. Input a Reddit username to return a probability of that user being a scammer.
Python scripts in my_app directory can be used individually to scrape a Reddit user's basic profile information and comment history.

## Background

On Reddit, there are over 60 subreddits for buy/sell/trade, and for each subreddit, they can range from 10k to over 100k users. Using online buying/selling/trading groups have always carried a risk of running into scammers. I was curious if it was possible for a classification model to predict if a person is a scammer or not based off of their Reddit profile information and comment history. I mainly wanted to find out if scammers use distinctive language (or more toxic language) compared to a regular user.

## Summary

Check out my [Medium post IN PROGRESS](https://medium.com/p/1bdc024c8d69/) for a detailed analysis of my investigation and findings!

TL;DR: Initial training and validation set using only users from watchexchange and banned users from USL yielded great results. Naive bayes was able to identify scammers alone on certain keywords with an accuracy of 90% and recall of 95%. Ensemble model of Naive Bayes with XGBoost, LightGBM, and Random Forest was able to increase the ROC AUC score to .96 and increase accuracy and recall to 96%. Negative comment sentiment was identified as an important feature in the initial training set.

After adding more users from all buy/sell/trade subreddits, the model classification accuracy decreased and negative comment sentiment was reduced from being an important feature. As only unbanned users are being classified as non-scammers, there is a chance that there are scammers being mixed into the non-scammer pool. P-value still shows that banned users still have increased negative sentiment compared to regular users, but not enough to identify scammers by their comment vocabulary. Negative comment sentiment can still be expanded on, as a true toxic comment is currently grouped in the same category as a comment showing disagreement. Lots of tuning can still be done to this baseline classification model to improve accuracy. 

## Directory

* **my_app/** 
  * api.py: Flask app for the Redditor Scammer Detector
  * ensemble.py - Includes class object that can load dataframe of processed Reddit user information into and can train/test split df; voting classifier ready for use (models used include XGBoost, LightGBM, Random Forest, Multinomial Naive Bayes)
  * user.py - Collects and processes Reddit user data for ensemble.py to predict on

* **jupyter/** 
  * Unorganized jupyter notebook files used during EDA and model training

* **presentation.ipynb** 
  * A summarized and organized jupyter notebook file of the week's work


## Models to Load into Flask App

| Model      | Description |
|------------|-------------|
|**[watchex](https://reddit-scammer-detector.s3-us-west-1.amazonaws.com/watchex_eclf.pkl)** | Model trained with r/watchexchange users + [USL](universalscammerlist.com) banned users |
|**[all](https://reddit-scammer-detector.s3-us-west-1.amazonaws.com/eclf.pkl)**     | Model trained with 50 users from all 60+ subreddits participating in USL |