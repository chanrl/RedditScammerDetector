# RedditScammerDetector

Flask app for a Reddit scammer detector. Input a Reddit username to return a probability of that user being a scammer.

## Summary

Check out my [Medium post IN PROGRESS](https://medium.com/p/1bdc024c8d69/) for a detailed analysis of my investigation and findings!
Initial results showed promise, but model accuracy needs to increase before public use.

## Directory

* **my_app/** 
  * Python scripts to scrape user's profile and predict scammer probability
   * ensemble.py - Class object to load dataframe into and can train/test split df; voting classifier ready for use (models used include XGBoost, LightGBM, Random Forest, Multinomial Naive Bayes)
   * user.py - Collects and processes data for ensemble.py to predict on
  * Flask app for the Scammer Detector

* **presentation.ipynb** 
  * A summarized and organized jupyter notebook file of the week's work

* **reddit.py**
  * Python script for scraping a subreddit for Reddit usernames, profile information, and comment history

* **jupyter/** 
  * Unorganized jupyter notebook files used during EDA and model training

## Models to Load into Flask App

| Model      | Description |
|------------|-------------|
|**[watchex]()** | Model trained with r/watchexchange users + [USL](universalscammerlist.com) banned users |
|**[all]()**     | Model trained with 50 users from all 60+ subreddits participating in USL |
