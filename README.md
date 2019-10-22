# RedditScammerDetector

Flask app for a Reddit scammer detector. Input a Reddit username to return a probability of that user being a scammer.

## Summary

Check out my [Medium post IN PROGRESS](https://medium.com/p/1bdc024c8d69/) for a detailed analysis of my investigation and findings!
Initial results showed promise, but model accuracy needs to increase before public use.

## Directory

* *my_app/* 
  * Python scripts to scrape user's profile and predict scammer probability
  * Flask app for the Scammer Detector

* *presentation.ipynb* 
  * A summarized and organized jupyter notebook file of the week's work

* *reddit.py*
  * Python script to scrape a subreddit for Reddit usernames, profile information, and comment history

* *jupyter/* 
  * Unorganized jupyter notebook files used during EDA and model training

## Models to Load into Flask App

| Model      | Description |
|------------|-------------|
|[watchex]() | Model trained with watchexchange users + [USL](universalscammerlist.com) banned users |
|[all]()     | Model trained with 50 users from all subreddits participating in USL|
