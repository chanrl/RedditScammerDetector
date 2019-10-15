import praw
import time
from datetime import datetime
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import defaultdict, Counter
import statistics as stats

reddit = praw.Reddit(user_agent='Comment History Parser',
                    client_id='nkVxbwp1RsHHCA',
                     client_secret='SlzWUhAhV5nIXPy4_1PTJSOaLrA')

def fetch_comments_id(user, limit=1000):
    all_comments = list(user.comments.controversial(limit=limit))
    for comment in user.comments.hot(limit=limit):
        if comment not in all_comments:
            all_comments.append(comment)
    for comment in user.comments.new(limit=limit):
        if comment not in all_comments:
            all_comments.append(comment)
    return all_comments

def retrieve_text(user, limit=1000):
  all_text = [comment.body for comment in fetch_comments_id(user, limit=limit)]
  return all_text

def get_user_details(user):
  #self-explanatory
  try:
    link_karma = user.link_karma
  except:
    return -1
  comment_karma = user.comment_karma
  verified = user.has_verified_email
  mod = user.is_mod
  gold = user.is_gold
  days_old =(datetime.fromtimestamp(1571093268) - datetime.fromtimestamp(user.created_utc)).days
  return link_karma, comment_karma, verified, mod, gold, days_old

def populate_df(df):
    df['details'] = df['users'].map(get_user_details)
    df = df[df['details'] != -1]
    df['link_karma'] = df['details'].map(lambda x : x[0])
    df['comment_karma'] = df['details'].map(lambda x : x[1])
    df['verified'] = df['details'].map(lambda x : x[2])
    df['mod'] = df['details'].map(lambda x : x[3])
    df['gold'] = df['details'].map(lambda x : x[4])
    df['days_old'] = df['details'].map(lambda x : x[5])
    return df

def apply_vader(comments):
    sid = SentimentIntensityAnalyzer()
    scores = defaultdict(int)
    for comment in comments:
        if sid.polarity_scores(comment)['compound'] >= 0.05:
            scores['positive'] += 1
        elif sid.polarity_scores(comment)['compound'] > -0.05 and sid.polarity_scores(comment)['compound'] < 0.05:
            scores['neutral'] += 1
        elif sid.polarity_scores(comment)['compound'] <= -0.05:
            scores['negative'] += 1
        else:
            scores['somethingwrong'] += 1
    return scores

def add_features(df):
  df['comments'] = df['users'].map(retrieve_text)
  df['total_comments'] = df.comments.map(lambda x: len(x))
  df['polarity'] = df['comments'].map(apply_vader)
  df['positive'] = df['polarity'].map(lambda x: x['positive'])/df['total_comments']
  df['neutral'] = df['polarity'].map(lambda x: x['neutral'])/df['total_comments']
  df['negative'] = df['polarity'].map(lambda x: x['negative'])/df['total_comments']
  return df

def more_features(df):
  df['len_cs'] = df['comments'].map(lambda x: [len(comment) for comment in x])
  df['mean_comment_length'] = df['len_cs'].map(lambda x: stats.mean(x))
  df['mode_comment_length'] = df['len_cs'].map(lambda x: Counter(x).most_common()[0][0])
  df['median_comment_length'] = df['len_cs'].map(lambda x: stats.median(x))
  df['duplicate_comments'] = df['comments'].map(lambda x: len(x) - len(set(x)))
  return df

if __name__ == "__main__":
  sub_name = 'watchexchange'

  subreddit = reddit.subreddit(sub_name)
  #extract as much users as possible in subreddit
  #hot, new, controversial, rising

  #start by collecting all submissions in subreddit

  submissions_id = []
  for submission in reddit.subreddit(sub_name).hot(limit=1000):
    if submission not in submissions_id:
      submissions_id.append(submission)
  for submission in reddit.subreddit(sub_name).rising(limit=1000):
    if submission not in submissions_id:
      submissions_id.append(submission)
  for submission in reddit.subreddit(sub_name).controversial(limit=1000):
    if submission not in submissions_id:
      submissions_id.append(submission)
  for submission in reddit.subreddit(sub_name).new(limit=1000):
    if submission not in submissions_id:
      submissions_id.append(submission)

  #initial scrape, authors of submissions
  users = []
  for submission in submissions_id:
    if submission.author not in users:
      users.append(submission.author)
  
  #secondary scrape, iterate through submission comment forest and extract all users inside

  for submission in submissions_id:
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        if comment.author not in users:
            users.append(comment.author)

  users.remove('AutoModerator')
  
  df = pd.DataFrame(users[500:], columns=['users'])
  df = populate_df(df)
  sid = SentimentIntensityAnalyzer()
  df = add_features(df)
  df = df.drop(columns = ['details','polarity'])
  #maybe this will be useful later
  no_comments = df[df.total_comments == 0]
  df = df[df.total_comments != 0]
  df = more_features(df)
  df.to_csv('data/df_watchexchange_2.csv', index=False)