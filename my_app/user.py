import praw
import time
from datetime import datetime
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import defaultdict, Counter
import statistics as stats
import language_check
from joblib import load

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

def cap_check(row):
  caps = []
  for comment in row:
    if len(comment) == 0:
      pass
    else:
      c = Counter("upper" if x.isupper() else "rest" for x in comment)
      caps.append(c['upper']/(c['rest']+c['upper']))
  return caps

def grammar_check(row):
  tool = language_check.LanguageTool('en-US')
  errors = []
  for comment in row:
      errors.append(len(tool.check(comment.replace('\n', ' '))))
  return (np.average(errors), np.sum(errors))

def grammar_feats(df):
  df['grammar'] = df['comments'].map(grammar_check)
  df['cap_freq'] = df['comments'].map(cap_check)
  df['avg_grammar'] = df['grammar'].map(lambda x: x[0])
  df['total_grammar'] = df['grammar'].map(lambda x: x[1])
  df['cap_freq_mean'] = df['cap_freq'].map(lambda x: np.mean(x))
  return df

def get_user_profile(user_input):
  reddit = praw.Reddit(user_agent='Comment History Parser',
                    client_id='nkVxbwp1RsHHCA',
                     client_secret='SlzWUhAhV5nIXPy4_1PTJSOaLrA')
  user = reddit.redditor(user_input)
  users = [user]
  u = pd.DataFrame(users, columns=['users'])
  u = populate_df(u)
  u = add_features(u)
  u = u.drop(columns = ['details','polarity'])
  u = more_features(u)
  u['comments_new'] = u['comments'].map(lambda x: " ".join(x))
  u = grammar_feats(u)
  if u.verified[0] != bool:
    u.verified = u.verified.fillna(True)
  return u

if __name__ == "__main__":

  # user_input = input('Username Here: ')
  # u = get_user_profile(user_input)

  # clf = load('watch_clf.joblib')
  # clf.predict(u)
  pass