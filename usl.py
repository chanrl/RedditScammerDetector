import json
import urllib.request
from reddit import *
import praw
import time
from datetime import datetime
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import defaultdict

url = "https://universalscammerlist.com/api/bulk_query.php"
x = urllib.request.urlopen(url)
raw_data = x.read()
encoding = x.info().get_content_charset('utf8')  # JSON default
# print(raw_data)   #this is data in string format
data = json.loads(raw_data.decode(encoding))
# print(data)   #this would be your json data
banned_users = [row['username'] for row in data['data']]

reddit = praw.Reddit(user_agent='Comment History Parser',
                    client_id='nkVxbwp1RsHHCA',
                     client_secret='SlzWUhAhV5nIXPy4_1PTJSOaLrA')

banned_users = [reddit.redditor(user) for user in banned_users]

ban_df = pd.DataFrame(banned_users, columns=['users'])
ban_df = populate_df(ban_df)
ban_df = add_features(ban_df)
ban_df = ban_df.drop(columns = ['details','polarity'])
#if you have no comments how are you scamming?
#maybe this will be useful later
no_comments = ban_df[ban_df.total_comments == 0]
ban_df = ban_df[ban_df.total_comments != 0]
ban_df = more_features(ban_df)
ban_df.to_csv('data/ban_df.csv', index=False)