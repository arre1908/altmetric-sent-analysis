import json
import os
from pprint import pprint
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

""" Need to store JSON files into a list or dicts that can be used to build a chart
    PSEUDOCODE:
    create data list
    iterate through a root directory
        for each json file found
            load json
            skip if altmetric score < 100
            try to get twitter data
                create dict of useful values from json
                skip tweets already processed (retweets)
                remove title from post summary
                apply sentiment analysis and store it
                append dict to data list
            try to get facebook data
                blah
            try to get googleplus data
                blah
            try to get blog data
                blah
    build chart: sentiment score histogram
    build chart: platform comparison
"""
sa = SentimentIntensityAnalyzer()
countries = ["US", "AU", "UK", "NZ", "IE", "JM", "BS", "GD"]  # english-speaking countries
jsonData = []
total_sent_scores = []
total_avg_sent_scores = []
total_post_counts = []

all_twitter_sent_scores = []
all_twitter_avg_scores =[]
all_tweet_counts = []
tCount = 0

all_facebook_sent_scores = []
all_facebook_avg_scores = []
all_facebook_counts = []
fCount = 0

all_blog_sent_scores = []
all_blog_avg_scores = []
all_blog_counts = []
bCount = 0

all_googleplus_sent_scores = []
all_googleplus_avg_scores = []
all_googleplus_counts = []
gCount = 0
rootdir = "altmetric_data/"

for rootpath, subdirectories, files in os.walk(rootdir):
    for filename in files:
        filepath = os.path.join(rootpath, filename)
        jsonfile = json.load(open(filepath))
        if (jsonfile["altmetric_score"]["score"] < 100):  # filter out papers with score < 100
            continue
        """ tempDict = {
            "altmetric_id": jsonfile["altmetric_id"],
            "title": jsonfile["citation"]["title"],
            "altmetric_score": jsonfile["altmetric_score"]["score"]
            } """
        """ try:
            tUsers = jsonfile["counts"]["twitter"]["unique_users_count"]
        except KeyError as ke:
            tUsers = 0
            tempDict["counts"] = {
                "twitter": 0
                "tweet_sentiment": 0
            } """
        
        try:
            # tw = []
            retweets = []
            tAnalyzed = 0
            for t in jsonfile["posts"]["twitter"]:
                """ try:
                    if t["author"]["geo"]["country"] not in countries:  # exclude non-english-speaking country
                        # print("Non-US", t["author"]["geo"]["country"])
                        continue
                except KeyError as ke:  # exclude tweets with no country listed
                    # print("Non-US")
                    continue """

                if t["summary"] in retweets:  # exclude retweet
                    continue
                retweets.append(t["summary"])

                try:
                    if (detect(t["summary"]) != "en"):
                        continue
                except LangDetectException as le:
                    continue

                tCount += 1  # running count of ALL tweets
                tAnalyzed += 1  # running count of tweets analyzed for the current json
                saScores = sa.polarity_scores(t["summary"].replace(jsonfile["citation"]["title"], ""))  # exclude title and analyze
                all_twitter_sent_scores.append(saScores["compound"])
                total_sent_scores.append(saScores["compound"])
                """ tw.append({
                    "summary": t["summary"],
                    "tweet_id": t["tweet_id"],
                    "sent_score": saScores["compound"]
                }) """
        except KeyError as ke:
            pass
        if (tAnalyzed > 0):
            avgscore = sum(all_twitter_sent_scores) / len(all_twitter_sent_scores)  # find average sentiment score in for the current JSON
            total_avg_sent_scores.append(avgscore)
            total_post_counts.append(jsonfile["counts"]["twitter"]["unique_users_count"])
            """ tempDict["counts"] = {
                "tweets": twCount,
                "avg_tweet_sentiment": avgscore
                }
            tempDict["twitter"] = tw
        jsonData.append(tempDict) """

        try:
            fRepost = []
            fAnalyzed = 0
            for f in jsonfile["posts"]["facebook"]:
                try:
                    if f["summary"] in fRepost:
                        continue
                    fRepost.append(f["summary"])
                except KeyError as ke2:  # skip facebook posts w/o summary
                    continue
                
                fCount += 1  # running count of ALL facebook posts
                fAnalyzed += 1  # running count of facebook posts analyzed for the current json
                # saScores = sa.polarity_scores(f["summary"].replace(jsonfile["citation"]["title"], ""))  # exclude title and analyze
                # all_facebook_sent_scores.append(saScores["compound"])
                # total_sent_scores.append(saScores["compound"])

        except KeyError as ke:
            pass
        
        try:
            bReposts = []
            bAnalyzed = 0
            for b in jsonfile["posts"]["blogs"]:
                try:
                    if b["summary"] in bReposts:
                        continue
                    bReposts.append(b["summary"])
                except KeyError as ke2:
                    continue

                bCount += 1
                bAnalyzed += 1
        except KeyError as ke:
            pass
            
        try:
            gRepost = []
            gAnalyzed = 0
            for g in jsonfile["posts"]["googleplus"]:
                try:
                    if g["summary"] in gRepost:
                        continue
                    gRepost.append(g["summary"])
                except KeyError as ke2:
                    continue
                
                gCount += 1
                gAnalyzed += 1
        except KeyError as ke:
            pass

# pprint(sorted(jsonData, key=lambda k: k["score"], reverse=True))  # sorted by altmetric score
# pprint(jsonData)
print("twitter posts:", tCount)
# print("papers:", len(jsonData))
print("papers: ", len(total_avg_sent_scores), len(total_post_counts))
print("facebook posts: ", fCount)
print("blog posts: ", bCount)
print("googleplus posts: ", gCount)

## PLOTS

num_bins = 50
# num_bins = [-1, -0.3, 0.3, 1]  # preset ranges for bins
n, bins, patches = plt.hist(total_sent_scores, num_bins, facecolor='blue', alpha=0.5)
plt.xlim(-1, 1)
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.title("Histogram of Twitter Sentiment Scores")
plt.show()
# print(n)  # number of frequency in each category

plt.scatter(total_avg_sent_scores, total_post_counts)
plt.xlim(-1, 1)
plt.xlabel("Sentiment Score")
plt.ylabel("Number of Tweets")
plt.title("Twitter Sentiment Scores vs. Tweet Count")
plt.show()