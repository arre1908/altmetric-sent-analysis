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
    create variables
    iterate through a root directory
        for each json file found
            load json
            skip if altmetric score < 100
            try to get twitter data
                for each twitter post
                    skip tweets already processed (retweets)
                    skip tweets not in english
                    remove title from post summary and apply sentiment analysis
                    append result to lists
                find average sentiment score for current JSON file and store it
                store posts count for JSON
            try to get facebook data
                (repeat process)
                ...
            try to get googleplus data
                (repeat process)
                ...
            try to get blog data
                (repeat process)
                ...
    build charts: sentiment score histogram
    build charts: scatter plot sentiment vs. post count
    build chart: platform comparison
"""
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

all_blogs_sent_scores = []
all_blogs_avg_scores = []
all_blogs_counts = []
bCount = 0

all_googleplus_sent_scores = []
all_googleplus_avg_scores = []
all_googleplus_counts = []
gCount = 0

pCount = 0
sa = SentimentIntensityAnalyzer()
rootdir = "some/folder/path"

for rootpath, subdirectories, files in os.walk(rootdir):
    for filename in files:
        filepath = os.path.join(rootpath, filename)
        jsonfile = json.load(open(filepath))

        if (jsonfile["altmetric_score"]["score"] < 100):  # filter out papers with score < 100
            continue
        pCount += 1
        
        # get Twitter data
        try:
            # tw = []
            paper_twitter_sent_scores = []
            retweets = []
            tAnalyzed = 0
            for t in jsonfile["posts"]["twitter"]:
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
                paper_twitter_sent_scores.append(saScores["compound"])  # add to list of sentiment scores for the current JSON
                all_twitter_sent_scores.append(saScores["compound"])  # add to the list of all twitter sentiment scores
                total_sent_scores.append(saScores["compound"])  # add to the list of all sentiment scores

            if (tAnalyzed > 0):
                avgscore = sum(paper_twitter_sent_scores) / len(paper_twitter_sent_scores)  # find average sentiment score in for the current JSON
                total_avg_sent_scores.append(avgscore)
                all_twitter_avg_scores.append(avgscore)
                total_post_counts.append(jsonfile["counts"]["twitter"]["posts_count"])
                all_tweet_counts.append(jsonfile["counts"]["twitter"]["posts_count"])
        except KeyError as ke:
            pass

        # get Facebook data
        try:
            paper_facebook_sent_scores = []
            fRepost = []
            fAnalyzed = 0
            for f in jsonfile["posts"]["facebook"]:
                try:
                    if f["summary"] in fRepost:  #skip reposts
                        continue
                    fRepost.append(f["summary"])
                except KeyError as ke2:  # skip facebook posts w/o summary
                    continue

                try:
                    if (detect(f["summary"]) != "en"):
                        continue
                except LangDetectException as le:
                    continue
                
                fCount += 1  # running count of ALL facebook posts
                fAnalyzed += 1  # running count of facebook posts analyzed for the current json
                saScores = sa.polarity_scores(f["summary"].replace(jsonfile["citation"]["title"], ""))  # exclude title and analyze
                paper_facebook_sent_scores.append(saScores["compound"])
                all_facebook_sent_scores.append(saScores["compound"])
                total_sent_scores.append(saScores["compound"])
            
            if (fAnalyzed > 0):
                avgScore = sum(paper_facebook_sent_scores) / len(paper_facebook_sent_scores)
                all_facebook_avg_scores.append(avgScore)
                total_avg_sent_scores.append(avgScore)
                all_facebook_counts.append(jsonfile["counts"]["facebook"]["posts_count"])
                total_post_counts.append(jsonfile["counts"]["facebook"]["posts_count"])
        except KeyError as ke:
            pass
        
        # get blogs data
        try:
            paper_blogs_sent_scores = []
            bReposts = []
            bAnalyzed = 0
            for b in jsonfile["posts"]["blogs"]:
                try:
                    if b["summary"] in bReposts:  # skip reposts
                        continue
                    bReposts.append(b["summary"])
                except KeyError as ke2:  # skip blogs w/o summary
                    continue

                try:
                    if (detect(b["summary"]) != "en"):
                        continue
                except LangDetectException as le:
                    continue

                bCount += 1  # running count of ALL blog posts
                bAnalyzed += 1  # running count of blog posts analyzed for the current json
                saScores = sa.polarity_scores(b["summary"].replace(jsonfile["citation"]["title"], ""))  # exclude title and analyze
                paper_blogs_sent_scores.append(saScores["compound"])
                all_blogs_sent_scores.append(saScores["compound"])
                total_sent_scores.append(saScores["compound"])
            
            if (bAnalyzed > 0):
                avgScore = sum(paper_blogs_sent_scores) / len(paper_blogs_sent_scores)
                all_blogs_avg_scores.append(avgScore)
                total_avg_sent_scores.append(avgScore)
                all_blogs_counts.append(jsonfile["counts"]["blogs"]["posts_count"])
                total_post_counts.append(jsonfile["counts"]["blogs"]["posts_count"])
        except KeyError as ke:
            pass
        
        # get Google Plus data
        try:
            paper_googleplus_sent_scores = []
            gRepost = []
            gAnalyzed = 0
            for g in jsonfile["posts"]["googleplus"]:
                try:
                    if g["summary"] in gRepost:
                        continue
                    gRepost.append(g["summary"])
                except KeyError as ke2:
                    continue
                
                try:
                    if (detect(g["summary"]) != "en"):
                        continue
                except LangDetectException as le:
                    continue
                
                gCount += 1
                gAnalyzed += 1
                saScores = sa.polarity_scores(g["summary"].replace(jsonfile["citation"]["title"], ""))  # exclude title and analyze
                paper_googleplus_sent_scores.append(saScores["compound"])
                all_googleplus_sent_scores.append(saScores["compound"])
                total_sent_scores.append(saScores["compound"])

            if (gAnalyzed > 0):
                avgScore = sum(paper_googleplus_sent_scores) / len(paper_googleplus_sent_scores)
                all_googleplus_avg_scores.append(avgScore)
                total_avg_sent_scores.append(avgScore)
                all_googleplus_counts.append(jsonfile["counts"]["googleplus"]["posts_count"])
                total_post_counts.append(jsonfile["counts"]["googleplus"]["posts_count"])
        except KeyError as ke:
            pass

print("papers:", pCount)
print("sources: ", len(total_avg_sent_scores), len(total_post_counts))
print("twitter posts:", tCount)
print("facebook posts: ", fCount)
print("blog posts: ", bCount)
print("googleplus posts: ", gCount)

## PLOTS

num_bins = 50
range_bins = [-1, -0.3, 0.3, 1]  # preset ranges for bins

n, bins, patches = plt.hist(all_twitter_sent_scores, num_bins, facecolor='darkturquoise', alpha=0.5)
plt.xlim(-1, 1)
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.title("Histogram of Twitter Sentiment Scores")
plt.show()
# print(n)  # number of frequency in each categorynum_bins = 50

n, bins, patches = plt.hist(all_facebook_sent_scores, num_bins, facecolor='blue', alpha=0.5)
plt.xlim(-1, 1)
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.title("Histogram of Facebook Sentiment Scores")
plt.show()
# print(n)  # number of frequency in each category

n, bins, patches = plt.hist(all_blogs_sent_scores, num_bins, facecolor='purple', alpha=0.5)
plt.xlim(-1, 1)
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.title("Histogram of Blog Sentiment Scores")
plt.show()
# print(n)  # number of frequency in each category

n, bins, patches = plt.hist(all_googleplus_sent_scores, num_bins, facecolor='orange', alpha=0.5)
plt.xlim(-1, 1)
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.title("Histogram of Google+ Sentiment Scores")
plt.show()
# print(n)  # number of frequency in each category

n, bins, patches = plt.hist(total_sent_scores, num_bins, facecolor='red', alpha=0.5)
plt.xlim(-1, 1)
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.title("Histogram of ALL Sentiment Scores")
plt.show()
# print(n)  # number of frequency in each category

plt.scatter(all_twitter_avg_scores, all_tweet_counts, facecolor='darkturquoise', alpha=0.5)
plt.xlim(-1, 1)
plt.xlabel("Sentiment Score")
plt.ylabel("Number of Posts")
plt.title("Twitter: Sentiment Scores vs. Tweet Count")
plt.show()

plt.scatter(all_facebook_avg_scores, all_facebook_counts, facecolor='blue', alpha=0.5)
plt.xlim(-1, 1)
plt.xlabel("Sentiment Score")
plt.ylabel("Number of Posts")
plt.title("Facebook: Sentiment Scores vs. Post Count")
plt.show()

plt.scatter(all_blogs_avg_scores, all_blogs_counts, facecolor='purple', alpha=0.5)
plt.xlim(-1, 1)
plt.xlabel("Sentiment Score")
plt.ylabel("Number of Posts")
plt.title("Blogs: Sentiment Scores vs. Post Count")
plt.show()

plt.scatter(all_googleplus_avg_scores, all_googleplus_counts, facecolor='orange', alpha=0.5)
plt.xlim(-1, 1)
plt.xlabel("Sentiment Score")
plt.ylabel("Number of Posts")
plt.title("Google+: Sentiment Scores vs. Post Count")
plt.show()

plt.scatter(total_avg_sent_scores, total_post_counts, facecolor='red', alpha=0.5)
plt.xlim(-1, 1)
plt.xlabel("Sentiment Score")
plt.ylabel("Number of Posts")
plt.title("Total: Sentiment Scores vs. Post Count")
plt.show()