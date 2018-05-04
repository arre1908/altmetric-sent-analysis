import json
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

""" PSEUDOCODE:
    create variables
    iterate through a root directory
        for each json file found
            load json
            skip if altmetric score < 10
            try to get twitter data
                for each twitter post
                    skip tweets already processed (retweets)
                    skip tweets not in english
                    remove title from tweet text and apply sentiment analysis
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
            find overall average sentiment score and store it
            store altmetric score
    build charts: sentiment score histogram
    build charts: sentiment vs. altmetric score scatter plot
    build chart: platform comparison
    Find correlation between sentiment score and altmetric score
"""
total_sent_scores = []
total_paper_avg_sent_scores = []
total_paper_altmetric_scores = []

all_twitter_sent_scores = []
all_twitter_avg_scores =[]
all_twitter_altmetric_scores = []
tCount = 0

all_facebook_sent_scores = []
all_facebook_avg_scores = []
all_facebook_altmetric_scores = []
fCount = 0

all_blogs_sent_scores = []
all_blogs_avg_scores = []
all_blogs_altmetric_scores = []
bCount = 0

all_googleplus_sent_scores = []
all_googleplus_avg_scores = []
all_googleplus_altmetric_scores = []
gCount = 0

pCount = 0
sa = SentimentIntensityAnalyzer()
rootdir = "some/path/to/json/"

for rootpath, subdirectories, files in os.walk(rootdir):  # iterate through directory
    for filename in files:
        filepath = os.path.join(rootpath, filename)
        jsonfile = json.load(open(filepath))

        if (jsonfile["altmetric_score"]["score"] < 10):  # filter out papers with score < 10
            continue
        paper_total_sent_scores = []
        
        # get Twitter data
        try:
            paper_twitter_sent_scores = []
            retweets = []
            tAnalyzed = 0
            for t in jsonfile["posts"]["twitter"]:
                if t["summary"] in retweets:  # skip retweet
                    continue
                retweets.append(t["summary"])

                try:
                    if (detect(t["summary"]) != "en"):  # skip non-english text
                        continue
                except LangDetectException as le:
                    continue

                tCount += 1  # running count of ALL tweets
                tAnalyzed += 1  # running count of tweets analyzed for the current json
                saScores = sa.polarity_scores(t["summary"].replace(jsonfile["citation"]["title"], ""))  # exclude title and analyze
                paper_twitter_sent_scores.append(saScores["compound"])  # add to list of twitter sentiment scores for the current JSON
                all_twitter_sent_scores.append(saScores["compound"])  # add to the list of twitter sentiment scores
                total_sent_scores.append(saScores["compound"])  # add to the list of all sentiment scores

            if (tAnalyzed > 0):
                avgscore = sum(paper_twitter_sent_scores) / len(paper_twitter_sent_scores)  # find average twitter sentiment score in for the current JSON
                all_twitter_avg_scores.append(avgscore)
                all_twitter_altmetric_scores.append(jsonfile["altmetric_score"]["score"])
                paper_total_sent_scores.extend(paper_twitter_sent_scores)
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
                    if (detect(f["summary"]) != "en"):  # skip non-english text
                        continue
                except LangDetectException as le:
                    continue
                
                fCount += 1  # running count of ALL facebook posts
                fAnalyzed += 1  # running count of facebook posts analyzed for the current json
                saScores = sa.polarity_scores(f["summary"].replace(jsonfile["citation"]["title"], ""))  # exclude title and analyze
                paper_facebook_sent_scores.append(saScores["compound"])  # add to list of facebook sentiment scores for the current JSON
                all_facebook_sent_scores.append(saScores["compound"])  # add to the list of facebook sentiment scores
                total_sent_scores.append(saScores["compound"])  # add to the list of all sentiment scores
            
            if (fAnalyzed > 0):
                avgScore = sum(paper_facebook_sent_scores) / len(paper_facebook_sent_scores)  # find average facebook sentiment score in for the current JSON
                all_facebook_avg_scores.append(avgScore) 
                all_facebook_altmetric_scores.append(jsonfile["altmetric_score"]["score"])
                paper_total_sent_scores.extend(paper_facebook_sent_scores)
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
                    if (detect(b["summary"]) != "en"):  # skip non-english text
                        continue
                except LangDetectException as le:
                    continue

                bCount += 1  # running count of ALL blog posts
                bAnalyzed += 1  # running count of blog posts analyzed for the current json
                saScores = sa.polarity_scores(b["summary"].replace(jsonfile["citation"]["title"], ""))  # exclude title and analyze
                paper_blogs_sent_scores.append(saScores["compound"])  # add to list of blog sentiment scores for the current JSON
                all_blogs_sent_scores.append(saScores["compound"])  # add to the list of blog sentiment scores
                total_sent_scores.append(saScores["compound"])  # add to the list of all sentiment scores
            
            if (bAnalyzed > 0):
                avgScore = sum(paper_blogs_sent_scores) / len(paper_blogs_sent_scores)  # find average blog sentiment score in for the current JSON
                all_blogs_avg_scores.append(avgScore)
                all_blogs_altmetric_scores.append(jsonfile["altmetric_score"]["score"])
                paper_total_sent_scores.extend(paper_blogs_sent_scores)
        except KeyError as ke:
            pass
        
        # get Google Plus data
        try:
            paper_googleplus_sent_scores = []
            gRepost = []
            gAnalyzed = 0
            for g in jsonfile["posts"]["googleplus"]:
                try:
                    if g["summary"] in gRepost:  # skip reposts
                        continue
                    gRepost.append(g["summary"])
                except KeyError as ke2:  # skip googleplus posts w/o summary
                    continue
                
                try:
                    if (detect(g["summary"]) != "en"):  # skip non-english text
                        continue
                except LangDetectException as le:
                    continue
                
                gCount += 1
                gAnalyzed += 1
                saScores = sa.polarity_scores(g["summary"].replace(jsonfile["citation"]["title"], ""))  # exclude title and analyze
                paper_googleplus_sent_scores.append(saScores["compound"])  # add to list of googleplus sentiment scores for the current JSON
                all_googleplus_sent_scores.append(saScores["compound"])  # add to the list of googeplus sentiment scores
                total_sent_scores.append(saScores["compound"])  # add to the list of all sentiment scores

            if (gAnalyzed > 0):
                avgScore = sum(paper_googleplus_sent_scores) / len(paper_googleplus_sent_scores)  # find average googleplus sentiment score in for the current JSON
                all_googleplus_avg_scores.append(avgScore)
                all_googleplus_altmetric_scores.append(jsonfile["altmetric_score"]["score"])
                paper_total_sent_scores.extend(paper_googleplus_sent_scores)
        except KeyError as ke:
            pass
        if (tAnalyzed + fAnalyzed + bAnalyzed + gAnalyzed > 0):  # if posts were processed
            pCount += 1
            paperAvgSentiment = sum(paper_total_sent_scores) / len(paper_total_sent_scores)  # find overall average sentiment score in for the current JSON
            total_paper_avg_sent_scores.append(paperAvgSentiment)
            total_paper_altmetric_scores.append(jsonfile["altmetric_score"]["score"])

print("papers:", pCount)
print("twitter posts:", tCount)
print("facebook posts: ", fCount)
print("blog posts: ", bCount)
print("googleplus posts: ", gCount)
print("total posts: ", tCount + fCount + bCount + gCount)

## HISTOGRAM PLOTS

num_bins = np.linspace(-1, 1, 50)
range_bins = [-1, -0.4, 0.4, 1]  # sentiment ranges for bins

n, bins, patches = plt.hist(all_twitter_sent_scores, num_bins, facecolor='darkturquoise', alpha=0.5)  # twitter
plt.xlim(-1, 1)
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.title("Histogram of Twitter Sentiment Scores")
plt.show()

n, bins, patches = plt.hist(all_facebook_sent_scores, num_bins, facecolor='blue', alpha=0.5)  # facebook
plt.xlim(-1, 1)
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.title("Histogram of Facebook Sentiment Scores")
plt.show()

n, bins, patches = plt.hist(all_blogs_sent_scores, num_bins, facecolor='purple', alpha=0.5)  # blogs
plt.xlim(-1, 1)
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.title("Histogram of Blog Sentiment Scores")
plt.show()

n, bins, patches = plt.hist(all_googleplus_sent_scores, num_bins, facecolor='orange', alpha=0.5)  # google+
plt.xlim(-1, 1)
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.title("Histogram of Google+ Sentiment Scores")
plt.show()

n, bins, patches = plt.hist(total_sent_scores, num_bins, facecolor='red', alpha=0.5)  # all
plt.xlim(-1, 1)
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.title("Histogram of ALL Sentiment Scores")
plt.show()

## SCATTER PLOTS

plt.scatter(all_twitter_avg_scores, all_twitter_altmetric_scores, facecolor='darkturquoise', alpha=0.5)  # twitter
plt.xlim(-1, 1)
plt.xlabel("Sentiment Score")
plt.ylabel("Altmetric Score")
plt.title("Twitter: Sentiment Score (Average) vs. Altmetric Score")
plt.show()

plt.scatter(all_facebook_avg_scores, all_facebook_altmetric_scores, facecolor='blue', alpha=0.5)  # facebook
plt.xlim(-1, 1)
plt.xlabel("Sentiment Score")
plt.ylabel("Altmetric Score")
plt.title("Facebook: Sentiment Score (Average) vs. Altmetric Score")
plt.show()

plt.scatter(all_blogs_avg_scores, all_blogs_altmetric_scores, facecolor='purple', alpha=0.5)  # blogs
plt.xlim(-1, 1)
plt.xlabel("Sentiment Score")
plt.ylabel("Altmetric Score")
plt.title("Blogs: Sentiment Score (Average) vs. Altmetric Score")
plt.show()

plt.scatter(all_googleplus_avg_scores, all_googleplus_altmetric_scores, facecolor='orange', alpha=0.5)  # google+
plt.xlim(-1, 1)
plt.xlabel("Sentiment Score")
plt.ylabel("Altmetric Score")
plt.title("Google+: Sentiment Score (Average) vs. Altmetric Score")
plt.show()

plt.scatter(total_paper_avg_sent_scores, total_paper_altmetric_scores, facecolor='red', alpha=0.5)  # all
plt.xlim(-1, 1)
plt.xlabel("Sentiment Score")
plt.ylabel("Altmetric Score")
plt.title("Paper Sentiment Score (Average) vs. Altmetric Score")
plt.show()

## COMPARISON CHART

positive = [0, 0, 0, 0]
neutral = [0, 0, 0, 0]
negative = [0, 0, 0, 0]

# split lists sentiments for each platform into three arrays (negative, neutral, positive)
for t in all_twitter_sent_scores:
    if (t <= -0.4):
        negative[0] += 1
    elif (t <= 0.4):
        neutral[0] += 1
    elif (t <= 1):
        positive[0] += 1
for f in all_facebook_sent_scores:
    if (f <= -0.4):
        negative[1] += 1
    elif (f <= 0.4):
        neutral[1] += 1
    elif (f <= 1):
        positive[1] += 1
for g in all_googleplus_sent_scores:
    if (g <= -0.4):
        negative[2] += 1
    elif (g <= 0.4):
        neutral[2] += 1
    elif (g <= 1):
        positive[2] += 1
for b in all_blogs_sent_scores:
    if (b <= -0.4):
        negative[3] += 1
    elif (b <= 0.4):
        neutral[3] += 1
    elif (b <= 1):
        positive[3] += 1
print("tw: ", positive[0], neutral[0], negative[0], positive[0] + neutral[0] + negative[0])
print("fb: ", positive[1], neutral[1], negative[1], positive[1] + neutral[1] + negative[1])
print("g+: ", positive[2], neutral[2], negative[2], positive[2] + neutral[2] + negative[2])
print("bl: ", positive[3], neutral[3], negative[3], positive[3] + neutral[3] + negative[3])

# turn raw numbers into percentages
positive[0] /= tCount / 100
neutral[0] /= tCount / 100
negative[0] /= tCount / 100
positive[1] /= fCount / 100
neutral[1] /= fCount / 100
negative[1] /= fCount / 100
positive[2] /= gCount / 100
neutral[2] /= gCount / 100
negative[2] /= gCount / 100
positive[3] /= bCount / 100
neutral[3] /= bCount / 100
negative[3] /= bCount / 100
print("tw: ", positive[0], neutral[0], negative[0], positive[0] + neutral[0] + negative[0])
print("fb: ", positive[1], neutral[1], negative[1], positive[1] + neutral[1] + negative[1])
print("g+: ", positive[2], neutral[2], negative[2], positive[2] + neutral[2] + negative[2])
print("bl: ", positive[3], neutral[3], negative[3], positive[3] + neutral[3] + negative[3])

labels = ('Twitter', 'Facebook', 'Google+', 'Blogs')
index = np.arange(4)

# create graph
barWidth = 0.25
index1 = np.arange(4)
index2 = [x + barWidth for x in index1]
index3 = [x + barWidth for x in index2]
plt.bar(index1, neutral, width=barWidth, facecolor='silver', edgecolor='white', alpha=0.5, label='neutral')
plt.bar(index2, positive, width=barWidth, facecolor='green', edgecolor='white', alpha=0.5, label='positive')
plt.bar(index3, negative, width=barWidth, facecolor='red', edgecolor='white', alpha=0.5, label='negative')
plt.xlabel("Platform")
plt.ylabel("% of Posts")
plt.title("Comparison of Platform Sentiments")
plt.xticks(index2, labels)
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.legend()
plt.show()

## CORRELATION

# create DataFrames to store values
tf = pd.DataFrame({
    "all_twitter_avg_scores": all_twitter_avg_scores,
    "all_twitter_altmetric_scores": all_twitter_altmetric_scores
    })
ff = pd.DataFrame({
    "all_facebook_avg_scores": all_facebook_avg_scores,
    "all_facebook_altmetric_scores": all_facebook_altmetric_scores
    })
bf = pd.DataFrame({
    "all_blogs_avg_scores": all_blogs_avg_scores,
    "all_blogs_altmetric_scores": all_blogs_altmetric_scores
    })
gf = pd.DataFrame({
    "all_googleplus_avg_scores": all_googleplus_avg_scores,
    "all_googleplus_altmetric_scores": all_googleplus_altmetric_scores
    })
af = pd.DataFrame({
    "total_paper_avg_sent_scores": total_paper_avg_sent_scores,
    "total_paper_altmetric_scores": total_paper_altmetric_scores
})

#print correlation results
print("CORR: Twitter sent vs altmetric score - ", tf["all_twitter_avg_scores"].corr(tf["all_twitter_altmetric_scores"]))
print("CORR: Facebook sent vs altmetric score - ", ff["all_facebook_avg_scores"].corr(ff["all_facebook_altmetric_scores"]))
print("CORR: Google+ sent vs altmetric score - ", gf["all_googleplus_avg_scores"].corr(gf["all_googleplus_altmetric_scores"]))
print("CORR: Blogs sent vs altmetric score - ", bf["all_blogs_avg_scores"].corr(bf["all_blogs_altmetric_scores"]))
print("CORR: Paper sent vs altmetric score - ", af["total_paper_avg_sent_scores"].corr(af["total_paper_altmetric_scores"]))
