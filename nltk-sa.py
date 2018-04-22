import json
import os
from pprint import pprint
from nltk.sentiment.vader import SentimentIntensityAnalyzer

"""
from nltk import tokenize
from nltk.corpus import subjectivity, stopwords, sentiwordnet
from twython import Twython
import keys

 # --- BASIC NLTK PROCESSING ---

text = "Here is journal on something. It showcases some interesting blah blah. Read it here."
text_words = tokenize.word_tokenize(text)
custom_sentences = tokenize.sent_tokenize(text)
text_filtered = []
stop_words = set(stopwords.words("english"))
for i in text_words:
    if i not in stop_words:
        text_filtered.append(i)

print("Words:")
print(text_words)
print("\nSentences:")
print(custom_sentences)
print("\nNo stop words:")
print(text_filtered)

print("\nNLTK data: subjectivity")
print(subjectivity.words())


# --- NLTK SENTIMENT ANALYSIS ---

print("\n\n--- Sentiwordnet: ---")
breakdown = sentiwordnet.senti_synset('breakdown.n.03')
print(breakdown)

sentences = [
    "VADER is smart, handsome, and funny.", # punctuation emphasis handled correctly (sentiment intensity adjusted)
    "VADER is very smart, handsome, and funny.",  # booster words handled correctly (sentiment intensity adjusted)
    "VADER is VERY SMART, handsome, and FUNNY.",  # emphasis for ALLCAPS handled
    "VADER is VERY SMART, handsome, and FUNNY!!!",# combination of signals - VADER appropriately adjusts intensity
    "VADER is VERY SMART, really handsome, and INCREDIBLY FUNNY!!!",# booster words & punctuation make this close to ceiling for score
    "The book was good.",         # positive sentence
    "The book was kind of good.", # qualified positive sentence is handled correctly (intensity adjusted)
    "The plot was good, but the characters are uncompelling and the dialog is not great.", # mixed negation sentence
    "A really bad, horrible book.",       # negative sentence with booster words
    "At least it isn't a horrible book.", # negated negative sentence with contraction
    ":) and :D",     # emoticons handled
    "",              # an empty string is correctly handled
    "Today sux",     #  negative slang handled
    "Today sux!",    #  negative slang with punctuation emphasis handled
    "Today SUX!",    #  negative slang with capitalization emphasis
    "Today kinda sux! But I'll get by, lol" # mixed sentiment example with slang and constrastive conjunction "but"
]

paragraph = "It was one of the worst movies I've seen, despite good reviews for Justice League. \
Unbelievably bad acting!! Poor direction. VERY poor production. \
The movie was bad. Very bad movie. VERY bad movie. VERY BAD movie. VERY BAD movie!"
paragraph_sentences = tokenize.sent_tokenize(paragraph)

tricky_sentences = [
    "Most automated sentiment analysis tools are shit.",
    "VADER sentiment analysis is the shit.",
    "Sentiment analysis has never been good.",
    "Sentiment analysis with VADER has never been this good.",
    "Warren Beatty has never been so entertaining.",
    "I won't say that the movie is astounding and I wouldn't claim that the movie is too banal either.",
    "I like to hate Michael Bay films, but I couldn't fault this one",
    "It's one thing to watch an Uwe Boll film, but another thing entirely to pay for it",
    "The movie was too good",
    "This movie was actually neither that funny, nor super witty.",
    "This movie doesn't care about cleverness, wit or any other kind of intelligent humor.",
    "Those who find ugly meanings in beautiful things are corrupt without being charming.",
    "There are slow and repetitive parts, BUT it has just enough spice to keep it interesting.",
    "The script is not fantastic, but the acting is decent and the cinematography is EXCELLENT!",
    "Roger Dodger is one of the most compelling variations on this theme.",
    "Roger Dodger is one of the least compelling variations on this theme.",
    "Roger Dodger is at least compelling as a variation on the theme.",
    "they fall in love with the product",
    "but then it breaks",
    "usually around the time the 90 day warranty expires",
    "the twin towers collapsed today",
    "However, Mr. Carter solemnly argues, his client carried out the kidnapping under orders and in the ''least offensive way possible.''"
]

sentences.extend(paragraph_sentences)
sentences.extend(tricky_sentences)
sentences.extend(custom_sentences)

sid = SentimentIntensityAnalyzer()

print("\n\n--- VADER SentimentIntensityAnalyzer: ---")
for sentence in sentences:
    print(sentence)
    print("  ", end='')
    ss = sid.polarity_scores(sentence) # dict stored in {string: float} format
    for key in sorted(ss):
        print('{0}: {1}, '.format(key, ss[key]), end='')
    print()


# --- ALTMETRIC DATA ---

twitter = Twython(keys.CONSUMER_KEY, keys.CONSUMER_SECRET, keys.OAUTH_TOKEN, keys.OAUTH_TOKEN_SECRET)
filepath = "altmetric_data/2580030.json"
data = json.load(open(filepath))

title = data['citation']['title']
tweet_id = data['posts']['twitter'][12]['tweet_id']
tweet = twitter.show_status(id=tweet_id)['text']

print("\n\n--- Altmetric data: --")
print(tweet)
print("** RESEARCH TITLE: ", title)
print("** TWEET W/O TITLE: ", tweet.replace(title, ''))
ss = SentimentIntensityAnalyzer().polarity_scores(tweet.replace(title, '')) # remove publication title from tweet and process it
for key in sorted(ss):
    print('{0}:  {1}, '.format(key, ss[key]), end='')
print() """

""" Need to store JSON files into a list of dicts that can be used to build a chart
    PSEUDOCODE:
    create data list
    iterate through a root directory
        for each json file found
            load json
            create dict of necessary values from json
            apply sentiment analysis and store it
            append dict to data list
    build chart: sentiment score vs. altmetric score
    build chart: platform vs. altmetric score 
"""
data = []
rootdir = "altmetric_data"
for rootpath, subdirectories, files in os.walk(rootdir):
    for filename in files:
        # if (filename.endswith(".json")):
        filepath = rootpath + "/" + filename
        jsonfile = json.load(open(filepath))
        tw = []
        for k in jsonfile["posts"]["twitter"]:
            tw.append({
                "tweet_id": k["tweet_id"]
            })
        data.append({
            "altmetric_id": jsonfile["altmetric_id"],
            "title": jsonfile["citation"]["title"],
            "score": jsonfile["altmetric_score"]["score"],
            "twitter": tw
        })
pprint(data)
