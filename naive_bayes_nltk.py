#!/usr/bin/python

import argparse

import codecs

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk import FreqDist


from xml.etree import ElementTree

import pickle
import csv

import sys
import pandas as pd
import os

tknzr = TweetTokenizer()


reload(sys)
sys.setdefaultencoding('ISO-8859-1')

def get_words_in_tweets(tweets):

    all_words = []

    for (words, sentiment) in tweets:

      all_words.extend(words)

    return all_words
    
def get_word_features(wordlist):

    wordlist = FreqDist(wordlist)

    word_features = wordlist.keys()

    return word_features


def extract_features(document, word_features):

    document_words = set(document)

    features = {}

    for word in word_features:

        features['contains(%s)' % word] = (word in document_words)

    return features



def age_gender(Genderclassifier, Ageclassifier, test_data_dir, output_dir, word_featuresAge, word_featuresGender):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    pos_sent = open("/home/itadmin/positive.txt").read()
    positive_words = pos_sent.split('\n')
    
    neg_sent = open("/home/itadmin/negative.txt").read()
    negative_words = neg_sent.split('\n')

    profile_file_path = os.path.join(test_data_dir, "profile/profile.csv")
    print profile_file_path
    with open(profile_file_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            uuid = row['userid']
            positive =    0
            negative = 0
            neutral = 0
            df = pd.DataFrame()

            df['positive'] = 0
            df['negative'] = 0
            df['neutral'] = 0
            test_file = os.path.join(test_data_dir, "text", uuid+".txt")
            with codecs.open(test_file, "r", "ISO-8859-1") as fo:
                try: 
                    temp = []
                    for line2 in fo:
                        stopword = stopwords.words('english')
                        for word in tknzr.tokenize(line2):
                    	    temp2 = word.lower()
                       	    
                            if temp2 not in stopword:
                                
                                temp.append(temp2)
                            	if temp2 in positive_words:
			            positive = positive + 1
                                    
                                elif temp2 in negative_words:
                                    negative = negative + 1
                                else :
                                    neutral = neutral + 1
                    df.set_value(0, 'positive', positive)
                    df.set_value(0, 'negative', negative)
                    df.set_value(0, 'neutral', neutral)

                   
                    gender = Genderclassifier.classify(extract_features(temp, word_featuresGender))
                    age = Ageclassifier.classify(extract_features(temp, word_featuresAge))
                    print("Gender Prediction ", gender)
		         print("Age Prediction ", age)
                except UnicodeDecodeError: 
		    gender = "0"
                output_file = os.path.join(output_dir, uuid+".xml")
                with open(output_file, "w") as out_f:
                    attrs = {"userId": uuid,
                             "gender" : "male" if gender == "0" else "female",
                             "age_group" : age,
                             }
                    tree = ElementTree.Element("user", attrs)
                    out_f.write(ElementTree.tostring(tree))


def run(test_dir, output_dir):
    f1 = open("/home/itadmin/Gender_classifier2.pickle", 'rb')
    Genderclassifier = pickle.load(f1)
    f1.close()
    f2 = open("/home/itadmin/Age_classifier2.pickle", 'rb')
    Ageclassifier = pickle.load(f2)
    f2.close()
    print ("works")
    with open("/home/itadmin/AgeTweets.pickle", 'rb') as Ageresults:
    	Agetweets = pickle.load(Ageresults)
    Ageresults.close()
    with open("/home/itadmin/GenderTweets.pickle", 'rb') as Genderresults:
    	Gendertweets = pickle.load(Genderresults)
    Genderresults.close()
    word_featuresAge = get_word_features(get_words_in_tweets(Agetweets))
    word_featuresGender = get_word_features(get_words_in_tweets(Gendertweets))
    
    age_gender(Genderclassifier, Ageclassifier, test_dir, output_dir, word_featuresAge, word_featuresGender)




def main(args):
   run(args.test_dir, args.output_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="""Script takes full input path to
                         test directory, output directory and training directory""")


    parser.add_argument('-i',
                        "--test_dir",
                        type=str,
                        required=True,
                        help='Full path to input test directory containing profile and text dir')

    parser.add_argument('-o', "--output_dir",
                        type=str,
                        required=True,
                        help='The path to output directory')


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)



