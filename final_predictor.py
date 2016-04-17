# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 19:18:38 2016

@author: Vikhyathi, Sai Prajna
"""

#!/usr/bin/python

import argparse
import csv
import math
import numpy as np
import pandas as pd
import nltk
import os
import sys
import pickle

from random import randint
from xml.etree import ElementTree
import codecs

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk import FreqDist

from sklearn import cross_validation
from sklearn import tree, linear_model


tknzr = TweetTokenizer()


reload(sys)
sys.setdefaultencoding('ISO-8859-1')

def compute_train_error():
    """
    Computes the trainig error
    :return:
    """
    pred = open("error.csv")
    pred_dict = {}
    for line in pred:
        values = line.strip().split(",")
        pred_dict[values[0]] = [values[1], values[2]]
    
    gender_bad = 0
    age_good = 0
    train_f = open("profile.csv")
    train_f.readline() # skip first line
    for line in train_f:
        values = line.strip().split(",")
        id = values[1]
        age = float(values[2])
        gender = "male" if values[3] == "0.0" else "female"

        if gender != pred_dict[id][0]:
            gender_bad += 1
            
        pred_age =pred_dict[id][1]
        if age < 25 and pred_age == "xx-24":
            age_good += 1
        elif age >= 25 and age <= 34 and pred_age == "25-34":
            age_good += 1
        elif age >= 35 and age <= 49 and pred_age == "35-49":
            age_good += 1
        elif age >= 50 and pred_age == "50-xx":
            age_good += 1

    print "Gender Acc ", (1.0 - ( gender_bad/9500.0)) * 100 , " Age Acc ", (age_good/9500.0) * 100
    
def get_words_in_Status(Status):

    all_words = []

    for (words, sentiment) in Status:

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

    error_f = open("error.csv", "w")
    pos_sent = open("positive.txt").read()
    positive_words = pos_sent.split('\n')
    
    neg_sent = open("negative.txt").read()
    negative_words = neg_sent.split('\n')

    profile_file_path = os.path.join(test_data_dir, "profile/profile.csv")
    print profile_file_path
    with open(profile_file_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
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
                    print "Gender Prediction ", gender
                    #print(age)
		         #print "Age Prediction ", age
                    error_f.write(",".join([uuid,  str(gender), str(age)] )+ "\n")
                except UnicodeDecodeError: 
		     gender = "0"
                output_file = os.path.join(output_dir, uuid+".xml")
                with open(output_file, "w") as out_f:
                    attrs = {"userId": uuid,
                             "gender" : "male" if gender == "0" else "female",
                             "age_group" : age,
                             "open" : get_personality_trait("ope",i),
                    	     "extrovert" : get_personality_trait("ext",i),
                             "conscientious" : get_personality_trait("con", i),
                             "agreeable" : get_personality_trait("agr", i),
                             "neurotic" : get_personality_trait("neu", i)
                             }
                    tree = ElementTree.Element("user", attrs)
                    out_f.write(ElementTree.tostring(tree))
    error_f.close()
    
def get_df(training_dir, test_dir):
    # load train data as data frame
    df_LIWC_training = pd.read_csv(training_dir + 'LIWC.csv', sep=',')
    df_nrc_training = pd.read_csv(training_dir + 'nrc.csv', sep=',')
    df_profile_training = pd.read_csv(training_dir + 'profile/profile.csv', sep=',')
    
    df_LIWC_training = df_LIWC_training.rename(columns={'userId': 'userid'})
    df_nrc_training = df_nrc_training.rename(columns={'userId': 'userid'})

    # merging LIWC, nrc and profile to get personality scores
    df_train1 = pd.merge(left=df_LIWC_training, right=df_profile_training, on='userid', how='outer')
    df_train = pd.merge(left=,df_train1 right=df_nrc_training, on='userid', how='outer')

    # load test data as data frame
    df_LIWC_test = pd.read_csv(test_dir + 'LIWC.csv', sep=',')
    df_nrc_test = pd.read_csv(test_dir + 'nrc.csv', sep=',')
    df_profile_test = pd.read_csv(test_dir + 'profile/profile.csv', sep=',')

    df_LIWC_test = df_LIWC_test.rename(columns={'userId': 'userid'})
    df_nrc_test = df_nrc_test.rename(columns={'userId': 'userid'})

    # merging LIWC, nrc and profile to add personality columns
    
    df_test1 = pd.merge(left=df_LIWC_test, right=df_profile_test, on='userid', how='outer')
    df_test = pd.merge(left=df_test1, right=df_nrc_test, on='userid', how='outer')
    
    # adding some values to age, gender and personality scores of test data to avoid null/value errors
    df_test = default_df_test(df_test)

    # Add test and train data together
    df = pd.concat([df_train, df_test])

    # Remove common name columns to avoid index errors
    df = df.reset_index()

    # Drop age and gender
    df = df.drop('age', 1)
    df = df.drop('gender', 1)

    return df


def default_df_test(df_test):
    df_test.age = 0
    df_test.gender = 1.0

    df_test.ope=3.90
    df_test.ext=3.48
    df_test.neu=2.73
    df_test.con=3.44
    df_test.agr=3.58
    return df_test


def predict_big5(training_dir, test_dir, output_dir):
    df = get_df(training_dir, test_dir)

    # making data frames for each personality scores
    big5 = ['ope', 'ext', 'con', 'agr', 'neu']
    dict_userid_personality = {}

    for personality in big5:
        df_train_personality = df
        df_train_personality = df_train_personality.drop('userid', 1)
        for trait in big5:
            if trait != personality:
                df_train_personality = df_train_personality.drop(trait, 1)
        df_train_personality_fix = df_train_personality
        df_train_personality = df_train_personality.drop(personality, 1)

        # making feature list for personality scores
        feature_list_personality = df_train_personality.columns.tolist()[:]

        # regression models
        X = df_train_personality[feature_list_personality]
        Y = df_train_personality_fix[[personality]]

        # Split the data and 10fold validation

        X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y[personality], train_size=9500,
                                                                             random_state=0)
        regr = linear_model.LinearRegression()

        # Train the model using the training sets
        regr.fit(X_train, Y_train)

        # run regression on test
        result = regr.predict(X_test)

        # root mean square error  
        rmse(result, Y_test, personality)

        write_personality_traits_to_separate_files(personality, result)


def write_personality_traits_to_separate_files(personality, result):
    if (personality == 'ope'):
        np.savetxt("ope.csv", result, delimiter=",", fmt='%0.2f')
    elif (personality == 'ext'):
        np.savetxt("ext.csv", result, delimiter=",", fmt='%0.2f')
    elif (personality == 'con'):
        np.savetxt("con.csv", result, delimiter=",", fmt='%0.2f')
    elif (personality == 'agr'):
        np.savetxt("agr.csv", result, delimiter=",", fmt='%0.2f')
    else:
        np.savetxt("neu.csv", result, delimiter=",", fmt='%0.2f')


# return personality trait for each user
def get_personality_trait(personality_trait, i):
    personality_trait = open(personality_trait + ".csv", mode="r").readlines()[i]
    return personality_trait.replace("\n", "")



def run(test_dir, output_dir):
    f1 = open("Gender_classifier2.pickle", 'rb')
    Genderclassifier = pickle.load(f1)
    f1.close()
    f2 = open("Age_classifier2.pickle", 'rb')
    Ageclassifier = pickle.load(f2)
    f2.close()
    print ("works")
    with open("AgeStatus.pickle", 'rb') as Ageresults:
    	AgeStatus = pickle.load(Ageresults)
    Ageresults.close()
    with open("GenderStatus.pickle", 'rb') as Genderresults:
    	GenderStatus = pickle.load(Genderresults)
    Genderresults.close()
    word_featuresAge = get_word_features(get_words_in_Status(AgeStatus))
    word_featuresGender = get_word_features(get_words_in_Status(GenderStatus))
    
    age_gender(Genderclassifier, Ageclassifier, test_dir, output_dir, word_featuresAge, word_featuresGender)
    predict_age_and_gender(test_dir, output_dir)
    print "Prediction for age and gender done"
    
# compute rmse
def rmse(result, expected, personality):
    print 'Evaluation results of ' + personality
    print("Residual sum of squares: %.2f" % np.mean((result - expected) ** 2))


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



