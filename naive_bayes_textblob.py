#!/usr/bin/python
import argparse
import io
import os
import codecs
import traceback
import logging
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from textblob.classifiers import NaiveBayesClassifier
from textblob import formats
from xml.etree import ElementTree

import pickle
import csv
import json
import sys, getopt
class PipeDelimitedFormat(formats.DelimitedFormat):
    delimiter = '|'

formats.register('psv', PipeDelimitedFormat)


separator="|"
processed_data="processed_data.psv"

reload(sys)
sys.setdefaultencoding("utf-8")

import csv


def preprocess(training_data_dir):
    """
    taining_data_dir : path to training dir. This dir should contain
    profile and text dir.
    """
    f = open(processed_data,'w+')     
    profile_file_path = os.path.join(training_data_dir, "profile/profile.csv")
    print profile_file_path
    counter = 0
    with open(profile_file_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if counter > 100:
                continue
            uuid = row['userid']
            train_file = os.path.join(training_data_dir, "text", uuid+".txt")
            with open(train_file, "r+") as fo:
                try:
                    file_content = fo.read().strip()             
                    file_content = file_content.replace(r"|"," ")
                    file_content = file_content.replace(r"\n", " ")   
                    values = file_content.split()

                    f.write(" ".join(values)+separator+row['gender']+"\n")
                except UnicodeDecodeError:
                    print("Unicode bad data")
            counter += 1
    f.close()
    csvfile.close()
	
           
def predict(classifier_obj, test_data_dir, output_dir):
    """
    Classifier_obj : A naive bayesing text blob object (Can be also uploaded from
    pickel file)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    profile_file_path = os.path.join(test_data_dir, "profile/profile.csv")
    print profile_file_path
    with open(profile_file_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:	
            uuid = row['userid']
            test_file = os.path.join(test_data_dir, "text", uuid+".txt")
            with open(test_file, "r+") as fo:
                try:
                    file_content = fo.read().strip()
                    prediction = classifier_obj.classify(file_content)
                    print("Prediction ", prediction )
                except UnicodeDecodeError:
                    prediction = ("Prediction", "1.0")
                    
                output_file = os.path.join(output_dir, uuid+".xml")
                with open(output_file, "w") as out_f:
                    attrs = {'userId': uuid,
                             'gender' : "male" if prediction[1] == "1.0" else "female",
                             'age_group' : "xx-24",
                             'extrovert': str(3.4869),
                     	     'neurotic': str(2.7324),
                             'agreeable': str(3.5839),
                             'conscientious': str(3.4456),
                             'open': str(3.9087)
                             }
                    tree = ElementTree.Element('user', attrs)
                    out_f.write(ElementTree.tostring(tree))

               

def jsonify():
    csvfile = open(processed_data, 'r')
    jsonfile = open('jsondata.json', 'w+')
    result = []
    fieldnames = ("text","label")
    reader = csv.DictReader(csvfile, fieldnames, delimiter = '|')
    for row in reader:
        try:
            row['text'] = row['text'].encode("ascii", errors="ignore")
            result.append(row)
        except UnicodeDecodeError:
            print("Bad unicode data in jsonify")
            
    json.dump(result, jsonfile)
    return 'jsondata.json'
    
def train_model(json_data_file):
    """
    Given a json file, train the data and return the classifier.
    """
        
    with open(json_data_file, 'r') as fp:
        cl = NaiveBayesClassifier(fp, format="json")
        pickle_file = open("classifier.pickle", "wb")
        pickle.dump(cl, pickle_file)
        pickle_file.close()
        return cl
    return None


def run(training_dir, test_dir, output_dir):
    if os.path.exists("classifier.pickle"):
        print("Loading pickel")
        classifier = pickle.load(open("classifier.pickle"))
    else:
        print("preoprocessing ..")
        preprocess(training_dir)
        print("preoprocessing done.")
        print("jsonify..")
        json_file_name = jsonify()
        print("jsonify done.")
        print("Training file")
        classifier = train_model(json_file_name)
    
    print("Training done.")

    predict(classifier, test_dir, output_dir)

    
def main(args):
   run(args.training_dir, args.test_dir, args.output_dir)

def parse_args():
    """
    """
    parser = argparse.ArgumentParser(description="""Script takes full input path to
                         test directory, output directory and training directory""")

    parser.add_argument('-d',
                        "--training_dir",
                        default='',
                        type=str, 
                        help='Full path to input trainig directory')

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
    if not args.training_dir:
        args.training_dir = "training"
    main(args)

