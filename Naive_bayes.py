#!/usr/bin/python
"""
An example of using this script is as following
python team5.py -i public-test-data -o output
or
./team5.py -i public-test-data -o output
This file assume following three files in the same location as this scipt:
1. profile.csv (containing information on 9500 proficles)
2. frequencies.csv
3. quantities.csv
Following modules must be intalled
scikit-learn
numpy
pandas

"""
import sys, getopt, os
import csv,math
     
import pandas as pd

import numpy as np

def train_error():
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
        #print id, age, gender
        if gender != pred_dict[id][0]:
            print gender, pred_dict[id][0]
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
    print  gender_bad, age_good
    
def main(inputfile, outputfile):
   """
   inputfile : test data directory containing text file on which we have to 
   make prediction
   outputfile : the directory where output predictions will be written 
   """
   error_f = open("error.csv", "w")
   male, female, age24, age34, age49, age50, n = 0,0,0,0,0,0,0
   with open("profile.csv", 'rb') as f:
       reader=csv.DictReader(f)
       for row in reader:
           n=n+1
           if float(row['gender'])==1:
               female=female+1
           else:
               male=male+1

           if float(row['age'])<25:
               age24=age24+1
           elif float(row['age'])>=25 and float(row['age'])<35:
               age34=age34+1
           elif float(row['age'])>=35 and float(row['age'])<50:
               age49=age49+1
           else:
               age50=age50+1

   age24=float(age24)/n
   age34=float(age34)/n
   age49=float(age49)/n
   age50=float(age50)/n
   female=float(female)/n
   male=float(male)/n
   femConProb = dict();
   maleConProb=dict();
   age24ConProb=dict();
   age34ConProb=dict();
   age49ConProb=dict();
   age50ConProb=dict();
   quantities=[0,0,0,0,0,0]
   vocSize=0
   with open("quantites.csv", 'rb') as fr:
       reader=csv.DictReader(fr)
       i=0
       for row in reader:
           if(i==6):
               vocSize=float(row['quantity'])
               continue
           quantities[i]=float(row['quantity'])
           i=i+1

   with open("frequencies.csv", 'rb') as fr:
       reader=csv.DictReader(fr)
       for row in reader:
           word=row['Word']
           maleConProb[word]=(float(row['M-Count'])+1)/(quantities[1]+vocSize)
           femConProb[word]=(float(row['F-Count'])+1)/(quantities[0]+vocSize)
           age24ConProb[word]=(float(row['Zero-Count'])+1)/(quantities[2]+vocSize)
           age34ConProb[word]=(float(row['One-Count'])+1)/(quantities[3]+vocSize)
           age49ConProb[word]=(float(row['Two-Count'])+1)/(quantities[4]+vocSize)
           age50ConProb[word]=(float(row['Three-Count'])+1)/(quantities[5]+vocSize)

   #tok=Tokenizer(preserve_case=False)
   dir = inputfile+"/text/"
   instr=''
   i=0
   for file in os.listdir(dir):
       if file.endswith(".txt"):
           instr = open(dir+file, 'r').read()
       file=file.rsplit('.', 1)[0]
       
       lst = [x.lower() for x in instr.strip().split()]

       userID=file
       maxP=0.0
       ages=[0.0,0.0,0.0,0.0]
       sumP=0.0
       gender='female'
       age='xx-24'
       for a in lst:
           if a in maleConProb:
               sumP=sumP+math.log(maleConProb[a])
               maxP=maxP+math.log(femConProb[a])
               ages[0]=ages[0]+math.log(age24ConProb[a])
               ages[1]=ages[1]+math.log(age34ConProb[a])
               ages[2]=ages[2]+math.log(age49ConProb[a])
               ages[3]=ages[3]+math.log(age50ConProb[a])
       sumP=sumP+math.log(male)
       maxP=maxP+math.log(female)
       if sumP>maxP:
           gender='male'
       maxP=0.0
       ages[0]=ages[0]+math.log(age24)
       ages[1]=ages[1]+math.log(age34)
       ages[2]=ages[2]+math.log(age49)
       ages[3]=ages[3]+math.log(age50)
       maxP=ages[0]
       if ages[1]>maxP:
           maxP=ages[1]
           age='25-34'
       if ages[2]>maxP:
           maxP=ages[2]
           age='35-49'
       if ages[3]>maxP:
           age='50-xx'
       file = open(outputfile+"/"+userID+".xml","w")
       file.write("<userId=\"{"+userID+"}\"\n")
       file.write("age_group=\""+age+"\"\n")
       file.write("gender="+"\""+gender+"\"\n")
       file.write("extrovert="+"\"3.49\"\n")
       file.write("neurotic="+"\"2.73\"\n")
       file.write("agreeable="+"\"3.58\"\n")
       file.write("conscientious="+"\"3.45\"\n")
       file.write("open="+"\"3.91\"\n")
       file.write("/>")
       file.close()
       
       error_f.write(userID+","+gender+","+age+"\n")
	#print userID
   error_f.close()
if __name__ == "__main__":
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(sys.argv[1:],"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print 'test.py -i <inputfile> -o <outputfile>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'test.py -i <inputfile> -o <outputfile>'
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   print 'Input file is "', inputfile
   print 'Output file is "', outputfile

   main(inputfile, outputfile)
   train_error()