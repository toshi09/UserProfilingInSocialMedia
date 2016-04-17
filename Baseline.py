#!/usr/bin/env python
# Authors : Peggy, Vikhyati Singh, Alka
# Please add your names here
import csv
import os
import random
import argparse

from xml.etree import ElementTree

#user_profile = ((18.1,1.1,2.3,4.5, 3.9,4.0,2.5))

def main(args):
    if not os.path.exists(args.output_dir):
         os.makedirs(args.output_dir)
         
    with open(args.input_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            #tree = ElementTree.Element('')
            #for age,gender,ext,neu,agr,con,ope  in user_profile:
            age_grps = ['18-24', '25-34', '35-49', '50-xx']
            selected_age = age_grps[random.randint(0,3)]
            attrs = {'userId': row['userid'],
                     'age': str(random.randint(10,70)),
                     'age_group' : selected_age,                      
                     'gender': "male" if random.randint(0,1) else "female",
                     'extrovert': str(random.randint(2,4)),
                     'neurotic': str(random.randint(1,4)),
                     'agreeable': str(random.randint(1,4)),
                     'conscientious': str(random.randint(3,5)),
                     'open': str(random.randint(1,4))
                     }
            tree = ElementTree.Element('', attrs)
        
            #print minidom.parseString(ElementTree.tostring(tree)).toprettyxml()
            path = os.path.join(args.output_dir, row['userid']+".xml")
            #print(filename)
            #tree.write("/outputFiles/"+filename)
            with open(path,'w') as f: ## Write document to file
              f.write(ElementTree.tostring(tree))

def parse_args():
    """
    """
    parser = argparse.ArgumentParser(description='Script takes full input path to test file and directory. Please ensure output directory exist')
    parser.add_argument('-i',
                        "--input_path",
                        type=str, 
                        required=True,
                        help='Full path to input test data')
                        
    parser.add_argument('-o', "--output_dir",
                        type=str,
                        required=True,
                        help='The path to output directory')

    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
