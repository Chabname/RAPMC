import os
import argparse
import pandas as pd
import re

def main():
    """Launch the application"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-trt', 
                        type=str, 
                        default="datas/training_text", 
                        help='File path of ')
    parser.add_argument('-trv', 
                        type=str, 
                        default="result/training_variants", 
                        help='File path of ')
    opt = parser.parse_args()

    print("Training text = " + str(opt.trt))
    print("Training variants = " + str(opt.trv))

    train_text = pd.read_csv("datas/training_text", sep = "\|\|", engine = 'python')
    #print(train_text)

    train_variants = pd.read_csv("datas/training_variants", engine = 'python')
    #print(train_variants)


    pattern = re.compile('^The(\s)receptor(\s)protein(\s)tyrosine(\s)phosphatase(\s)T.*$', re.IGNORECASE)
    for row in range(1, train_text.shape[0]):
        if(re.match(pattern, str(train_text.iloc[row][0]))):
            print(train_text.iloc[row])
       

main()