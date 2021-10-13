import os
import argparse
import pandas as pd
import re

from Datas import Articles
from Datas import Variants
import EmbW2V 


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

    parser.add_argument('-caf',
                        type=str, 
                        default="datas/all_data_clean.txt", 
                        help='File all cleaned article')
    parser.add_argument('--type', 
                        type=str, 
                        default="both", 
                        help='Choose the type of the model between cbow or skipgram')
    parser.add_argument('--winsize', 
                        type=str, 
                        default="5", 
                        help='Give the context window size for cbow or skipgram')
    opt = parser.parse_args()

#    print("Training text = " + str(opt.trt))
#    print("Training variants = " + str(opt.trv))
#
#    train_text = Articles(opt.trt)
#    #print(train_text)
#
#    train_variants = Variants(opt.trv)
#    #print(train_variants)

    print(str(opt.caf))
    print(str(opt.type))

    EmbW2V.model_test( str(opt.caf), str(opt.type), str(opt.winsize))

       

main()