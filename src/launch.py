import os
import argparse
import pandas as pd
import re

from Datas import Articles
from Datas import Variants



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

    train_text = Articles(opt.trt)
    #print(train_text)

    train_variants = Variants(opt.trv)
    #print(train_variants)


       

main()