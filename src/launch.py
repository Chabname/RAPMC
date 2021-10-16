#import os
import argparse
#import pandas as pd
#import re

from Datas import Articles
#from Datas import Variants
import EmbW2V 
import tensorflow as tf


def main():
    """Launch the application"""
    parser = argparse.ArgumentParser()
#    parser.add_argument('-trt', 
#                        type=str, 
#                        default="datas/training_text", 
#                        help='File path of ')
#    parser.add_argument('-trv', 
#                        type=str, 
#                        default="result/training_variants", 
#                        help='File path of ')

    parser.add_argument('-caf',
                        type=str, 
                        default="datas/all_data_clean.txt", 
                        help='File all cleaned article')
    parser.add_argument('--type', 
                        type=str, 
                        default="both", 
                        help='Choose the type of the model between cbow or skipgram')
    parser.add_argument('--winsize', 
                        type=int, 
                        default=20, 
                        help='Give the context window size for cbow or skipgram')
    parser.add_argument('--epoch',
                        type=int, 
                        default=100, 
                        help='Number of epoch : fraining word 2 vec modele')
    parser.add_argument('--batch', 
                        type=int, 
                        default=10000, 
                        help='Batch size for word 2 vec')
    opt = parser.parse_args()

#    print("Training text = " + str(opt.trt))
#    print("Training variants = " + str(opt.trv))
#
#    train_text = Articles(opt.trt)
#    #print(train_text)
#
#    train_variants = Variants(opt.trv)
#    #print(train_variants)

    print("Clean file : " + str(opt.caf))
    print("Modele type : " + str(opt.type))
    print("Context window size : " + str(opt.winsize))
    print("Number of epoch : " + str(opt.epoch))
    print("Batch size : " + str(opt.batch))

    

    #EmbW2V.model_test(opt.caf, opt.type, opt.winsize, opt.epoch, opt.batch)
    print("Num GPUs Available: ", len(tf.config.list_logical_devices('GPU')))
       

main()