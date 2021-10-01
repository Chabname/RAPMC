import argparse
import pandas as pd
import numpy as np
import time
import os


## function : 
#       extract_training_by_gene
## input :
#       text_path : (string) File path of training_text which needed to be extracted to get datas only for a gene
#       variant_path : (string) File path of training_variants which needed to be extracted to get datas only for a gene
#       gene : (string) Name of the Gene we want to have an extraction
## output :
#       (files) 
## details : 
#       Create two files file having one header and all datas extracted
def extract_training_by_gene(text_path, variant_path, gene):
    """Extract datas by gene name"""
    train_variants = pd.read_csv(variant_path, engine = 'python')
    train_variants.set_index("ID", inplace = True)

    train_text = pd.read_csv(text_path, sep = "\|\|", engine = 'python')
    train_text.index.name = "ID"
    train_text.columns = ["Text"]    

    select_variants = train_variants[train_variants["Gene"] == gene ]
    select_text = train_text.loc[select_variants.index,]

    extract_text_path = text_path + "_" + gene

    select_variants.to_csv(variant_path + "_" + gene, sep=',', encoding='utf-8')

    dtf = pd.merge(pd.DataFrame(select_variants.index), select_text, on = "ID")
    np.savetxt(extract_text_path, dtf, fmt = '%d||%s', header = ','.join(dtf.columns), comments = '')



def main():
    """Extract datas"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', 
                        type=str, 
                        default="datas/training_text", 
                        help='File path of text file to extract')
    parser.add_argument('-v', 
                        type=str, 
                        default="datas/training_variants", 
                        help='File path of gene file to extract')
    parser.add_argument('-g', 
                        type=str, 
                        default="CBL", 
                        help='Gene name')                  
    opt = parser.parse_args()

    print("_________________Starting the extraction_______________")
    start_time = time.perf_counter()

    print("Training text = " + str(opt.t))
    print("Training variants = " + str(opt.v))

    extract_training_by_gene(opt.t, opt.v, opt.g )

    stop_time = time.perf_counter()
    print("extraction ended in {} seconds".format(stop_time-start_time))
    print("______________________________________________________")

main()