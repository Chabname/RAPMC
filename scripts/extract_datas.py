import argparse
import pandas as pd
import csv
import time


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
    train_text = pd.read_csv(text_path, sep = "\|\|", engine = 'python')
    train_variants = pd.read_csv(variant_path, engine = 'python')


    select_variants = train_variants[train_variants["Gene"].isin([gene])]
    print(select_variants)

    variant_header = [None]*4
    variant_header[0] = "ID"
    variant_header[1] = "Gene"
    variant_header[2] = "Variation"
    variant_header[3] = "Class"

    select_texts = []

    with open(variant_path + "_" + gene, 'w', newline='', encoding='utf-8') as parse_file:
        writer = csv.writer(parse_file)
        writer.writerow(variant_header)
        for index, row in select_variants.iterrows():
            select_texts.append(train_text.loc[index])
            writer.writerow(row)
       

    text_header = [None]*4
    text_header[0] = "ID"
    text_header[1] = "Text"
    with open(text_path + "_" + gene, 'w', newline='', encoding='utf-8') as parse_file:
        writer = csv.writer(parse_file, delimiter="||" )
        writer.writerow(text_header)
        for index, row in select_texts.iterrows():
            writer.writerow(row)


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