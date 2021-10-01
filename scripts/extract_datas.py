import argparse
import pandas as pd
import csv
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

    train_text = pd.read_csv(text_path, sep = "\|\|", engine = 'python')
    

    select_variants = pd.DataFrame(train_variants[train_variants["Gene"].isin([gene])])
    select_text = []

    
    select_variants.to_csv(variant_path + "_" + gene, sep=',', encoding='utf-8', index=False)

    #  Find the text with the same ID and store it
    for index, row in select_variants.iterrows():
        select_text.append(train_text.loc[index])

    # Doing like that add " at the bgining ant the end of the line, so we need a temporary file
    text_header = [None]*2
    text_header[0] = "ID"
    text_header[1] = "Text"
    with open(text_path + "_" + gene + "_tmp", 'w', newline='', encoding='utf-8') as extract_file:
        writer = csv.writer(extract_file)
        writer.writerow(text_header)
        for index, val in enumerate(select_text):
            value = str(index) + "||" + val
            writer.writerow(value)
    extract_file.close()

    # Here we strip the begining and the endind of each line from temporary file and write into final file
    with open(text_path + "_" + gene + "_tmp", "r") as file_in , open(text_path + "_" + gene ,'w') as fileout:
        for line in file_in:
            line = line.strip('"')
            fileout.write(line)

    # Remove temporary file
    os.remove(text_path + "_" + gene + "_tmp")


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