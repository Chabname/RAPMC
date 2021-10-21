import pandas as pd
import numpy as np
from Datas import Articles

from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
import string
import time
import multiprocessing

class Vector():
    def __init__(self, data_file, model_path, is_training):
        """ Initialize Low matrix before completing

        Keyword arguments:
                data_file -- 
        """
        self.model_path = model_path
        self.data_file = data_file
        self.is_training = is_training

    def get_vector_datas(self):
        model = Word2Vec.load(self.model_path)
        all_datas = Articles(self.data_file, self.is_training)
        datas_df = all_datas.datas.drop(['ID', 'Text', 'Score'], axis=1)
        datas_df["Gene"] = datas_df["Gene"].str.lower()
        datas_df["Variation"] = datas_df["Variation"].str.lower()


        #TODO is_training ?
        datas_vect = pd.DataFrame(columns =['Gene', 'Variation', 'Sum', 'Class'])
        new_data = []
        not_found_var_list =[]

        for index, row in datas_df.iterrows():
            var_clean=""
            try:
                gene_clean = clean_word(row['Gene'])
                vec_gene = model.wv.get_vector(gene_clean)
                var_list = row['Variation'].split(" ")
            
                var_clean = clean_word(var_list[0])
                vec_var = model.wv.get_vector(var_clean)
                if(len(var_list)>1):
                    for ind_var in range(1,len(var_list)):
                        var_clean = clean_word(var_list[ind_var])
                        vec_var = vec_var + model.wv.get_vector(var_clean)

                new_data.append([vec_gene, vec_var, vec_gene + vec_var, row['Class']])
            except:
                line = row
                line['Target'] = var_clean
                not_found_var_list.append(line)
                continue

        not_found_var = pd.DataFrame(not_found_var_list, columns =['Gene', 'Variation', 'Class', 'Target'])
        split_filename = self.model_path.split("/")
        modele_name = split_filename[len(split_filename) - 1]
        log_file_path = "results/log_not_found_words_" + modele_name + ".txt"
        np.savetxt(log_file_path, not_found_var, fmt = "%s\t\t\t%s\t\t\t%d\t\t\t%s", header= "\t\t\t".join(not_found_var.columns), comments='')
        
        datas_vect = pd.DataFrame(new_data, columns =['Gene', 'Variation', 'Sum', 'Class'])
        return datas_vect



def clean_word(word):
    lemmatizer = WordNetLemmatizer()
    clean_var = word.translate(str.maketrans('', '', string.punctuation))
    clean_var = lemmatizer.lemmatize(clean_var)
    return clean_var


def main(data_file, model_path, is_training):
    vector = Vector(data_file, model_path, is_training)
    vect = vector.get_vector_datas()
    print(vect)
    

main("datas/training_clean", "datas/cbow_A3316_WS20_E20_B10000_R2000_CTrue.model", True)
#main("datas/training_clean", "datas/cbow_3284.model", True)
#main("datas/training_clean", "datas/skipgram_3284.model", True)