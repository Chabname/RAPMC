import re
import pandas as pd
import numpy as np
import time

amino_acid_dict = {'C' : 'CYS', 'D' : 'ASP', 'S' : 'SER', 'Q' : 'GLN', 'K' : 'LYS',
    'I' : 'ILE', 'P' : 'PRO', 'T' : 'THR', 'F' : 'PHE', 'N' : 'ASN', 
    'G' : 'GLY', 'H' : 'HIS', 'L' : 'LEU', 'R' : 'ARG', 'W' : 'TRP', 
    'A' : 'ALA', 'V' : 'VAL', 'E' : 'GLU', 'Y' : 'TYR', 'M' : 'MET'}

def amino_three(amino):
    return amino_acid_dict[amino]
    
def decompose_variation(variation):
    decompose_aa_pos_aa = re.compile("([a-z]{1,})(\d+)([a-z]{1,})")
    list_variation = decompose_aa_pos_aa.search(variation)
    if list_variation:
        aa1 = list_variation.group(1)
        aa2 = list_variation.group(3)

        if len(aa1) + len(aa2) == 2:
            amino1 = amino_acid_dict[aa1.upper()].lower()
            position = list_variation.group(2)
            amino2 = amino_acid_dict[aa2.upper()].lower()
            return [amino1,position,amino2]
    return False

def decompose_fusion(variation):
    decompose_g1_g2_fusion = re.compile("(\w+)\s?(\?|-)\s?(\w+)\? fusion")
    list_variation = decompose_g1_g2_fusion.search(variation)
    if list_variation:
        gene1 = list_variation.group(1)
        gene2 = list_variation.group(3)
        return "(" + gene1 + "|" + gene2 + ")"
    return False
        
    
def decompose_dup(variation):
    decompose_mut_pos = re.compile("([a-z]{1,})(\d+)dup")
    list_mut = decompose_mut_pos.search(variation)
    if list_mut:
        mut = list_mut.group(1)
        pos = list_mut.group(2)
        return mut + "" + pos
    return False


def clean_text(article):
    dot3 = re.compile("[\.]{2,}")
    fig = re.compile("fig[s]?\.")
    decimal = re.compile("\d+\.\d+")    
    etal = re.compile("et al\.")
    ie = re.compile("i\.e\.")
    inc = re.compile("inc\.")
    mutation_point = re.compile("[p|c]\.")
    
    clean_article = article.lower()
    clean_article = dot3.sub(".", clean_article)
    clean_article = fig.sub("", clean_article)
    clean_article = decimal.sub("", clean_article)
    clean_article = etal.sub("", clean_article)
    clean_article = ie.sub("", clean_article)    
    clean_article = inc.sub("", clean_article)    
    clean_article = mutation_point.sub("", clean_article)    
    
    
    return clean_article

def join_tuple_string(strings_tuple):
    return ' '.join(strings_tuple)

def find_match(text, word):
    clean = clean_text(text)
    word = word.lower()
    target_sentence = "([^.]*{}[^.]*\.)".format(word)
    before_after_target = "([^.]*\.){0,1}"    
    match_exp = re.compile(before_after_target + target_sentence + before_after_target)
    match_text = match_exp.findall(clean)
    final_match = "".join(list(map(join_tuple_string, match_text)))
    return final_match


def extract_match(line):
    
    
    # Cleaning text --> Already clean, no need    
    text = line["Text"]
    variation = line["Variation"]
    gene = line["Gene"].lower()
    
    if len(text) < 10000:
        return text,6
    
    if "r1627" == variation:
        return find_match(text, "162[0-9]"), 4
    if "c1385" == variation:
        return find_match(text, "p300"), 4
    
    if "hypermethylation" in variation:
        match_meth = find_match(text, "methylat")
        if len(match_meth) != 0:
            return match_meth, 2
        
    if "casp" in variation:
        match_casp = find_match(text, "casp")
        if len(match_casp) != 0:
            return match_casp, 1
    # Splice
    if "splice" in variation:
        match_splice = find_match(text, "splice")
        if len(match_splice) != 0:
            return match_splice, 2
        
    if "fs" in variation:
        match_fs = find_match(text, "frameshift")
        if len(match_fs) != 0:
            return match_fs, 2
        
    # Amplification
    if 'ampli' in variation:
        match_ampli = find_match(text, "(amplif|increse)")
        if len(match_ampli) != 0:
            return match_ampli,3
    
    # Duplication
    if "dup" in variation:
        decomp_dup = decompose_dup(variation)
        if decomp_dup:
            match_mut_pos = find_match(text, decomp_dup)
            if len(match_mut_pos) != 0:
                return match_mut_pos,2
        
        match_dup = find_match(text, "dup")
        if len(match_dup) != 0:
            return match_dup, 3
        
    # Try with * --> w802*
    if "*" in variation:
        new_var = variation.replace("*", "\\*")
        match_star = find_match(text, new_var)
        if len(match_star) != 0:
            return match_star, 1
        if "fs" in variation:
            match_fs = find_match(text, "fs\\*")
            if len(match_fs) != 0:
                return match_fs, 3
            
        match_stop_nonsense = find_match(text,"(stop|nonsense)")
        if len(match_stop_nonsense) != 0:
            return match_stop_nonsense, 2

    # Try first match with inital variation value
    # Quality score = 1 
    initial_match = find_match(text, variation)    
    if len(initial_match) != 0:
        #print("First match ! ", variation)
        return initial_match, 1
    
    
    # deletion and insertion
    if "del" in variation or "ins" in variation:
        match_delins = find_match(text, "(deletion|insertion|delet|insert)")
        if len(match_delins) != 0:
            return match_delins,2
        
        match_delins_sentence = find_match(text, "(del|ins)(\w|\s){0,}(del|ins)")
        if len(match_delins_sentence) != 0:
            return match_delins_sentence, 3
    
    # Trunc mutations
    if "trunc" in variation:
        match_trunc = find_match(text, "trunc")
        if len(match_trunc) != 0:
            #print("Trunc", variation)
            return match_trunc, 2
        
        match_shorte = find_match(text, "(shorte|delet)")
        if len(match_shorte) != 0:
            return match_shorte,4


    # Fusion of two genes
    # Quality score = 2
    if "fusion" in variation:
        fusion_gene = decompose_fusion(variation)
        if fusion_gene:
            match_fusion_gene = find_match(text, fusion_gene)
            if len(match_fusion_gene) != 0:
                #print("Fusion gene1 | gene2", variation)
                return match_fusion_gene, 2

        # Try to match the word fusion at least..
        # Quality score 4 (bad)
        match_fusion = find_match(text,"fusion")
        if len(match_fusion) != 0:
            #print("FUSION", variation)
            return match_fusion, 4
        
    aa_pos_aa = decompose_variation(variation)
    if aa_pos_aa :
        if len(aa_pos_aa) == 3:
            # If we success to split variation in 3 group --> aa1 pos aa2
            # Second try without the last amino acid --> y371
            match_variation_aa_pos = find_match(text, variation[:-1])
            if len(match_variation_aa_pos) != 0:
                #print("y371 aa_pos", variation)
                
                return match_variation_aa_pos, 2

            # Third try with 3 letter code of amino acid --> tyr371ser
            match_aa_pos_aa = find_match(text, "".join(aa_pos_aa))
            if len(match_aa_pos_aa) != 0:
                #print("aa_pos_aa", variation)
                return match_aa_pos_aa, 1
            
            # Try with 3 letter code without the last aa --> tyr371
            match_aa_pos = find_match(text, aa_pos_aa[0] + aa_pos_aa[1])
            if len(match_aa_pos) != 0:
                #print("aa_pos", variation)
                
                return match_aa_pos,2
            
            # Match position only --> 371
            match_pos = find_match(text, aa_pos_aa[1])
            if len(match_pos) != 0:
                #print("pos", variation)
                
                return match_pos,4
            # Search word Substitution
            match_substitution = find_match(text, "substitu")
            if len(match_substitution) != 0:
                return match_substitution, 3
                
            # Match position around the real position --> 370 - 379
            match_pos_weak = find_match(text, aa_pos_aa[1][:-1] + "[0-9]")
            if len(match_pos_weak) != 0:
                #print("pos weak", variation)
                
                return match_pos_weak,5
     
    match_gene = find_match(text, gene)
    if len(match_gene) != 0:
        return match_gene, 5
            
    # score 6 ?
    return text,7



def prepare_datas(file_text, file_variant, file_out, is_training):
    print("____________________________Cleaning Datas__________________________")
    print("____________________________________________________________________")
    start_time = time.perf_counter()

    text = pd.read_csv(file_text, sep = '\|\|', engine='python')
    text.index.name = "ID"
    text.columns = ["Text"]
    variant = pd.read_csv(file_variant)
    variant.set_index("ID",inplace = True)
    
    concatenate_data = pd.merge(variant, text, on="ID").dropna()
    concatenate_data["Text"] = concatenate_data.apply(lambda line: clean_text(line["Text"]), axis = 1)
    concatenate_data["Variation"] = variant["Variation"].apply(lambda line: line.lower())
   

    clean_match_data = concatenate_data.apply(lambda x: extract_match(x), axis = 1)
    clean_match = pd.DataFrame(list(clean_match_data), columns = ["Text","Score"], index = clean_match_data.index)
    clean_match.index.name = "ID"

    new_data = pd.merge(concatenate_data,clean_match, on = "ID")
    if(is_training):
        final_data = new_data.loc[:,["Gene","Variation","Class","Text_y","Score"]]
        final_data.columns = ["Gene","Variation","Class","Text","Score"]
        dtf = pd.merge(pd.DataFrame(final_data.index), final_data, on ="ID")
        np.savetxt(file_out,dtf, fmt = "%d|||%s|||%s|||%d|||%s|||%d", header= "|||".join(dtf.columns), comments='')
    else:
        final_data = new_data.loc[:,["Gene","Variation","Text_y","Score"]]
        final_data.columns = ["Gene","Variation","Text","Score"]
        dtf = pd.merge(pd.DataFrame(final_data.index), final_data, on ="ID")
        np.savetxt(file_out,dtf, fmt = "%d|||%s|||%s|||%s|||%d", header= "|||".join(dtf.columns), comments='')

   
    stop_time = time.perf_counter()
    print("____________________________________________________________________")
    print("Cleaning datas finished in {} seconds".format(stop_time-start_time))



def main(is_training):
    if is_training:
        file_text = "datas/training_text"
        file_variant = "datas/training_variants"
        file_out = "datas/training_clean"
    else:
        file_text = "datas/test_text"
        file_variant = "datas/test_variants"
        file_out = "datas/test_clean"
    prepare_datas(file_text, file_variant, file_out, is_training)

main(True)