import argparse
import EmbW2V 



def main():
    """Launch the application"""
    parser = argparse.ArgumentParser()

    parser.add_argument('-caf',
                        type=str, 
                        default="datas/all_data_clean.txt", 
                        help='File all cleaned article')
    parser.add_argument('-t', '--type',
                        type=str, 
                        default="both", 
                        help='Choose the type of the model between cbow or skipgram')
    parser.add_argument('-ws', '--winsize', 
                        type=int, 
                        default=20, 
                        help='Give the context window size for cbow or skipgram')
    parser.add_argument('-e', '--epoch',
                        type=int, 
                        default=20, 
                        help='Number of epoch : fraining word2vec modele')
    parser.add_argument('-b', '--batch', 
                        type=int, 
                        default=10000, 
                        help='Batch size for word 2 vec')
    parser.add_argument('-sw', '--stopword', 
                        type=bool, 
                        default=True,
                        help='Cleaning stop_words') 
    parser.add_argument('-r', '--repeat', 
                        type=int, 
                        default=2000, 
                        help='Reapeating article vector to amplifie datas')
    parser.add_argument('-c', '--concat', 
                        type=bool, 
                        default=True,
                        help='Concat all articles and replace each articles by the concatenation')
    opt = parser.parse_args()



    print("Clean file : " + str(opt.caf))
    print("Modele type : " + str(opt.type))
    print("Context window size : " + str(opt.winsize))
    print("Number of epoch : " + str(opt.epoch))
    print("Batch size : " + str(opt.batch))
    print("Cleaning StopWords : " + str(opt.stopword))
    print("Repeat : " + str(opt.repeat))
    print("concatening all articles : " + str(opt.concat))

 

    EmbW2V.main(opt.caf, opt.type, opt.winsize, opt.epoch, opt.batch, opt.stopword, opt.repeat, opt.concat)


       

main()