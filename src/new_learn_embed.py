import argparse
import W2VModel 
import os

def choose_type():
    type_model = input("Enter the type of the model (cbow or skipgram) ")
    if type_model == "cbow" or type_model == "skipgram":
        return type_model
    else:
        print("You must choose between cbow or skipgram")
        return choose_type()

def main():
    """Launch the application"""
    parser = argparse.ArgumentParser()

    parser.add_argument('-tc','--testclean',
                        type=str, 
                        default="datas/test_clean", 
                        help='File with all test text cleaned concatenate with gene')
    parser.add_argument('-tm','--trainedmodel',
                        type=str, 
                        help='File with all test text cleaned concatenate with gene',
                        required=True)
    parser.add_argument('-t', '--type',
                        type=str, 
                        help='Choose the type of the model between cbow or skipgram',
                        required=True,)
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

    

    print("Clean file : " + str(opt.testclean))
    print("Modele path : " + str(opt.trainedmodel))
    print("Modele type : " + str(opt.type))
    print("Context window size : " + str(opt.winsize))
    print("Number of epoch : " + str(opt.epoch))
    print("Batch size : " + str(opt.batch))
    print("Cleaning StopWords : " + str(opt.stopword))
    print("Repeat : " + str(opt.repeat))
    print("concatening all articles : " + str(opt.concat))

    W2VModel.main(str(opt.testclean), 
                str(opt.trainedmodel),
                type = str(opt.type), 
                win_size = str(opt.winsize), 
                epoch = str(opt.epoch), 
                batch = str(opt.batch), 
                stopword = str(opt.stopword), 
                repeat = str(opt.repeat), 
                concat = str(opt.concat))

main()