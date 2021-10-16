import BERT_script
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        dest = "data_file",                  
                        type=str, 
                        default="../../datas/all_data_clean.txt", 
                        help='Cleaned articles')

    args = parser.parse_args()
    print(args)
    BERT_script.model(args.data_file)

if __name__ == "__main__":
    main()