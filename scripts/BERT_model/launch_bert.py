import BERT_script
import argparse
import os
import tensorflow as tf
import sys
import multiprocessing as mp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        dest = "data_file",                  
                        type=str, 
                        default="../../datas/all_data_clean_011.txt", 
                        help='Cleaned articles')
    parser.add_argument('-c',
                        dest = "cpu",                  
                        type=int, 
                        default=-1, 
                        help='Cleaned articles')                        

    args = parser.parse_args()
    
    cpu = args.cpu

    if cpu != -1 and cpu <= 0:
        sys.exit(f"Can't use {cpu} CPU. Please enter a valid number (>=1 or -1) of CPU to use.")
    else:
        max_cpu = mp.cpu_count()
        if cpu == -1:
            n_cpu = mp.cpu_count() - 1
            args.cpu = n_cpu
        elif cpu >= 1:			
            if cpu >= max_cpu:
                args.cpu = max_cpu

    print(f"\nUsing {args.cpu} CPUs for parallel jobs.")


    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True 
    config.gpu_options.per_process_gpu_memory_fraction = 0.8

    session = tf.compat.v1.Session(config=config)
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    print(tf.__version__)
    with tf.device('/gpu:0'):
        BERT_script.model(args)


    # 379s 100row 1 cpu
    # 293s 100 row 5 cpu
    # 277s




if __name__ == "__main__":
    main()