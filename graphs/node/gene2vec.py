# adapted from jingcheng-du/Gene2vec

import os
import random
import logging
import datetime
import argparse

import numpy as np

import gensim
from gensim.models.keyedvectors import KeyedVectors

def load_embeddings(file_name):
    model = KeyedVectors.load(file_name)
    wordVector = model.wv
    vocabulary, wv = zip(*[[word, wordVector[word]] for word, vocab_obj in wordVector.vocab.items()])
    return np.asarray(wv), vocabulary

def outputTxt(embeddings_file):
    embeddings_file = embeddings_file  # gene2vec file address
    wv, vocabulary = load_embeddings(embeddings_file)
    index = 0
    matrix_txt_file = os.path.splitext(embeddings_file)[0]+".txt"  # gene2vec matrix txt file address
    with open(matrix_txt_file, 'w') as out:
        for ele in wv[:]:
            out.write(str(vocabulary[index]))
            index = index + 1
            for elee in ele:
                out.write(" " + str(elee))
            out.write("\n")
    out.close()

def gene2vec(sourceDir, export_dir, ending_pattern,
    dimension = 100,  # dimension of the embedding
    num_workers = 32,  # number of worker threads
    sg = 1,  # sg =1, skip-gram; sg =0, CBOW
    max_iter = 10,  # number of iterations
    window_size = 1,  # maximum distance between the gene and predicted gene within a gene list
    txtOutput = True  # output text file of gene embeddings
    ):

    # sourceDir = "../data"

    # training file format:
    #   TOX4 ZNF146
    #   TP53BP2 USP12
    #   TP53BP2 YRDC

    num_db = 0
    files = os.listdir(sourceDir)
    size = len(files)
    gene_pairs = list()
    random.shuffle(files)

    #load all the data
    for fname in files:
        if not fname.endswith(ending_pattern):
            continue
        num_db = num_db + 1
        now = datetime.datetime.now()
        print(now)
        print("current file "+ fname + " num: " + str(num_db) + " total files " + str(size))
        f = open(os.path.join(sourceDir, fname), 'r', encoding='windows-1252')
        for line in f:
            gene_pair = line.strip().split()
            gene_pairs.append(gene_pair)
        f.close()

    current_time = datetime.datetime.now()
    print(current_time)
    print("shuffle start " + str(len(gene_pairs)))
    random.shuffle(gene_pairs)
    current_time = datetime.datetime.now()
    print(current_time)
    print("shuffle done " + str(len(gene_pairs)))

    # export_dir = "../emb/"

    for current_iter in range(1,max_iter+1):
        if current_iter == 1:
            print("gene2vec dimension "+ str(dimension) +" iteration "+ str(current_iter)+ " start")
            model = gensim.models.Word2Vec(gene_pairs, size=dimension, window=window_size, min_count=1, workers=num_workers, iter=1, sg=sg)
            model.save(os.path.join(export_dir, "gene2vec_dim_"+str(dimension)+"_iter_"+str(current_iter)+".pth"))
            if txtOutput:
                outputTxt(os.path.join(export_dir, "gene2vec_dim_"+str(dimension)+"_iter_"+str(current_iter)+".pth"))
            print("gene2vec dimension "+ str(dimension) +" iteration "+ str(current_iter)+ " done")
            del model
        else:
            current_time = datetime.datetime.now()
            print(current_time)
            print("shuffle start " + str(len(gene_pairs)))
            random.shuffle(gene_pairs)
            current_time = datetime.datetime.now()
            print(current_time)
            print("shuffle done " + str(len(gene_pairs)))

            print("gene2vec dimension " + str(dimension) + " iteration " + str(current_iter) + " start")
            model = gensim.models.Word2Vec.load(os.path.join(export_dir, "gene2vec_dim_"+str(dimension)+"_iter_"+str(current_iter-1)+".pth"))
            model.train(gene_pairs,total_examples=model.corpus_count,epochs=model.iter)
            model.save(os.path.join(export_dir, "gene2vec_dim_"+str(dimension)+"_iter_"+str(current_iter)+".pth"))
            if txtOutput:
                outputTxt(os.path.join(export_dir, "gene2vec_dim_"+str(dimension)+"_iter_"+str(current_iter)+".pth"))
            print("gene2vec dimension " + str(dimension) + " iteration " + str(current_iter) + " done")
            del model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Please specify data directory, embedding output directory and data file ending pattern')
    parser.add_argument('fileAddress', metavar='N', type=str, nargs='+',
                        help='python gene2vec.py data_directory output_directory txt')

    args = parser.parse_args()
    sourceDir = args.fileAddress[0]  # source directory of the files
    export_dir = args.fileAddress[1]
    ending_pattern = args.fileAddress[2]

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    print("start!")

    gene2vec(sourceDir, export_dir, ending_pattern)
