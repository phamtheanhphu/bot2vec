import sys
from model.bot2vec import Bot2Vec


def main():
    osn_input_file = './data/inputs/cresci-2015/cresci-2015.edgelist.txt'
    emb_output_file = './data/outputs/cresci-2015/cresci-2015.emb'
    bot2vec = Bot2Vec(osn_input_file, emb_output_file)
    bot2vec.train()


if __name__ == "__main__":
    sys.exit(main())
