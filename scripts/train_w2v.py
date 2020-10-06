import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collections import Counter
from gensim.models import Word2Vec
import spacy
import pickle
import pandas as pd


LOW_NUM = "<low_num>"
MEDIUM_NUM = "<medium_num>"
HIGH_NUM = "<high_num>"


class Corpus(object):

    def __init__(self, dataset, core_model):
        self.dataset = dataset
        self.core_model = core_model
        self.counter = Counter()

    def create_mapping(self, min_count=5):
        mapping = dict()
        mapping['<pad>'] = 0
        for word, count in self.counter.items():
            if count >= min_count:
                mapping[word] = len(mapping)
        mapping['<unk>'] = len(mapping)
        return mapping

    def _cleaning(self, sentence):
        cleaned_sentence = []
        for token in sentence:
            if token.is_punct or token.is_space:
                continue
            elif token.is_digit:
                if int(token.text) < 20:
                    cleaned_sentence.append(LOW_NUM)
                    self.counter.update([LOW_NUM])
                elif int(token.text) >= 20 and int(token.text) < 1000:
                    cleaned_sentence.append(MEDIUM_NUM)
                    self.counter.update([MEDIUM_NUM])
                else:
                    cleaned_sentence.append(HIGH_NUM)
                    self.counter.update([HIGH_NUM])
            else:
                cleaned_sentence.append(token.lemma_)
                self.counter.update([token.lemma_])
        return cleaned_sentence

    def get_vocabulary(self):
        return self.create_mapping()

    def __call__(self):
        sentences = []
        doc_count = 0
        for text in self.dataset.text:
            if isinstance(text, str):
                doc = self.core_model(text)
                doc_count += 1
                if doc_count % 20 == 0:
                    print(f"\nProcessed {doc_count} / {len(self.dataset)} documents")
                for sent in doc.sents:
                    cleaned_sent = self._cleaning(sent)
                    if cleaned_sent:
                        sentences.append(cleaned_sent)
        return sentences


if __name__ == "__main__":
    core_model = spacy.load('en_core_web_sm')
    dataset = pd.read_csv("data/clean_fake_news_dataset.csv")

    corpus = Corpus(dataset=dataset, core_model=core_model)

    sentences = corpus()
    with open("data/w2v_sentences.pkl", "wb") as f:
        pickle.dump(sentences, f)

    vocabulary = corpus.get_vocabulary()
    with open("models_files/w2v_vocabulary.pkl", "wb") as f:
        pickle.dump(vocabulary, f)

    model = Word2Vec(sentences,
                     min_count=5,  # Ignore words that appear less than this
                     size=100,  # Dimensionality of word embeddings
                     window=5,  # Context window for words during training
                     iter=3)

    model.wv.save_word2vec_format('models_files/w2v_model.bin', binary=True)
