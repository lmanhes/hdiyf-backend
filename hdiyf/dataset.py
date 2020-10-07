import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from hdiyf.utils import calculate_maxlen


class TrainingDataset(Dataset):

    def __init__(self, dataset_file, vocabulary, core_model, config):
        self.dataset = pd.read_csv(dataset_file)
        self.dataset = self.dataset.sample(frac=1).iloc[:5000]
        self.vocab = vocabulary
        self.core_model = core_model
        self.config = config

        print("\nProcess texts ...")
        self.text_docs = []
        self.labels = []
        for i, doc in enumerate(self.dataset.text):
            if type(doc) == str:
                self.text_docs.append(self.preprocessing(self.core_model(doc)))
                self.labels.append(self.dataset.iloc[i].label)
        #self.maxlen_doc = calculate_maxlen(self.text_docs)
        self.maxlen_doc = 300
        print("maxlen doc : ", self.maxlen_doc)

        config["maxlen_doc"] = self.maxlen_doc

    def __len__(self):
        return len(self.text_docs)

    @staticmethod
    def is_a_good_one(token):
        if (token.is_stop or token.is_punct or token.is_digit
                or token.is_space or not token.is_alpha):
            return False
        return True

    def preprocessing(self, doc):
        return [token.lemma_ for token in doc if self.is_a_good_one(token)]

    def vectorize(self, clean_doc):
        X_word = np.ones((self.maxlen_doc,), dtype=np.int64) * self.vocab.get("<pad>")
        X_len_word = np.array(len(clean_doc))

        for j, word in enumerate(clean_doc):
            if j >= self.maxlen_doc:
                continue

            X_word[j] = self.vocab.get(word, self.vocab.get("<unk>"))

        sample = {
            "word": X_word,
            "length": X_len_word
        }

        return sample

    def __getitem__(self, idx):

        text_doc = self.text_docs[idx]
        vectorized_text = self.vectorize(text_doc)

        label = int(self.labels[idx])

        return vectorized_text, label