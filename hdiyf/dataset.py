import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from hdiyf.utils import calculate_maxlen


class TrainingDataset(Dataset):

    def __init__(self, dataset_file, vocabulary, core_model, config):
        self.dataset = pd.read_csv(dataset_file)
        self.dataset = self.dataset.iloc[:20]
        self.vocab = vocabulary
        self.core_model = core_model
        self.config = config

        print("\nProcess texts ...")
        self.text_docs = list(self.core_model.pipe(self.dataset.text))
        self.maxlen_doc = calculate_maxlen(self.text_docs)

        config["maxlen_doc"] = self.maxlen_doc

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def is_a_good_one(token):
        if (token.is_stop or token.is_punct or token.is_digit
                or token.is_space or not token.is_alpha):
            return False
        return True

    def preprocessing(self, doc):
        return [token.lemma_ for token in doc if self.is_a_good_one(token)]

    def vectorize(self, clean_doc):
        X_word = np.ones((self.maxlen_doc,), dtype=np.int64) * self.vocab['word'].get("<pad>")
        X_len_word = np.array(len(clean_doc))

        for j, word in enumerate(clean_doc):
            if j >= self.maxlen_doc:
                continue

            X_word[j] = self.vocab['word'].get(word, self.vocab['word'].get("<unk>"))

        sample = {
            "word": X_word,
            "length": X_len_word
        }

        return sample

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        text_doc = self.text_docs[idx]
        vectorized_text = self.vectorize(text_doc)

        label = int(self.dataset[idx].label)

        return vectorized_text, label