import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import spacy

from hdiyf.utils import load_word2vec_embeddings
from hdiyf.nnet import AttentiveBilstm
from hdiyf.dataset import TrainingDataset
from hdiyf.trainer import Trainer


config = {
    "type": "templstm",
    "training":
        {
            "learning_rate": 0.001,
            "test_size": 0.1,
            "batch_size": 32,
            "sampling": True,
            "max_epochs": 30,
            "grad_norm_clip": 1.0,
            "weight_decay": 1e-5,
            "lr_decay": False,
            "ckpt_path": "models_files",
            "num_workers": 0, # for DataLoader
            "min_delta": 1e-3,
            "patience": 3
        },
    "preprocessing":
        {
        },
    "model":
        {
            "class_num": None,
            "embedding_dropout": 0.3,
            "fc_dropout": 0.5,
            "contextual_hidden_dim": 256,
            "hidden_dim": 256,
            "num_layers": 1
        }
}


if __name__ == "__main__":

    core_model = spacy.load('en_core_web_sm')

    with open("models_files/w2v_vocabulary.pkl", "rb") as f:
        vocabulary = pickle.load(f)

    config["model"]["word_vocab_size"] = len(vocabulary)
    embeddings, embedding_dim = load_word2vec_embeddings('models_files/w2v_model.bin',
                                                         vocabulary)
    config['model']['word_embedding_dim'] = embedding_dim

    # create dataset
    dataset = TrainingDataset(dataset_file="data/clean_fake_news_dataset.csv",
                              vocabulary=vocabulary,
                              core_model=core_model,
                              config=config['preprocessing'])
    config["model"]["class_num"] = 2

    model = AttentiveBilstm(embeddings=embeddings, args=config['model'])

    trainer = Trainer(model, dataset, config)
    trainer.train()