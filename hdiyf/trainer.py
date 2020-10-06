import json
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from sklearn.metrics import f1_score


class Trainer(object):

    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.training_config = self.config['training']

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()

    def save_checkpoint(self):
        model_path = os.path.join(self.training_config['ckpt_path'], f"model.pth")
        torch.save(self.model.state_dict(), model_path)

        config_path = os.path.join(self.training_config['ckpt_path'], "config.json")
        json.dump(self.config, open(config_path, 'w'))

    def train(self):
        model, config = self.model, self.training_config
        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                               lr=config['learning_rate'],
                               weight_decay=config['weight_decay'])

        # split dataset into train and test
        dataset = self.dataset
        test_size = int(config["test_size"] * len(dataset))
        train_size = len(dataset) - test_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = train_dataset if is_train else test_dataset

            loader = DataLoader(data, shuffle=is_train, pin_memory=True,
                                batch_size=config['batch_size'],
                                num_workers=config['num_workers'])

            losses = []
            scores = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (inputs, targets) in pbar:

                x = inputs['word'].to(self.device)
                y = targets.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    logit, loss = model(x, y)

                    y_pred = torch.max(logit, -1)[1].numpy()
                    score = f1_score(y.numpy(), y_pred, average='macro')
                    scores.append(score)

                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_norm_clip'])
                    optimizer.step()

                    # report progress
                    pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}.")

            if not is_train:
                test_loss = float(np.mean(losses))
                test_f1 = float(np.mean(scores))

                print(f"test loss: {test_loss}\ttest f1 : {test_f1}\n\n")

                return test_loss

        best_loss = float('inf')
        best_epoch = 1
        wait = 0  # counter used for early stopping
        for i, epoch in enumerate(range(config['max_epochs'])):

            run_epoch('train')
            test_loss = run_epoch('test')
            #scheduler.step(test_loss)

            # supports early stopping based on the test loss
            if (test_loss - best_loss) < -self.training_config['min_delta']:
                best_loss = test_loss
                best_epoch = i + 1
                wait = 1
                self.save_checkpoint()
            else:
                wait += 1

            if wait >= self.training_config['patience']:
                print(f'\nTerminated Training for Early Stopping at Epoch {best_epoch} with loss {best_loss}')
                break