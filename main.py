import os

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import logging, AutoTokenizer, AutoModel

from config import get_config
from data import load_dataset
from model import Transformer, Gru_Model, BiLstm_Model, Lstm_Model, Rnn_Model, Transformer_Extend_LSTM
from sklearn.metrics import f1_score


class Niubility:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.logger.info('> creating model {}'.format(args.model_name))
        # Create model
        if args.model_name == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.base_model = AutoModel.from_pretrained('bert-base-uncased')
        elif args.model_name == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
            self.base_model = AutoModel.from_pretrained('roberta-base')
        elif args.model_name == 'sentiWSP-large':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.base_model = AutoModel.from_pretrained("shuaifan/SentiWSP")
        else:
            raise ValueError('unknown model')
        # Operate the method
        if args.method_name == 'fnn':
            self.Mymodel = Transformer(self.base_model, args.num_classes)
        elif args.method_name == 'gru':
            self.Mymodel = Gru_Model(self.base_model, args.num_classes)
        elif args.method_name == 'lstm':
            self.Mymodel = Lstm_Model(self.base_model, args.num_classes)
        elif args.method_name == 'bilstm':
            self.Mymodel = BiLstm_Model(self.base_model, args.num_classes)
        elif args.method_name == 'rnn':
            self.Mymodel = Rnn_Model(self.base_model, args.num_classes)
        elif args.method_name == 'wsp-lstm':
            self.Mymodel = Transformer_Extend_LSTM(self.base_model, args.num_classes)
        else:
            raise ValueError('unknown method')

        self.Mymodel.to(args.device)
        if args.device.type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
        self._print_args()

    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

    def _train(self, dataloader, criterion, optimizer):
        global score
        train_loss, n_train = 0, 0
        pre, tar = [], []
        # Turn on the train mode
        self.Mymodel.train()
        for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii='>='):
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}

            targets = targets.to(self.args.device)
            predicts = self.Mymodel(inputs)
            loss = criterion(predicts, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            n_train += targets.size(0)
            predicts = torch.argmax(predicts, dim=1)
            pre += predicts.cpu().tolist()
            tar += targets.cpu().tolist()
        score = f1_score(pre, tar, average='macro')
        return train_loss / n_train, score

    def _test(self, dataloader, criterion):
        test_loss, n_test = 0, 0
        pre, tar = [], []
        # Turn on the eval mode
        self.Mymodel.eval()

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}

                targets = targets.to(self.args.device)
                predicts = self.Mymodel(inputs)
                loss = criterion(predicts, targets)

                test_loss += loss.item() * targets.size(0)
                n_test += targets.size(0)

                predicts = torch.argmax(predicts, dim=1)
                pre += predicts.cpu().tolist()
                tar += targets.cpu().tolist()

        score = f1_score(predicts.cpu(), targets.cpu(), average='macro')
        return test_loss / n_test, score

    def _submit(self, dataloader):
        submit = []
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                predicts = self.Mymodel(inputs)
                predicts = torch.argmax(predicts, dim=1)
                submit += predicts.cpu().tolist()
        return submit

    def run(self):
        train_dataloader, test_dataloader, submit_dataloader = load_dataset(tokenizer=self.tokenizer,
                                                                            train_batch_size=self.args.train_batch_size,
                                                                            test_batch_size=self.args.test_batch_size,
                                                                            model_name=self.args.model_name,
                                                                            method_name=self.args.method_name,
                                                                            workers=self.args.workers)
        _params = filter(lambda x: x.requires_grad, self.Mymodel.parameters())
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.weight_decay)

        # Get the best_loss, the best_score and save the model
        submit = []
        best_loss, best_score = 0, 0
        for epoch in range(self.args.num_epoch):
            train_loss, train_score = self._train(train_dataloader, criterion, optimizer)
            test_loss, test_score = self._test(test_dataloader, criterion)
            if (epoch > 20):
                if test_score > best_score or (test_score == best_score and test_loss < best_loss):
                    best_score, best_loss = test_score, test_loss
                    submit = self._submit(submit_dataloader)
            self.logger.info(
                '{}/{} - {:.2f}%'.format(epoch + 1, self.args.num_epoch, 100 * (epoch + 1) / self.args.num_epoch))
            self.logger.info('[train] loss: {:.4f}, f1_score: {:.2f}'.format(train_loss, train_score * 100))
            self.logger.info('[test] loss: {:.4f}, f1_score: {:.2f}'.format(test_loss, test_score * 100))
        self.logger.info('best loss: {:.4f}, best f1_score: {:.2f}'.format(best_loss, best_score * 100))
        self.logger.info('log saved: {}'.format(self.args.log_name))

        # Get the final submit text
        with open('result.csv', "w") as f:
            f.write('ID,Label\n')
            index = 25000
            for line in submit:
                f.write(str(index) + ',' + str(line) + '\n')
                index += 1
        print("----------Congratulations!----------")


if __name__ == '__main__':
    logging.set_verbosity_error()
    args, logger = get_config()
    nb = Niubility(args, logger)
    nb.run()
