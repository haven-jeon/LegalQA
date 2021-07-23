# coding=utf-8
# Modified MIT License

# Software Copyright (c) 2021 Heewon Jeon

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# The above copyright notice and this permission notice need not be included
# with content created by the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import logging
import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from jina import DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torchmetrics import Accuracy
from pytorch_lightning import loggers as pl_loggers
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import BertForNextSentencePrediction

from kobert_tokenizer import KoBERTTokenizer

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument('--max_seq_len',
                            type=int,
                            default=128,
                            help='')
        return parser


class ReRankerDataset(Dataset):
    label_to_int = {'negative': 0, 'positive': 1}

    def __init__(self, dataset: List, model_name: str = 'skt/kobert-base-v1', max_seq_len: int = 128):
        self.tokenizer = KoBERTTokenizer.from_pretrained(model_name)
        self.dataset = dataset
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict:
        q, contents = self.dataset[index]
        numbers = list(range(0, index)) + list(range(index  + 1, self.__len__()))
        _, n_contents = self.dataset[random.choice(numbers)]
        c_encoding = self.tokenizer.encode(text=q,
                                     text_pair=contents)
        while len(c_encoding) < self.max_seq_len:
            c_encoding += [self.tokenizer.pad_token_id]
        nc_encoding = self.tokenizer.encode(text=q,
                                text_pair=n_contents)
        while len(nc_encoding) < self.max_seq_len:
            nc_encoding += [self.tokenizer.pad_token_id]       
        labels = [self.label_to_int['positive'], self.label_to_int['negative']]
        return {
            'input_ids': np.array([c_encoding[:self.max_seq_len]] + [nc_encoding[:self.max_seq_len]], dtype=np.int_),
            'labels': np.array(labels, dtype=np.int_)
        }

    @staticmethod
    def collate_fn(batch):
        input_ids = torch.LongTensor(np.concatenate([item['input_ids'] for item in batch]))
        labels = torch.LongTensor(np.concatenate([item['labels'] for item in batch]))

        return {
            'input_ids': input_ids,
            'labels': labels
        }


class ReRankDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_dataset: List,
                 last_n_test: int = 100,
                 max_seq_len: int = 128,
                 model_name: str = 'skt/kobert-base-v1',
                 batch_size: int = 32, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.train = train_dataset
        self.test = train_dataset[-last_n_test: ]
        self.model_name = model_name

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--model_name',
                            type=str,
                            default='skt/kobert-base-v1',
                            help='huggingface.co model name')
        return parser

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        self.train_ds = ReRankerDataset(self.train, self.model_name,
                                        self.max_seq_len)
        self.test_ds = ReRankerDataset(self.test, self.model_name,
                                       self.max_seq_len)

    # return the dataloader for each split
    def train_dataloader(self):
        train_dl = DataLoader(self.train_ds,
                              batch_size=int(self.batch_size / 2),
                              num_workers=5, shuffle=True, collate_fn=ReRankerDataset.collate_fn)
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(self.test_ds,
                            batch_size=int(self.batch_size / 2),
                            num_workers=5, shuffle=False, collate_fn=ReRankerDataset.collate_fn)
        return val_dl

    def test_dataloader(self):
        test_dl = DataLoader(self.test_ds,
                             batch_size=int(self.batch_size / 2),
                             num_workers=5, shuffle=False, collate_fn=ReRankerDataset.collate_fn)
        return test_dl



class BaseModule(pl.LightningModule):
    def __init__(self, hparams, **kwargs) -> None:
        super(BaseModule, self).__init__()
        self.save_hyperparameters(hparams)

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument('--batch_size',
                            type=int,
                            default=128,
                            help='batch size for training (default: 128)')

        parser.add_argument('--lr',
                            type=float,
                            default=2e-5,
                            help='The initial learning rate')

        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')

        return parser

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_workers = (self.hparams.gpus if self.hparams.gpus is not None else 1) * (self.hparams.num_nodes if self.hparams.num_nodes is not None else 1)
        data_len = len(self.train_dataloader().dataset)
        logging.info(f'number of workers {num_workers}, data length {data_len}')
        num_train_steps = int(data_len / (self.hparams.batch_size * num_workers * self.hparams.accumulate_grad_batches) * self.hparams.max_epochs)
        logging.info(f'num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        logging.info(f'num_warmup_steps : {num_warmup_steps}')
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]


class KoBERTReRanker(BaseModule):
    def __init__(self, hparams, **kwargs):
        super(KoBERTReRanker, self).__init__(hparams, **kwargs)
        self.model = BertForNextSentencePrediction.from_pretrained(self.hparams.model_name)
        self.valid_acc = Accuracy()
        self.train_acc = Accuracy()
        self.loss_f = CrossEntropyLoss() 

    def forward(self, input_ids, labels):
        return self.model(input_ids=input_ids, labels=labels)


    def training_step(self, batch, batch_idx):
        input_ids= batch['input_ids']
        labels = batch['labels']
        out = self(input_ids, labels=labels)
        self.log('train_loss', out.loss, prog_bar=True)
        self.train_acc(torch.nn.functional.softmax(out.logits, dim=1), labels)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False, prog_bar=True)
        return out.loss

    def validation_step(self, batch, batch_idx):
        input_ids= batch['input_ids']
        labels = batch['labels']
        out = self(input_ids, labels=None)
        self.valid_acc(torch.nn.functional.softmax(out.logits, dim=1), labels)
        self.log('val_acc', self.valid_acc, on_step=True, on_epoch=True, prog_bar=True)


def train():
    global parser
    parser = BaseModule.add_model_specific_args(parser)
    parser = ArgsBase.add_model_specific_args(parser)
    parser = ReRankDataModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    logger.info(args)
    return args



class BertReRanker(Executor):

    def __init__(self, model_name: str ='skt/kobert-base-v1',
                 max_seq_len: int = 128, batch_size: int = 64, max_epochs: int = 3,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams = {'batch_size': batch_size,
                        'max_seq_len': max_seq_len,
                        'max_epochs': max_epochs,
                        'model_name': model_name}
        self.logger = JinaLogger(self.__class__.__name__)
    
    def _init_(self):
        parser = argparse.ArgumentParser(description='BERT ReRanker')
        parser = BaseModule.add_model_specific_args(parser)
        parser = ArgsBase.add_model_specific_args(parser)
        parser = ReRankDataModule.add_model_specific_args(parser)
        parser = pl.Trainer.add_argparse_args(parser)
        parser.add_argument('-t', type=str, default='train')
        args = parser.parse_args()
        self.logger.info(args)
        return args

    @requests(on='/train')
    def train(self, docs: Optional[DocumentArray],
                     parameters: Dict,
                     **kwargs):
        args = self._init_()
        args.batch_size = self.hparams['batch_size']
        args.max_seq_len = self.hparams['max_seq_len']
        args.max_epochs = self.hparams['max_epochs']
        args.model_name = self.hparams['model_name']
        args.default_root_dir = "training"
        
        # prepare docs to train dataset, valdataset
        dataset = list(zip(docs.get_attributes('tags__title'), docs.get_attributes('tags__question')))

        model = KoBERTReRanker(args)
        dm = ReRankDataModule(train_dataset=dataset, **vars(args))
        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_acc',
                                                    dirpath=args.default_root_dir,
                                                    filename='model_chp/{epoch:02d}-{val_acc:.3f}',
                                                    verbose=True,
                                                    save_last=True,
                                                    mode='max',
                                                    save_top_k=-1)
        tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.default_root_dir, 'tb_logs'))
        lr_logger = pl.callbacks.LearningRateMonitor()
        trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger,
                                            callbacks=[checkpoint_callback, lr_logger]) 
        trainer.fit(model, dm)
        
