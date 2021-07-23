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
import sys
import random

import pandas as pd
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torchmetrics import Accuracy, PearsonCorrcoef, SpearmanCorrcoef
from pytorch_lightning import loggers as pl_loggers
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import BartModel
from pytorch_lightning.trainer.supporters import CombinedLoader

from kobart import get_kobart_tokenizer, get_pytorch_kobart_model


parser = argparse.ArgumentParser(description='subtask for KoBART')

parser.add_argument('--subtask',
                    type=str,
                    default='NLI',
                    help='NLI or STS or NLISTS')

parser.add_argument('--checkpoint_path',
                    type=str,
                    help='checkpoint path')

parser.add_argument('--avg_type',
                    type=str,
                    default='norm_avg',
                    help='norm_avg or avg')

parser.add_argument('--alpha',
                    type=float,
                    default=0.7,
                    help='')

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--train_file',
                            nargs='+',
                            help='train file')

        parser.add_argument('--test_file',
                             nargs='+',
                            help='test file')

        parser.add_argument('--max_seq_len',
                            type=int,
                            default=128,
                            help='')
        return parser


class NLIDataset(Dataset):

    label_to_int = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

    def __init__(self, filepath=None, max_seq_len=128):
        self.filepath = filepath
        if self.filepath:
            self.data = pd.read_csv(self.filepath, sep='\t').dropna()
        self.max_seq_len = max_seq_len
        self.tokenizer = get_kobart_tokenizer()

    def __len__(self):
        return len(self.data)

    def _encode(self, text):
        tokens = [self.tokenizer.bos_token] + \
            self.tokenizer.tokenize(text) + [self.tokenizer.eos_token]
        encoder_input_id = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(encoder_input_id)
        if len(encoder_input_id) < self.max_seq_len:
            while len(encoder_input_id) < self.max_seq_len:
                encoder_input_id += [self.tokenizer.pad_token_id]
                attention_mask += [0]
        else:
            encoder_input_id = encoder_input_id[:self.max_seq_len - 1] + [
                self.tokenizer.eos_token_id]
            attention_mask = attention_mask[:self.max_seq_len]
        return np.array(encoder_input_id, dtype=np.int_), np.array(attention_mask, dtype=np.float32)

    def __getitem__(self, index):
        record = self.data.iloc[index]
        premise, hypothesis, label = str(record['premise']), str(record['hypothesis']), str(record['gold_label'])

        p_input_ids, p_attention_mask = self._encode(premise)
        h_input_ids, h_attention_mask = self._encode(hypothesis)

        return {'p_input_ids': p_input_ids,
                'p_attention_mask': p_attention_mask,
                'h_input_ids': h_input_ids,
                'h_attention_mask': h_attention_mask,
                'labels': np.array(self.label_to_int[label], dtype=np.int_)}

class NLIDataModule(pl.LightningDataModule):
    def __init__(self, train_file,
                 test_file,
                 max_seq_len=128,
                 batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.train_file_path = train_file
        self.test_file_path = test_file

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        return parser

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        self.nli_train = NLIDataset(self.train_file_path,
                                      self.max_seq_len)
        self.nli_test = NLIDataset(self.test_file_path,
                                     self.max_seq_len)

    # return the dataloader for each split
    def train_dataloader(self):
        nli_train = DataLoader(self.nli_train,
                                batch_size=self.batch_size,
                                num_workers=5, shuffle=True)
        return nli_train

    def val_dataloader(self):
        nli_val = DataLoader(self.nli_test,
                              batch_size=int(self.batch_size / 2),
                              num_workers=5, shuffle=False)
        return nli_val

    def test_dataloader(self):
        nli_test = DataLoader(self.nli_test,
                               batch_size=int(self.batch_size / 2),
                               num_workers=5, shuffle=False)
        return nli_test



class NLISTSDataModule(pl.LightningDataModule):
    def __init__(self, 
                 nli_train_file,
                 nli_test_file,
                 sts_train_file,
                 sts_test_file,
                 max_seq_len=128,
                 batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.nli_train_file_path = nli_train_file
        self.nli_test_file_path = nli_test_file
        self.sts_train_file_path = sts_train_file
        self.sts_test_file_path = sts_test_file
        

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        return parser

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        self.nli_train = NLIDataset(self.nli_train_file_path,
                                      self.max_seq_len)
        self.nli_test = NLIDataset(self.nli_test_file_path,
                                     self.max_seq_len)
        self.sts_train = STSDataset(self.sts_train_file_path,
                                      self.max_seq_len)
        self.sts_test = STSDataset(self.sts_test_file_path,
                                     self.max_seq_len)

    # return the dataloader for each split
    def train_dataloader(self):
        nli_train = DataLoader(self.nli_train,
                                batch_size=int(self.batch_size),
                                num_workers=5, shuffle=True)
        sts_train = DataLoader(self.sts_train,
                                batch_size=int(self.batch_size),
                                num_workers=5, shuffle=True)
        return CombinedLoader({'sts': sts_train, 'nli': nli_train}, 'min_size')

    def val_dataloader(self):
        nli_val = DataLoader(self.nli_test,
                              batch_size=int(self.batch_size / 2),
                              num_workers=5, shuffle=False)
        sts_val = DataLoader(self.sts_test,
                              batch_size=int(self.batch_size / 2),
                              num_workers=5, shuffle=False)
        return CombinedLoader({'sts': sts_val, 'nli': nli_val}, 'min_size')

    def test_dataloader(self):
        nli_test = DataLoader(self.nli_test,
                               batch_size=int(self.batch_size / 2),
                               num_workers=5, shuffle=False)
        sts_test = DataLoader(self.sts_test,
                              batch_size=int(self.batch_size / 2),
                              num_workers=5, shuffle=False)
        return CombinedLoader({'sts': sts_test, 'nli': nli_test}, 'min_size')

class STSDataset(Dataset):

    def __init__(self, filepath=None, max_seq_len=128):
        self.filepath = filepath
        if self.filepath:
            self.data = pd.read_csv(self.filepath, sep='\t').dropna()
        self.max_seq_len = max_seq_len
        self.tokenizer = get_kobart_tokenizer()

    def __len__(self):
        return len(self.data)

    def _encode(self, text):
        tokens = [self.tokenizer.bos_token] + \
            self.tokenizer.tokenize(text) + [self.tokenizer.eos_token]
        encoder_input_id = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(encoder_input_id)
        if len(encoder_input_id) < self.max_seq_len:
            while len(encoder_input_id) < self.max_seq_len:
                encoder_input_id += [self.tokenizer.pad_token_id]
                attention_mask += [0]
        else:
            encoder_input_id = encoder_input_id[:self.max_seq_len - 1] + [
                self.tokenizer.eos_token_id]
            attention_mask = attention_mask[:self.max_seq_len]
        return np.array(encoder_input_id, dtype=np.int_), np.array(attention_mask, dtype=np.float32)

    def __getitem__(self, index):
        record = self.data.iloc[index]
        st1, st2, label = str(record['sentence1']), str(record['sentence2']), float(record['label'])
        sh = [st1, st2]
        random.shuffle(sh)
        st1, st2 = sh
        s1_input_ids, s1_attention_mask = self._encode(st1)
        s2_input_ids, s2_attention_mask = self._encode(st2)

        return {'p_input_ids': s1_input_ids,
                'p_attention_mask': s1_attention_mask,
                'h_input_ids': s2_input_ids,
                'h_attention_mask': s2_attention_mask,
                'labels': np.array(label, dtype=np.float_)}


class STSDataModule(pl.LightningDataModule):
    def __init__(self, train_file,
                 test_file,
                 max_seq_len=128,
                 batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.train_file_path = train_file
        self.test_file_path = test_file

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        return parser

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        self.sts_train = STSDataset(self.train_file_path,
                                      self.max_seq_len)
        self.sts_test = STSDataset(self.test_file_path,
                                     self.max_seq_len)

    # return the dataloader for each split
    def train_dataloader(self):
        sts_train = DataLoader(self.sts_train,
                                batch_size=self.batch_size,
                                num_workers=5, shuffle=True)
        return sts_train

    def val_dataloader(self):
        sts_val = DataLoader(self.sts_test,
                              batch_size=int(self.batch_size / 2),
                              num_workers=5, shuffle=False)
        return sts_val

    def test_dataloader(self):
        sts_test = DataLoader(self.sts_test,
                               batch_size=int(self.batch_size / 2),
                               num_workers=5, shuffle=False)
        return sts_test

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

class PoolingHead(nn.Module):

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        return hidden_states


class KoBARTClassification(BaseModule):
    def __init__(self, hparams, **kwargs):
        super(KoBARTClassification, self).__init__(hparams, **kwargs)
        self.model = BartModel.from_pretrained(get_pytorch_kobart_model())
        self.pooling = PoolingHead(input_dim=self.model.config.d_model,
            inner_dim=self.model.config.d_model,
            pooler_dropout=0.1)
        self.classification = nn.Linear(self.model.config.d_model * 3, len(NLIDataset.label_to_int))
        self.valid_acc = Accuracy()
        self.train_acc = Accuracy()
        self.loss_f = CrossEntropyLoss() 

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

    def _get_encoding(self, input_ids, attention_mask, typ='norm_avg'):
        if typ == 'norm_avg':
            outs = self(input_ids, attention_mask)
            lengths = attention_mask.sum(dim=1)
            mask_3d = attention_mask.unsqueeze(dim=-1).repeat_interleave(repeats=self.model.config.d_model, dim=2)

            masked_encoder_out = (outs['last_hidden_state'] * mask_3d).sum(dim=1)
            # to avoid 0 division 
            norm_encoder_out = masked_encoder_out / (lengths + 1).unsqueeze(dim=1)
            return norm_encoder_out
        elif typ == 'avg':
            last_hidden = self(input_ids, attention_mask)['last_hidden_state']
            return torch.mean(last_hidden, dim=1)

    def encoding(self, input_ids, attention_mask):
        return self.pooling(self._get_encoding(input_ids, attention_mask, typ=self.hparams.avg_type))

    def _step(self, batch):
        p_input_ids= batch['p_input_ids']
        p_attention_mask = batch['p_attention_mask']
        h_input_ids= batch['h_input_ids']
        h_attention_mask = batch['h_attention_mask']
        # encode of premise [batch, 768]
        p_encoding = self._get_encoding(p_input_ids, p_attention_mask, typ=self.hparams.avg_type)
        h_encoding = self._get_encoding(h_input_ids, h_attention_mask, typ=self.hparams.avg_type)
        u = self.pooling(p_encoding)
        v = self.pooling(h_encoding)
        logits = self.classification(torch.cat((u, v, torch.abs(u - v)), dim=1))
        return logits

    def training_step(self, batch, batch_idx):
        labels = batch['labels']
        logits = self._step(batch)
        loss = self.loss_f(logits, labels)
        self.log('train_loss', loss, prog_bar=True)
        self.train_acc(torch.nn.functional.softmax(logits, dim=1), labels)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch['labels']
        logits = self._step(batch)
        self.valid_acc(torch.nn.functional.softmax(logits, dim=1), labels)
        self.log('val_acc', self.valid_acc, on_step=True, on_epoch=True, prog_bar=True)


class KoBARTRegression(BaseModule):
    def __init__(self, hparams, **kwargs):
        super(KoBARTRegression, self).__init__(hparams, **kwargs)
        self.model = BartModel.from_pretrained(get_pytorch_kobart_model())
        self.pooling = PoolingHead(input_dim=self.model.config.d_model,
            inner_dim=self.model.config.d_model,
            pooler_dropout=0.1)
        self.valid_pearson = PearsonCorrcoef()
        self.valid_spearman = SpearmanCorrcoef()


    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

    def _get_encoding(self, input_ids, attention_mask, typ='norm_avg'):
        if typ == 'norm_avg':
            outs = self(input_ids, attention_mask)
            lengths = attention_mask.sum(dim=1)
            mask_3d = attention_mask.unsqueeze(dim=-1).repeat_interleave(repeats=self.model.config.d_model, dim=2)

            masked_encoder_out = (outs['last_hidden_state'] * mask_3d).sum(dim=1)
            # to avoid 0 division 
            norm_encoder_out = masked_encoder_out / (lengths + 1).unsqueeze(dim=1)
            return norm_encoder_out
        elif typ == 'avg':
            last_hidden = self(input_ids, attention_mask)['last_hidden_state']
            return torch.mean(last_hidden, dim=1)

    def encoding(self, input_ids, attention_mask):
        return self.pooling(self._get_encoding(input_ids, attention_mask, typ=self.hparams.avg_type))

    def _step(self, batch):
        p_input_ids= batch['p_input_ids']
        p_attention_mask = batch['p_attention_mask']
        h_input_ids= batch['h_input_ids']
        h_attention_mask = batch['h_attention_mask']
        # encode of premise [batch, 768]
        p_encoding = self._get_encoding(p_input_ids, p_attention_mask, typ=self.hparams.avg_type)
        h_encoding = self._get_encoding(h_input_ids, h_attention_mask, typ=self.hparams.avg_type)
        u = self.pooling(p_encoding)
        v = self.pooling(h_encoding)
        return torch.cosine_similarity(u, v)

    def training_step(self, batch, batch_idx):
        labels = batch['labels']
        logits = self._step(batch)
        loss = torch.nn.functional.mse_loss(logits.squeeze(), labels.float() / 5.0)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch['labels']
        logits = self._step(batch)
        self.valid_pearson(logits.squeeze(), labels.float() / 5.0)
        self.valid_spearman(logits.squeeze(), labels.float() / 5.0)
        self.log('val_pearson', self.valid_pearson, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_spearman', self.valid_spearman, on_step=True, on_epoch=True, prog_bar=True)



class KoBARTMultiTask(BaseModule):
    def __init__(self, hparams, **kwargs):
        super(KoBARTMultiTask, self).__init__(hparams, **kwargs)
        self.model = BartModel.from_pretrained(get_pytorch_kobart_model())
        self.pooling = PoolingHead(input_dim=self.model.config.d_model,
            inner_dim=self.model.config.d_model,
            pooler_dropout=0.1)
        self.classification = nn.Linear(self.model.config.d_model * 3, len(NLIDataset.label_to_int))
        self.loss_f = CrossEntropyLoss() 
        self.valid_pearson = PearsonCorrcoef()
        self.valid_spearman = SpearmanCorrcoef()
        self.valid_acc = Accuracy()
        self.train_acc = Accuracy()


    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

    def _get_encoding(self, input_ids, attention_mask, typ='norm_avg'):
        if typ == 'norm_avg':
            outs = self(input_ids, attention_mask)
            lengths = attention_mask.sum(dim=1)
            mask_3d = attention_mask.unsqueeze(dim=-1).repeat_interleave(repeats=self.model.config.d_model, dim=2)

            masked_encoder_out = (outs['last_hidden_state'] * mask_3d).sum(dim=1)
            # to avoid 0 division 
            norm_encoder_out = masked_encoder_out / (lengths + 1).unsqueeze(dim=1)
            return norm_encoder_out
        elif typ == 'avg':
            last_hidden = self(input_ids, attention_mask)['last_hidden_state']
            return torch.mean(last_hidden, dim=1)

    def encoding(self, input_ids, attention_mask):
        return self.pooling(self._get_encoding(input_ids, attention_mask, typ=self.hparams.avg_type))

    def _step_sts(self, batch):
        u, v = self._step(batch)
        return torch.cosine_similarity(u, v)

    def _step(self, batch):
        p_input_ids= batch['p_input_ids']
        p_attention_mask = batch['p_attention_mask']
        h_input_ids= batch['h_input_ids']
        h_attention_mask = batch['h_attention_mask']
        # encode of premise [batch, 768]
        p_encoding = self._get_encoding(p_input_ids, p_attention_mask, typ=self.hparams.avg_type)
        h_encoding = self._get_encoding(h_input_ids, h_attention_mask, typ=self.hparams.avg_type)
        u = self.pooling(p_encoding)
        v = self.pooling(h_encoding)
        return u, v

    def _step_nli(self, batch):
        u, v = self._step(batch)
        logits = self.classification(torch.cat((u, v, torch.abs(u - v)), dim=1))
        return logits

    def training_step(self, batch, batch_idx):
        b_sts = batch['sts']
        b_nli = batch['nli']
        nli_logits = self._step_nli(b_nli)
        sts_logits = self._step_sts(b_sts)
        sts_loss = torch.nn.functional.mse_loss(sts_logits.squeeze(), b_sts['labels'].float() / 5.0)
        nli_loss = self.loss_f(nli_logits, b_nli['labels'])
        self.log('train_sts_loss', sts_loss, prog_bar=True)
        self.log('train_nli_loss', nli_loss, prog_bar=True)
        return (1.0 - self.hparams.alpha) * nli_loss + (self.hparams.alpha) * sts_loss

    def validation_step(self, batch, batch_idx):
        labels = batch['sts']['labels']
        sts_logits = self._step_sts(batch['sts'])
        self.valid_pearson(sts_logits.squeeze(), labels.float() / 5.0)
        self.valid_spearman(sts_logits.squeeze(), labels.float() / 5.0)
        self.log('val_pearson', self.valid_pearson, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_spearman', self.valid_spearman, on_step=True, on_epoch=True, prog_bar=True)
        labels = batch['nli']['labels']
        nli_logits = self._step_nli(batch['nli'])
        self.valid_acc(torch.nn.functional.softmax(nli_logits, dim=1), labels)
        self.log('val_acc', self.valid_acc, on_step=True, on_epoch=True, prog_bar=True)


if __name__ == '__main__':
    parser = BaseModule.add_model_specific_args(parser)
    parser = ArgsBase.add_model_specific_args(parser)
    parser = NLIDataModule.add_model_specific_args(parser)
    parser = STSDataModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    logging.info(args)

    if args.subtask == 'NLISTS':
        if args.checkpoint_path:
            model = KoBARTMultiTask.load_from_checkpoint(checkpoint_path=args.checkpoint_path)
            logging.info('model loaded from checkpoints')
            sys.exit()
        else:
            # init model
            model = KoBARTMultiTask(args)
        dm = NLISTSDataModule(args.train_file[0], #nli
                              args.test_file[0],
                              args.train_file[1], #sts
                              args.test_file[1],
                              batch_size=args.batch_size, max_seq_len=args.max_seq_len)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_acc',
                                                           dirpath=args.default_root_dir,
                                                           filename='model_chp/{epoch:02d}-{val_acc:.3f}',
                                                           verbose=True,
                                                           save_last=True,
                                                           mode='max',
                                                           save_top_k=-1)
    elif args.subtask == 'NLI':
        # init data
        if args.checkpoint_path:
            model = KoBARTClassification.load_from_checkpoint(checkpoint_path=args.checkpoint_path)
            logging.info('model loaded from checkpoints')
            sys.exit()
        else:
            # init model
            model = KoBARTClassification(args)
        dm = NLIDataModule(args.train_file[0],
                           args.test_file[0],
                           batch_size=args.batch_size, max_seq_len=args.max_seq_len)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_acc',
                                                           dirpath=args.default_root_dir,
                                                           filename='model_chp/{epoch:02d}-{val_acc:.3f}',
                                                           verbose=True,
                                                           save_last=True,
                                                           mode='max',
                                                           save_top_k=-1)
    elif args.subtask == 'STS':
        if args.checkpoint_path:
            model = KoBARTRegression.load_from_checkpoint(strict=False, **vars(args))
            logging.info('model loaded from checkpoints')
        else:
            # init model
            model = KoBARTRegression(args)
        dm = STSDataModule(args.train_file[0],
                           args.test_file[0],
                           batch_size=args.batch_size, max_seq_len=args.max_seq_len)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_pearson',
                                                           dirpath=args.default_root_dir,
                                                           filename='model_chp/{epoch:02d}-{val_pearson:.3f}',
                                                           verbose=True,
                                                           save_last=True,
                                                           mode='max',
                                                           save_top_k=-1)
    else:
        # add more subtasks
        assert False
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.default_root_dir, 'tb_logs'))
    # train
    lr_logger = pl.callbacks.LearningRateMonitor()
    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger,
                                            callbacks=[checkpoint_callback, lr_logger])  
    trainer.fit(model, dm)
