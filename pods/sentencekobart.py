from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch

from jina import DocumentArray, Executor, requests
from jina_commons.batching import get_docs_batch_generator
from jina.logging.logger import JinaLogger
from kobart import get_kobart_tokenizer, get_pytorch_kobart_model
from transformers import BartModel


class PoolingHead(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = torch.nn.Linear(input_dim, inner_dim)
        self.dropout = torch.nn.Dropout(p=pooler_dropout)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        return hidden_states


class KoBARTRegression(pl.LightningModule):
    def __init__(self, hparams, **kwargs):
        super(KoBARTRegression, self).__init__(**kwargs)
        self.save_hyperparameters(hparams)
        self.model = BartModel.from_pretrained(get_pytorch_kobart_model())
        self.pooling = PoolingHead(input_dim=self.model.config.d_model,
                                   inner_dim=self.model.config.d_model,
                                   pooler_dropout=0.1)
        self.classification = torch.nn.Linear(self.model.config.d_model * 3, 1)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          return_dict=True)

    def _get_encoding(self, input_ids, attention_mask, typ='norm_avg'):
        if typ == 'norm_avg':
            outs = self(input_ids, attention_mask)
            lengths = attention_mask.sum(dim=1)
            mask_3d = attention_mask.unsqueeze(dim=-1).repeat_interleave(
                repeats=self.model.config.d_model, dim=2)

            masked_encoder_out = (outs['last_hidden_state'] *
                                  mask_3d).sum(dim=1)
            # to avoid 0 division
            norm_encoder_out = masked_encoder_out / (lengths +
                                                     1).unsqueeze(dim=1)
            return norm_encoder_out
        elif typ == 'avg':
            last_hidden = self(input_ids, attention_mask)['last_hidden_state']
            return torch.mean(last_hidden, dim=1)

    def encoding(self, input_ids, attention_mask):
        return self.pooling(
            self._get_encoding(input_ids,
                               attention_mask,
                               typ=self.hparams.avg_type))


class KoSentenceBART(Executor):
    def __init__(
        self,
        pretrained_model_path: str = 'model/SentenceKoBART.bin',
        max_length: int = 128,
        device: str = 'cpu',
        default_traversal_paths: Optional[List[str]] = None,
        default_batch_size: int = 32,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if default_traversal_paths is not None:
            self.default_traversal_paths = default_traversal_paths
        else:
            self.default_traversal_paths = ['r']
        self.default_batch_size = default_batch_size
        self.pretrained_model_path = pretrained_model_path
        self.max_length = max_length

        self.logger = JinaLogger(self.__class__.__name__)
        if not device in ['cpu', 'cuda']:
            self.logger.error(
                'Torch device not supported. Must be cpu or cuda!')
            raise RuntimeError(
                'Torch device not supported. Must be cpu or cuda!')
        if device == 'cuda' and not torch.cuda.is_available():
            self.logger.warning(
                'You tried to use GPU but torch did not detect your'
                'GPU correctly. Defaulting to CPU. Check your CUDA installation!'
            )
            device = 'cpu'
        self.device = device
        self.tokenizer = get_kobart_tokenizer()
        self.model = KoBARTRegression.load_from_checkpoint(
            self.pretrained_model_path, hparams={'avg_type': 'norm_avg'})
        self.model.eval()
        self.model.to(torch.device(device))

    @requests
    def encode(self, docs: Optional[DocumentArray], parameters: Dict,
               **kwargs):
        """
        Encode text data into a ndarray of `D` as dimension, and fill the embedding of each Document.
        :param docs: DocumentArray containing text
        :param parameters: dictionary to define the `traversal_paths` and the `batch_size`. For example,
               `parameters={'traversal_paths': ['r'], 'batch_size': 10}`.
        :param kwargs: Additional key value arguments.
        """
        for batch in get_docs_batch_generator(
                docs,
                traversal_path=parameters.get('traversal_paths',
                                              self.default_traversal_paths),
                batch_size=parameters.get('batch_size',
                                          self.default_batch_size),
                needs_attr='text',
        ):
            texts = batch.get_attributes('text')
            processed_content = []
            for cont in texts:
                processed_content.append(self.tokenizer.bos_token + cont +
                                         self.tokenizer.eos_token)
            input_tensors = self.tokenizer(list(processed_content),
                                           return_tensors='pt',
                                           max_length=self.max_length,
                                           padding=True)
            with torch.no_grad():
                if self.device == 'cuda':
                    embedding = self.model.encoding(
                        input_tensors['input_ids'].cuda(),
                        input_tensors['attention_mask'].cuda())
                elif self.device == 'cpu':
                    embedding = self.model.encoding(
                        input_tensors['input_ids'],
                        input_tensors['attention_mask'])
                else:
                    assert False
                for doc, embed in zip(batch, embedding):
                    doc.embedding = embed.cpu().detach().numpy()
