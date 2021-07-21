# coding=utf-8
# Modified MIT License

# Software Copyright (c) 2021 Heewon Jeon, Jina AI

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

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from jina import Document, DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger
from jina.types.arrays.memmap import DocumentArrayMemmap
from jina_commons.batching import get_docs_batch_generator
from kobart import get_kobart_tokenizer, get_pytorch_kobart_model
from transformers import BartModel

from jinahub.indexers.storage.LMDBStorage import LMDBStorage


class Segmenter(Executor):
    def __init__(self,
                 min_sent_len: int = 1,
                 max_sent_len: int = 512,
                 punct_chars: Optional[List[str]] = None,
                 uniform_weight: bool = True,
                 default_traversal_path: Optional[List[str]] = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.min_sent_len = min_sent_len
        self.max_sent_len = max_sent_len
        self.punct_chars = punct_chars
        self.uniform_weight = uniform_weight
        self.logger = JinaLogger(self.__class__.__name__)
        self.default_traversal_path = default_traversal_path or ['r']
        if not punct_chars:
            self.punct_chars = [
                '!', '.', '?', '։', '؟', '۔', '܀', '܁', '܂', '‼', '‽', '⁇',
                '⁈', '⁉', '⸮', '﹖', '﹗', '！', '．', '？', '｡', '。', '\n'
            ]
        if self.min_sent_len > self.max_sent_len:
            self.logger.warning(
                'the min_sent_len (={}) should be smaller or equal to the max_sent_len (={})'
                .format(self.min_sent_len, self.max_sent_len))
        self._slit_pat = re.compile('\s*([^{0}]+)(?<!\s)[{0}]*'.format(''.join(
            set(self.punct_chars))))

    def _split(self, text: str) -> List:
        results = []
        ret = [(m.group(0), m.start(), m.end())
               for m in re.finditer(self._slit_pat, text)]
        if not ret:
            ret = [(text, 0, len(text))]
        for ci, (r, s, e) in enumerate(ret):
            f = re.sub('\n+', ' ', r).strip()
            f = f[:self.max_sent_len]
            if len(f) > self.min_sent_len:
                results.append(
                    dict(text=f,
                         offset=ci,
                         weight=1.0 if self.uniform_weight else len(f) /
                         len(text),
                         location=[s, e]))
        return results

    @requests
    def segment(self, docs: DocumentArray, parameters: Dict, **kwargs):
        traversal_path = parameters.get('traversal_paths', self.default_traversal_path)
        f_docs = docs.traverse_flat(traversal_path)
        for doc in f_docs:
            text = doc.text
            chunks = self._split(text)
            for c in chunks:
                doc.chunks += [(Document(**c,
                                         mime_type='text/plain'))]


class DocVectorIndexer(Executor):
    def __init__(self, index_file_name: str, aggr_chunks: str, **kwargs):
        super().__init__(**kwargs)
        self.aggr_chunks = aggr_chunks.lower()
        self._docs = DocumentArrayMemmap(self.workspace +
                                         f'/{index_file_name}')

    @requests(on='/index')
    def index(self, docs: DocumentArray, **kwargs):
        self._docs.extend(docs)

    @requests(on='/search')
    def search(self, docs: DocumentArray, parameters: Dict, **kwargs):
        if docs is None:
            return
        a = np.stack(docs.get_attributes('embedding'))
        q_emb = _ext_A(_norm(a))
        # get chunk embeddings and 'min' aggr
        if self.aggr_chunks == 'none':
            embedding_matrix = _ext_B(
                _norm(np.stack(self._docs.get_attributes('embedding'))))
            dists = _cosine(q_emb, embedding_matrix)
        else:
            aggr_chunk_dist = []
            for d in self._docs:
                b = np.stack(d.chunks.get_attributes('embedding'))
                d_emb = _ext_B(_norm(b))
                dists = _cosine(q_emb, d_emb)  # cosine distance
                if self.aggr_chunks == 'min':
                    aggr_chunk_dist.append(dists[:, np.argmin(dists)])
                elif self.aggr_chunks == 'avg':
                    aggr_chunk_dist.append(np.average(dists, axis=1))
                else:
                    assert False
            dists = np.stack(aggr_chunk_dist, axis=1)
        idx, dist = self._get_sorted_top_k(dists, int(parameters['top_k']))
        for _q, _ids, _dists in zip(docs, idx, dist):
            for _id, _dist in zip(_ids, _dists):
                d = Document(self._docs[int(_id)], copy=True)
                d.scores['cosine'] = 1 - _dist  # cosine sim.
                _q.matches.append(d)

    @staticmethod
    def _get_sorted_top_k(dist: 'np.array',
                          top_k: int) -> Tuple['np.ndarray', 'np.ndarray']:
        if top_k >= dist.shape[1]:
            idx = dist.argsort(axis=1)[:, :top_k]
            dist = np.take_along_axis(dist, idx, axis=1)
        else:
            idx_ps = dist.argpartition(kth=top_k, axis=1)[:, :top_k]
            dist = np.take_along_axis(dist, idx_ps, axis=1)
            idx_fs = dist.argsort(axis=1)
            idx = np.take_along_axis(idx_ps, idx_fs, axis=1)
            dist = np.take_along_axis(dist, idx_fs, axis=1)

        return idx, dist


class KeyValueIndexer(Executor):
    def __init__(self, aggr_chunks: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aggr_chunks = aggr_chunks.lower()
        self._docs = DocumentArrayMemmap(self.workspace + '/kv-idx')

    @requests(on='/index')
    def index(self, docs: DocumentArray, **kwargs):
        self._docs.extend(docs)

    @requests(on='/search')
    def query(self, docs: DocumentArray, **kwargs):
        if self.aggr_chunks == 'none':
            for doc in docs:
                for match in doc.matches:
                    extracted_doc = self._docs[match.parent_id]
                    match.update(extracted_doc)


class FilterBy(Executor):
    def __init__(self, cutoff: float = -1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cutoff = cutoff

    @requests
    def query(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            filtered_matches = DocumentArray()
            for match in doc.matches:
                if match.scores__cosine__value >= self.cutoff:
                    filtered_matches.append(match)
            doc.matches = filtered_matches


class WeightedRanker(Executor):
    @requests(on='/search')
    def rank(self, docs_matrix: List['DocumentArray'], parameters: Dict,
             **kwargs) -> 'DocumentArray':
        """
        :param docs_matrix: list of :class:`DocumentArray` on multiple requests to
          get bubbled up matches.
        :param parameters: the parameters passed into the ranker, in this case stores :attr`top_k`
          to filter k results based on score.
        :param kwargs: not used (kept to maintain interface)
        """

        result_da = DocumentArray(
        )  # length: 1 as every time there is only one query
        for d_mod1, d_mod2 in zip(*docs_matrix):

            final_matches = {}  # type: Dict[str, Document]

            for m in d_mod1.matches:
                m.scores[
                    'relevance'] = m.scores['cosine'].value * d_mod1.weight
                final_matches[m.parent_id] = Document(m, copy=True)

            for m in d_mod2.matches:
                if m.parent_id in final_matches:
                    final_matches[
                        m.parent_id].scores['relevance'] = final_matches[
                            m.parent_id].scores['relevance'].value + (
                                m.scores['cosine'].value * d_mod2.weight)
                else:
                    m.scores[
                        'relevance'] = m.scores['cosine'].value * d_mod2.weight
                    final_matches[m.parent_id] = Document(m, copy=True)

            da = DocumentArray(list(final_matches.values()))
            da.sort(key=lambda ma: ma.scores['relevance'].value, reverse=True)
            d = Document(matches=da[:int(parameters['top_k'])])
            result_da.append(d)
        return result_da


def _get_ones(x, y):
    return np.ones((x, y))


def _ext_A(A):
    nA, dim = A.shape
    A_ext = _get_ones(nA, dim * 3)
    A_ext[:, dim:2 * dim] = A
    A_ext[:, 2 * dim:] = A**2
    return A_ext


def _ext_B(B):
    nB, dim = B.shape
    B_ext = _get_ones(dim * 3, nB)
    B_ext[:dim] = (B**2).T
    B_ext[dim:2 * dim] = -2.0 * B.T
    del B
    return B_ext


def _euclidean(A_ext, B_ext):
    sqdist = A_ext.dot(B_ext).clip(min=0)
    return np.sqrt(sqdist)


def _norm(A):
    return A / np.linalg.norm(A, ord=2, axis=1, keepdims=True)


def _cosine(A_norm_ext, B_norm_ext):
    return A_norm_ext.dot(B_norm_ext).clip(min=0) / 2
