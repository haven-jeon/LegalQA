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
from docarray import Document, DocumentArray
from jina import Executor, requests
from jina.logging.logger import JinaLogger
from docarray.array.sqlite import SqliteConfig

# from jinahub.indexers.storage.LMDBStorage import LMDBStorage
# from jinahub.indexers.searcher.AnnoySearcher import AnnoySearcher
# from jinahub.indexers.searcher.HnswlibSearcher import HnswlibSearcher
# from jinahub.indexers.searcher.FaissSearcher import FaissSearcher


class Preprocess(Executor):
    def __init__(self, default_traversal_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_traversal_path = default_traversal_path or '@r'

    @requests(on=['/index', '/update'])
    def preprocess(self, docs: DocumentArray, parameters: Dict, **kwargs):
        traversal_path = parameters.get('traversal_paths',
                                        self.default_traversal_path)
        f_docs = docs[traversal_path]
        for doc in f_docs:
            doc.text = doc.tags['title'] + '. ' + doc.tags['question']


class Segmenter(Executor):
    def __init__(self,
                 min_sent_len: int = 1,
                 max_sent_len: int = 512,
                 punct_chars: Optional[List[str]] = None,
                 uniform_weight: bool = True,
                 default_traversal_path: str = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.min_sent_len = min_sent_len
        self.max_sent_len = max_sent_len
        self.punct_chars = punct_chars
        self.uniform_weight = uniform_weight
        self.logger = JinaLogger(self.__class__.__name__)
        self.default_traversal_path = default_traversal_path or '@r'
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

    @requests(on=['/index', '/update'])
    def segment(self, docs: DocumentArray, parameters: Dict, **kwargs):
        traversal_path = parameters.get('traversal_paths',
                                        self.default_traversal_path)
        f_docs = docs[traversal_path]
        for doc in f_docs:
            chunks = self._split(doc.text)
            for c in chunks:
                doc.chunks += [(Document(**c, mime_type='text/plain'))]


class DocVectorIndexer(Executor):
    def __init__(self, index_file_name: str, aggr_chunks: str, **kwargs):
        super().__init__(**kwargs)
        self.aggr_chunks = aggr_chunks.lower()
        cfg = SqliteConfig(connection=f'{self.workspace}/{index_file_name}.db', table_name='legalqa')
        self._docs = DocumentArray(storage='sqlite', config=cfg)

    @requests(on='/index')
    def index(self, docs: DocumentArray, **kwargs):
        self._docs.extend(docs)

    @requests(on='/delete')
    def delete(self, docs: DocumentArray, **kwargs):
        if docs is None:
            return
        for doc in docs:
            if doc.id in self._docs:
                del self._docs[doc.id]
    
    @requests(on='/update')
    def update(self, docs: DocumentArray, **kwargs):
        if docs is None:
            return
        for doc in docs:
            if doc.id in self._docs:
                self._docs[doc.id] = doc
            else:
                self._docs.append(doc)

    @requests(on='/search')
    def search(self, docs: DocumentArray, parameters: Dict, **kwargs):
        # if docs is None:
        #     return
        # a = docs.embeddings
        # q_emb = _ext_A(_norm(a))
        # # get chunk embeddings and 'min' aggr
        # if self.aggr_chunks == 'none':
        #     embedding_matrix = _ext_B(_norm(self._docs.embeddings))
        #     dists = _cosine(q_emb, embedding_matrix)
        # else:
        #     aggr_chunk_dist = []
        #     # assume, it processes just one root query.
        #     doc_ids = []
        #     for d in self._docs:
        #         b = d.chunks.embeddings
        #         d_emb = _ext_B(_norm(b))
        #         dists = _cosine(q_emb, d_emb) * d.chunks[:,'weight']  # cosine distance
        #         if self.aggr_chunks == 'min':
        #             aggr_chunk_dist.append(dists[:, np.argmin(dists)])
        #         elif self.aggr_chunks == 'avg':
        #             aggr_chunk_dist.append(np.average(dists, axis=1))
        #         else:
        #             assert False
        #         doc_ids.append(d.id)
        #     dists = np.stack(aggr_chunk_dist, axis=1)
        # idx, dist = self._get_sorted_top_k(dists, int(parameters['top_k']))
        # ids = np.expand_dims(np.array(doc_ids), 0).repeat(idx.shape[0], axis=0)
        # assert idx.shape[0] == ids.shape[0]
        # for _q, _idx, _ids,  _dists in zip(docs, idx, ids, dist):
        #     _ids = _ids[_idx]
        #     for _id, _dist in zip(_ids, _dists):
        #         d = Document(self._docs[_id], copy=True)
        #         d.scores['cosine'] = 1 - _dist  # cosine sim.
        #         _q.matches.append(d)
        return docs.match(self._docs)

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
        self._docs = DocumentArray(self.workspace + '/kv-idx')

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


# class AnnoyFastSearcher(AnnoySearcher):
#     def __init__(self, index_file_name: str, buffer_k: int = 5, **kwargs):
#         super().__init__(**kwargs)
#         self.buffer_k = buffer_k
#         self._docs = DocumentArray(
#             DocumentArray(self.workspace + f'/{index_file_name}'))
#         self._docs_flat = self._docs.traverse_flat(
#             self.default_traversal_paths)

#     @requests(on='/search')
#     def search(self, docs: DocumentArray, parameters: Dict, **kwargs):
#         if not hasattr(self, '_indexer'):
#             self.logger.warning('Querying against an empty index')
#             return

#         traversal_paths = parameters.get('traversal_paths',
#                                          self.default_traversal_paths)

#         top_k = parameters.get(
#             'top_k',
#             self.default_top_k,
#         )

#         for doc in docs.traverse_flat(traversal_paths):
#             # multiply avg. number of setences on each document
#             indices, dists = self._indexer.get_nns_by_vector(
#                 doc.embedding,
#                 int(top_k * self.buffer_k),
#                 include_distances=True)
#             id_score = {}
#             for idx, dist in zip(indices, dists):
#                 idx_id = str(self._ids[idx])
#                 p_id = self._docs_flat[idx_id].parent_id if self._docs_flat[
#                     idx_id].parent_id != '' else str(idx)
#                 match = Document(self._docs[p_id], copy=True)
#                 match.embedding = self._vecs[idx]
#                 if self.is_distance:
#                     if self.metric == 'dot':
#                         match.scores[self.metric] = 1 - dist
#                     else:
#                         match.scores[self.metric] = dist
#                 else:
#                     if self.metric == 'dot':
#                         match.scores[self.metric] = dist
#                     elif self.metric == 'angular' or self.metric == 'hamming':
#                         if self.metric == 'angular':
#                             match.scores['cosine'] = 1 - dist
#                         else:
#                             match.scores[self.metric] = 1 - dist
#                     else:
#                         match.scores[self.metric] = 1 / (1 + dist)
#                 if p_id not in id_score:
#                     id_score[p_id] = True
#                     doc.matches.append(match)
#                     if len(doc.matches) >= top_k:
#                         break
#             if len(doc.matches) < top_k:
#                 self.logger.warning("Please increase 'buffer_k'")


# class HnswlibFastSearcher(HnswlibSearcher):
#     def __init__(self, index_file_name: str, buffer_k: int = 5, **kwargs):
#         super().__init__(**kwargs)
#         self.buffer_k = buffer_k
#         self._docs = DocumentArray(
#             DocumentArray(self.workspace + f'/{index_file_name}'))
#         self._docs_flat = self._docs.traverse_flat(
#             self.default_traversal_paths)

#     @requests(on='/search')
#     def search(self, docs: DocumentArray, parameters: Dict, **kwargs):
#         if not hasattr(self, '_indexer'):
#             self.logger.warning('Querying against an empty index')
#             return

#         traversal_paths = parameters.get('traversal_paths',
#                                          self.default_traversal_paths)

#         top_k = parameters.get(
#             'top_k',
#             self.default_top_k,
#         )

#         for doc in docs.traverse_flat(traversal_paths):
#             # multiply avg. number of setences on each document
#             indices, dists = self._indexer.knn_query(doc.embedding,
#                                                      k=int(top_k *
#                                                            self.buffer_k))
#             id_score = {}
#             for idx, dist in zip(indices[0], dists[0]):
#                 idx_id = str(self._ids[int(idx)])
#                 p_id = self._docs_flat[idx_id].parent_id if self._docs_flat[
#                     idx_id].parent_id != '' else int(idx)
#                 match = Document(self._docs[p_id], copy=True)
#                 match.embedding = self._vecs[int(idx)]
#                 if self.is_distance:
#                     match.scores[self.metric] = dist
#                 else:
#                     if self.metric == 'cosine' or self.metric == 'ip':
#                         match.scores[self.metric] = 1 - dist
#                     else:
#                         match.scores[self.metric] = 1 / (1 + dist)
#                 if p_id not in id_score:
#                     id_score[p_id] = True
#                     doc.matches.append(match)
#                     if len(doc.matches) >= top_k:
#                         break
#             if len(doc.matches) < top_k:
#                 self.logger.warning("Please increase 'buffer_k'")


# class FaissFastSearcher(FaissSearcher):
#     def __init__(self, index_file_name: str, buffer_k: int = 5, **kwargs):
#         super().__init__(**kwargs)
#         self.buffer_k = buffer_k
#         self._docs = DocumentArray(
#             DocumentArray(self.workspace + f'/{index_file_name}'))
#         self._docs_flat = self._docs.traverse_flat(
#             self.default_traversal_paths)

#     @requests(on='/search')
#     def search(self,
#                docs: DocumentArray,
#                parameters: Optional[Dict] = None,
#                *args,
#                **kwargs):
#         """Find the top-k vectors with smallest ``metric`` and return their ids in ascending order.
#         :param docs: the DocumentArray containing the documents to search with
#         :param parameters: the parameters for the request
#         """
#         if not hasattr(self, 'index'):
#             self.logger.warning('Querying against an empty Index')
#             return

#         if parameters is None:
#             parameters = {}

#         top_k = parameters.get('top_k', self.default_top_k)
#         traversal_paths = parameters.get('traversal_paths',
#                                          self.default_traversal_paths)

#         query_docs = docs.traverse_flat(traversal_paths)

#         vecs = np.array(query_docs.get_attributes('embedding'))

#         if self.normalize:
#             from faiss import normalize_L2
#             normalize_L2(vecs)
#         dists, ids = self.index.search(vecs, int(top_k * self.buffer_k))
#         id_score = {}
#         if self.metric == 'inner_product':
#             dists = 1 - dists
#         for doc_idx, dist in enumerate(zip(ids, dists)):
#             for m_info in zip(*dist):
#                 idx, distance = m_info
#                 idx_id = str(self._ids[int(idx)])
#                 p_id = self._docs_flat[idx_id].parent_id if self._docs_flat[
#                     idx_id].parent_id != '' else int(idx)
#                 match = Document(self._docs[p_id], copy=True)
#                 match.embedding = self._vecs[int(idx)]
#                 if self.is_distance:
#                     match.scores[self.metric] = distance
#                 else:
#                     if self.metric == 'inner_product':
#                         match.scores[self.metric] = 1 - distance
#                     else:
#                         match.scores[self.metric] = 1 / (1 + distance)

#                 if p_id not in id_score:
#                     id_score[p_id] = True
#                     query_docs[doc_idx].matches.append(match)
#                     if len(query_docs[doc_idx].matches) >= top_k:
#                         break
#             if len(query_docs[doc_idx].matches) < top_k:
#                 self.logger.warning("Please increase 'buffer_k'")


class FilterBy(Executor):
    def __init__(self, cutoff: float = -1.0, **kwargs):
        super().__init__(**kwargs)
        self.cutoff = cutoff

    @requests
    def query(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            filtered_matches = DocumentArray()
            for match in doc.matches:
                if match.scores__cosine__value >= self.cutoff:
                    filtered_matches.append(match)
            doc.matches = filtered_matches

class DocCount(Executor):
    def __init__(
        self,
        default_traversal_path: str = '@r',
        n_dim: int = 768,
        data_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        config = {
                'n_dim': n_dim,
                'data_path': data_path or self.workspace or './workspace',
            }
        self._index = DocumentArray(storage='annlite', config=config)
        self.traversal_path = default_traversal_path

    @requests(on='/count')
    def get_count(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            doc.tags['doc_count'] = len(self._index[self.traversal_path])


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
