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
import requests as req

import numpy as np
from docarray import Document, DocumentArray
from jina import Executor, requests
from jina.logging.logger import JinaLogger
from docarray.array.sqlite import SqliteConfig

from langchain.chat_models import ChatOpenAI

import kss

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate


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

class RestAPIEmbedding(Executor):
    def __init__(self, endpoint: str = None, dim: int = 764, whitening: bool = False, default_batch_size: int = 32, default_traversal_paths: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger(
            getattr(self.metas, 'name', self.__class__.__name__))
        self.endpoint = endpoint
        self.dim = dim
        if default_traversal_paths is not None:
            self.default_traversal_paths = default_traversal_paths
        else:
            self.default_traversal_paths = '@r'
        self.default_batch_size = default_batch_size
        self.whitening = whitening

    
    def api(self, texts: List[str], endpoint: str, whitening:bool, dim: int):
        headers = {
            'Content-Type': 'application/json; charset=utf-8'
        }
        response = req.post(endpoint, headers=headers, json={"texts": texts, "whitening":whitening, "dim":dim})
        if response.status_code == 200:
            embedding = response.json()['embeddings']
            embedding = np.array(embedding)
            assert embedding.shape[0] ==  len(texts)
            return embedding
        else:
            self.logger.error(f'API call failed: {response.status_code}')


    @requests(on=['/search', '/index', '/update'])
    def restapi_encode(self, docs: DocumentArray, parameters: Dict, **kwargs):
        traversal_path=parameters.get('traversal_paths',self.default_traversal_paths)
        batch_size=int(parameters.get('batch_size',
                                      self.default_batch_size))
        dim = int(parameters.get('dim', self.dim))
        endpoint = parameters.get('endpoint', self.endpoint)
        whitening = parameters.get('whitening', self.whitening)
        docs_flatten = docs[traversal_path]
        for batch in DocumentArray(filter(lambda x: bool(x.text), docs_flatten)).batch(batch_size=batch_size):
            texts = batch[:, 'text']
            embedding = self.api(texts, endpoint, whitening, dim)
            for doc, embed in zip(batch, embedding):
                doc.embedding = embed[:dim, ]



class AddAIResponse(Executor):
    def __init__(self, cutoff: float = 0.3, api_type: str = 'openaichat', model_name:str = "text-davinci-003", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger(
            getattr(self.metas, 'name', self.__class__.__name__))
        self.api_type = api_type
        self.model_name = model_name
        self.cutoff = cutoff

    def chatgpt_prompt(self, query: str, search_results: str, answer_max_length: int = 100):
        system_prompt = SystemMessagePromptTemplate.from_template(
                'When a Legal information for the Question is given, Write an explanatory answer related to the Question is created based on it. However, be sure to follow the instructions below.'
                '1. Write an explanatory answer by referring to the Legal information only related to the Question.'
                '2. Reference the number(ex: [1]-(1), [2]-(3),..) of the Legal information that is the basis when writing an explanation answer.'
                '3. Keep it short and easy to understand.'
                '4. Answer with Korean.'
                '5. You keep responses to no more than {character_limit} characters long (including whitespace).'
            )
        human_template = HumanMessagePromptTemplate.from_template(
            'Question: {query}\n\n'
            'Legal information: {search_results}'
        )

        # create the human message
        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_template])
        # format with some input
        chat_prompt_value = chat_prompt.format_prompt(
            character_limit=answer_max_length,
            query=query,
            search_results=search_results
        )
        #chat_prompt_value.to_string()
        #chat_prompt.append(chat_prompt_value.to_messages()[0])
        return chat_prompt_value.to_messages()

    @requests(on=['/search'])
    def add_ai_response(self, docs: DocumentArray, parameters: Dict, **kwargs):
        api_type = parameters.get('api_type', self.api_type)
        cutoff = float(parameters.get('cutoff', self.cutoff))
        answer_max_length = float(parameters.get('max_length', 100))
        llm = ''
        if api_type == 'openaichat':
            llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        else:
            self.logger.error('invalid api_type')
            assert False
        answers: List[str] = []
        r_num = 0
        for doc in docs:
            for match in doc.matches:
                if match.scores["cosine"].value <= cutoff:
                    answers.append(f'[{r_num + 1}]: ' + '"' + match.tags['answer'] + '"')
                    r_num += 1
        self.logger.info('\n'.join(answers))
        if len(answers) > 0:
            try:
                ai_response:str = llm(self.chatgpt_prompt(query=docs[0].text, search_results='\n'.join(answers[0]), answer_max_length=answer_max_length)).content
            except:
                sents = []
                for sent in kss.split_sentences(answers[0]):
                    sents.append(sent)
                ai_response:str = llm(self.chatgpt_prompt(query=docs[0].text, search_results='[0]: "' + ' '.join(sents[-3:]) + "\"", answer_max_length=answer_max_length)).content
            docs[0].tags['ai_response'] = ai_response
        else:
            docs[0].tags['ai_response'] = ''



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


if __name__ == '__main__':
    from jina import Flow

    f = Flow().add(uses=AddAIResponse)
    with f:
         res = f.search([Document(text="아버지가 방에 들어가신다."), Document(text="아버지가 방에 들어가신다.")], parameters={'api_type':'openai', 'cutoff': 0.3})
         print(res[0].tags)