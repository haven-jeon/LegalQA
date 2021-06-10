# Simple LegalQA with KoBART and jina

Sentence-KoBART 모델을 기반으로 하는 Neural Search Engine 기반의 QA 예제

- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)  (EMNLP 2019) 기반 KoBART 튜닝(_실험중_)
- Neural Search Engine Jina 기반의 서치 플랫폼 구현
- 법률 QA 데이터 수집(156 쌍)

## Setup

```sh
sh get_model.sh
pip install -r requirements.txt
```

## Index

```sh
python app.py -t index
```

## Search

### With REST API

To start the Jina server for REST API:

```sh
python app.py -t query_restful
```

Then use a client to query:

```sh
curl --request POST -d '{"top_k": 1, "mode": "search",  "data": ["상속 관련 문의"]}' -H 'Content-Type: application/json' 'http://0.0.0.0:7875/api/search'
````

Or use [Jinabox](https://jina.ai/jinabox.js/) with endpoint `http://127.0.0.1:7875/api/search`

### From the terminal

```sh
python app.py -t query
```
