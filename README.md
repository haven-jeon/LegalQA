# Korean LegalQA using Neural Search

| ![](data/demo.gif)|
| ------ |

Sentence-KoBART 모델을 기반으로 하는 Neural Search Engine 기반의 QA 예제 구현

- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)  활용 KoBART 튜닝(_실험중_)
- Neural Search Engine [Jina](https://github.com/jina-ai/jina) 활용
- 법률 QA 데이터 수집(156 pairs)




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
