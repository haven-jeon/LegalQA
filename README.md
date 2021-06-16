# LegalQA using SentenceKoBART

| ![](data/demo.gif)|
| ------ |

SentenceKoBART 모델을 기반으로 하는 법률 QA 시스템

- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)  활용 KoBART 튜닝(_실험중_)
- Neural Search Engine [Jina](https://github.com/jina-ai/jina) 활용
- 법률 QA 데이터 수집(1,830 pairs)


## Setup

```bash
git clone https://github.com/haven-jeon/LegalQA.git
cd LegalQA
sh get_model.sh
pip install -r requirements.txt
```

## Index


```sh
python app.py -t index
```

![](index.svg)

필자의 머신(AMD Ryzen 5 PRO 4650U)에서는 인덱싱을 완료하는데 44분 소요됨. 이는 `encoder`에서 `SentenceKoBART`가 동작하기 때문이며 인덱싱 시간을 줄이기 위해 GPU를 사용하는걸 추천한다. GPU를 사용하기 위해서는 `pods/encoder.yml` `on_gpu: true`로 셋업한다. 

## Search

### With REST API

To start the Jina server for REST API:

```sh
python app.py -t query_restful
```

![](query.svg)

Then use a client to query:

```sh
curl --request POST -d '{"top_k": 1, "mode": "search",  "data": ["상속 관련 문의"]}' -H 'Content-Type: application/json' 'http://0.0.0.0:1234/api/search'
````

Or use [Jinabox](https://jina.ai/jinabox.js/) with endpoint `http://127.0.0.1:1234/api/search`

### From the terminal

```sh
python app.py -t query
```



## Citation

```
@misc{heewon2021,
author = {Heewon Jeon},
title = {LegalQA using SentenceKoBART},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/haven-jeon/LegalQA}}
```

## License

- QA 데이터인 `data/legalqa.jsonlines`는 [FREELAWFIRM](http://www.freelawfirm.co.kr/lawqnainfo)에서 `robots.txt`에 준거하여 크롤링한 데이터이며 학술적인 용도 이외에 상업적인 이용을 금합니다.(`CC BY-NC-SA 4.0`)
