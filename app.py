__copyright__ = "Copyright (c) 2021 Heewon Jeon"

import os
import json
from timeit import default_timer as timer

import click
from jina import Flow, Document


def config():
    os.environ["JINA_DATA_FILE"] = os.environ.get("JINA_DATA_FILE",
                                                  "data/legalqa.jsonlines")
    os.environ["JINA_WORKSPACE"] = os.environ.get("JINA_WORKSPACE",
                                                  "workspace")

    os.environ["JINA_PORT"] = os.environ.get("JINA_PORT", str(1234))


def print_topk(resp, sentence):
    for doc in resp.data.docs:
        print(f"\n\n\nTa-Dah🔮, here's what we found for: {sentence}")
        for idx, match in enumerate(doc.matches):
            score = match.scores['cosine'].value
            bert_score = match.scores['bert_rerank'].value
            print(f'> {idx:>2d}({score:.2f}, {bert_score:.2f}). {match.text}')
        print('\n\n\n')


def _pre_processing(texts):
    print('start of pre-processing')
    results = []
    for i in texts:
        d = json.loads(i)
        d['text'] = d['title'].strip() + '. ' + d['question']
        results.append(Document(json.dumps(d, ensure_ascii=False)))
    return results


def index():
    f = Flow().load_config("flows/index.yml").plot(output='index.svg')
    work_place = os.path.join(os.path.dirname(__file__),
                                os.environ.get('JINA_WORKSPACE', None))

    with f:
        data_path = os.path.join(os.path.dirname(__file__),
                                 os.environ.get('JINA_DATA_FILE', None))
        f.post('/index',
               _pre_processing(open(data_path, 'rt').readlines()),
               show_progress=True, parameters={'traversal_paths': ['r', 'c']})
        f.post('/dump',
               target_peapod='KeyValIndexer',
               parameters={
                   'dump_path': os.path.join(work_place, 'dumps/'),
                   'shards': 1,
                   'timeout': -1
               })

# queries = ['아버지가 돌아가시고 조문객들이 낸 부의금의 분배', '유실수의 소유권은 누구에게 귀속되는지 여부',
#            '사위가 장인 재산을 상속받을 수 있는지 여부',
#            '경매신청된 토지수용 시 기업자의 수용절차는 경락자를 상대로 해야 하는지 여부',
#            '녹색등화가 점멸되고 있을때 횡단보도 진입 후 사고당한 경우', 
#            '편도 1차로에 정차한 버스 앞서려고 황색실선 중앙선 넘어간 경우',
#            '건너가는 피해자를 발견하고 급정거하였으나 피하지 못하고 충격',
#            '달려오던 영업택시에 충격 당하여 전치 3주의 상해를', 
#            '행정소송 진행 중 원고 사망 시 상속인의 승계 가능 여부',
#            '부(父)의 사망과 인지(認知)'] * 10

def query(top_k):
    f = Flow().load_config("flows/query_hnswlib_rerank.yml").plot(output='query.svg')
    with f:
        f.post('/load', parameters={'model_path': 'gogamza/kobert-legalqa-v1'})
        # t = []
        # for text in queries:
        while True:
            text = input("Please type a sentence: ")
            if not text:
                break
            def ppr(x):
                print_topk(x, text)
            # start = timer()
            f.search(Document(text=text),
                     parameters={'top_k': top_k, 'model_path': 'gogamza/kobert-legalqa-v1'},
                     on_done=ppr)
            # end = timer()
            # print(f'elapse time : {end - start} sec.')
            # t.append(end - start)
        # print(sum(t)/len(t))


def query_restful():
    f = Flow().load_config("flows/query.yml",
                           override_with={
                               'protocol': 'http',
                               'port_expose': int(os.environ["JINA_PORT"])
                            })
    with f:
        f.post('/load', parameters={'model_path': 'gogamza/kobert-legalqa-v1'})
        f.block()


def dryrun():
    f = Flow().load_config("flows/index.yml")
    with f:
        f.dry_run()


def train():
    f = Flow().load_config("flows/train.yml").plot(output='train.svg')
    with f:
        data_path = os.path.join(os.path.dirname(__file__),
                                 os.environ.get('JINA_DATA_FILE', None))
        f.post('/train',
              _pre_processing(open(data_path, 'rt').readlines()),
              show_progress=True, parameters={'traversal_paths': ['r', 'c']},
              request_size=0)
        #f.post('/load',
        #      parameters={'model_path': 'kobert_model'})


def dump():
    f = Flow().add(uses='pods/keyval_lmdb.yml').plot(output='dump.svg')
    with f:
        f.post('/dump', parameters={
                'dump_path': 'dumps/',
                'shards': 1,
                'timeout': -1}
        )



@click.command()
@click.option(
    "--task",
    "-t",
    type=click.Choice(["index", "query", "query_restful", "dryrun", "train", "dump"],
                      case_sensitive=False),
)
@click.option("--top_k", "-k", default=3)
def main(task, top_k):
    config()
    workspace = os.environ["JINA_WORKSPACE"]
    if task == "index":
        if os.path.exists(workspace):
            print(
                f'\n +----------------------------------------------------------------------------------+ \
                    \n |                                   🤖🤖🤖                                         | \
                    \n | The directory {workspace} already exists. Please remove it before indexing again.  | \
                    \n |                                   🤖🤖🤖                                         | \
                    \n +----------------------------------------------------------------------------------+'
            )
        index()
    if task == "query":
        if not os.path.exists(workspace):
            print(
                f"The directory {workspace} does not exist. Please index first via `python app.py -t index`"
            )
        query(top_k)
    if task == "query_restful":
        if not os.path.exists(workspace):
            print(
                f"The directory {workspace} does not exist. Please index first via `python app.py -t index`"
            )
        query_restful()
    if task == "dryrun":
        dryrun()
    if task == 'train':
        train()
    if task == 'dump':
        dump()


if __name__ == "__main__":
    main()
