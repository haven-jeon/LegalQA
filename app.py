__copyright__ = "Copyright (c) 2021 Heewon Jeon"

import os
import json

import click
from jina import Flow
from docarray import Document


def config():
    os.environ["JINA_DATA_FILE"] = os.environ.get("JINA_DATA_FILE",
                                                  "data/legalqa_small.jsonlines")
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
    for i in texts:
        d = json.loads(i)
        yield Document(id=d['id'], text='', tags={'title': d['title'], 'question': d['question'], 'answer': d['answer']})
            


# def _pre_processing(texts):
#     print('start of pre-processing')
#     results = []
#     for i in texts:
#         d = json.loads(i)
#         results.append(Document.from_dict(d))
#     return results


def index():
    f = Flow.load_config("flows/index.yml")
    f.plot(output='index.svg')
    work_place = os.path.join(os.path.dirname(__file__),
                              os.environ.get('JINA_WORKSPACE', None))

    with f:
        data_path = os.path.join(os.path.dirname(__file__),
                                 os.environ.get('JINA_DATA_FILE', None))
        f.post('/index',
               inputs=_pre_processing(open(data_path, 'rt').readlines()),
               show_progress=True)
               #parameters={'traversal_paths': '@r,c'})
        f.post('/dump',
               target_peapod='KeyValIndexer',
               parameters={
                   'dump_path': os.path.join(work_place, 'dumps/'),
                   'shards': 1,
                   'timeout': -1
               })


def query(top_k, query_flow):
    f = Flow.load_config(query_flow)
    f.plot(output='query.svg')
    with f:
        f.post('/load', parameters={'model_path': 'gogamza/kobert-legalqa-v1'})
        while True:
            text = input("Please type a sentence: ")
            if not text:
                break

            def ppr(x):
                print_topk(x, text)

            f.search([Document(text=text)],
                     parameters={
                         'top_k': top_k,
                         'model_path': 'gogamza/kobert-legalqa-v1'
                     },
                     on_done=ppr)


def query_restful(query_flow):
    f = Flow().load_config(query_flow,
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
               show_progress=True,
               parameters={'traversal_paths': '@r,c'},
               request_size=0)


def dump():
    f = Flow().add(uses='pods/keyval_lmdb.yml').plot(output='dump.svg')
    with f:
        f.post('/dump',
               parameters={
                   'dump_path': 'dumps/',
                   'shards': 1,
                   'timeout': -1
               })


@click.command()
@click.option(
    "--task",
    "-t",
    type=click.Choice(
        ["index", "query", "query_restful", "dryrun", "train", "dump"],
        case_sensitive=False),
)
@click.option("--top_k", "-k", default=3)
@click.option('--query_flow', type=click.Path(exists=True), default='flows/query_numpy_rerank.yml')
def main(task, top_k, query_flow):
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
        query(top_k, query_flow)
    if task == "query_restful":
        if not os.path.exists(workspace):
            print(
                f"The directory {workspace} does not exist. Please index first via `python app.py -t index`"
            )
        query_restful(query_flow)
    if task == "dryrun":
        dryrun()
    if task == 'train':
        train()
    if task == 'dump':
        dump()


if __name__ == "__main__":
    main()
