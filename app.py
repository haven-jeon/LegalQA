__copyright__ = "Copyright (c) 2021 Heewon Jeon"

import os
import json

import click
from jina import Flow
from docarray import Document


def config():
    os.environ["JINA_DATA_FILE"] = os.environ.get(
        "JINA_DATA_FILE", "data/legalqa_small.jsonlines")
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
        yield Document(id=d['id'],
                       text=d['title'] + '. ' + d['question'],
                       tags={
                           'title': d['title'],
                           'question': d['question'],
                           'answer': d['answer']
                       })

def _pre_processing_note(data_path: str):
    jd = json.load(open(data_path, 'r'))
    for l in jd:
        #l.pop('id', None)
        yield Document(id = l['id'], text= l['text'], tags=l['tags'])


def index(index_flow):
    f = Flow.load_config(index_flow)
    f.plot(output='index.svg')

    with f:
        data_path = os.path.join(os.path.dirname(__file__),
                                 os.environ.get('JINA_DATA_FILE', None))
        f.post('/index',
               inputs=_pre_processing(open(data_path, 'rt').readlines()),
               show_progress=True)

def count():
    f = Flow(protocol='http').add(uses='pods/counts.yml')
    with f:
        f.post('/count', inputs=['count'], on_done=get_result)

def get_result(d):
    print(f'Num of docs: {d.docs[0].tags["doc_count"]}')

def query(top_k, query_flow):
    f = Flow.load_config(query_flow)
    f.plot(output='query.svg')
    with f:
        while True:
            text = input("Please type a sentence: ")
            if not text:
                break

            def ppr(x):
                print_topk(x, text)

            f.search([Document(text=text)],
                     parameters={
                         'limit': top_k,
                         'ef_search': top_k * 10
                     },
                     on_done=ppr)

def status():
    f = Flow.load_config("flows/status.yml",
                         uses_with={
                             'protocol': 'http',
                             'port': int(os.environ["JINA_PORT"]),
                         })
    f.plot(output='status.svg')
    f.expose_endpoint('/status', summary='document status')
    with f:
        f.block()

def query_restful(query_flow):
    f = Flow.load_config(query_flow,
                         uses_with={
                             'protocol': 'http',
                             'port': int(os.environ["JINA_PORT"]),
                         })
    with f:
        f.block()


@click.command()
@click.option(
    "--task",
    "-t",
    type=click.Choice(["index", "query", "query_restful", "count"],
                      case_sensitive=False),
)
@click.option("--top_k", "-k", default=3)
@click.option('--flow',
              type=click.Path(exists=True),
              default='flows/query_numpy.yml')
def main(task, top_k, flow):
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
        index(flow)
    if task == "query":
        if not os.path.exists(workspace):
            print(
                f"The directory {workspace} does not exist. Please index first via `python app.py -t index`"
            )
        query(top_k, flow)
    if task == "query_restful":
        if not os.path.exists(workspace):
            print(
                f"The directory {workspace} does not exist. Please index first via `python app.py -t index`"
            )
        query_restful(flow)
    if task == 'count':
        if not os.path.exists(workspace):
            print(
                f"The directory {workspace} does not exist. Please index first via `python app.py -t index`"
            )
        count()


if __name__ == "__main__":
    main()
