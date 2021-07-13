__copyright__ = "Copyright (c) 2021 Heewon Jeon"

import os
import json

import click
from jina import Flow, Document


MAX_DOCS = int(os.environ.get("JINA_MAX_DOCS", 2000))


def config():
    os.environ["JINA_DATA_FILE"] = os.environ.get(
        "JINA_DATA_FILE", "data/legalqa.jsonlines"
    )
    os.environ["JINA_WORKSPACE"] = os.environ.get("JINA_WORKSPACE", "workspace")

    os.environ["JINA_PORT"] = os.environ.get("JINA_PORT", str(1234))


def print_topk(resp, sentence):
    for doc in resp.data.docs:
        print(f"\n\n\nTa-Dah🔮, here's what we found for: {sentence}")
        for idx, match in enumerate(doc.matches):
            score = match.scores['cosine'].value
            if score < 0.0:
                continue
            print(f'> {idx:>2d}({score:.2f}). {match.text}')
        print('\n\n\n')


def _pre_processing(texts):
    print('start of pre-processing')
    results = []
    for i in texts:
        d = json.loads(i)
        d['text'] = d['title'].strip() + '. ' + d['question']
        results.append(Document(json.dumps(d, ensure_ascii=False)))
    return results


def index(num_docs):
    f = Flow().load_config("flows/index.yml").plot(output='index.svg')

    with f:
        data_path = os.path.join(os.path.dirname(__file__), os.environ.get('JINA_DATA_FILE', None))
        f.post('/index',
                _pre_processing(open(data_path, 'rt').readlines()),
                how_progress=True)


def query(top_k):
    f = Flow().load_config("flows/query.yml").plot(output='query.svg')
    with f:
        while True:
            text = input("Please type a sentence: ")
            if not text:
                break

            def ppr(x):
                print_topk(x, text)

            f.search(Document(text=text), parameters={'top_k': top_k}, on_done=ppr)


def query_restful():
    f = Flow().load_config("flows/query.yml", override_with={'protocol': 'http', 'port_expose': int(os.environ["JINA_PORT"])})
    with f:
        f.block()


def dryrun():
    f = Flow().load_config("flows/index.yml")
    with f:
        f.dry_run()


@click.command()
@click.option(
    "--task",
    "-t",
    type=click.Choice(
        ["index", "query", "query_restful", "dryrun"], case_sensitive=False
    ),
)
@click.option("--num_docs", "-n", default=MAX_DOCS)
@click.option("--top_k", "-k", default=3)
def main(task, num_docs, top_k):
    config()
    workspace = os.environ["JINA_WORKSPACE"]
    if task == "index":
        if os.path.exists(workspace):
            print(f'\n +----------------------------------------------------------------------------------+ \
                    \n |                                   🤖🤖🤖                                         | \
                    \n | The directory {workspace} already exists. Please remove it before indexing again.  | \
                    \n |                                   🤖🤖🤖                                         | \
                    \n +----------------------------------------------------------------------------------+')
        index(num_docs)
    if task == "query":
        if not os.path.exists(workspace):
            print(f"The directory {workspace} does not exist. Please index first via `python app.py -t index`")
        query(top_k)
    if task == "query_restful":
        if not os.path.exists(workspace):
            print(f"The directory {workspace} does not exist. Please index first via `python app.py -t index`")
        query_restful()
    if task == "dryrun":
        dryrun()


if __name__ == "__main__":
    main()
