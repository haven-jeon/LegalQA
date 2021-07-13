import json
import argparse
import csv
import pandas as pd

parser = argparse.ArgumentParser(description='combine dataset')

parser.add_argument('--kluenli_file',
                    nargs='+',
                    help='')

parser.add_argument('--kornli_file',
                    nargs='+',
                    help='')

parser.add_argument('--output',
                    type=str,
                    default='output.tsv',
                    help='')


parser.add_argument('--kluests_file',
                    nargs='+',
                    help='')

parser.add_argument('--korsts_file',
                    nargs='+',
                    help='')


def load_klue_nli(filename):
    nli_list = []
    for f in filename:
        with open(f, 'rt') as fp:
            nli = json.load(fp)
        for d in nli:
            nli_list.append([d['premise'], d['hypothesis'], d['gold_label']])
    return nli_list


def load_klue_sts(filename):
    stt_list = []
    for f in filename:
        with open(f, 'rt') as fp:
            nli = json.load(fp)
        for d in nli:
            stt_list.append([d['content']['sentence1'], d['content']['sentence2'], d['labels']['label']])
    return stt_list


def load_kornli(filename):
    nli_list = []
    for f in filename:
        data = pd.read_csv(f, delimiter='\t', error_bad_lines=False)
        for _, i in data.iterrows():
            nli_list.append([i['sentence1'], i['sentence2'], i['gold_label']])
    return nli_list


def load_korsts(filename):
    stt_list = []
    for f in filename:
        data = pd.read_csv(f, delimiter='\t', error_bad_lines=False)
        for _, i in data.iterrows():
            stt_list.append([i['sentence1'], i['sentence2'], i['score']])
    return stt_list


if __name__ == '__main__':
    args = parser.parse_args()

    if args.kluenli_file and args.kornli_file:
        data = load_klue_nli(args.kluenli_file) + load_kornli(args.kornli_file)
        with open(args.output, 'w', newline='') as csvfile:
            allnli = csv.writer(csvfile, delimiter='\t')
            allnli.writerow(['premise', 'hypothesis', 'gold_label'])
            for row in data:
                allnli.writerow(row)
    elif args.kluests_file and args.korsts_file:
        data = load_klue_sts(args.kluests_file) + load_korsts(args.korsts_file)
        with open(args.output, 'w', newline='') as csvfile:
            allnli = csv.writer(csvfile, delimiter='\t')
            allnli.writerow(['sentence1', 'sentence2', 'label'])
            for row in data:
                allnli.writerow(row)

