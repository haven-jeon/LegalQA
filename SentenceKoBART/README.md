# SentenceKoBART

- Fine tuning is essential for good search performance.

![](../data/sbart.png)

- The idea came from [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084), and the encoder performance of BART was utilized.

### Dataset

- [KorSTS](https://github.com/kakaobrain/KorNLUDatasets)
  - Train : 17,373
  - Test : 2,867
- [KLUE](https://github.com/KLUE-benchmark/KLUE) STS Dataset
  - Train : 11,668
  - Test : 10

### Preprocessing

```sh
cd data
unzip data.zip
python combine.py --kluests_file KLUE_STS/klue_sts_train.json --korsts_file KorSTS/sts-train.tsv --output sts_train.tsv
python combine.py --kluests_file KLUE_STS/klue_sts_test.json --korsts_file KorSTS/sts-dev.tsv KorSTS/sts-test.tsv  --output sts_test.tsv
```

### Hwo to train

```
python train.py --gpus 1 --max_epochs 20 --default_root_dir training_log --gradient_clip_val 1.0  --train_file sts_train.tsv --test_file sts_test.tsv --batch_size 64 --avg_type norm_avg --subtask STS
```

### Performance

| Dataset  | Pearson corr. | Spearman corr.  |
|:--------:|:-------------:|:---------------:|
| KorSTS(test) + KLUE STS(test) | 0.82  |   0.83  |

