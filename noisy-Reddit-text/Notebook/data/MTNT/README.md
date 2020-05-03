# MTNT

MTNT is a collection of comments from the [Reddit](www.reddit.com) discussion website in English, French and Japanese, translated to and from English. The particularity of this dataset is that the data consists of "noisy" text, that exhibits typos, grammar errors, code switching and more. For more details, check out the [paper](http://www.cs.cmu.edu/~pmichel1/hosting/mtnt-emnlp.pdf).

## Data

This folder should have the following structure:

```
MTNT
├ monolingual
│   ├ dev.en
│   ├ dev.fr
│   ├ dev.ja
│   ├ dev.tok.en
│   ├ dev.tok.fr
│   ├ dev.tok.ja
│   ├ train.en
│   ├ train.fr
│   ├ train.ja
│   ├ train.tok.en
│   ├ train.tok.fr
│   └ train.tok.ja
├ README.md
├ split_tsv.sh
├ test
│   ├ test.en-fr.tsv
│   ├ test.en-ja.tsv
│   ├ test.fr-en.tsv
│   └ test.ja-en.tsv
├ train
│   ├ train.en-fr.tsv
│   ├ train.en-ja.tsv
│   ├ train.fr-en.tsv
│   └ train.ja-en.tsv
└ valid
    ├ valid.en-fr.tsv
    ├ valid.en-ja.tsv
    ├ valid.fr-en.tsv
    └ valid.ja-en.tsv
```

The monolingual data is distributed with and without tokenization, in raw text format. The parallel data is split into training, validation and test set. Each tsv file contains 3 columns:

- Comment ID
- Source sentence
- Target sentence

Some source sentences are from a same original comment, and you can use the comment ID to group them together and leverage the contextual information.

If you're only interested in the source and target sentence, you can run the `split_tsv.sh` script to split the files into source and target files.

## Code

The code to reproduce the collection process and the Machine Translation experiments is available [on github](https://github.com/pmichel31415/mtnt).

## Citing

If you use this dataset or the associated code, please cite:

```
@InProceedings{michel2018mtnt,
  author    = {Michel, Paul  and  Neubig, Graham},
  title     = {MTNT: A Testbed for Machine Translation of Noisy Text},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing}
}
```

## Contact

If you have any issue with the data, please contact `pmichel1[at]cs.cmu.edu`. For any question regarding the code, please [open an issue on Github](https://github.com/pmichel31415/mtnt/issues).

## License

This data is released under the terms of the [Reddit API](https://www.reddit.com/wiki/api).

## Changelog

- 01/22/2019: fixed a missing tab in MTNT/test/test.fr-en.tsv (https://github.com/pmichel31415/mtnt/issues/3)
