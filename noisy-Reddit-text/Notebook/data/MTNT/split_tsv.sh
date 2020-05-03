#!/usr/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

for split in train valid test
do
  for lang in fr ja
  do
    # en -> $lang
    cut -f2 $DIR/$split/$split.en-$lang.tsv > $DIR/$split/$split.en-$lang.en
    cut -f3 $DIR/$split/$split.en-$lang.tsv > $DIR/$split/$split.en-$lang.$lang
    # $lang -> en
    cut -f2 $DIR/$split/$split.$lang-en.tsv > $DIR/$split/$split.$lang-en.$lang
    cut -f3 $DIR/$split/$split.$lang-en.tsv > $DIR/$split/$split.$lang-en.en
  done
done
