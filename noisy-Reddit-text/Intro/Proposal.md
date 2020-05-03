# Project Proposal for Machine Translation for Noisy Reddit Text
Group members: Serena, Niki, Roy

### Introduction

In our project, we aim to build a neural machine translation model that translates noisy Reddit posts from French into English. "Noisy" is defined as text with typos, grammar errors, code switching, textspeak, and other features common to social media / online domains. The input to our model will be the noisy Reddit post in French, and the output will be the equivalent Reddit post in English. 

### Motivation and Contributions

Given that social media is becoming increasingly prevalent in our world, it is important to build machine translation models that are able to handle not only standard texts, such as those found in newspapers and novels, but also less formal texts, such as those found online or in social media. Thus, this project is important as it aims to focus on machine translation as applied in more informal settings, which may require a slightly different model as compared to those used in standard machine translation settings.

We hope that our contribution will result in a more robust machine translation model which is able to handle not only standard text but also informally-written text. In this way, it will be a better system than what already exists, and additionally it will be more prepared for an online world more dominated by social media and informal language. 

### Data

We will be using [MTNT dataset](http://www.cs.cmu.edu/~pmichel1/mtnt/#data)1 as in-domain parallel data, for English and French language pair. MTNT is a collection of comments from the [Reddit website](https://reddit.com), in particular this dataset consists of "noisy" text that exhibits typos, grammar errors, code switching and more.

Example of data:

| Language pair     | Source | Target |
|-------------------|-------------------------------------------------------------------|---------------------------|
| en-fr | Just got called into work tho so I won’t be in til tomorrow night | Mais on vient de m'appeler pour le travail donc je n'y serai pas avant demain soir |
| fr-en | je demande lazil politique pr janluk # Il ressuscitera ! | I demand political asylum for jean luc # He will resurrect! |


<br>

<br>

| Data   | Size (comments) | Style |
|-----------------|-------------------------------|-------|
| Training Data   | En-fr : 36058 Fr-en : 19161   | .tsv |
| Validation Data | En-fr : 886 Fr-en : 852 | .tsv |
| Test Data | En-fr : 1022 Fr-en : 1020 | .tsv |

### Engineering:

We will be using personal computers and we will use [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true) so we can take advantage of a GPU. We will be implementing machine translation with attention-based  BiLSTMs. We will be using PyTorch and we will take codebase from COLX 531 [Lab3](https://github.ubc.ca/MDS-CL-2019-20/COLX_531_translation_students/blob/master/lab_solutions/lab3_sol.ipynb) and [Lab4](https://github.ubc.ca/MDS-CL-2019-20/COLX_531_translation_students/blob/master/lab_solutions/lab4%20-%20sol.ipynb). 

### Previous Works:

Previously we have built neural machine translation models, specifically, seq2seq with attention models as well as BLEU evaluation systems for English-French and English-Portuguese translation. 

### Evaluation:

Since our task is neural machine translation, we will use the BLEU evaluation system to evaluate the output and the gold standard will be our original reddit texts in English. For one block of texts, our system would sum clipped n-gram(unigram to 4-gram) matches, divided by the counts of n-grams. It would take an average logarithm with uniform weights and Brevity penalty on sentence lengths. 

### Conclusion:

Online-community domain is of greater importance to investigate and with the knowledge and skills we learned so far, we hope our project helps better understanding this specific field in terms of automated translation systems. 

### References:

http://www.cs.cmu.edu/~pmichel1/mtnt/#data

