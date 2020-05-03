## Machine Translation for Noisy Reddit Text
##### Serena Huang, Niki Hajmoshir, Roy Guo
##### University of British Columbia

--------

### Abstract
In this paper, we describe the neural machine translation model that we built for the task of translating noisy Reddit posts from French into English. "Noisy" is defined as text with typos, grammar errors, code switching, textspeak, and other features common to social media / online domains which can decrease the accuracy of machine translation models to an extent. The input to our model is noisy Reddit posts in French, and the output is the equivalent Reddit post in English. We use an attention-based bidirectional Seq2Seq model built in PyTorch and experiment with a number of preprocessing methods and different hyperparameters to achieve a BLEU score of 0.20 on our best model. 

### Introduction and Motivation 
Given that social media is becoming increasingly prevalent in our world, it is important to build machine translation models that are able to handle not only standard texts, such as those found in newspapers and novels, but also less formal texts, such as those found online or in social media. Thus, this project is important as it aims to focus on machine translation as applied in more informal settings, which may require a slightly different model as compared to those used in standard machine translation settings.

We hope that our contribution will result in a more robust machine translation model which is able to handle not only standard text but also informally-written text. In this way, it will 
be a better system than what already exists, and additionally it will be more prepared for an online world more dominated by social media and informal language.

### Related Works 
Previously in the UBC’s COLX 531 course we have built neural machine translation models, specifically, seq2seq with attention models as well as BLEU evaluation systems for English-French and English-Portuguese translation. We also took a look at some papers to shed some more light on out of domain text in the social media domain for machine translation. [Vaibhav et al](https://www.aclweb.org/anthology/N19-1190.pdf)provides a clear and detailed analysis in “Improving Robustness of Machine Translation with Synthetic Noise”. [Vaibhav et al](https://www.aclweb.org/anthology/N19-1190.pdf) approaches two methods to approach robustness. First they introduce synthetic noise induction model which heuristically introduces types of noise unique to social media text and second, labeled back translation by implementing [Sennrich et al., 2015a](https://arxiv.org/abs/1508.07909) methods, a data-driven method to emulate target noise. The result suggests that although both methods increase the BLEU score, the tagged back-translation technique produces the most increase in BLEU score of +6.07 points (14.42 → 20.49) compared to vanilla version. 
[MT. P.Michel et al](https://arxiv.org/pdf/1809.00388.pdf) qualitatively and quantitatively examines types of noises in MTNT dataset itself and shows poor performance of  MT models on these noises. Their result shows a Blue score of 21.77 for English -> French and 23.27 for French -> English for a bidirectional LSTM with 2 layers and [BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding). [Xiaoyi Ma](https://www.cs.brandeis.edu/~marc/misc/proceedings/lrec-2006/pdf/746_pdf.pdf) introduces Champollion,“a lexicon-based parallel text sentence aligner ” which increases the robustness of neural machine translation methods for noisy text by assigning greater weights to less frequent translated words. Experiments on a manually aligned Chinese – English parallel corpus show that Champollion achieves 97% precision and recall on noisy data. 

### Data
We use the [MTNT dataset](http://www.cs.cmu.edu/~pmichel1/mtnt/#data) as in-domain parallel data for the language pair of French and English, where French is the source language and English is the target language. MTNT is a collection of comments from the [Reddit website](https://reddit.com), and in particular this dataset consists of "noisy" text that exhibits typos, grammar errors, code switching and more. An example of data and it’s statistics can be seen in Appendix1.

### Methods and Experiments
For this model, we will be building it primarily on Google Colab, both because this allows all three of us to contribute to the engineering in real-time and because it gives us access to a powerful tool,GPU. However, some preprocessing work occurred locally, such as preparing the files from the raw form in which it was downloaded to a form that was ready to be fed into our model. Cleaning up the data, such as removing the line numbers for each file, also occurred locally. 
The model itself is an attention-based bidirectional Seq2Seq model built in PyTorch, loosely based off of the model that we built during [Lab3](https://github.ubc.ca/MDS-CL-2019-20/COLX_531_translation_students/blob/master/lab_solutions/lab3_sol.ipynb) and [Lab4](https://github.ubc.ca/MDS-CL-2019-20/COLX_531_translation_students/blob/master/lab_solutions/lab4%20-%20sol.ipynb) in the Machine COLX 531 Translation course. In our preliminary model, we use a teacher forcing rate of 0.5, a learning rate of 0.001, encoder dropout of 0.5 and decoder dropout of 0.3. We initialize the model with random weights and train it for fifteen epochs using batch sizes of (4,64,64), and we measure loss using CrossEntropy Loss. Notebook can be found [here](https://github.ubc.ca/nikihm/COLX-585-noisy-Reddit-text/blob/master/Milestone2/Trends_project-baseline.ipynb) on Github.
In the third week, we tried building a multilayer neural net by changing the number of layers from one to two. Additionally, we tried using word embeddings from FastText instead of using weights randomly initialized from a normal distribution with a mean of 0 and a standard deviation of 0.01. However, when we trained the model after implementing the changes, there were no noticeable improvements in the results when we looked at the model's translations. Notebook can be found [here](https://github.ubc.ca/nikihm/COLX-585-noisy-Reddit-text/blob/master/Milestone3/Trends_project-baseline_week3.ipynb) on Github. Since the result of the first model was better than a multilayer RNN, we re-trained the first model using some hyperparameter tuning to achieve the best result. Notebook can be found [here](https://github.ubc.ca/nikihm/COLX-585-noisy-Reddit-text/blob/master/Milestone4/20_BLEU_model.ipynb) on Github. 


### Evaluation and Results
Since our task is neural machine translation, we will use the [BLEU evaluation](https://en.wikipedia.org/wiki/BLEU)system to evaluate the output and the gold standard will be our original Reddit texts in English. For one block of texts, our system would sum clipped n-gram (unigram to 4-gram) matches, divided by the counts of n-grams. It would take an average logarithm with uniform weights and Brevity penalty on sentence lengths.
First a seq2seq with Attention model was trained for 15 epochs, which each epoch took around 15 minutes, a total around 225 minutes. The results show a train loss below 1.8 and validation loss around 5 and BLEU score 0.0839 (Appendix2, table1) . We use our model to generate a sample French-English translation, as shown in the following: 
Source language: [‘<sos>’, ‘a’, ‘partir’, ‘de’, ‘ce’, ‘moment’, ‘la’, ‘,’, “j’“, ‘arrive’, ‘plus’, ‘a’, ‘contenir’, ‘mes’, ‘sentiments’, ‘mais’, ‘je’, ‘sais’, ‘que’, ‘la’, ‘distance’, ‘fait’, ‘que’, ‘rien’, ‘ne’, ‘se’, ‘passera’, ‘et’, “j’“, ‘essaye’, ‘de’, ‘passer’, ‘outre’, ‘,’, ‘mais’, “j’“, ‘y’, ‘arrive’, ‘pas’, ‘.’]
    
Our model translation:  [‘from’, ‘this’, ‘moment’, ‘,’, ‘i’, “’m”, ‘not’, ‘anymore’, ‘my’, ‘feelings’, ‘but’, ‘i’, ‘know’, ‘that’, ‘the’, ‘distance’, ‘prevents’, ‘anything’, ‘over’, ‘and’, ‘i’, “’m”, ‘trying’, ‘to’, ‘get’, ‘over’, ‘it’, ‘,’, ‘but’, ‘i’, ‘ca’, “n’t”, ‘get’, ‘it’, ‘.’]
We see that there are some misalignment and incorrectness in the translation compare to source, for example, [“j’“, ‘arrive’, ‘plus’, ‘a’, ‘contenir’] has translation of [‘i’, “’m”, ‘not’, ‘anymore’] which is not complete meaningful result.
Another  example, the model failed to learn that in English words like “the” or  “and” would not appear at the end of the sentence. 
To improve the results, we trained the model again on Google Colab, and trained the model for fifteen epochs, with an average time of 20 minutes per epoch and a total time of around 300 minutes (approximately five hours). The lowest validation PPL was from Epoch 4, at 113, with a train loss of around 22. The validation PPL for the final epoch,
Epoch 15, was around 174, and the train loss was around 6. 
In our final model, we attempted to use a TPU with the torch_xla package, although we discovered that a GPU still yielded better results. Thus, in our final model training using a GPU and using the same hyperparameters that we had used originally, we obtained a BLEU score of 0.20 after training for 15 epochs. The validation loss was 5.325, while the validation PPL was 205.33. The lack of reproducible results is a bit concerning. 

### Challenges
During this project we faced a few challenges including data type, data size, preprocessing steps and getting correct results.

The data is gathered from reddit and it was in a zip file for multiple language pairs. Each file in data had three columns of index, source language, and target language. The first time we ran the model our translation took indices as source language therefore the model was useless and did not translate correctly. We fixed this issue by removing the indices column from all train, validation and test sets. The Training data contains around 19000 lines of text which takes a very long time and the cuda in collab would get out of memory. To tackle this problem for this milestone we reduced the batch size to 2 to see if we get any results and the problem is not from the model, next we increased the batchsize to (4,64,64) for train, validation and test. However we are planning to cut the training data in half to save time for the next train. 

During preprocessing, to tokenize the data we decided to not use regular white space tokenizer since social media text contains emojis and emoticons, therefore white space tokenizer would not be ideal in this scenario. After searching for multiple tokenizers for social media content we decided to use [Reddit Crazy Tokenizer](https://redditscore.readthedocs.io/en/master/apis/tokenizer.html). This tokenizer seemed to cover all the aspects that need to be suitable for social media text tokenization, such as emojis and emoticons. Upon installation of this tokenizer we faced a problem to import the library into google collab. There were so many dependencies needed for the tokenizer to work on collab. To solve this problem we decided to preprocess in a local notebook and use the outputs and train model on collab. 

One more challenge is that the data size is relatively big. The first time we ran it we encountered CUDA out of memory issue, so we had to switch to a smaller batch size and fewer epochs. After we believe our model is working, we increased the batch size by a bit, and we also increased the number of epochs to 15. However, because of the low number of the batch size, the model took 3.5 hours to run, which is relatively time consuming. 

In week 3, we try several different values for hyperparameters. We change the number of layers from 1 to 2; add pre-trained word embedding models (Fasttext); change learning rates, teaching force rates, dropout rates. We also change batch sizes and epochs to higher numbers. We find that the translation started to spill out random, long sentences which does not make sense and the performance and BLEU scores do not beat the baseline model we built in week two. Therefore we will be trying some more methods to improve our model, such as BPE, sentence piece, different encodings like BERT models. 

### Conclusion and Future Directions
Given that the use of social media and the data that is gathered from this field is becoming increasingly prevalent in our world, it is important to build machine translation models that are able to handle the noise associated with this data. We built robust machine translation models to see which overcome the noisy text obstacle and give reasonable translation from French to English for Reddit comments. We investigated a seq2seq attention model with one layer and multilayers for translating French to English. The result suggested that one layer seq2seq attention with hyper parameter tuning gives the highest BLEU score of 20%. In future, few improvements must be done to result in a more robust machine translation. These further improvements are preprocessing of tokens so we can remove repeated letters in the words (eg. Hiiiii, LOOOL), use a different tokenizer/embedding such as BERT, and  Beam search instead of greedy search approach, which preserve the top(k) tokens instead of remembering the most probable one. This approach could be generating better sequences and not deviated sentences.  Lastly, the traditional RNN LSTM models are our interest, but transformers are the state-to-art models that take advantage of matrix multiplication, which is supposedly faster than quadratic time complexity and we can compare the model with the transformer model by utilizing more of Attention mechanism.

 Online-community domain is of greater importance to investigate and with the knowledge and skills we learned so far, we hope our project helps better understanding of this specific field in terms of automated translation systems.

### References

Michel, P., & Neubig, G. (2018). MTNT: A Testbed for Machine Translation of Noisy Text. Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. doi: 10.18653/v1/d18-1050

Sennrich, R.,Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). doi: 10.18653/v1/p16-1162

Vaibhav, V., Singh, S., Stewart, C., & Neubig, G. (2019). Improving Robustness of Machine Translation with Synthetic Noise. Proceedings of the 2019 Conference of the North. doi: 10.18653/v1/n19-1190


### Appendices: 

Appendix 1.

An example of MTNT dataset:
    
| Language pair | Source | Target |
|---------------|-------------------------------------------------------------------|------------------------------------------------------------------------------------|
| en-fr | Just got called into work tho so I won’t be in til tomorrow night | Mais on vient de m'appeler pour le travail donc je n'y serai pas avant demain soir |
| fr-en | je demande lazil politique pr janluk # Il ressuscitera ! | I demand political asylum for jean luc # He will resurrect! |




Number of examples in the datasets are:

| Data | Number of comments | Type |
|-----------------|------------------------------|------|
| Training Data | En-fr : 36058  Fr-en : 19161 | .tsv |
| Validation Data | En-fr : 886  Fr-en : 852 | .tsv |
| Test Data | En-fr : 1022  Fr-en : 1020 | .tsv |



Appendix 2. 

Table1. Summary of models with results

| Model | Components | Hyperparameters Tuned | Results |
|------------------------------------|--------------------|----------------------------------------------------------------------------------------------------|------------------------------------------------------|
| seq2seq attention | Bidirectional LSTM | Vocab size,embedding dimensions, Dropouts, learning rates, teaching force rate, epochs, batch size | Train loss = 1.8,validation loss = 5BLEU score = 8%. |
| Multi-layer Seq2seq with Attention | Bidirectional LSTM | Pre-trained word embeddings | validation PPL = 113,Train loss of = 22 |
| seq2seq attention | Bidirectional LSTM | Vocab size,embedding dimensions, Dropouts, learning rates, teaching force rate, epochs, batch size |  |




