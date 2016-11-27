# Emotion recognition from speech
An HMM-based classifier to recognise emotions from Italian speech and its transcription.

### The corpus
The corpus (which cannot be disclosed) contains 899 sentences pronounced by six Italian actors, 180 for each emotion: joy, neutral, fear, anger and sadness. Each actor read out 50 different sentences, 10 per emotion, several times.

Each utterance is segmented into a number of tokens (i.e. the "words" of the sentence) and for each of these a set of audio and textual features are extracted: the average pitch, its variation, the average intensity, its variation, the average value of the first two formants and their variations, the harmonicity, the phone rate and finally the part-of-speech tag. The continuous features, after being normalised, are discretised into “high”, “mid”, “low” and “--slope“, “-slope “, “flat”, “+slope”, “++slope”.

### The Bayesian model
The classifier is based on five Baum-Welch-trained Hidden Markov Models, one for each emotion we want to recognise.

The system is assumed to be a first-order Markov model (i.e. the behaviour of the system is only based on its past state) with hidden states and observable states. The hidden states for each HMM are two; the observable states are the eleven-dimensional feature vectors.
Formally, HMMs are defined by the hidden states *Q*, the observation symbols *O*, the transition probabilities *A* and the emission probabilities *B*. Since we're dealing with discrete features, *A* and *B* will be matrices.

The task is to find the most likely set of state-transition and emission probabilities, given the observations collected for each emotion. To do so, we use the Baum-Welch algorithm which is a variant of the Expectation-Maximization algorithm, an iterative procedure that finds the best HMM parameters given a set of observations.

Once the classifiers have been trained, every new utterance will be classified as belonging to the HMM that returns the highest likelihood.

### Feature extraction
In order to extract all the different audio features we used the speech-analysis free software PRAAT. The script can be found in the file *audio feature extraction.Praat*.

We used the alignment and silence TextGrids in order to remove the non-spoken parts from the computation of the features.

Due to incorrect alignments in some utterances (e.g. initial tokens were framed on a speechless part of the waveform) and due to the fact that some tokens are pitchless consonants (e.g. the article *l’*), PRAAT returned --undefined-- values for either pitch or harmonicity; we decided to “mend the holes” by averaging the first defined pitch values before and after the undefined one and assigning this value to the missing one (if one of these two values did not exist, for example because the undefined pitch belonged to the first or last token of a sentence, we simply used the first useful value in either direction).

We decided to assign to the delta features “on the border” their preceding or following value (e.g. the delta pitch value assigned to the first token in a sentence is always the same as that of the second token).

Despite this, the script generated a number of outliers in the delta features due to the fact that some tokens lasted only a few milliseconds (e.g. *è*). This introduced a few problems in the discretisation phase since simply taking the range of values for a particular feature and cutting it in five classes would have resulted in assigning the wide majority of tokens to the “flat” class, with a very small number of tokens belonging to the “++slope” and “--slope” classes, thus making the delta features very weak predictors.

In order to extract the POS tags, we used TreeTagger and the Italian POS tagset provided by Marco Baroni (http://sslmit.unibo.it/~baroni/collocazioni/itwac.tagset.txt) trained on the La Repubblica corpus. The output of TreeTagger was modified because the tokenisation it produced wasn’t compatible with that of the corpus (e.g. TreeTagger does not tokenise *c’era* into *c’* and *era*).

### Classifier implementation
Python was used to import the corpus and to discretise it into classes as described. The outlier problem for the delta features was solved using the data-mining Python library Orange, which discretises the data considering its distribution and finding cutting points to create significant classes.

The NLTK library was used for the HMM classification task.

We implemented two feature selection algorithms: “best subset selection” and “forward stepwise selection”. Best subset selection selects the optimal subset of predictors in terms of accuracy for a given subset size, but its complexity makes it computationally impractical even for a small number of features such as ours; a good alternative is forward selection, a greedy procedure that converges to a local optimum, since it chooses the best new feature to add to the active set at every iteration, without changing the previous subset.

We also implemented “k-fold cross-validation” and “random cross-validation”. K-fold cross-validation partitions the dataset in k equal-size subsamples and, of these, a single subsample is retained as the validation data for testing the model, while the remaining k-1 subsamples are used as training data. The cross-validation process is then repeated k times, with each of the k subsamples used exactly once as the training set. Random cross-validation, on the other hand, is a naïve method that simply shuffles the corpus so that the training and testing sets are different every time the programme is run.


### Results
Classifying using the whole set of predictors gives very low accuracies: depending on how the corpus is partitioned, sometimes the accuracies are below the 20%, i.e. it is better to classify an utterance randomly rather than use the classifier. This is due to the fact that some features aren’t correlated with the emotion and have to be discarded. 

Best subset selection without cross-validation gives the following result:

(11)  78%

(1, 11)  89%

(1, 7, 11)  88%

(1, 3, 4, 5)  76%

(1, 3, 5, 7, 10)  72%

(1, 3, 5, 7, 9, 10)  48%

(2, 5, 7, 8, 9, 10, 11)  30%

(2, 4, 5, 6, 7, 8, 10, 11)  30%

(1, 2, 4, 5, 6, 8, 9, 10, 11)  27%

(2, 3, 4, 5, 6, 7, 8, 9, 10, 11)  23%

(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)  21%

It would seem that the POS tag (the eleventh feature) is a very good predictor for this task. However, given that there are only 50 sentences in the corpus and that each sentence represents only one emotion, this gives a bias to the classification (certain POS sequences only appear for some emotions for reasons related to how the corpus was created and not because those sequences inherently belong to one emotion). While POS features give rise to overfitting given this corpus, they might turn out to be useful predictors if the corpus contained different sentences for each utterance.

The best subset of predictors is therefore composed by the following features: average pitch, average intensity, delta intensity and average F1. This subset returned a classification accuracy of 76%.

Finally, using forward stepwise selection together with 4-fold cross-validation (i.e. running the feature selection algorithm 4 times, each time with a different training set) without considering the POS tags, the best accuracy is 78% and it is given by the following features: average pitch, average F1, average F2 and delta F2.

In conclusion the average pitch is positively correlated with speech emotion, since it is chosen by every algorithm, and if we used it to classify an emotion we’d be right 50% of the time. The formants and average intensity are also good predictors.
