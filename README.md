# Natural-Language-Processing
 Building a Multi-Class Text Classification for Emotion using the Go Emotions Dataset

There are a total of 28 labels pertaining to emotions. The predictor column, ‘labels’ consists of a list of labels
per row, however given that we are not doing multilabel prediction, we can disregard rows with more than
one label/class. We chose which emotion labels to include by looking at the value counts per label (in
descending order). This was done to ensure that we have a sufficient quantity of data per label. The resulting
labels are: 27, 0, 15, 4, 1, 3, 10, 18, 7, 2, 20, 6, 17 and 25.
