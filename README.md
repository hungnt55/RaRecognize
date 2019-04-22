# RaRecognize

Instructions to prepare data and run RaRecognize:

1. Prepare data: This step generates 5 random train/test splits, transform text into numerical vector using TFIDF (1K), PCA (99%) and ICA (same #dimension as PCA) and store them in test_data folder.

python setup_experiment_data.py

Here the random splits are indexed from 0 to 4.

2. Run RaRecognize:

1K TFIDF: to run RaRecognize when 1K TFIDF features are use and for a random split, e.g. 0,

./run_RaRecognize_1k.sh 0

PCA: to run RaRecognize when PCA with 99% variance is preserved and for a random split, e.g. 0,

./run_RaRecognize_pca.sh 0

ICA: to run RaRecognize when ICA with the same number of features as PCA is used and for a random split, e.g. 0,

./run_RaRecognize_ica.sh 0
