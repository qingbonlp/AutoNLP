## WAIC AutoNLP 3th Solution(txta)

## Data cleaning and feature selection

1.Do some data cleaning on Chinese and English texts respectively  
2.Do something about the data imbalance  
3.Use automated feature filtering  
4.Automated processing of long and short text  
        We tried hashingvctorizer to reduce the dimension of long text and to deal with sparse short text densely  
5.Character level tf-idf is used for feature selection in Chinese, while word level feature selection is used in English  

## Sub-training, multi-layer sampling training

1.Stratified sampling based on incremental model  
2.Oversampling of the sampled samples  
3.Control the proportion of training sample class quantity  
4.Oversampling is carried out for the categories with too small data volume  

## Linear-SVM+ probability calibration  
1. Unbalanced category of automatic adjustment  
2. Number of iterations of automatic search model  
3. Automatic search for superparameters  


1. Use the cross-validation generator and estimate the calibration of training samples and test samples for each split model parameter  
2. Then average the probability of folding prediction  
3. Since these probabilities are not always consistent, post-processing is performed to normalize them.  
