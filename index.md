# PracticalMachineLearning
Raphael Turcotte  
April 26, 2016  



## Background
Using personaldevices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: [linked phrase](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset).

The goal of this project is to predict the manner in which participants did the exercise. This is the "classe" variable. The report describes how the data was clean and the model was built built, what cross validation was selected and expected out of sample error is. All decisions are explained.

## Loading data and libraries
The data for this project come from [linked phrase](http://groupware.les.inf.puc-rio.br/har) and have been very generously provided by Velloso, E. et al. ("Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.) We load the data and ensure that all empty or '#DIV/0' cells are NA.


```r
download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "pml-training.csv")
download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "pml-testing.csv")
pmltrain <- read.csv('pml-training.csv',na.strings=c("","#DIV/0","NA"))
pmltest <- read.csv('pml-testing.csv',na.strings=c("","#DIV/0","NA"))
```

We also load the required libraries:

```r
library(ggplot2)
library(caret)
library(gbm)
library(randomForest)
library(MASS)
library(plyr)
```

## Cleaning the data

First, we remove all variables which contains more than 95% of NA values. The limit is arbitrary as only 60 variables remains (from initially 160) for any threshold between 0.1% and 97% of NA values. 

```r
namesNAmin <- names(pmltrain[, colSums(is.na(pmltrain)) < nrow(pmltrain) * 0.95])
position_minNA = match(namesNAmin,names(pmltrain))
pmltrain_minNA = pmltrain[,position_minNA]
```

It is explained in the study design that all participants performed the different lifts. We also want to avoid that any non-sensor-related variable contributes to the predictions. We therefore remove the first 6 columns.

```r
pmltrain_minNA_noID = pmltrain_minNA[,7:dim(pmltrain_minNA)[2]]
```

Finally, we split the training set into a training and validation set. We also applied all transformations to the test set.

```r
set.seed(861)
inTrain <- createDataPartition(pmltrain_minNA_noID$classe, p = 0.6)[[1]]
pmltrain_clean <- pmltrain_minNA_noID[ inTrain,]
pmltrain_crossvalid <- pmltrain_minNA_noID[ -inTrain,]

pmltest_clean <-  pmltest[,position_minNA]
pmltest_clean <- pmltest_clean[,7:dim(pmltrain_minNA)[2]]

remove(pmltest,pmltrain,pmltrain_minNA_noID,pmltrain_minNA,namesNAmin,inTrain,position_minNA)
```

## Building the model

We built three models using the train() function and will evaluate which one is the best based on the accuracy of the validation set. The three models are: 

i) random forest ("rf"),

```r
set.seed(861)
rf_Model = train(classe~., data=pmltrain_clean, method="rf")
```

ii) boosted trees ("gbm"), and

```r
gbm_Model = train(classe~., data=pmltrain_clean, method="gbm")
```

iii) linear discriminant analysis ("lda"). 


```r
lda_Model = train(classe~., data=pmltrain_clean, method="lda")
```

## Validating the model

We perfom a cross validation and simultaneously estimate the out of sample error rate using the validation set that was previously created. We will use the accuracy of the validation set to select the best model.


```r
rf_Results <- predict(rf_Model,pmltrain_crossvalid)
gbm_Results <- predict(gbm_Model,pmltrain_crossvalid)
lda_Results <- predict(lda_Model,pmltrain_crossvalid)

rfAcc <- confusionMatrix(pmltrain_crossvalid$classe, rf_Results)$overall['Accuracy']
gbmAcc <- confusionMatrix(pmltrain_crossvalid$classe, gbm_Results)$overall['Accuracy']
ldaAcc <- confusionMatrix(pmltrain_crossvalid$classe, lda_Results)$overall['Accuracy']
c(rfAcc, gbmAcc, ldaAcc)
```

```
##  Accuracy  Accuracy  Accuracy 
## 0.9955391 0.9882743 0.7190925
```

```r
outOfSampleError <- 1-sum(rf_Results == pmltrain_crossvalid$classe)/length(rf_Results)
outOfSampleError
```

```
## [1] 0.004460872
```

The cross-validation accuracy for the random forest model is the largest: 99.554. The random forest model is of particular interest because it is not sensitive to the interactions between variables which are unknown in this case. They likely exist as all sensors are attached to one human body. It can also handle variables that are unscaled; we didn't rescale the data because we didn't the composition/distribution of the 54 remaining variables after cleaning the data. The out-of-sample error was calculated to be of 0.446%, which is simply 1-accurary for a random forest model as this model provide an unbiased estimate of the out-of-sample error. The best model is therefore a random forest model.


```r
bestModel <- rf_Model
bestModel
```

```
## Random Forest 
## 
## 11776 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 11776, 11776, 11776, 11776, 11776, 11776, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9904944  0.9879689
##   27    0.9951897  0.9939128
##   53    0.9919450  0.9898049
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

## Prediction of the 20 test cases

We can now predict the answers to the 20 test cases for the quiz using the best model:

```r
Quiz <- predict(bestModel, pmltest_clean)
Quiz
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

## Summary

The random forest model provided an accuracy (0.9955) superior to the boosted tree and the linear discrimant model. This model was of particular interest because it is not sensitive to the interactions between variables. More importantly, it predicted correctly all of the 20 test cases.
