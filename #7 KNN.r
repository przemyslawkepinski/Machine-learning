# lets load needed packages

# install.packages("class") 
library(class)

require(caret)
require(dplyr)

# load also the functions which we created
# in lab 5 (stored in separate files 
# in the folder called functions)

source("functions/F_summary_binary.R")

# 2nd variant if only classes predicted,
# not probabilities - check its definition!

source("functions/F_summary_binary_class.R")

# lets get back again to modelling churn
load("data/churn_train_test.RData")


#----------------------------------------------
# application of KNN to predict values
# of Churn variable

# we will use a function train() from the caret
# package which in case of KNN method uses
# knn() function from the class package behind

# WARNING! this model requires defining hyperparameters

# list of (hyper)parameters required by a specific model
# can be previewed using the modelLookup() function
# from the caret package

modelLookup("knn")

# model parameter      label forReg forClass probModel
#   knn         k #Neighbors   TRUE     TRUE      TRUE

# here only k - #Neighbors

# however the function knn() has a default value
# of k = 5, so we can run model training without
# setting this parameter

# lets apply knn algorithm on training dataset

# we define appropriate control
# - lets start with no cross-validation

ctrl_nocv <- trainControl(method = "none")

# and run the train() function

churn_train_knn5 <- 
  train(Churn ~ ., 
        data = churn_train %>% 
          # we exclude customerID
          dplyr::select(-customerID),
        # model type - now knn!!
        method = "knn",
        # train control
        trControl = ctrl_nocv)

churn_train_knn5

# the default parameter value in this case is 5

# it is saved in the "finalModel" element
# results of train()

churn_train_knn5$finalModel
churn_train_knn5$finalModel$k

# if we want to set another value of k, we have
# to create a data frame with a column called
# like the parameter and specify one (or more values)

# lets check K = sqrt(n)

sqrt(nrow(churn_train))

# lets use 71 - the first ODD number larger than 
# the square root of n

k_value <- data.frame(k = 71)

# and then use this information
# in the train() function with the option
# tuneGrid=
# WARNING! if we do NOT use resampling
# specify ONE parameter (one set of parameters!

churn_train_knn71 <- 
  train(Churn ~ ., 
        data = churn_train %>% 
          # we exclude customerID
          dplyr::select(-customerID),
        # model type - now knn!!
        method = "knn",
        # train control
        trControl = ctrl_nocv,
        # we give the parameter(s)
        # required by the model
        tuneGrid = k_value)

churn_train_knn71$finalModel$k

# lets calculate fitted values - prediction on the training sample

churn_train_knn71_fitted <- predict(churn_train_knn71,
                                   churn_train)

# it may take a while - prediction might be
# time-consuming for knn!

# Let's see the frequencies of fitted values

table(churn_train_knn71_fitted)

# check different accuracy measures

summary_binary_class(predicted_classes = churn_train_knn71_fitted,
                     real = churn_train$Churn) # level_positive  = "Yes" by default

# accuracy seemingly high but in fact only "negatives"
# predicted well (very high specificity and low
# sensitivity), balanced accuracy and F1 quite low

# lets compare prediction results in the test sample

churn_train_knn71_forecasts <- predict(churn_train_knn71,
                                       churn_test)

table(churn_train_knn71_forecasts)

# check different measures

summary_binary_class(predicted_classes = churn_train_knn71_forecasts,
                     real = churn_test$Churn) # level_positive  = "Yes" by default

# results similar to the training sample
# (even slightly better)


# lets compare the results for k = 5

# lets calculate fitted values - prediction on the training sample

churn_train_knn5_fitted <- predict(churn_train_knn5,
                                   churn_train)

# check different accuracy measures

summary_binary_class(predicted_classes = churn_train_knn5_fitted,
                     real = churn_train$Churn) 

# accuracy measures improved significantly!!!

# lets compare prediction results in the test sample

churn_train_knn5_forecasts <- predict(churn_train_knn5,
                                      churn_test)


summary_binary_class(predicted_classes = churn_train_knn5_forecasts,
                     real = churn_test$Churn) 

# there is is also visible, but it is 
# not as large as in the training sample

# this can be a problem of overfitting
# - small k causes a larger fit to data

# lets try to find optimal value of k

# we will use cross validation for tuning model parameters

# we need to assume a set of differenct values of k
# and analyze model performance for each parameter value

# we create a data frame with the values 
# of parameter k from 1 to 99 by 4

different_k <- data.frame(k = seq(1, 99, 4))

# define the training control -
# use 5-fold cross validation

ctrl_cv5 <- trainControl(method = "cv",
                         number = 5)

# and run the training
# REMEMBER about setting the seed
# to obtain reproducible results!

set.seed(987654321)

churn_train_knn_tuned <- 
  train(Churn ~ ., 
        data = churn_train %>% 
          # we exclude customerID
          dplyr::select(-customerID),
        # model type - now knn!!
        method = "knn",
        # validation used!
        trControl = ctrl_cv5,
        # parameters to be compared
        tuneGrid = different_k)

# now validation is applied to EVERY
# SINGLE value of k, which takes few seconds

# lets check the results

churn_train_knn_tuned

# the result of cross-validation  can be plotted

plot(churn_train_knn_tuned)

# accuracy quite low - one can see
# the reversed U shape where accuracy
# in the validation sample initially increases
# with k but then there is a turning point and
# it starts to decrease again


# CAUTION! remember that in the case of knn
# variables should be rescaled to the range [0,1]

# this can be done automatically
# within the train() function

# in the case of cross-validation, data
# transformations will ALWAYS use ONLY the data
# from the TRAINING sample (here 4/5 of the set
# at each stage of validation) and in the test data 
# (1/5) analogous transformations will be used 

# to apply transformations of input data in the
# train() function one should use preProcess = 
# option providing the transformation method

# possible values: "scale", "center", "range", etc.
# (also several together)

# value "range" automatically scales 
# all variables to range [0, 1]

set.seed(987654321)

churn_train_knn_tuned_scaled <- 
  train(Churn ~ ., 
        data = churn_train %>% 
          # we exclude customerID
          dplyr::select(-customerID),
        # model type - now knn!!
        method = "knn",
        # validation used!
        trControl = ctrl_cv5,
        # parameters to be compared
        tuneGrid = different_k,
        # data transformation
        preProcess = c("range"))


churn_train_knn_tuned_scaled

plot(churn_train_knn_tuned_scaled)

# here the best accuracy obtained for 
# k = 61 and the accuracy seems to increase
# with k


# lets check which value of k 
# gives the highest AUC

ctrl_cv5a <- trainControl(method = "cv",
                          number = 5,
                          # probabilities of each level
                          # predicted in cross-validation
                          classProbs = TRUE,
                          # and summary function
                          # that includes ROC
                          summaryFunction = twoClassSummary)


set.seed(987654321)

churn_train_knn_tuned_scaled2 <- 
  train(Churn ~ ., 
        data = churn_train %>% 
          # we exclude customerID
          dplyr::select(-customerID),
        # model type - now knn!!
        method = "knn",
        # validation used - now with probabilities
        # and twoClassSummary!!!
        trControl = ctrl_cv5a,
        # parameters to be compared
        tuneGrid = different_k,
        # data transformation
        preProcess = c("range"),
        metric = "ROC")

churn_train_knn_tuned_scaled2

# plot 

plot(churn_train_knn_tuned_scaled2)

# from the perspective of of AUC
# the best value of k = 81, but
# looks like it is not much higher
# than for k = 40


# we can also use own summary function 
# and optimize a different statistic

source("functions/F_own_summary_functions.R")

ctrl_cv5b <- trainControl(method = "cv",
                          number = 5,
                          # probabilities of each level
                          # predicted in cross-validation
                          classProbs = TRUE,
                          # and summary function
                          # that includes F1
                          summaryFunction = mySummary)


set.seed(987654321)

churn_train_knn_tuned_scaled3 <- 
  train(Churn ~ ., 
        data = churn_train %>% 
          # we exclude customerID
          dplyr::select(-customerID),
        # model type - now knn!!
        method = "knn",
        # validation used - now with probabilities
        # and mySummary!!!
        trControl = ctrl_cv5b,
        # parameters to be compared
        tuneGrid = different_k,
        # data transformation
        preProcess = c("range"),
        metric = "F1")

churn_train_knn_tuned_scaled3

# plot 

plot(churn_train_knn_tuned_scaled3)

# same as based on ROC

# lets compare prediction on the test 
# sample for all above models

# we store all predictions in 
# a single data frame

churn_test_forecasts <- 
  data.frame(churn_train_knn5 = predict(churn_train_knn5,
                                        churn_test),
             churn_train_knn71 = predict(churn_train_knn71,
                                         churn_test),
             churn_train_knn_tuned = predict(churn_train_knn_tuned,
                                             churn_test),
             churn_train_knn_tuned_scaled = predict(churn_train_knn_tuned_scaled,
                                                    churn_test),
             churn_train_knn_tuned_scaled2 = predict(churn_train_knn_tuned_scaled2,
                                                     churn_test),
             churn_train_knn_tuned_scaled3 = predict(churn_train_knn_tuned_scaled3,
                                                     churn_test)
             )


head(churn_test_forecasts)

# and lets apply the summary_binary_class()
# function to each of the columns with sapply

sapply(churn_test_forecasts,
       function(x) summary_binary_class(predicted_classes = x,
                                        real = churn_test$Churn)) %>% 
  # transpose the results to have statistics in columns
  # for easier comparison
  t()