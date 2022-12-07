# lets load needed packages

library(caret)
library(dplyr)

install.packages("pROC")
library(pROC)

# install.packages("DMwR") # for the SMOTE method

# package removed from CRAN:
# https://cran.r-project.org/web/packages/DMwR/index.html

# to install the latest archived version available
install.packages("https://cran.r-project.org/src/contrib/Archive/DMwR/DMwR_0.4.1.tar.gz", 
                 repos = NULL, 
                 type = "source")

library(DMwR)

install.packages("ROSE") # for the ROSE method
library(ROSE)
# https://journal.r-project.org/archive/2014/RJ-2014-008/RJ-2014-008.pdf

library(janitor)

# lets load churn data
setwd("C:\\Users\\pk393785\\Desktop\\Machine Learning\\11")
load("data/churn_train_test.RData")



# BEFORE we proceed!

# one of the resampling methods, namely "ROSE" requires 
# that all the levels of factor variables (dummy variables 
# after recoding) do not include special characters, like " ", etc.

# lets clean this 

# first we store a list of names
# of variables that are factors

churn_vars_factors <- 
  churn_train %>% 
  sapply(is.factor) %>% 
  which() %>% 
  names()

# check for which variables 
# levels include a space

for(var_ in churn_vars_factors) {
  # store the levels of tha variable as a character vector
  levels_ <- levels(churn_train[[var_]])
  # check if any level includes a space
  which_include_spaces <- grep(" ", levels_)
  # if none, jump to the next variable
  # (iteration of the loop)
  if (length(which_include_spaces) == 0) next else {
    # otherwise print a message
    message(paste0("variable: ", 
                   var_, 
                   " has a space in levels: "))
    # and a list of 
    print(levels_[which_include_spaces])
  }
} # end of the loop


# lets "correct" the issue by transforming
# the levels using a function make_clean_names()
# from the janitor package (it replaces all 
# non-standard characters into "_")

# we can see the example

levels(churn_train$MultipleLines)

levels(churn_train$MultipleLines) <- 
  make_clean_names(levels(churn_train$MultipleLines) )

# check what has changed
levels(churn_train$MultipleLines)

# in fact we can easily apply this to
# all factor variables in our dataset

for(var_ in churn_vars_factors) {
  levels(churn_train[[var_]]) <- 
    make_clean_names(levels(churn_train[[var_]]) )
}

# of course the same transformation of variables
# values should be applied to the test dataset

for(var_ in churn_vars_factors) {
  levels(churn_test[[var_]]) <- 
    make_clean_names(levels(churn_test[[var_]]) )
}



#-------------------------------------------------------
# application of sample balancing methods


# lets check the effect of balancing the sample

# CAUTION!
# rebalancing technique should be used 
# ONLY on the TRAINING DATA SET !!!!!

# Similarly like data transformations - it makes
# no sense in applying transformations first
# and dividing the transformed data in the
# training and testing dataset -
# this is information leakage.

# It makes no sense to create instances based on 
# the current minority class and then exclude 
# an instance for validation, pretending we didn’t
# generate it using data that is still in the training set.

tabyl(churn_train$Churn)


# lets check how down-sampling works

set.seed(987654321)

churn_train_down <-
  downSample(# a matrix or data frame of predictor variables
    x = churn_train, 
    # a factor variable with the class memberships
    y = churn_train$Churn,
    # a label for the class column
    yname = "Churn") 

tabyl(churn_train_down$Churn)


# lets check how up-sampling works - same syntax as above

set.seed(987654321)

churn_train_up <- 
  upSample(# a matrix or data frame of predictor variables
    x = churn_train, 
    # a factor variable with the class memberships
    y = churn_train$Churn,
    # a label for the class column
    yname = "Churn") 

tabyl(churn_train_up$Churn)




#--------------------------------------------------------
# lets estimate the simple logistic regression model

# lets define a summary function to evaluate
# models with all 5 measured

fiveStats <- function(...) c(twoClassSummary(...), 
                             defaultSummary(...))

# in the twoClassSummary() the first level
# of the dependent variable is automatically 
# assumed as the level of "success" and 
# sensitivity refers to it

ctrl_cv5 <- trainControl(method = "cv",
                         number = 5,
                         classProbs = TRUE,
                         # and use it in trControl
                         summaryFunction = fiveStats)

# and use it in model estimation

set.seed(987654321)

churn_logit_train <- 
  train(Churn ~ ., 
        data = churn_train %>% 
          # we exclude customerID
          dplyr::select(-customerID), 
        # model type
        method = "glm",
        # family of models
        family = "binomial",
        # train control
        trControl = ctrl_cv5)

# lets see the result

churn_logit_train


# let's try to use different methods of balancing the sample

# applying all methods in the caret package is very easy

# CAUTION!!
# the issue here is not even WHICH method to use, 
# but WHEN to use it. 
# Using oversampling before cross-validation may lead
# to almost perfect accuracy, due to information leakage
# discussed above.

# caret's train() applies resampling on EACH FOLD
# of cross-validation INDPENDENTLY


#-------------------------
# weighing observations

# train() function has the argument weights =
# (of course, it all depends on whether 
# the applied model allows weighting the data)

# we have to generate weights (adding up to 1 or n)
# in such a way that the weights for each group sum up
# up to 0.5 (or 0.5*n)

# so we generate weights in each group as
# 0.5/n_i, where n_i is the size of a given group

(freqs <- table(churn_train$Churn))

myWeights <- ifelse(churn_train$Churn == "yes",
                    0.5/freqs[2], 
                    0.5/freqs[1]) * nrow(churn_train)

tabyl(myWeights)

sum(myWeights) == nrow(churn_train)

# and estimate the model with weights

set.seed(987654321)

churn_logit_train_weighted <- 
  train(Churn ~ ., 
        data = churn_train %>% 
          # we exclude customerID
          dplyr::select(-customerID), 
        # model type
        method = "glm",
        # family of models
        family = "binomial",
        # train control
        trControl = ctrl_cv5,
        # we add weights
        weights = myWeights)

# lets see the result

churn_logit_train_weighted

# ROC is slightly worse, overall accuracy lower, 
# but specificity improved a lot and
# is much closer to sensitivity



#------------------
# down-sampling

# to define the resampling method in caret
# one only needs to define it in trainControl()
# by adding the appropriate sampling = argument

# we can use the trainControl() function again
# to define all entries from scratch or
# simply modify the selected argument

# lets see its current value

ctrl_cv5

# lets just change the $sampling entry to "down"

ctrl_cv5$sampling <- "down"

set.seed(987654321)

churn_logit_train_down <- 
  train(Churn ~ ., 
        data = churn_train %>% 
          # we exclude customerID
          dplyr::select(-customerID),
        # model type
        method = "glm",
        # family of models
        family = "binomial",
        # train control
        trControl = ctrl_cv5)

# lets see the result

churn_logit_train_down

# comparable with simple weighting,



#-----------------------
# up-sampling

# lets just change the $sampling entry to "up"

ctrl_cv5$sampling <- "up"

set.seed(987654321)

churn_logit_train_up <- 
  train(Churn ~ ., 
        data = churn_train %>% 
          # we exclude customerID
          dplyr::select(-customerID),
        # model type
        method = "glm",
        # family of models
        family = "binomial",
        # train control
        trControl = ctrl_cv5)

# lets see the result

churn_logit_train_up

# comparable to down-sampling



#--------------------
# SMOTE model

# we will use the SMOTE method in the same way

ctrl_cv5$sampling <- "smote"

set.seed(987654321)

# as it takes longer, lets use smaller dataset
# for training

churn_logit_train_smote <- 
  train(Churn ~ ., 
        data = churn_train %>% 
          # we exclude customerID
          dplyr::select(-customerID),
        # model type
        method = "glm",
        # family of models
        family = "binomial",
        # train control
        trControl = ctrl_cv5)

# which takes a bit longer
# lets see the result

churn_logit_train_smote

# seems to be better than up/down sampling


#--------------------
# ROSE model

ctrl_cv5$sampling <- "rose"

set.seed(987654321)

churn_logit_train_rose <- 
  train(Churn ~ ., 
        data = churn_train %>% 
          # we exclude customerID
          dplyr::select(-customerID),
        # model type
        method = "glm",
        # family of models
        family = "binomial",
        # train control
        trControl = ctrl_cv5)


churn_logit_train_rose


# once again lets compare directly all
# models on validation stage

# the results of validation are stored
# in the element of each result called 
# $results, for example

churn_logit_train$results

# wee need only columns 2-6
churn_logit_train$results[, 2:6]

# lets compare this for all the models
# considered above!

models_all <- ls(pattern = "churn_logit_train")

models_all

# remember that if the object name is stored
# as a string one can easily access this object
# with the get function:

get("churn_logit_train_weighted")

# not lets use this knowledge to get 
# the direct comparison of models

sapply(models_all,
       function(x) (get(x))$results[,2:6]) %>% 
  # transposition in the end to have
  # models in rows and statistics in columns
  t()

# lets also compare the results on 
# the full training and test sample

# we can use a function that for a trained model 
# - result of the train() function,
# will return selected measures

source("functions/F_summary_binary_class.R")

# full training data

# lets show an example for a selected model

# take the name
"churn_logit_train" %>% 
  # get the model
  get() %>% 
  # use it for prediction
  # in the TRAIN sample
  predict(newdata = churn_train) %>% 
  # apply the summary
  summary_binary_class(level_positive = "yes",
                       level_negative = "no",
                       real = churn_train$Churn)

# we can easily apply this to all the models  
# in the training sample

models_all %>% 
  sapply(function(x) get(x) %>% 
           # use it for prediction
           # in the TRAIN sample
           predict(newdata = churn_train) %>% 
           # apply the summary
           summary_binary_class(level_positive = "yes",
                                level_negative = "no",
                                real = churn_train$Churn)) %>% 
  # transpose the result to # have models in rows
  t()

# and do the same for the TEST data

models_all %>% 
  sapply(function(x) get(x) %>% 
           # use it for prediction
           # in the TEST sample
           predict(newdata = churn_test) %>% 
           # apply the summary
           summary_binary_class(level_positive = "yes",
                                level_negative = "no",
                                real = churn_test$Churn)) %>% 
  # transpose the result to # have models in rows
  t()

# in both cases (train and test) the balanced accuracy 
# and F1 increase when reweighting or resampling is
# applied.
# SMOTE seems to make a better job than ROSE, but
# simple down or upsampling is also not bad