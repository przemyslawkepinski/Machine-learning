# lets load needed packages

install.packages("janitor")

library(caret)
library(verification)
library(janitor) # tabyl()


# lets get back to logistic regression from labs 04b
# modelling the churn of telecommunication company

load("data/churn_train_test.RData")

# lets compare the distribution of the dependent
# variable in both samples

tabyl(churn_train$Churn)

tabyl(churn_test$Churn)

# almost the same


# WARNING!
# to obtain decoding of ordinal variables into dummies with 
# a reference level for we need to change the appropriate 
# system option contrasts on "contr.treatment"

# for more information, see:
# http://faculty.nps.edu/sebuttre/home/R/contrasts.html

options(contrasts = c("contr.treatment",  # for non-ordinal factors
                      "contr.treatment")) # for ordinal factors


# we will use the function train() from caret package,
# which allows to automatically apply cross-validation

# train() gives easy access to many (over 200)
# modeling methods using a unified interface.

# current list of models:
# http://topepo.github.io/caret/train-models-by-tag.html

# the simplest syntax:
# train(model_formula, 
#       data = training_sample,
#       method = "method_name",
#       trControl = trainControl(...)) # or object saved earlier

# potential additional argument:
# metric =  measure of model assessment - default:
#           "Accuracy" for classification 
#           "RMSE" for regression

# trControl = trainControl() define additional assumptions 
# for the train() function - one can earlier save them
# as a separate object

# syntax of trainControl()
# trainControl(method = "resampling_method", # eg. none, cv, repeatedcv, LOOCV
#              number = number,  # number of samples or iterations
#              repeats = no_repeats) # number of repeats for repeatedcv


# if a particular method requires additional parameters 
# they also have to provided - an example will be shown 
# on the next lab


# lets estimate the same logistic regression as before, but 
# with the train() function and on the training sample  
# lets start WITHOUT cross validation

# we define an appropriate training control

ctrl_nocv <- trainControl(method = "none")

# and use it in model estimation
# (it was called logit2 - lets keep consistency)

churn_logit2_train <- 
  train(Churn ~ ., # simplfied formula:
        # use all other variables (apart from Churn)
        # training sample
        data = churn_train %>% 
          # we exclude customerID
          dplyr::select(-customerID),
        # model type
        method = "glm",
        # family of models
        family = "binomial",
        # train control
        trControl = ctrl_nocv)

# lets see the result

churn_logit2_train

summary(churn_logit2_train)

# looks similar and if estimated on the same sample,
# parameters will the same as with glm() used directly



#---------------------------------------
# prediction

# having trained the model one can use it
# for forecasting using the predict() method
# syntax:
# predict(model, data_to_forecast)

churn_logit2_fitted <- predict(churn_logit2_train,
                               churn_train)

head(churn_logit2_fitted)

# if we want to forecast probabilities for
# individual levels of a dependent variable
# we must use an additional function argument
# predict(), type = "prob" (default type = "raw")

churn_logit2_fitted <- predict(churn_logit2_train,
                               churn_train,
                               type = "prob")

head(churn_logit2_fitted)


# lets see the forecast error on the TEST sample

churn_logit2_forecasts <- predict(churn_logit2_train,
                                  churn_test,
                                  type = "prob")

# confusion matrix 

confusionMatrix(data = as.factor(ifelse(churn_logit2_forecasts["Yes"] > 0.5, 
                                        "Yes",
                                        "No")), 
                  reference = churn_test$Churn, 
                positive = "Yes") 

# Accuracy : 0.813 on test data

# ROC/AUC
roc.area(ifelse(churn_test$Churn == "Yes", 1, 0),
         churn_logit2_forecasts[,"Yes"])$A

# 0.8597783 on TEST data


# but NOW we do not have any data left for 
# potential model selection!

# That is where cross-validation may come into
# - to ASSESS the forecast accuracy (on new data)
# WITHOUT using the test sample


#-----------------------------------------------
# cross validation 

# by using cross validation on the training sample
# we can determine the EXPECTED forecast error
# WITHOUT looking into the test data


# lets use 10-fold cross validation.
# Define the appropriate "control" of the model training

ctrl_cv10 <- trainControl(method = "cv",
                          number = 10)

# and let's apply it during the 
# model training process

# WARNING! during the validation training data is
# randomly divided into several parts.
# If we want to get reproducible result, 
# we need to define a specific random seed

set.seed(987654321)

churn_logit2_train2 <- 
  train(Churn ~ ., 
        data = churn_train %>% 
          # we exclude customerID
          dplyr::select(-customerID),
        method = "glm",
        family = "binomial",
        # train control - now WITH cross-validation
        trControl = ctrl_cv10)

# training with validation lasts a bit longer

# lets check the summary

churn_logit2_train2

# "Resampling results" show averaged estimated
# errors derived from cross-validation
# (based on all validation test samples)

#  Accuracy   Kappa    
# 0.8004412  0.4596673

# Remember that real accuracy on the test sample was
# Accuracy : 0.813 - so here we slightly underestimate it

# one can see results for all folds

churn_logit2_train2$resample

summary(churn_logit2_train2$resample$Accuracy)

# For different folds it shows values from 0.79 to 0.81.

# For 5 folds the variance of results should be lower as 
# we estimate model on larger sample and logistic
# regression is not over-fitting too much

# By default train() function shows
# Accuracy and kappa for classification,
# RMSE and R2 for regression.

# We can also calculate other measures
# of model quality within cross-validation

# It can be defined in the model training control.

# e.g. to display measures based on
# probabilities of individual levels
# (e.g. ROC / AUC), we must "turn on"
# their calculation in all validation steps
# (remember that predict() function by default
# predicts the LEVEL of the variable, NOT probabilities)

# to have probabilities of individual levels
# calculated, one has to use classProbs = TRUE
# option in the trainControl() function

# to calculate non-standard measures of model accuracy
# we need to define/use a specific summary function that
# calculates them - this is controlled by the argument
# summaryFunction= of the function trainControl()


# e.g. summaryFunction = defaultSummary
# (default for binary classification)
# calculates Accuracy and Kappa
# (averaged for all validation tests !!)

# e.g. summaryFunction = twoClassSummary
# will calculate the most common measures for models
# with a binary explanatory variable
# (averaged for all validation tests !!)
# ROC   Sens  Spec

# (for classification with multiple groups
# one can use summaryFunction=multiClassSummary)

# lets see the example 

ctrl_cv10a <- trainControl(method = "cv",
                           number = 10,
                           # we enable calculating probabilities
                           # of individual levels in all validation steps
                           classProbs = TRUE,
                           # we change the function for summary measures
                           summaryFunction = twoClassSummary)

# to have the same division into folds,
# lets use identical random seed as
# in the previous validation process
# on this data

set.seed(987654321)

churn_logit2_train3 <- 
  train(Churn ~ ., 
        data = churn_train %>% 
          # we exclude customerID
          dplyr::select(-customerID),
        method = "glm",
        family = "binomial",
        # we use ROC/AUC as the main summary metric
        # !!! it has to be returned by the 
        # summary function used
        metric = "ROC",
        # and the new training control
        trControl = ctrl_cv10a)

churn_logit2_train3

# ROC        Sens       Spec     
# 0.8382249  0.8939767  0.5416089

# Lets remind real AUC on test data:
# 0.8597783 on test data
# so we again underestimate
# the accuracy based on cross-validation


# in the end lets see the example of repeated cross validation
# (due to the size of the data, let's use 5 * 3 validation

# with a standard measure of accuracy - accuracy)

ctrl_cv5x3 <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 3)

# again the same seed

set.seed(987654321)

churn_logit2_train4 <- 
  train(Churn ~ ., 
        data = churn_train %>% 
          # we exclude customerID
          dplyr::select(-customerID),
        method = "glm",
        family = "binomial",
        # and the new training control
        trControl = ctrl_cv5x3)


churn_logit2_train4

# Accuracy   Kappa    
# 0.8009859  0.4614326

# and ROC

ctrl_cv5x3a <- trainControl(method = "repeatedcv",
                            number = 5,
                            classProbs = TRUE,
                            summaryFunction = twoClassSummary,
                            repeats = 3)

set.seed(987654321)

churn_logit2_train5 <- 
  train(Churn ~ ., 
        data = churn_train %>% 
          # we exclude customerID
          dplyr::select(-customerID),
        method = "glm",
        family = "binomial",
        metric = "ROC",
        # and the new training control
        trControl = ctrl_cv5x3a)

churn_logit2_train5

# ROC      Sens       Spec     
# 0.83858  0.8941666  0.5431517


# one can also see a summary of the results
# the final model - estimated on the entire 
# (training) sample

summary(churn_logit2_train5)

# IMPORTANT!!!
# Using different cross validation methods does not
# have impact on the finally estimated model.

# The final model is estimated on the whole sample
# and is identical in ALL ABOVE examples.

# Cross-validation in the above examples is
# used ONLY to assess the quality of prediction
# of the considered model, and NOT FOR
# model selection.


# the final model is stored in the results of
# the train() function as $finalModel

# lets check that ALL the models above 
# have identical coefficients

identical(coef(churn_logit2_train$finalModel),
          coef(churn_logit2_train2$finalModel))

identical(coef(churn_logit2_train$finalModel),
          coef(churn_logit2_train3$finalModel))

identical(coef(churn_logit2_train$finalModel),
          coef(churn_logit2_train4$finalModel))

identical(coef(churn_logit2_train$finalModel),
          coef(churn_logit2_train5$finalModel))