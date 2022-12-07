# lets load needed packages

install.packages("caret")
install.packages("verification")

library(dplyr)
library(caret)
library(verification)
library(nnet)

#-------------------------------------------------------------
# evaluation of regression

# lets load the data about houses previously
# divided into train and test samples

load("data/houses_train_test.RData")

# The description ov variables (types and definition)
# can be found for example here (names might be slightly different):
# https://rdrr.io/cran/subgroup.discovery/man/ames.html

# The detailed description of the variable levels
# can be found here 
# http://jse.amstat.org/v19n3/decock/DataDocumentation.txt
# and in the text file "DataDocumentation.txt" in lab materials

# lets load the variables lists 
# saved on labs 3

load("data/houses_variables_lists.RData")

houses_variables_all

# Lets exclude Order and PID
houses_variables_all <- houses_variables_all[-1:-2]

# and estimate the model lm4 (see labs 04a)

# WARNING!
# to obtain decoding of ordinal variables into dummies with 
# a reference level for we need to change the appropriate 
# system option contrasts on "contr.treatment"

# for more information, see:
# http://faculty.nps.edu/sebuttre/home/R/contrasts.html

options(contrasts = c("contr.treatment",  # for non-ordinal factors
                      "contr.treatment")) # for ordinal factors

# in fact if your task is just prediction, the way
# of recoding categorical variables does not matter.
# It does NOT have any impact on the predicted values

houses_lm4 <- lm(Sale_Price ~ ., # simplified formula
                 # ALL variables from the dataset
                 # apart from Sale_Price are used
                 # as predictors
                 data = houses_train %>% 
                   dplyr::select(houses_variables_all)) # training data

summary(houses_lm4)

# R2 and adjusted R2 is printed automatically

# among other results the model stores residuals

head(houses_lm4$residuals)

hist(houses_lm4$residuals, breaks = 30)

# and MSE is basically the average square residual

mean(houses_lm4$residuals^2)

# and MAE is the average absolute residual

mean(abs(houses_lm4$residuals))

# similarly we can calculate Median Absolute Error

median(abs(houses_lm4$residuals))

# lets write a simple function that will 
# summarize all the errors and R2

regressionMetrics <- function(real, predicted) {
  # Mean Square Error
  MSE <- mean((real - predicted)^2)
  # Root Mean Square Error
  RMSE <- sqrt(MSE)
  # Mean Absolute Error
  MAE <- mean(abs(real - predicted))
  # Mean Absolute Percentage Error
  MAPE <- mean(abs(real - predicted)/real)
  # Median Absolute Error
  MedAE <- median(abs(real - predicted))
  # Mean Logarithmic Absolute Error
  MSLE <- mean((log(1 + real) - log(1 + predicted))^2)
  # Total Sum of Squares
  TSS <- sum((real - mean(real))^2)
  # Residual Sum of Squares
  RSS <- sum((predicted - real)^2)
  # R2
  R2 <- 1 - RSS/TSS
  
  result <- data.frame(MSE, RMSE, MAE, MAPE, MedAE, MSLE, R2)
  return(result)
}

# lets apply it to our model

regressionMetrics(real = houses_train$Sale_Price,
                  predicted = predict(houses_lm4))


#------------------------------------------------------
# binary classification

# lets get back to logistic regression from labs 04b
# modelling the churn of telecomunication company

load("data/churn_train_test.RData")

# Lets estimate a model with ALL potential predictors
# (we do not use CustomerID)

churn_logit2 <- glm(Churn ~ ., # simplfied formula:
                    # use all other variables (apart from Churn)
                    # as predictors
                    family =  binomial(link = "logit"),
                    data = churn_train %>% 
                      # we exclude customerID
                      dplyr::select(-customerID))

summary(churn_logit2)

# lets calculate predicted probabilities 

churn_logit2_fitted <- predict(churn_logit2,
                               type = "response")

# lets see the classification table with 
# the default cut-off probability of 0.5

table(churn_train$Churn,
      ifelse(churn_logit2_fitted > 0.5, "Yes", "No"))

# based on these numbers we can calculate
# accuracy, sensitivity, specifity, NPV, PPV, etc.

# but we can also use one of may R functions
# that do it automatically
# lets use confusionMatrix() from caret package

# syntax:
# confusionMatrix(data,
#                 reference,
#                 positive)

# where 
# data - refers to predicted values (on the scale/levels of the dependent)
# reference - real values of the dependent variable (our reference)
# positive - indicates which value refers to "success" (provided in
#            quotation marks), by default the first value treated as "success"

# CAUTION!
# both `data` and `reference` should be factors with the same levels

( ctable <- confusionMatrix(data = as.factor(ifelse(churn_logit2_fitted > 0.5, 
                                                    "Yes", 
                                                    "No")
                                             ), 
                                             reference = churn_train$Churn, 
                                             positive = "Yes") 
)

# the output includes

# - classification table
# TN = True Negative
# FN = False Negative
# TP = True Positive
# FP = False Positive
# n = TN+TP+FN+FP


#               Reference 	
# Prediction  	 No    	Yes   
#----------------------------
#     No 	      TN      FN   
#     Yes	      FP      TP   
#-----------------------------

# - additional measures

# Accuracy = (TN + TP) / n      - accuracy, percent of correct preditions
# Sensitivity = TP / (TP + FN)  - sensitivity
# Specificity = TN / (TN + FP)  - specificity
# PPV = TP / (TP + FP)          - positive predictive value - the share of 
#                                 correctly predicted "successes"
# NPV = TN / (TN + FN)          - negative predictive value - the share of 
#                                 correctly predicted "defaults"
# BalancedAccuracy = (sensitivity + specificity)/2

# Precision - the same as PPV
# Recall - the same as sensitivity

# F1 - harmonic mean of Precision and Recall (range between 0 and 1)
#      equally weights the precision and recall, which is criticized
#      as different errors may have different cost for the company



# - other measures, used less often

# No Information Rate - share of largest group in the sample -
#                       i.e. accuracy of the "model" which always
#                       predicts the mode

# Prevalence = (TP + FN) / n  - share of "successes" in the sample
# DetectionRate = TP / n      - share of correctly predicted "successes"
#                               in the whole sample
# DetectionPrevalence = (TP + FP) / n - share of predicted "successes" 
#                                    in the sample
# Kappa - the measure of agreement between predicted and real values.
#          The higher the better.


# lets check the structure of the object ctable

str(ctable)

ctable$overall
ctable$byClass

# we can easily get selected measures into a vector

c(ctable$overall[1],
  ctable$byClass[c(1:4, 7, 11)])

# lets write a simple function that will 
# calculate these measures for any model

summary_binary <- function(predicted_probs,
                           real,
                           cutoff = 0.5,
                           level_positive = "Yes",
                           level_negative = "No") {
  # save the classification table
  ctable <- confusionMatrix(as.factor(ifelse(predicted_probs > cutoff, 
                                             level_positive, 
                                             level_negative)), 
                            real, 
                            level_positive) 
  # extract selected statistics into a vector
  stats <- round(c(ctable$overall[1],
                   ctable$byClass[c(1:4, 7, 11)]),
                 5)
  # and return them as a function result
  return(stats)
}


# lets check how it works

summary_binary(predicted_probs = churn_logit2_fitted,
               real = churn_train$Churn)



# in addition we can plot a ROC curve using 
# a function roc.plot() from verification package

# syntax
# roc.plot(real_values_in_terms_of_0_1,
#          predicted_probabilities_of_success)

roc.plot(ifelse(churn_train$Churn == "Yes", 1, 0),
         # predicted probabilities
         churn_logit2_fitted) 

# on the vertical axis we have sensitivity (True Positive Rate),
# and on a horizontal one 1 - specifity (False Positive Rate)

# similarly roc.area() calculates area under curve
# (the same syntax)

roc.area(ifelse(churn_train$Churn == "Yes", 1, 0),
         # predicted probabilities
         churn_logit2_fitted)

# A = 0.8426294


#--------------------------------------------------
# assessment of the accuracy
# of the multiclass classification

# lets apply multinomial logistic regression for
# the status of the credit card account (see labs 04b)

load("data/cards_train_test.RData")

cards_mlogit1 <- multinom(status ~ credit_limit + n_contracts +
                            utilization + gender, 
                          data = cards_train)

# and save predicted values

cards_mlogit1_fitted <- predict(cards_mlogit1)


# and see the comparison of real and predicted values
# (equivalent to the classification table)

( ctable_m <- table(cards_mlogit1_fitted,
                    cards_train$status) 
)

# predicted values are in rows and real in columns

# let's count the total % of correct predictions
# (sum of values from diagonal / n)

# the diag() function returns values from
# the diagonal of the square matrix 

100 * sum(diag(ctable_m)) / sum(ctable_m)

# seems quite high - above 85%...

# let's count the share of correctly predicted values
# for each group - the multiclass equivalent of 
# sensitivity / specificity
# (the probability of predicting a particular level,
# provided that it actually occurred)

100 * diag(ctable_m) / colSums(ctable_m)

# almost 95% of customers that had status "O" predicted correctly!
# a bit lower for the remaining statuses, but still high - 73-76%

# These values can be averaged into 
# the equivalent for balanced accuracy
# (or avg macro recall)

mean(100 * diag(ctable_m) / colSums(ctable_m))


# lets look at the equivalents of PPV and NPV,
# i.e. percent of correctly predicted values
# among predictions for a particular level

100 * diag(ctable_m) / rowSums(ctable_m)

# again this values can be averaged into 
# the equivalent for balanced correctly
# predicted measure (avg macro precision)

mean(100 * diag(ctable_m) / rowSums(ctable_m))

# one can also calculate the harmonic mean
# of recall and precision for each level (F1)


# one can write a simple function to summarize
# multinomial model with three above discussed
# measures (averaged)

accuracy_multinom <- function(predicted, real) {
  ctable_m <- table(predicted, 
                    real)
  # accuracy
  accuracy <- (100 * sum(diag(ctable_m)) / sum(ctable_m))
  # recall
  recall <- 100 * diag(ctable_m) / colSums(ctable_m)
  # precision
  precision <- 100 * diag(ctable_m) / rowSums(ctable_m)
  # F1 - harmonic mean of each above pair
  F1 <- mapply(function(x, y) 1/mean(1/c(x, y)),
                         recall,
                         precision)
    
  return(c(accuracy = accuracy, 
           recall = recall,
           precision = precision,
           F1 = F1,
           avg_macro_recall = mean(recall),
           avg_macro_precision = mean(precision),
           avg_macro_F1 = mean(F1)))
}

accuracy_multinom(predicted = cards_mlogit1_fitted, 
                  real = cards_train$status)