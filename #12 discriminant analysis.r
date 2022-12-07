# lets load needed packages

library(dplyr)
library(MASS)
library(janitor)
library(caret)
library(caret)

# lets load churn dataset

load("data/churn_train_test.RData")

# again we will now deal with the model classifying whether 
# a person has churned based on set of characteristics


# !!!!!!!!
# As later we will "manually" recode categorical variables
# into dummies, lets apply the same cleaning of levels
# as last time (removing special characters from levels
# of categorical variables)

# lets store a list of names of factor variables

churn_vars_factors <- 
   churn_train %>% 
   sapply(is.factor) %>% 
   which() %>% 
   names()

# and clean the levels from special characters

for(var_ in churn_vars_factors) {
   # in the training data
   levels(churn_train[[var_]]) <- 
      make_clean_names(levels(churn_train[[var_]]) )
   # and test data
   levels(churn_test[[var_]]) <- 
      make_clean_names(levels(churn_test[[var_]]) )
}


# to apply the linear discriminant analysis
# we can use the function lda() from the MASS package

# in the simplest syntax the lda() function is analogous
# for other modeling functions - requires providing two
# arguments: formula of the model and data set:
# lda(model_formula, data)

# lets use the same variables as in logistic regression

churn_lda1 <- lda(Churn ~ .,
                  data = churn_train %>% 
                     # we exclude customerID
                     dplyr::select(-customerID))
                     
# Let's see the structure of the resulting object

str(churn_lda1)

# it is very complex

# by default the function assumes that the a-priori probabilities 
# are equal to the proportions of the groups observed in the data

# they are stored in a result element named $priors

churn_lda1$prior

# compare to sample frequencies

tabyl(churn_train$Churn)

# discriminant function coefficients are stored
# in the $scaling element

churn_lda1$scaling

# in the case of the binary target variable we have
# only one function (there are always k-1)

# group means 'mu_k' - average values of individual
# variables in each group (coordinates of group centroids)
# ($means)

# Let's see them in the form transposed into a column vector

t(churn_lda1$means)

# lets calculate fitted values

churn_lda1_fitted <- predict(churn_lda1)

str(churn_lda1_fitted)

# predicted classes have been saved
# in the element named $class

# in turn, the $x element contains values
# of discriminant functions (here just one)

head(churn_lda1_fitted$x)

# while $posterior includes final (resulting from the model)
# probabilities of belonging to each group for
# individual observations

head(churn_lda1_fitted$posterior)

# in the case of modeling a binary variable
# we can treat these values as the probability 
# of "positive" and "negative" value

# if we want to select other a-priori probabilities,
# we need to use an additional argument prior=
# in the lda() function

# lets estimate an alternative model with the assumption
# that a priori probabilities are equal across all classes

churn_lda2 <- lda(Churn ~ .,
                  data = churn_train %>% 
                     # we exclude customerID
                     dplyr::select(-customerID),
                     # vector of values adding up to 1,
                     # with a length equal to the number of groups
                     prior = c(0.5, 0.5))

# Let's calculate the fitted values for this model

churn_lda2_fitted <- predict(churn_lda2)

# and compare them with the results of the previous model

table(churn_lda1_fitted$class,
      churn_lda2_fitted$class)

# 749 observations were qualified differently
# - in the first model they were matched as "No",
# - in the second they have been matched as "Yes"


#------------------------------------------
# quadratic discriminant analysis - 2 groups

# we can use a quadratic DA in the same way
# as linear using a function qda() from 
# the MASS package

# its basic syntax is identical
# as in the case of lda()

# let's estimate both above models with qda

# a priori probabilities equal to the frequency of groups

churn_qda1 <- qda(Churn ~ .,
                  data = churn_train %>% 
                     # we exclude customerID
                     dplyr::select(-customerID))

# This error message: “Error in qda.default(x, grouping, …)
# : rank deficiency in group 1” indicates that there is
# a rank deficiency, i.e. some variables are collinear 
# and one or more covariance matrices cannot be inverted 
# to obtain the estimates in group No!


# To deal successfully with this we have to prepare
# a data matrix WITHOUT multicolinear variables.

# !!! but here collinearity appears AFTER one-hot encoding
# of categorical variables ! (the level "No Internet service"
# appearing in many categorical variables)

# So we have to manually make the encoding...
# for example with the function:
# model.matrix(model_formula, data)

churn_train_model_matrix <- 
   model.matrix(Churn ~ .,
                data = churn_train %>% 
                   # we exclude customerID
                   dplyr::select(-customerID))

head(churn_train_model_matrix)

# !!! It does NOT include the dependent variable
# from the above formula (Churn)

# the first column refers to the intercept term (column of 1s)
# which can be skipped here

churn_train_model_matrix <- churn_train_model_matrix[, -1]

# lets also convert the matrix into a data.frame

churn_train_model_matrix <- data.frame(churn_train_model_matrix)

# then lets identify linearly dependent columns

# there is a useful function in the caret package
# that can do it automatically:

(churn_linearCombos <- findLinearCombos(churn_train_model_matrix))

# there are 12 pairs of variables which are 
# multicollinear 

# lets check the first pair

churn_linearCombos$linearCombos[[1]]

# these are indexes of colinear columns
# from the data matrix

names(churn_train_model_matrix)[churn_linearCombos$linearCombos[[1]]]

names(churn_train_model_matrix)[churn_linearCombos$linearCombos[[2]]]

names(churn_train_model_matrix)[churn_linearCombos$linearCombos[[12]]]

# the element $remove includes indexes of columns
# removing which reduces the problem of multicolinearity

churn_linearCombos$remove


names(churn_train_model_matrix)[churn_linearCombos$remove]

# lets remove them from a data matrix

churn_train_model_matrix <- 
   churn_train_model_matrix[, -churn_linearCombos$remove]

# of course we need to add a Churn column to the dataset

churn_train_model_matrix$Churn <- churn_train$Churn
 
  
# lets try to estimate LDA once again

churn_qda1 <- qda(Churn ~ .,
                  data = churn_train_model_matrix)

# still a problem...
   
# lets check the structure of the data

str(churn_train_model_matrix)

# Looks like two columns:
# PhoneServiceyes and MultipleLinesno_phone_service
# refer include the same information

table(churn_train_model_matrix$PhoneServiceyes,
      churn_train_model_matrix$MultipleLinesno_phone_service)

# but were not recognized as colinear

# lets remove the second column

churn_train_model_matrix$MultipleLinesno_phone_service <- NULL

# and try again

churn_qda1 <- qda(Churn ~ .,
                  data = churn_train_model_matrix)

# now it works fine :)

# the structure of the resulting object is similar 
# to the result of LDA - also includes for example
# the element called $prior with a priori probs

churn_qda1$prior

# lets use equal a priori probabilities for all groups

churn_qda2 <- qda(Churn ~ .,
                  data = churn_train_model_matrix,
                  prior = c(0.5, 0.5))

# fitted values are also generated in the same way

churn_qda1_fitted <- predict(churn_qda1)
churn_qda2_fitted <- predict(churn_qda2)

# and they have a similar structure as from the linear model

str(churn_qda1_fitted)

# $class contains the predicted group
# $posterior contains posterior probabilities
# of each level of the target variable

head(churn_qda1_fitted$posterior)


# lets reestimate LDA models on the model matrix

churn_lda1 <- lda(Churn ~ .,
                  data = churn_train_model_matrix)

churn_lda2 <- lda(Churn ~ .,
                  data = churn_train_model_matrix,
                  prior = c(0.5, 0.5))


# and generate fitted values

churn_lda1_fitted <- predict(churn_lda1)
churn_lda2_fitted <- predict(churn_lda2)


# lets see if probabilities of "positive" (Churn = "yes")
# predicted by different models are correlated 

# lets combine columns nemad "yes" from all 
# $posterior elements

churn_fitted_all <- 
   data.frame(lda1 = churn_lda1_fitted$posterior[,"yes"],
              lda2 = churn_lda2_fitted$posterior[,"yes"],
              qda1 = churn_qda1_fitted$posterior[,"yes"],
              qda2 = churn_qda2_fitted$posterior[,"yes"])

cor(churn_fitted_all)

# they seem to be highly correlated


# lets compare the accuracy of all models in the training data

source("functions/F_summary_binary.R")


sapply(churn_fitted_all,
       function(x) summary_binary(predicted_probs = x,
                                  real = churn_train_model_matrix$Churn, 
                                  cutoff = 0.5, 
                                  level_positive = "yes", 
                                  level_negative = "no")
       )

# QDA is clearly better than LDA.

# In LDA using equal a priori probabilities (lda2) 
# does a better job, while in case of QDA 
# just the opposite


# of course LDA/QDA can be also applied via the train() function

fiveStats <- function(...) c(twoClassSummary(...), 
                             defaultSummary(...))

ctrl_cv5 <- trainControl(method = "cv",
                         number = 5,
                         classProbs = TRUE,
                         # and use it in trControl
                         summaryFunction = fiveStats)

set.seed(987654321)

train(Churn ~ ., 
      data = churn_train %>% 
        # we exclude customerID
        dplyr::select(-customerID), 
      method = "lda",
      trControl = ctrl_cv5)

set.seed(987654321)

train(Churn ~ ., 
      data = churn_train %>% 
        # we exclude customerID
        dplyr::select(-customerID), 
      method = "qda",
      trControl = ctrl_cv5)

# here manual recoding into dummies helps

set.seed(987654321)

train(Churn ~ ., 
      data = churn_train_model_matrix, 
      method = "qda",
      trControl = ctrl_cv5)