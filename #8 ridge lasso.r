# lets load all needed packages

library(caret)

# package with a function for estimating regularized models

install.packages("glmnet")
library(glmnet)

library(dplyr)

# We return to modelling the price of the house.
# dataset: houses, target variable: Sale_Price

# Let's load previously prepared data

load("data/houses_train_test.RData")

load("data/houses_variables_lists.RData")

# for the purposes of the first example we will use
# the function glmnet() from the glmnet package
# (Friedman, Hastie, and Tibshirani 2008)
# the algorithm used is very fast (written in Fortran),

# WARNING!
# in the glmnet() function, we do NOT provide the formula
# of the model, only
# y = target variable
# x = NUMERIC MATRIX with explanatory variables
# family = type of target variable ("gaussian" for the penalized
#   linear regression, "binomial" for penalized logistic regression
#    and "multinomial" for penalized multinomial logit model)

# the need to provide explanatory variables in the form
# of a matrix means that all categorical (factor) variables have 
# to be recoded into numeric (matrix in R can only store data
# of one type), that is, manually recode qualitative explanatory
# variables into dummy variables

# therefore later we will use its implementation in the caret
# package, which does it automatically

# glmnet() function has argument alpha, which is responsible
# for the type of regularized regression:
# alpha = 0 for ridge regression
# alpha = 1 for LASSO

# let's start with ridge regression using only
# quantitative variables

houses_quant_vars <- 
  sapply(houses_train, is.numeric) %>% 
  which() %>% 
  names()

# we need to exclude Sale_Price, but lets
# remove also Order, PID, Longitude and Latitude

houses_quant_vars <- 
  houses_quant_vars[-which(houses_quant_vars %in%
                             c("Sale_Price", 
                               "Order", "PID", 
                               "Longitude", "Latitude"))]

# and estimate the model

houses_ridge <- glmnet(# x needs to be a matrix here
  x = as.matrix(houses_train[, houses_quant_vars]),
  y = houses_train$Sale_Price,
  family = "gaussian", # if not provided, the function guesses it
  # based on the distribution of the variable y
  alpha = 0) # for ridge regression

# by default, the function estimates the model 
# with some automatically selected values of lambda

# we can see these values

houses_ridge$lambda

# one can also enter own lambda range
# let's specify 200 values between 10^(-2) and 1e9
# (at equal intervals on a logarithmic scale)
# i.e. from the value close to 0 (almost OLS)
# to a very high penalty for the values of betas

lambdas <- exp(log(10)*seq(-2, 9, length.out = 200))

houses_ridge <- 
  glmnet(x = as.matrix(houses_train[, houses_quant_vars]),
         y = houses_train$Sale_Price,
         family = "gaussian",
         # providing own values of lambdas
         lambda = lambdas,
         alpha = 0)

# WARNING! the glmnet() function standardizes variables by default
# so that they have the same scale.
# this can be turned off using the option standardize = FALSE
# (which is NOT recommended)

# for each lambda value we have a set of coefficients.
# the size of matrix of coefficients is therefore 
# 33 (number of parameters)
# by 200 (number of lambda values)

dim(coef(houses_ridge))

head(coef(houses_ridge))

# lambdas are in columns, variables in rows

# estimation results for FIRST lambda are in the first column
coef(houses_ridge)[, 1]

# estimated coefficients for the FIRST variable 
# (Lot_Frontage) is in the second row 
# (first includes intercept terms which are not
# interpretable here)

coef(houses_ridge)[2, ]

# lets check how these parameter changes with lambda

plot(houses_ridge$lambda,
     coef(houses_ridge)[2, ],
     log = "x",
     type = "l")

# we expect that for larger lambdas
# coefficients will be closer to 0
# - and that is what we observe here

# Let's see a graph for all 32 quantitative variables
# (we omit the constant term in the model - 1st row)

colors_ <- rainbow(33)

plot(houses_ridge$lambda, 
     coef(houses_ridge)[2,], 
     type = "l",
     ylim = c(-600, 600),
     lwd = 3, 
     col = colors_[2],
     xlab = "value of lambda", 
     ylab = "regression coefficient",
     # we use log-scale for x-axis
     # otherwise not much can be seen
     log = "x",
     main = "Ridge regression parameters for different lambdas")
# the remaining ones in a loop
for (i_col in 3:33) lines(houses_ridge$lambda, 
                          coef(houses_ridge)[i_col,], 
                          col = colors_[i_col],
                          lwd = 3)
abline(h = 0, lty = 2)

# interestingly, as lambda increases initially
# some parameters are rising to finally start
# fall asymptotically to 0

# lets compare the analogous graph for the LASSO method

houses_lasso <- 
  glmnet(x = as.matrix(houses_train[, houses_quant_vars]),
         y = houses_train$Sale_Price,
         family = "gaussian",
         # providing own values of lambdas
         lambda = lambdas,
         alpha = 1) # this defines LASSO

# lets plot the same figure as before - now for LASSO

plot(houses_lasso$lambda, 
     coef(houses_lasso)[2,], 
     type = "l",
     ylim = c(-600, 600),
     lwd = 3, 
     col = colors_[2],
     xlab = "value of lambda", 
     ylab = "regression coefficient",
     # we use log-scale for x-axis
     # otherwise not much can be seen
     log = "x",
     main = "Ridge regression parameters for different lambdas")
# the remaining ones in a loop
for (i_col in 3:33) lines(houses_lasso$lambda, 
                          coef(houses_lasso)[i_col,], 
                          col = colors_[i_col],
                          lwd = 3)
abline(h = 0, lty = 2)

# here for lambda = 1e5 all parameters become = 0 
abline(v = 1e5, 
       col = "red",
       lty = 2)

# now use the full model - to avoid manual recoding
# of qualitative variables into dummies, lets use 
# the train() function from the caret package

# we may wish to change the default recoding 
# for ordinal factor to make it consistent 
# with nominal factors (one-hot encoding)

options(contrasts = c("contr.treatment",  # for non-ordinal factors
                      "contr.treatment")) # for ordinal factors


# let's start with ridge regression

# Define cross-validation to find the optimal
# value of the lambda parameter

ctrl_cv5 <- trainControl(method = "cv",
                         number = 5)

# use the same lambda values as before -
# for the needs of train() we must provide them in 
# the data frame with columns having appropriate names

parameters_ridge <- expand.grid(alpha = 0, # ridge 
                                lambda = lambdas)


# lets remve Order, PID, Longitude and Latitude
# from the list of all variables

houses_variables_all <- 
  houses_variables_all[-which(houses_variables_all %in%
                                c("Order", "PID", 
                                  "Longitude", "Latitude"))]

set.seed(123456789)
houses_ridge <- train(Sale_Price ~ .,
                      data = houses_train %>% 
                        dplyr::select(all_of(houses_variables_all)),
                      method = "glmnet", 
                      tuneGrid = parameters_ridge,
                      trControl = ctrl_cv5)

houses_ridge

plot(houses_ridge)

# generally RMSE increases with lambda
# (the lower, the better!)

# let's see results for lambda between 10 to 10000 more precisely

parameters_ridge2 <- expand.grid(alpha = 0,
                                 lambda = seq(10, 1e4, 10))


set.seed(123456789)

houses_ridge <- train(Sale_Price ~ .,
                      data = houses_train %>% 
                        dplyr::select(all_of(houses_variables_all)),
                      method = "glmnet", 
                      tuneGrid = parameters_ridge2,
                      trControl = ctrl_cv5)


houses_ridge

plot(houses_ridge)

# looks like RMSE/R2 for lambda up to 5120 is equally
# good as for lambda close to 0 (OLS without restrictions)

# in this case in the $finalModel object
# the caret() function saves the full result
# of glmnet() functions, including parameter estimates
# for models for all lambda considered

# Let's see that the coefficients for the best
# models are not equal to 0

# best lambda
houses_ridge$bestTune$lambda

# on the results of glmnet() one can use
# the predict function to return parameters
# for any lambda

predict(houses_ridge$finalModel, # stored model
        s = houses_ridge$bestTune$lambda, # lambda
        type = "coefficients")

# we can check how they would look for 
# OLS model (lambda = 0) although
# it was not tested

predict(houses_ridge$finalModel, # stored model
        s = 0, # lambda
        type = "coefficients")

# for the sake of comparison, let's save the estimation 
# results of the OLS regression

houses_lm <- train(Sale_Price ~ .,
                   data = houses_train %>% 
                     dplyr::select(all_of(houses_variables_all)),
                   method = "glmnet", 
                   tuneGrid = expand.grid(alpha = 0, 
                                          lambda = 0),
                   trControl = trainControl(method = "none"))

# (remember that in the glmnet method
# variables are automatically standardized!)

houses_lm

# lets load the function that will
# summarize all the error measures 
# used on previous labs

source("functions/F_regression_metrics.R")


# and lets see a forecast error on the test data
# from regular linear regression (lambda = 0)

regressionMetrics(real = houses_test$Sale_Price,
                  predicted = predict(houses_lm, 
                                      houses_test)
                  )

# Missing value of MSLE might mean that 
# some predictions are negative and
# calculating it's log is not possible


# let's compare the result for the optimal 
# ridge regression (lambda 5120)

regressionMetrics(real = houses_test$Sale_Price,
                  predicted = predict(houses_ridge, 
                                      houses_test)
                  )

# identical!!! - regularization does not help here


# Let's see if the LASSO method will 
# improve the forecast errors


# let's take lambda parameters again from 10 to 10000

parameters_lasso <- expand.grid(alpha = 1,
                                lambda = seq(10, 1e4, 10))


set.seed(123456789)

houses_lasso <-  train(Sale_Price ~ .,
                       data = houses_train %>% 
                         dplyr::select(all_of(houses_variables_all)),
                       method = "glmnet", 
                       tuneGrid = parameters_lasso,
                       trControl = ctrl_cv5)

houses_lasso

plot(houses_lasso)

# what is the best lambda value (giving the lowest error
# forecasts based on cross validation)?

houses_lasso$bestTune$lambda

# 370 - quite close to 0 (similar to OLS?)

# let's see the coefficients for variables in this model

predict(houses_lasso$finalModel, # stored model
        s = houses_lasso$bestTune$lambda, # lambda
        type = "coefficients")

# despite the low lambda many variables fell out of the model
# (dots mean that the coefficients are equal to 0)

# Let's see what would happen for a larger lambda

predict(houses_lasso$finalModel,
        s = 4000,
        type = "coefficients")

# here already most variables have parameters equal to 0

# check the quality of the forecast from the LASSO model
# for the optimal lambda parameter value

regressionMetrics(real = houses_test$Sale_Price,
                  predicted = predict(houses_lasso, 
                                      houses_test)
                  )

# let's remind this summary for the ridge regression (and OLS)

regressionMetrics(real = houses_test$Sale_Price,
                  predicted = predict(houses_lm, 
                                      houses_test)
                  )

# thanks to the LASSO method, we managed to improve 
# the quality of forecasts in the test sample!


# we can still try the mix of LASSO and ridge
# - elastic net

parameters_elastic <- expand.grid(alpha = seq(0, 1, 0.2), 
                                  lambda = seq(10, 1e4, 10))

nrow(parameters_elastic)

# we wil compare 6000 different models with CV

set.seed(123456789)

houses_elastic <- train(Sale_Price ~ .,
                        data = houses_train %>% 
                          dplyr::select(all_of(houses_variables_all)),
                        method = "glmnet", 
                        tuneGrid = parameters_elastic,
                        trControl = ctrl_cv5)
houses_elastic

# looks like here the LASSO regression (alpha = 1)
# gives the best result

plot(houses_elastic)

# Mixing percentage = alpha
# Regularization parameter = lambda

regressionMetrics(real = houses_test$Sale_Price,
                  predicted = predict(houses_elastic, 
                                      houses_test)
                  )

# WARNING! in an analogous manner, one can apply regularization
# to the logit or multinomial logit model


# lets see what happens if p >>> n

set.seed(987654321)

x <- data.frame(matrix(rnorm(20000), nrow = 100))

y <- rnorm(100) + 0.3 * x[, 124] - 0.7 * x[,23]

data_xy <- cbind(y, x)

dim(data_xy)

# lets define a new set of parameters

parameters_elastic2 <- expand.grid(alpha = seq(0, 1, 0.2), 
                                   lambda = seq(0.01, 2, 0.01))

sample_elastic <- train(y ~ ., 
                        data = data_xy,
                        method = "glmnet", 
                        tuneGrid = parameters_elastic2,
                        trControl = ctrl_cv5)

sample_elastic

plot(sample_elastic)

predict(sample_elastic$finalModel, # stored model
        s = sample_elastic$bestTune$lambda, # lambda
        type = "coefficients")

# we obtain reasonable results


# lets try the same with a linear regression

sample_lm <- lm(y ~ ., data = data_xy)

summary(sample_lm)

# model cannot be estimated