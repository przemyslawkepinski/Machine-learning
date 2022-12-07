# lets load needed packages

install.packages("bestNormalize") # for boxcox(), yeojohnson()
library(bestNormalize)

install.packages("Information")
library(Information)

install.packages("smbinning")
library(smbinning)

library(caret)

library(dplyr)

library(lattice)
library(ggplot2)

# lets load houses data and variables lists

load("houses_train_test.RData")

load("houses_variables_lists.RData")

# lets remove Order, PID, Longitude and Latitude
# from the list of all variables

houses_variables_all <- 
  houses_variables_all[-which(houses_variables_all %in%
                                c("Order", "PID", 
                                  "Longitude", "Latitude"))]

# we need to define one-hot encoding 
# for ordinal factors

options(contrasts = c("contr.treatment",  # for non-ordinal factors
                      "contr.treatment")) # for ordinal factors


#---------------------------------------------------
# Feature generation in practice

# 0. No binning for reference (benchmark)

# Lets remind the prediction accuracy
# of the simple linear regression model 

# linear model on all variables

houses_lm0_all <- lm(Sale_Price ~ .,
                     data = houses_train %>% 
                       dplyr::select(all_of(houses_variables_all)))

# linear model using only selected variables
# - 10 most correlated quantitative and
# 10 most related qualitative

houses_lm1_selected <- lm(Sale_Price ~ .,
                          data = houses_train %>% 
                            dplyr::select(all_of(houses_variables_selected)))

# lets remind the distribution of the outcome
# variable (we check JUST THE TRAINING DATA)

ggplot(houses_train,
       aes(x = Sale_Price)) +
  geom_histogram(fill = "blue",
                 bins = 100) +
  theme_bw()

# clearly right-skewed distribution - let's see
# how it looks after log transformation
# we use log(x + 1) in case of zeros in x

ggplot(houses_train,
       aes(x = log(Sale_Price + 1))) +
  geom_histogram(fill="blue",
                 bins = 100) +
  theme_bw()

# much more "Normal-like" - might be
# easier to model


# lets apply the log-linear model
# on all variables

houses_lm2_all_log <- lm(log(Sale_Price + 1) ~ .,
                         data = houses_train %>% 
                           dplyr::select(all_of(houses_variables_all)))

# log-linear model on selected variables

houses_lm3_selected_log <- lm(log(Sale_Price + 1) ~ .,
                              data = houses_train %>% 
                                dplyr::select(all_of(houses_variables_selected)))


# lets generate predicted values 
# on the training sample from all models

# we can put them in a list

houses_models_list <- list(houses_lm0_all = houses_lm0_all,
                           houses_lm1_selected = houses_lm1_selected,
                           houses_lm2_all_log = houses_lm2_all_log,
                           houses_lm3_selected_log = houses_lm3_selected_log)

# and use vectorized approach to 
# make predictions easier
# (here: fitted values - in the training sample)

houses_models_list %>% 
  sapply(function(x) predict(x, newdata = houses_train)) %>% 
  # function sapply() returns a matrix here,
  # we immediately convert it to a data frame
  # and save as a new object
  data.frame() -> houses_fitted

head(houses_fitted)

# remember that models lm2 and lm3 operated 
# on transformed prices.
# To calculate the real fitted value
# of these models one should reverse 
# the transformation of the target variable
# if y2 = ln(y + 1)  then  y = exp (y2) - 1

houses_fitted$houses_lm2_all_log <- 
  exp(houses_fitted$houses_lm2_all_log) - 1

houses_fitted$houses_lm3_selected_log <-
  exp(houses_fitted$houses_lm3_selected_log) - 1

head(houses_fitted)

# in the same way we apply the models
# to generate predictions on the test sample

houses_models_list %>% 
  sapply(function(x) predict(x, newdata = houses_test)) %>% 
  data.frame() -> houses_forecasts

# reverse the transformation of the target
# y2 = ln(y + 1) , y = exp(y2) - 1 

houses_forecasts$houses_lm2_all_log <- 
  exp(houses_forecasts$houses_lm2_all_log) - 1

houses_forecasts$houses_lm3_selected_log <-
  exp(houses_forecasts$houses_lm3_selected_log) - 1

head(houses_forecasts)

# lets compare the quality of these forecasts
# for the training sample
# using the function we created 
# on one of the previous labs

source("F_regression_metrics.R")

sapply(houses_fitted,
       function(x) regressionMetrics(x,
                                     houses_train$Sale_Price)) %>% 
  t()

# models on transformed outcome variable
# seem to work better in the training sample!


# lets compare the test data

sapply(houses_forecasts,
       function(x) regressionMetrics(x,
                                     houses_test$Sale_Price)) %>% 
  t()

# which is not that clear on the test sample


#----------------------------------------------------
# power transformations

# lets try to apply boxcox transformation for Sale_Price

Sale_Price_boxcox <- boxcox(houses_train$Sale_Price)

# lets check the structure of results
str(Sale_Price_boxcox)

# $lambda is the optimal lambda

Sale_Price_boxcox$lambda

# But this transformation is very hard to interpret.
# Optimal lambda seems to be very close to 0,
# so log transformation of the Sale_Price
# should be good enough

# But if one wanted to use transformed data
# from boxcox() a new column can be added
# to the dataset based on the $x.t element
# that includes transformed data

houses_train$Sale_Price_boxcox <- 
  Sale_Price_boxcox$x.t

# in case of the need to apply the same
# transformation on the test data 
# one has to use a predict() method

predict(Sale_Price_boxcox, 
        newdata = houses_test$Sale_Price) %>%
  head()



# we can also check the Yeo-Johnson's transformation

Sale_Price_yeojohnson <- yeojohnson(houses_train$Sale_Price)

str(Sale_Price_yeojohnson)

Sale_Price_yeojohnson$lambda

# in this case it is exactly the same transformation
# as in Box-Cox, so lets skip it


# we can compare the histogram of values 
# before and after transformations

# lets divide the graphical window
# in three panels one below the other

layout(matrix(1:3, ncol = 1))

hist(houses_train$Sale_Price, 
     main = "Histogram of Sale_Price",
     breaks = 100)
hist(log(houses_train$Sale_Price + 1), 
     main = "Histogram of log(Sale_Price+1)",
     breaks = 100)
hist(houses_train$Sale_Price_boxcox, 
     main = "Histogram of Sale_Price_boxcox",
     breaks = 100)
# restore the original window w/o panels
layout(matrix(1))

# here results of the Box-Cox and 
# log-transformation are very similar
# so we prefer to use the simpler one


# CAUTION!
# Generally the Box-Cox and Yeo-Johnson 
# transformations were designed to be used
# for outcome variables, but one can try
# to apply these transformations also 
# to PREDICTOR VARIABLES


# Lets check the distribution of Lot_Area

ggplot(houses_train,
       aes(x = Lot_Area)) +
  geom_histogram(fill = "blue",
                 bins = 100) +
  theme_bw()

# one can consider log or Box-Cox transformation

ggplot(houses_train,
       aes(x = log(Lot_Area + 1))) +
  geom_histogram(fill = "blue",
                 bins = 100) +
  theme_bw()




#########################################################################
# binning

# alternatively we can consider binning, 
# i.e. combining continuous values
# into intervals.

# We will show this on the example
# of churn data, as this is more
# common for classification tasks
# and includes so called optimal binning


# lets load the churn data

load("churn_train_test.RData")

# and apply the simple logistic regression
# using all non-transformed variables
# (apart from CustomerID)

# This time we will compare models based
# on the validation sample (with the use
# of 5-fold cross-validation)

# lets define a summary function to evaluate
# models with all 5 measured

fiveStats <- function(...) c(twoClassSummary(...), 
                             defaultSummary(...))

ctrl_cv5 <- trainControl(method = "cv",
                         number = 5,
                         classProbs = TRUE,
                         # and use it in trControl
                         summaryFunction = fiveStats)

set.seed(987654321)

churn_logit0 <- 
  train(Churn ~ ., 
        data = churn_train %>% 
          # we exclude customerID
          dplyr::select(-customerID),
        method = "glm",
        family = "binomial",
        trControl = ctrl_cv5)

churn_logit0

#  ROC        Sens       Spec       Accuracy   Kappa    
#  0.8387552  0.8926043  0.5462022  0.8006465  0.4618858


# Lets check if/how the distribution of MonthlyCharges 
# differs for churning and non-churning clients

boxplot(MonthlyCharges ~ Churn,  # formulae
        data = churn_train) # shape

# looks like people with higher charges
# churn more often

# lets check some transformations of MonthlyCharges

# 1. "Full binning" - every value as a different level. 
# This is most certainly too much and does not make sense here
# as there are many different values of MonthlyCharges
# and they do not repeat for many customers.

# This could be applied for quantitative variables that take
# a limited number of values that repeat quite often - e.g. age


# 2. division into quantile groups

# one can  divide a quantitative variable into quantile
# groups (equal sized) - lets show this on the example
# of quintile groups (5)

# quantiles can be calculated with a function quantile()

quantile(churn_train$MonthlyCharges)

# by default it shows quartiles + min + max

# to obtain other quantiles one has to use

quantile(churn_train$MonthlyCharges,
         probs = seq(0, 1, 0.2))

# and use the result of quantile() as breaks 
# in the function cut()

churn_train$MonthlyCharges_quintiles <- 
  cut(churn_train$MonthlyCharges, 
      breaks = quantile(churn_train$MonthlyCharges, 
                        probs = seq(0, 1, 0.2)),
      include.lowest = TRUE)

# in this case groups will not always have the same size
# (all repeating values will always be in the same group)

table(churn_train$MonthlyCharges_quintiles)

# lets check how the model changes

set.seed(987654321)

churn_logit1_quintiles <- 
  train(Churn ~ ., 
        data = churn_train %>% 
          # we exclude customerID
          # AND original MonthlyCharges
          dplyr::select(-customerID, -MonthlyCharges),
        method = "glm",
        family = "binomial",
        trControl = ctrl_cv5)

churn_logit1_quintiles

# ROC        Sens       Spec       Accuracy   Kappa    
# 0.8388552  0.8931564  0.5461993  0.8010515  0.4626624

# ROC, Sens and Spec improved as compared to the benchmark


#-------------------------------------
# optimal binning - supervised binning

# Optionally, we can categorize quantitative variables 
# based on Weight of Evidence (WOE) and Information Value (IV)

# The WOE tells the predictive power of an independent 
# variable in relation to the dependent variable.
# generally:
# WOE_i = ln(% of non-events / % events) in group_i - 
#          ln(% of non-events / % events) in whole sample

# WOE helps to transform a continuous independent variable
# into a set of groups or bins based on similarity of dependent
# variable distribution

# For continuous independent variables:
# 1. start with bins (eg. 10-20 equal sized) for
#    a continuous independent variable 
# 2. combine categories with similar WOE values 
# 3. replace variable levels with the values of WOE
# 3a. Use WOE values rather than input values in your model.

# For categorical independent variables:
# 1. Combine categories with similar WOE 
# 2. create new categories of an independent variable 
#    with continuous WOE values.

# Information value

# Information value is one of the most useful technique 
# to select important variables in a predictive model. 
# It helps to rank variables on the basis of their importance. 
# The IV is calculated using the following formula:

# IV = Sum (% of non-events - % of events) * WOE

# Information Value 	Variable Predictiveness
#   Less than 0.02 	   Not useful for prediction
#   0.02 to 0.1 	     Weak predictive Power
#   0.1 to 0.3 	       Medium predictive Power
#   0.3 to 0.5 	       Strong predictive Power
#   > 0.5 	           Suspicious Predictive Power

# more details:
# http://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html

# To compute and print/plot information values
# one can use Information package

# Compute Information Value and WOE

# The function create_infotables() takes all the variables
# except a dependent variable as predictors from a dataset 
# and runs IV on them (so the dataset should be cleared from 
# any additional variables)

# In the first parameter, you need to define a data frame
# followed by your target variable y (numeric!!!). 
# In the bins= parameter, you need to specify the number of groups
# you want to create for WOE and IV.

# lets create a numeric version of Churn

churn_train <- 
  churn_train %>% 
  mutate(Churn_01 = ifelse(Churn == "Yes", 1, 0)) 

# and calculate their Information Values

churn_train_IV <-
  create_infotables(churn_train %>%  # input data
                      # we exclude customerID, Churn
                      # AND MonthlyCharges_quintiles
                      dplyr::select(-customerID, 
                                    -MonthlyCharges_quintiles,
                                    -Churn),
                    # dependent variable
                    y = "Churn_01", 
                    # number of bins created 
                    # for quantitative variables
                    bins = 20) 

# The parameter bins= is not used for factor variables

# result is a list - the element called Summary contains all IVs

churn_train_IV_values <- data.frame(churn_train_IV$Summary)

print(churn_train_IV_values)

# variables are sorted in the decreasing order of IV
# first 5-8 variables have huge predictive power (IV>=0.5),
# and the next 4-6 moderate (IV>0.1) - MonthlyCharges is here

# To get WOE table for variable MonthlyCharges, 
# one needs to call $Tables element from IV object.

print(churn_train_IV$Tables$MonthlyCharges)

# Plot WOE Scores

# To see trend of WOE values, you can plot them 
# by using plot_infotables() function.

plot_infotables(churn_train_IV, "MonthlyCharges")

# for a factor variable, the number of bins
# will be equal to the number of levels

plot_infotables(churn_train_IV, "Contract")

# based on the results one can decide 
# which values should be binned
# and do it manually


#-------------------------------------------
# another useful package which applies the optimal
# binning automatically (based on WOEs)

# Lets run the automated binning process for a variable 
# MonthlyCharges (here we do not specify the number of bins
# - instead the binning is applied automatically by recursive 
# partitioning of the quantitative variable into intervals)

# lets use the function 
# smbinning(df, y, x, p)
# on the full training dataset

# Caution!!! 
# df = pure data frame required as input
# y = - dependent variable -  should be binary and integer
#       its name cannot include a dot
# x = continuous variable (with at least 5 different values)

MonthlyCharges_binned <-
  smbinning(df = data.frame(churn_train), # data
            y = "Churn_01", # dependent - NUMERIC !!!!
            x = "MonthlyCharges", # continuous variable to be binned
            p = 0.05) # percentage of obs per bin (between 0 and 0.5)

# lets see the results

MonthlyCharges_binned$ivtable

# - Odds_i = GoodRate_i/BadRate_i
# - LnOdds_i = log(Odds_i)
# - WoE_i = LnOdds_i - LnOdds_total
# - IV_i = WoE_i * (CntGood_i/CntGood_total - CntBad_i/CntBad_total)
# - IV_total = sum(IV_i)

# MonthlyCharges was categorized into 6 intervals

# lets see some useful plots

# lets divide graphical window in 4 parts (2 rows and 2 columns)
par(mfrow = c(2, 2))
# boxplot of UAGE by UCURNINS
boxplot(MonthlyCharges ~ Churn_01, 
        data = churn_train,
        horizontal = TRUE, 
        frame = FALSE, 
        col = "lightgray",
        main = "Distribution of MonthlyCharges by Churn") 
# relative frequencies of UAGE bins
smbinning.plot(MonthlyCharges_binned, 
               option = "dist") 
# bad rates in UAGE bins
smbinning.plot(MonthlyCharges_binned, 
               option = "badrate") 
# WOE in UAGE bins
smbinning.plot(MonthlyCharges_binned, 
               option = "WoE")
# restore the original window  
par(mfrow = c(1, 1)) 

# how to create a new variable based on above results?

# we can easily extract interval borders

MonthlyCharges_binned$cuts # intervals are 

# in case of WoE one has to extract data 
# from the column WoE printed in results

MonthlyCharges_binned$ivtable

MonthlyCharges_binned$ivtable$WoE

# the first 6 values refer to WoE for subsequent bins

# to create a new variable we can manually use cut() function, 
# but it is easier to be done automatically

churn_train <- 
  smbinning.gen(churn_train, # original data
                MonthlyCharges_binned, # results of binning
                "MonthlyCharges_BINNED") # name of the new variable

# but is stores just the intervals of bins
table(churn_train$MonthlyCharges_BINNED)

# how to replace that with WoE?

# lets create a new variable
churn_train$MonthlyCharges_WoE <- churn_train$MonthlyCharges_BINNED

# check its levels
levels(churn_train$MonthlyCharges_WoE)

# we have to change it to WoE values 
# (!ALWAYS check the order)

levels(churn_train$MonthlyCharges_WoE) <- 
  MonthlyCharges_binned$ivtable$WoE[1:nlevels(churn_train$MonthlyCharges_WoE)]

levels(churn_train$MonthlyCharges_WoE)

# but these are still just labels of levels 

# the last thing we have to do is to convert
# a new variable from factor to numeric (with WoE values)

# a factor has numbers 1, 2,..., nlevels() behind

head(levels(churn_train$MonthlyCharges_WoE))

# if we try to convert a factor to numeric

head(as.numeric(churn_train$MonthlyCharges_WoE))

# we do not get a desired result...

# a small trick is needed here

as.numeric(levels(churn_train$MonthlyCharges_WoE))

churn_train$MonthlyCharges_WoE <-
  as.numeric(levels(churn_train$MonthlyCharges_WoE))[churn_train$MonthlyCharges_WoE] 

table(churn_train$MonthlyCharges_WoE)

# now it is a numeric variable that can be used 
# in a logit model (or any other)


# optimal binning, but w/o WOE (MothlyCharges_BINNED)

# CAUTION!
# if transformed variables added 
# to the dataset one should carefully 
# select predictors for thre model

names(churn_train)

# CAUTION!
# Running below code after binning applied above
# may produce an error due to incorrect default
# specification of parallel processing

# To solve the issue, please FIRST run the code
# stored in the file "99_doSNOW_solution.R"

# Source:
# https://github.com/tobigithub/R-parallel/wiki/R-parallel-Errors
# (at the bottom)

set.seed(987654321)

churn_logit2_binned <- 
  train(Churn ~ ., 
        data = churn_train %>% 
          # we exclude customerID, Churn_01
          # AND original MonthlyCharges,
          # MonthlyCharges_quintiles,
          # MonthlyCharges_WoE
          dplyr::select(-customerID,
                        -MonthlyCharges,
                        -MonthlyCharges_quintiles,
                        -MonthlyCharges_WoE,
                        -Churn_01),
        method = "glm",
        family = "binomial",
        trControl = ctrl_cv5)

churn_logit2_binned

# ROC        Sens      Spec       Accuracy  Kappa    
# 0.8392414  0.893433  0.5439092  0.800646  0.4610919

# ROC and spec best so far, spec slighly lower


# optimal binning, but with WOE in place (UAGE_WoE)

set.seed(987654321)

churn_logit3_WoE <- 
  train(Churn ~ ., 
        data = churn_train %>% 
          # we exclude customerID, Churn_01
          # AND original MonthlyCharges,
          # MonthlyCharges_quintiles,
          # MonthlyCharges_WoE
          dplyr::select(-customerID,
                        -MonthlyCharges,
                        -MonthlyCharges_quintiles,
                        -MonthlyCharges_BINNED,
                        -Churn_01),
        method = "glm",
        family = "binomial",
        trControl = ctrl_cv5)

churn_logit3_WoE

# ROC        Sens       Spec       Accuracy   Kappa    
# 0.8391149  0.8934319  0.5477289  0.8016596  0.4643997

# ROC slightly lower than the highest, but all other measures
# seem to be the best


# lets compare all the transformations again
# on the validation sample

models_list <- ls(pattern = "churn_logit")

sapply(models_list, 
       function(x) get(x)$resample[, 1:5] %>% 
         colMeans()) %>% 
  t()