# lets load needed packages

library(dplyr)
library(readr)
library(caret)

install.packages("AER")
library(AER)

# install.packages("lmtest")
library(lmtest) # lrtest(), waldtest()

# install.packages("nnet") # multinom()
library(nnet)

#--------------------------------------------------------------------
# binary logistic regression

# we will now deal with the model classifying whether
# a client of some telecomunications company churns
# (resigns)

# The raw data contains 7043 rows (customers) and 
# 21 columns (features).

# The “Churn” column is our target.

# variables in the dataset:
# customerID - Customer ID
# gender - Whether the customer is a male or a female
# SeniorCitizen - Whether the customer is a senior citizen or not (1, 0)
# Partner - Whether the customer has a partner or not (Yes, No)
# Dependents - Whether the customer has dependents or not (Yes, No)
# tenure - Number of months the customer has stayed with the company
# PhoneService - Whether the customer has a phone service or not (Yes, No)
# MultipleLines - Whether the customer has multiple lines or not (Yes, No, No phone service)
# InternetService - Customer’s internet service provider (DSL, Fiber optic, No)
# OnlineSecurity - Whether the customer has online security or not (Yes, No, No internet service)
# OnlineBackup - Whether the customer has online backup or not (Yes, No, No internet service)
# DeviceProtection - Whether the customer has device protection or not (Yes, No, No internet service)
# TechSupport - Whether the customer has tech support or not (Yes, No, No internet service)
# StreamingTV - Whether the customer has streaming TV or not (Yes, No, No internet service)
# StreamingMovies - Whether the customer has streaming movies or not (Yes, No, No internet service)
# Contract - The contract term of the customer (Month-to-month, One year, Two year)
# PaperlessBilling - Whether the customer has paperless billing or not (Yes, No)
# PaymentMethod - The customer’s payment method (Electronic check, Mailed check,
#                 Bank transfer (automatic), Credit card (automatic))
# MonthlyCharges - The amount charged to the customer monthly
# TotalCharges - The total amount charged to the customer
# Churn - Whether the customer churned or not (Yes or No)

churn <- read_csv("data/churn.csv")

glimpse(churn)

# Lets convert all character variables to factors
# (apart from the 1st column - customerID)

# create a vector of names

churn_categorical_vars <- 
  sapply(churn[, -1], is.character) %>% 
  which() %>% 
  names()

churn_categorical_vars

# and apply a conversion in a loop

for (variable in churn_categorical_vars) {
  churn[[variable]] <- as.factor(churn[[variable]])
}

glimpse(churn)

# SeniorCitizen is also categorical

churn$SeniorCitizen <- factor(churn$SeniorCitizen,
                              levels = c(0, 1),
                              labels = c("No", "Yes"))

glimpse(churn)


# Lets divide the data into training and testing sample

set.seed(987654321)

churn_which_training <- createDataPartition(churn$Churn,
                                            p = 0.7, 
                                            list = FALSE) 

churn_train <- churn[c(churn_which_training),]
churn_test <- churn[-c(churn_which_training),]

save(list = c("churn_train",
              "churn_test"),
     file = "data/churn_train_test.RData")



# Lets check if there are any missing values
# in the training sample

any(is.na(churn_train))

colSums(is.na(churn_train)) %>% 
  sort()

# 8 missings in `TotalCharges` - lets replace 
# them with the median of this column
# in the training sample

median_TotalCharges <- median(churn_train$TotalCharges, 
                              na.rm = TRUE)

churn_train$TotalCharges[is.na(churn_train$TotalCharges)] <- median_TotalCharges
   

# CAUTION!!!!!
# EXACTLY the same value of a median based on 
# a training sample should be used to fill
# missings in the TEST sample

churn_test$TotalCharges[is.na(churn_test$TotalCharges)] <- median_TotalCharges

# The only correct way of transforming data is to define
# transformation based on the distribution in the training
# sample. 

# Otherwise we would in fact use the test data 
# to build the model, which is not correct.


# logistic regression can be estimated using the function
# glm() (generalized linear models)
# we must provide information about:
# - the distribution of the dependent variable - binomial
#   means a distribution with two values
# - a link function (logit)

# WARNING! the glm() function also AUTOMATICALLY recodes
# categorical variables into dummies, assuming by default
# the FIRST level of the variable as the reference

# WARNING2!
# to obtain decoding of ordinal variables into dummies with 
# a reference level we need to change the appropriate 
# system option contrasts on "contr.treatment"

# for more information, see:
# http://faculty.nps.edu/sebuttre/home/R/contrasts.html

options(contrasts = c("contr.treatment",  # for non-ordinal factors
                      "contr.treatment")) # for ordinal factors

# lets try just two variables at the beginning:
# - one numeric and one categorical

churn_logit1 <- glm(Churn ~ tenure + Contract,
                       # here we define type of the model
                       family =  binomial(link = "logit"),
                       data = churn_train)

summary(churn_logit1)

# results are shown in a similar way as in case of lm():
# - we have model coefficients, standard errors and significance 
#   tests (instead of t statistics there are z statistics,
#   and their p-values)

# there is no joint significance test 
# (remember F test from lm() results)


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

# missing values at some coefficients mean that 
# they were automatically excluded from the model
# as these variables are colinear (contain the same
# information as other variables); here the level
# "No internet service" repeats in many categorical
# predictors and after recoding these variables
# into dummies we have this information repeated
# several times


# joint test of significance can be performed 
# by using the function lrtest() - Likelihood ratio test
# from the lmtest package

lrtest(churn_logit2)

# we reject the null hypothesis
# that the model is NOT jointly significant

# lets calculate fitted values and compare them with real values

churn_logit2_fitted <- predict(churn_logit2)

# lets check predicted values

head(churn_logit2_fitted)

# by default predictions from logit
# refer to log-odds: ln(p/(1-p))

# one can calculate predicted probabilities 
# of "success" (here the second level of the dependent 
# variable) by adding type="response"

churn_logit2_fitted <- predict(churn_logit2,
                               type = "response")

head(churn_logit2_fitted)

# these can be now converted into the levels of
# the dependent variable with the default
# cut-off probability of 0.5

table(churn_train$Churn,
      ifelse(churn_logit2_fitted > 0.5, # condition
             "Yes", # what returned if condition TRUE
             "No")) # what returned if condition FALSE


# lets also apply the probit model in similar two variants

churn_probit1 <- glm(Churn ~ tenure + Contract,
                     # here we define type of the model
                     # NOW "probit" is used
                     family =  binomial(link = "probit"),
                     data = churn_train)

summary(churn_probit1)

# summary of results looks similarly

churn_probit2 <- glm(Churn ~ ., # simplified formula:
                     # use all other variables (apart from Churn)
                     # as predictors
                     # NOW "probit" is used
                     family =  binomial(link = "probit"),
                     data = churn_train %>% 
                       # we exclude customerID
                       dplyr::select(-customerID))

summary(churn_probit2)


# joint significance can also be tested 

lrtest(churn_probit2)

# we reject the null hypothesis
# that the model is NOT jointly significant

# lets calculate fitted values and compare them with real values

churn_probit2_fitted <- predict(churn_probit2,
                                type = "response")

# and compare them with the predictions from logit2

plot(churn_logit2_fitted,
     churn_probit2_fitted)


cor(churn_logit2_fitted,
    churn_probit2_fitted)

# compare the predictions converted into "Yes" and "No"
# with the cut-off level of 0.5

table(ifelse(churn_logit2_fitted > 0.5,
             "Yes", 
             "No"),
      ifelse(churn_probit2_fitted > 0.5,
             "Yes",
             "No"))

# only 26 observations classified differently

#--------------------------------------------------------------------
# multinomial logistic regression

# The dataset "cards" contains data on customers who
# 2 years earlier received a credit card from a certain bank
# (e.g. as part of cross-selling, in installment purchases),
# but did not necessarily use it later. 

# The bank opened a card account to each client. 
# Now the bank wants to assess the status of this account.
# The status of the card might have 3 different levels:
# - "O" (like "open") means that the credit card is actively used
#   and loans are paid. Bank makes a profit and it is worth keeping 
#   such an account. 
# - "A" (like "attrition" means that the credit card was not used 
#   at all or was used sporadically. Mainatning such an account
#   generates costs for the bank and it should be closed. 
# - "C" (like "charge-off") means that the card is used by the customer
#   but the loans are not repaid or are significantly delayed,
#   such an account should be closed and transfered to the
#   debt collection department.
                                                             
# The dataset includes 7100 observations and 5 variables:
# - status - card account status explained above
# - credit_limit - average credit limit on other customer cards
# - n_contracts - the total number of contracts in the customer's 
#   credit history (credit cards, cash, mortgage, car loans, etc.)
# - gender - gender of the client
# - utilization - utilization of other customer credit cards (in %)


# lets import the data from file "credit_cards.csv".

cards <- read_csv("credit_cards.csv")

glimpse(cards)

# Convert qualitative data into factors

cards$status <- as.factor(cards$status)
cards$gender <- as.factor(cards$gender)

# divide the data into training and test sample (in proportion70%/30%)

set.seed(987654321)

cards_which_train <- createDataPartition(cards$status,
                                         p = 0.7, 
                                         list = FALSE) 

cards_train <- cards[c(cards_which_train),]
cards_test <- cards[-c(cards_which_train),]


# save these datasets for future applications

save(list = c("cards_train",
              "cards_test"),
     file = "data/cards_train_test.RData")

# Lets estimate a multinomial logistic regression model
# using the training data (status is a dependent variable.

cards_mlogit1 <- multinom(status ~ credit_limit + n_contracts +
                            utilization + gender, 
                          data = cards_train)

# It is an iterative proces - by default 100 iterations are done
# If it is not enough for the estimation to converge, one
# can use an additional option 'maxit = number_of_iterations'

cards_mlogit1 <- multinom(status ~ credit_limit + n_contracts +
                            utilization + gender, 
                          data = cards_train,
                          # lets increase a maximal
                          # number of iterations
                          maxit = 1000)

# lets calculate fitted values (for the training sample)
predict(cards_mlogit1)

cards_mlogit1_fitted <- predict(cards_mlogit1) 

# by default predicted values of 
# the dependent variable are stored

table(cards_mlogit1_fitted)

# one can also generate predicted probabilities 
# for each level with type = "probs"

predict(cards_mlogit1,
        type = "probs") %>% 
  head()

# by default it is assumed that the level with 
# the highest probability is predicted
# (e.g. "O" for the first six observations)

# Lets compare real and fitted values

table(cards_mlogit1_fitted, # fitted in rows
      cards_train$status) # real in columns

# looks like the model tends to predict
# the status "O" more often than it appears
# in the data - see the numbers in the last row


#--------------------------------------------------------------------
# Exercises 4b. - cont'd


# Exercises 4b.2
# Wine Quality Data Set: "data/wines.csv"

# estimate the multinomial logistic regression model with 
# the quality evaluation as the dependent variable,
# treating the explained variable as qualitative.

# we import the data

wines <- read_csv("wines.csv")

glimpse(wines)
table(wines$quality)

# lets combine levels 3 and 4
wines$quality[wines$quality == 3] <- 4

# and levels 8 and 9
wines$quality[wines$quality == 9] <- 8

# and convert this variable into a factor
wines$quality <- as.factor(wines$quality)

# type is also a categorical variable
wines$type <- as.factor(wines$type)

# Your tasks:
# - divide the data randomly into training and test sample (70/30),
#   and save both datasets to the file "data/wines_train_test_qualit.RData
# - estimate multinomial logistic regression model that 
#   can be used to explain the quality of the wine.
# - calculate fitted values and compare them with real values


set.seed(987654321)

wines_which_train <- createDataPartition(wines$quality,
                                         p = 0.7, 
                                         list = FALSE) 

wines_train <- wines[c(wines_which_train),]
wines_test <- wines[-c(wines_which_train),]


# save these datasets for future applications

save(list = c("wines_train",
              "wines_test"),
     file = "wines_train_test.RData")

# Lets estimate a multinomial logistic regression model
# using the training data (status is a dependent variable.

wines_mlogit1 <- multinom(quality ~ fixed_acidity + volatile_acidity +
                            citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide
                          + density + pH + sulphates + alcohol + type, 
                          data = wines_train)

# It is an iterative proces - by default 100 iterations are done
# If it is not enough for the estimation to converge, one
# can use an additional option 'maxit = number_of_iterations'

wines_mlogit1 <- multinom(quality ~ fixed_acidity + volatile_acidity +
                            citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide
                          + density + pH + sulphates + alcohol + type, 
                          data = wines_train,
                          # lets increase a maximal
                          # number of iterations
                          maxit = 1000)

# lets calculate fitted values (for the training sample)
predict(wines_mlogit1)

wines_mlogit1_fitted <- predict(wines_mlogit1) 


# by default predicted values of 
# the dependent variable are stored

table(wines_mlogit1_fitted)

# one can also generate predicted probabilities 
# for each level with type = "probs"

predict(wines_mlogit1,
        type = "probs") %>% 
  head()

# by default it is assumed that the level with 
# the highest probability is predicted
# (e.g. "O" for the first six observations)

# Lets compare real and fitted values

table(wines_mlogit1_fitted, # fitted in rows
      wines_train$quality) # real in columns

# looks like the model tends to predict
# the status "O" more often than it appears
# in the data - see the numbers in the last row