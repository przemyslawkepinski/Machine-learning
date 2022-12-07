# lets load needed packages

library(dplyr)
library(readr)

install.packages("olsrr")
library(olsrr)

library(ggplot2)

# see:
# https://cran.r-project.org/web/packages/olsrr/vignettes/variable_selection.html


# lets load the data about houses
# divided into train and test samples
# last week

load("data/houses_train_test.RData")

# The description of variables (types and definition)
# can be found for example here (names might be slightly different):
# https://rdrr.io/cran/subgroup.discovery/man/ames.html

# The detailed description of the variable levels
# can be found here 
# http://jse.amstat.org/v19n3/decock/DataDocumentation.txt
# and in the text file "DataDocumentation.txt" in lab materials


# lets also load the lists of all variables
# and selected variables (see last week's materials)

load("data/houses_variables_lists.RData")

# lets remind the list of selected variables

houses_variables_selected


#--------------------------------------------------------------------
# linear regression

# to estimate the linear model in R we will use the function lm()

# in the simplest syntax it requires the formula 
# and the data set on which the model is estimated:
# lm(formula, data)

# the model formula is given in the pattern:
# dependent_variable ~ independent_variables
# eg. Y ~ X for simple regression with one explanatory
# variable (X)

# if we want to include more explanatory variables, 
# they should be separated with plus "+" in the model formula,
# e.g. Y ~ X1 + X2 + X3

# lets see an example - we will try to explain 
# the price of a house

# at first, the only explanatory variable will be Gr_Liv_Area

# let's save the modeling result as an object

houses_lm1 <- lm(Sale_Price ~ Gr_Liv_Area, # formula
                 data = houses_train) # training data

# lets check the summary of results

summary(houses_lm1)

# in the `coefficients` part we can see estimates
# of individual model parameters (only two here)
# their standard errors, t statistics and their p-values

# it can be seen that Gr_Liv_Area DOES significantly affect 
# the price of a house (p-value < 2e-16)

# on the right hand side of p-value of statistically significant 
# parameters one can see asterisk(s) or dot indicating
# significance according to the legend shown below the table 
# with coefficients:
# Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 '' 1

# for p-value between 0 and 0.001, there will be three stars (***),
# for p-value between 0.001 and 0.01, there will be two asterisks (**),
# etc.

# At the bottom of the results we also have the displayed value:
# - Residual standard error, which has no special interpretation,
#   but the standard errors of the estimators depend on it
# - R2: Multiple R-squared: 0.5556
# - adjusted R2: Adjusted R-squared: 0.5554
#   which will be described on the next labs
# - F statistic testing the joint significance of the model and 
#   its p-value: F-statistic: 2420,  p-value: < 2.2e-16


# lets estimate the model with more explanatory variables (quantitative)

houses_lm2 <- lm(Sale_Price ~ Gr_Liv_Area + Garage_Area + Full_Bath +
                   Year_Built + TotRms_AbvGrd, # formula
                data = houses_train) # training data

summary(houses_lm2)

# using more variables we explain 74% of variability (R2)!
# ALL used variables are statistically significant

# lets add some qualitative explanatory variables to the model

# WARNING! 
# if qualitative variables are stored in a data.frame as factors
# they will be encoded into respective dummy variables AUTOMATICALLY

# WARNING2!
# to obtain decoding of ordinal variables into dummies with 
# a reference level we need to change the appropriate 
# system option contrasts on "contr.treatment"

# The default values are "contr.treatment" for variables
# nominal and "contr.poly" for ordinal.

# for more information, see:
# http://faculty.nps.edu/sebuttre/home/R/contrasts.html

options(contrasts = c("contr.treatment",  # for non-ordinal factors
                      "contr.treatment")) # for ordinal factors

# however, it does not have any impact on model fit
# and predictions, just for interpretation of parameters

houses_lm3 <- lm(Sale_Price ~ Gr_Liv_Area + Garage_Area + Full_Bath +
                   Year_Built + TotRms_AbvGrd +
                   # qualitative predictors are added
                   # in the same way 
                   Exter_Qual + Kitchen_Qual + Overall_Qual +
                   Central_Air + Foundation + Heating_QC, # formula
                 data = houses_train) # training data

summary(houses_lm3)

# lets calculate predicted values based on the last model

houses_predicted <- predict(houses_lm3)

# and chek the distribution of errors
# (differences between real and predicted values)

ggplot(data.frame(error = houses_train$Sale_Price - houses_predicted),
       aes(x = error)) +
  geom_histogram(fill = "blue",
                 bins = 100) +
  theme_bw()

# and plot real values against the predicted

ggplot(data.frame(real = houses_train$Sale_Price,
                  predicted = houses_predicted),
       aes(x = predicted, 
           y = real)) +
  geom_point(col = "blue") +
  theme_bw()


# check their correlation

cor(houses_train$Sale_Price,
    houses_predicted)


## Automated variable selection methods
# Model subset

# lets save the results for the model 
# including all potential predictors

load("data/houses_variables_lists.RData")

houses_variables_all

# Lets exclude Order and PID

houses_variables_all <- houses_variables_all[-1:-2]

houses_lm4 <- lm(Sale_Price ~ ., # simplified formula
                 # ALL variables from the dataset
                 # apart from Sale_Price are used
                 # as predictors
                 data = houses_train %>% 
                   dplyr::select(all_of(houses_variables_all))) # training data


summary(houses_lm4)

# some variables (after recoding into dummies)
# are multicollinear - have missing coefficients (NA)

coef(houses_lm4)[is.na(coef(houses_lm4))]

# Automated variable selection functions
# do not allow for that
# we can either merge these levels with
# some others in a sensible way
# or exclude these variables from the model

# lets apply the second solution

houses_variables_all2 <-
  houses_variables_all[-which(houses_variables_all %in% 
                         c("Bldg_Type", "Exterior_2nd",
                           "Exter_Cond", "Bsmt_Cond",
                           "BsmtFin_Type_1", "BsmtFin_SF_1",
                           "Garage_Qual", "Garage_Cond"))]

# reestimate the last model once again

houses_lm4a <- lm(Sale_Price ~ ., # simplified formula
                 # ALL variables from the dataset
                 # apart from Sale_Price are used
                 # as predictors
                 data = houses_train %>% 
                   dplyr::select(all_of(houses_variables_all2))) # training data

summary(houses_lm4a)

# no more missing coefficients

# lets apply the backward elimination 
# based on p-values - functions from 
# olsrr package

ols_step_backward_p(houses_lm4a,
                    # p-value for removing
                    # (default p = 0.3)
                    prem = 0.05,
                    # show progress
                    progress = TRUE) -> houses_lm4_backward_p

# lets check the final model details

summary(houses_lm4_backward_p$model)

# a list of removed terms is available here

houses_lm4_backward_p$removed


# the selection can also be based 
# on the values of AIC

ols_step_backward_aic(houses_lm4a, 
                      progress =  TRUE) -> houses_lm4_backward_AIC


# which takes a bit longer...

# lets check the final model details

summary(houses_lm4_backward_AIC$model)

# a list of removed terms is available here

houses_lm4_backward_AIC$predictors


# Do both approaches give the same result?

coef(houses_lm4_backward_AIC$model) %>% 
  names() %>% 
  sort() -> coef_list_AIC

coef(houses_lm4_backward_p$model) %>% 
  names() %>% 
  sort() -> coef_list_p


identical(coef_list_AIC, 
          coef_list_p)

# NO!!!

coef_list_AIC[which(! coef_list_AIC %in% coef_list_p)]

coef_list_AIC[which(! coef_list_p %in% coef_list_AIC)]

# selection based on AIC keeps 14 more variables 


# forward selection can be applied with the use of:
# ols_step_forward_p(model, penter = 0.3)
# ols_step_forward_aic(model)

# stepwise selection can be applied with the use of:
# ols_step_both_p(model, pent = 0.1, prem = 0.3)
# ols_step_both_aic(model)

# in the results of each the element called
# $predictors	includes variables added/retained
# in the model
# and the element called $model - the final model