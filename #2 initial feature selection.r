# lets install and load needed packages

# install.packages("DescTools") # for CramerV()

library(caret)
library(dplyr)
library(tibble)
library(purrr)
library(corrplot)
library(DescTools)

# lets import the data prepared last week

load("houses_prepared.RData")

# Lets divide the data into a learning and testing sample.
# It can be done for example with the use of
# of the createDataPartition() function from the caret package

# breakdown in this case takes into account 
# the distribution of the target variable.

# Because the split is random to be identical for 
# everyone we will use the defined random seed

set.seed(987654321)

houses_which_train <- createDataPartition(houses$Sale_Price, # target variable
                                          # share of the training sample
                                          p = 0.7, 
                                          # should result be a list?
                                          list = FALSE) 

# it is a vector of numbers - indexes 
# of 70% of observations selected 

head(houses_which_train, 10)

# we need to apply this index for data division

houses_train <- houses[houses_which_train,]
houses_test <- houses[-houses_which_train,]

# let's check the distribution of
# the target variable in both samples

summary(houses_train$Sale_Price)
summary(houses_test$Sale_Price)

# minimum and maximum are slightly different,
# but other statistics are similar

# Generally, we will divide the data into two parts
# in this way, but in the case of the "houses" data
# we have time dimension and in this case 
# it will be more natural to use older data as 
# a learning sample, and newer observations as
# a test sample

table(houses$Year_Sold)

# we will separate the houses sold in the last
# two years (2009, 2010) as a test sample 
# and earlier ones as a training set

houses_train <- houses[houses$Year_Sold < 2009,]
houses_test <- houses[houses$Year_Sold >= 2009,]


# WARNING!!!!
# all data analysis and transformations
# taking into account distribution 
# of variables should ONLY be based 
# on the TRAINING sample

#---------------------------------------------
# initial filtering of variables

# we can pre-select variables for the model
# checking the relationship of each of them
# separately with the outcome variable

# then we can, for example, choose k variables
# mostly individually related to the outcome

# lets do it separately for the quantitative
# and qualitative variables

#---------------------------------------------------
# quantitative explanatory variables - correlations

# correlation analysis with a target variable for
# quantitative variables - also allows one for
# identification of potential collinearity
# (strong correlation between predictors)

# lets store the list of names of 
# quantitative (numeric) variables into a vector

houses_numeric_vars <- 
  # check if variable is numeric
  sapply(houses, is.numeric) %>% 
  # select those which are
  which() %>% 
  # and keep just their names
  names()

houses_numeric_vars


#---------------------------------------------------
# mutually correlated (irrelevant) variables

# we can calculate correlations between 
# variables to identify redundant features
# (not always necessary - some methods are robust 
# to the problem of multicollinearity)

houses_correlations <- 
  cor(houses_train[, houses_numeric_vars],
      use = "pairwise.complete.obs")

houses_correlations

# In case of many variables table may be not readable .
# It is worth to look at graphical representation
# of correlation matrix

# there is a useful function corrplot() from
# a package with the same name - it requires 
# a correlation matrix as input data

# by default it shows a graph with colored dots
# - their size and color saturation indicates
# value of correlation

corrplot(houses_correlations)

# alternatively one can use pie plots

corrplot(houses_correlations, 
         method = "pie")

# available values of the "method" argument:
# "circle" (default), "square", "ellipse", 
# "number", "pie", "shade" and "color".


# one can also generate a mixed plot
# - showing correlations in two different
# ways - differently below and above the diagonal.
# To obtain this we should use corrplot.mixed()

corrplot.mixed(houses_correlations,
               upper = "square",
               lower = "number",
               tl.col = "black", # color of labels (variable names)
               tl.pos = "lt")  # position of labels (lt = left and top)


# lets put the variables in descending order
# correlations with dependent variable

houses_numeric_vars_order <- 
  # we take correlations with the Sale_Price
  houses_correlations[,"Sale_Price"] %>% 
  # sort them in the decreasing order
  sort(decreasing = TRUE) %>%
  # end extract just variables' names
  names()

houses_numeric_vars_order


# now we use thie vector to order rows and 
# columns of the correlation matrix on the plot

corrplot.mixed(houses_correlations[houses_numeric_vars_order, 
                                   houses_numeric_vars_order],
               upper = "square",
               lower = "number",
               tl.col = "black", # color of labels (variable names)
               tl.pos = "lt")  # position of labels (lt = left and top)


# lets focus on the first 22 variables
# most correlated with Sale_Price

corrplot.mixed(houses_correlations[houses_numeric_vars_order[1:22], 
                                          houses_numeric_vars_order[1:22]],
               upper = "square",
               lower = "number",
               tl.col = "black", # color of labels (variable names)
               tl.pos = "lt")  # position of labels (lt = left and top)

# strong correlations, although not extremely high
# you can also see the potential problem of collinearity
# (interrelationship between predictors): 
# Garage_Cars and Garage_Area


# let's see the relation of the target variable 
# with the three most strongly correlated variables

#---------------------------
# Gr_Liv_Area

ggplot(houses_train,
       aes(x = Gr_Liv_Area,
           y = Sale_Price)) +
  geom_point(col = "blue") +
  geom_smooth(method = "lm", se = FALSE) +
  theme_bw()

# 5 very large houses (area >4,000),
# including 3 outliers (area >4,500) - not 
# matching the relationship (low priced)
# - maybe exclude these observations from the sample

( houses_which_outliers <- which(houses_train$Gr_Liv_Area > 4500) )

#---------------------------
# Total_Bsmt_SF

ggplot(houses_train,
       aes(x = Total_Bsmt_SF,
           y = Sale_Price)) +
  geom_point(col = "blue") +
  geom_smooth(method = "lm", se = FALSE) +
  theme_bw()

# here again one can see these 3 outliers
# in the bottom right quadrant

which(houses_train$Total_Bsmt_SF > 3000 &
        houses_train$Sale_Price < 2e5)


#---------------------------
# Garage_Area 

# strongly correlated with Garage_Cars

ggplot(houses_train,
       aes(x = Garage_Area,
           y = Sale_Price)) +
  geom_point(col = "blue") +
  geom_smooth(method = "lm", se = FALSE) +
  theme_bw()


# the alternative from the caret package:
# findCorrelation(x = correlation_matrix, cutoff = 0.9)
# which identifies correlations above the accepted threshold
# and indicates variables to be deleted

# by default it returns a list of
# column numbers to be deleted

findCorrelation(houses_correlations)

# with the names = TRUE option it returns 
# the list names of correlated columns

# lets decrease the cutoff to 0.75

findCorrelation(houses_correlations,
                cutoff = 0.75,
                names = TRUE)

# these are potential candidates
# to be excluded from the model


#----------------------------------------
# qualitative (categorical) variables

houses_categorical_vars <- 
  # check if variable is a factor
  sapply(houses, is.factor) %>% 
  # select those which are
  which() %>% 
  # and keep just their names
  names()

houses_categorical_vars

# Let's check their relationship with the target variable

# because the target variable is quantitative and explanatory
# qualitative, one can use analysis of variance (ANOVA)

# Let's see an example for the selected variable `MS_SubClass`

aov(houses_train$Sale_Price ~ houses_train$MS_SubClass) ->
  houses_anova

summary(houses_anova)

# The F statistic is used to verify 
# the null hypothesis that:
# H0: MS_SubClass does NOT impact the Sale_Price
# i.e. average Sale_Price does NOT differ
# for different values of MS_SubClass

# The higher the F-statistic
# (or the lower its p-value)
# the stronger we reject H0

# lets see how to extract the value 
# of the test statistic

str(summary(houses_anova))

# it is a list with one element
# which is a data frame

# the value of F is in row number 1
# and column number 4

summary(houses_anova)[[1]][1, 4]

# let's write a function that retrieves this value
# for the explanatory categorical variable provided 
# as the function argument

houses_F_anova <- function(categorical_var) {
  anova_ <- aov(houses_train$Sale_Price ~ 
                  houses_train[[categorical_var]]) 
  
  return(summary(anova_)[[1]][1, 4])
}

# and check how it works on Ms_SubClass

houses_F_anova("MS_SubClass")

# we can apply the same for all
# categorical predictors

sapply(houses_categorical_vars,
       houses_F_anova) %>% 
  # in addition lets sort them
  # in the decreasing order of F
  #  and store as an object
  sort(decreasing = TRUE) -> houses_anova_all_categorical

houses_anova_all_categorical


# let's see the relation of the target variable
# with the three categorical variables most 
# strongly associated with it


#---------------
# Exter_Qual 

ggplot(houses_train,
       aes(x = Exter_Qual,
           y = Sale_Price)) +
  geom_boxplot(fill = "blue") +
  theme_bw()

# clear monotonic relationship

#---------------
# Kitchen_Qual 

ggplot(houses_train,
       aes(x = Kitchen_Qual,
           y = Sale_Price)) +
  geom_boxplot(fill = "blue") +
  theme_bw()


#---------------
# Overall_Qual 

ggplot(houses_train,
       aes(x = Overall_Qual,
           y = Sale_Price)) +
  geom_boxplot(fill = "blue") +
  theme_bw()


# WARNING!
# strength of relation between two CATEGORICAL variables
# can be tested, e.g. using the Cramer's V coefficient
# (calculated on the basis of Chi2 test statistic)
# - Cramer's V  takes values from 0 to 1, where
# higher values mean a stronger relationship
# (if both variables have only two levels 
# Cramer's V take values from -1 to 1)

# let's see an example

DescTools::CramerV(houses_train$Sale_Type,
                   houses_train$Neighborhood)



#---------------------------------------------------
# variables with zero or near zero variance

# to identify low-varying variables one can use
# the measure that is the ratio of the frequency 
# of most common and the second most frequent 
# value ("frequency ratio")

# for diversified variables it will take the value close to 1,
# and a very large value for unbalanced unbalanced

# the next measure is "percent of unique values":
# 100 * (number of unique values) / (number of observations)
# close to zero for little different data

# if the "frequency ratio" is greater than some threshold,
# and "percent of unique values" is less than a certain limit
# one can assume that the variable has a variance close to 0

# to identify such variables we can use the function
# nearZeroVar(data, saveMetrics = FALSE)
# from the caret package

# argument saveMetrics = TRUE will show 
# the details of the calculation
# default settings:
# freqCut = 95/5, uniqueCut = 10

nearZeroVar(houses_train,
            saveMetrics = TRUE)

# while the default value FALSE displays
# only indexes of problematic variables

nearZeroVar(houses_train)

# lets save all calculated statistics
# to the data frame

nearZeroVar(houses_train,
            saveMetrics = TRUE) -> houses_nzv_stats

# and display it sorted in descending order
# by the columns zeroVar, nzv and freqRatio

houses_nzv_stats %>% 
  # we add rownames of the frame
  # (with names of variables)
  # as a new column in the data
  rownames_to_column("variable") %>% 
  # and sort it in the descreasing order
  arrange(-zeroVar, -nzv, -freqRatio)

# there are several problematic variables

# maybe we should skip them
# in considerations, or transform

table(houses_train$Utilities)
table(houses_train$Pool_Area)
table(houses_train$Three_season_porch)
table(houses_train$Pool_QC)
table(houses_train$Low_Qual_Fin_SF)
table(houses_train$BsmtFin_SF_2)
table(houses_train$Street)
table(houses_train$Condition_2)

#--------------------------------------------
# taking all above results into account
# lets create two lists of variables

# first - we create a vector with a list 
# of ALL variables (including explanatory 
# variable) apart from those which have
# near-zero variance or are redundant

# start with all

houses_variables_all <- names(houses_train)

# exclude those with near-zero variance

( houses_variables_nzv <- nearZeroVar(houses_train, 
                                      names = TRUE) )

houses_variables_all <-
  houses_variables_all[!houses_variables_all %in% 
                         houses_variables_nzv]

# we also omit numeric variables 
# being linear combinations of others
# (here "Gr_Liv_Area" is the sum of particular areas)

houses_variables_all <- 
  houses_variables_all[-which(houses_variables_all == "Gr_Liv_Area")]


# in addition let's also create a vector 
# with a list of SELECTED variables:
# - 10 numerical most correlated with the target
# - 10 categorical most related (ANOVA) with the target


# WARNING! 
# in the object houses_numeric_vars_order 
# the first variable is Sale_Price, and then 
# the next most strongly correlated with it
# therefore we take the first 11 variables 
# from this vector

houses_numeric_vars_order

houses_variables_selected <- c(houses_numeric_vars_order[1:11],
                               names(houses_anova_all_categorical)[1:10]
                               )

# save the lists for further analyses

save(list = c("houses_variables_all",
              "houses_variables_selected"),
     file = "houses_variables_lists.RData")


# lets also remove 3 outliers from the training data

houses_train <- houses_train[-houses_which_outliers,]

# and finally save the data divided
# into training and test sample

save(list = c("houses_train", "houses_test"),
     file = "houses_train_test.RData")