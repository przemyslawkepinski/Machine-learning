# lets install and load needed packages

install.packages("dplyr")
install.packages("readr")
install.packages("ggplot2")

library(dplyr)
library(readr)
library(ggplot2)

# lets import the data
#setwd()
houses <- read_csv("houses.csv")

# The file contains data on the characteristics and prices of houses
# sold in the city of Ames (Iowa, USA) between 2006 and 2010.

# It includes 2930 observations and 81 variables.

# The dataset was used some time ago in the kaggle competition 
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques

# The task was to use data from before 2010 to build a model predicting 
# real estate prices sold in 2010.

# In class we will use the version of data available in the package 
# `AmesHousing`, but we will also learn how to prepare the data.
# That is why we import it from the csv file. 
# The data have been initially pre-ordered (variable names with 
# underscores, clearly defined individual levels of qualitative variables).

# The description of variables (types and definition)
# can be found for example here (names might be slightly different):
# https://rdrr.io/cran/subgroup.discovery/man/ames.html

# The detailed description of the variable levels
# can be found here 
# http://jse.amstat.org/v19n3/decock/DataDocumentation.txt
# and in the text file "DataDocumentation.txt" in lab materials

# lets check the structure of the dataset

glimpse(houses)

#dbl - double -numeric data
#chr - text data

# lets count missings in every column

sort(colSums(is.na(houses)))
a %>% f(args)



# there is only one missing in the column Electrical

table(houses$Electrical, useNA = "ifany")

# lets replace the missing with the mode
# (most frequent value) - SBrkr 

houses$Electrical[is.na(houses$Electrical)] <- "SBrkr"

any(is.na(houses))

# no more missings


#---------------------------------------------------
# Storing qualitative variables as factors

# qualitative variables may be nominal or ordinal.
# If they are stored as a text column this information cannot
# be included. They are often coded as numeric variables,
# but then R by default treats them as quantitative variables.

# Qualitative variables should be stored as factors
# and after importing are stored as text columns

# we can check it one by one

is.character(houses$MS_SubClass)

table(houses$MS_SubClass)

# looks like this is a nominal variable
# (see also its description here:
# https://rdrr.io/cran/subgroup.discovery/man/ames.html)

# it should be converted into a factor

# CAUTION! conversion to another type can be done
# using the functions of the group as .____

houses$MS_SubClass <- as.factor(houses$MS_SubClass)

# in NOMINAL variables the order of values DOES NOT matter

table(houses$MS_SubClass)

# for factors we can check their 
# levels (including their order) with

levels(houses$MS_SubClass)

glimpse(houses$MS_SubClass)

# THE SAME procedure  should be applied to 
# ALL nominal variables in the dataset



#--------------------------------------------
# then lets move to preparing ORDINAL variables

# Lot_Shape is the first one

table(houses$Lot_Shape)

# by default the order of levels is alphabetic
# which does not necessarily relate to the real
# order of values (if there is any)

# here we need to use a function factor() which
# has more options - it allows to define the 
# order of levels and also character of the variable
# (ordinal = TRUE/FALSE - false is default)

# !!! check the correct order in 
# DataDocumentation.txt if needed

houses$Lot_Shape <- factor(houses$Lot_Shape,
                          # levels from lowest to highest
                          levels = c("Regular",
                                     "Slightly_Irregular",
                                     "Moderately_Irregular",
                                     "Irregular"),
                          ordered = TRUE) # ordinal

# lets check the table of frequencies now

table(houses$Lot_Shape)

# now the values (levels) are shown
# in an increasing order

levels(houses$Lot_Shape)

glimpse(houses$Lot_Shape)


# lets check one more ordinal variable
# - Overall_Cond

table(houses$Overall_Cond)

# the level "Very Excellent" which is in 
# the description (see "DataDocumentation.txt")
# is not present in the data

# This might be problematic, so lets
# merge two extreme levels on both sides
# ("Very_Excellent" and "Excellent" on one side
# and "Poor" and "Very_Poor" on the other)

# first we define the factor
# for all levels which are in the data

houses$Overall_Cond <- factor(houses$Overall_Cond,
                           # levels from lowest to highest
                           levels = c("Excellent",
                                      "Very_Good",
                                      "Good",
                                      "Above_Average",
                                      "Average",
                                      "Below_Average",
                                      "Fair",
                                      "Poor",
                                      "Very_Poor"),
                           ordered = TRUE) # ordinal

table(houses$Overall_Cond)

# and merge two lowest levels

houses$Overall_Cond[houses$Overall_Cond == "Very_Poor"] <- "Poor"

table(houses$Overall_Cond)

# we still need to remove from the factor
# variable definition the level "Poor"
# which will not be used

houses$Overall_Cond <- droplevels(houses$Overall_Cond)

table(houses$Overall_Cond)

# now it is done correctly

# THE SAME should be applied to 
# ALL ordinal variables in the dataset


#-----------------------------------------
# lets check the distribution (histogram) 
# of the dependent variable Sale_Price

ggplot(houses,
       aes(x = Sale_Price)) +
  geom_histogram(fill = "blue",
                 bins = 100) +
  theme_bw()

# clearly right-skewed distribution

# lets check how it looks after 
# log transformation
# lets take log(x + 1) in case of zeroes

ggplot(houses,
       aes(x = log(Sale_Price + 1))) +
  geom_histogram(fill = "blue",
                 bins = 100) +
  theme_bw()

# the transformation might help
# to obtain a better model as
# symmetric variables are easier to model



#-----------------------------------------
# Numeric variables

# Lets identify ALL numeric variables

is.numeric(houses$Order)

# it can be easily applied to all columns
# with sapply()

sapply(houses, is.numeric)

# then we can easily filter out
# those which are numeric

# lets store their list in a vector

houses_numeric_vars <- 
  sapply(houses, is.numeric) %>% 
  which() %>% 
  names()

houses_numeric_vars

# how many unique values
# does each numeric variable have?
# (qualitative variables can also 
# be stored as factors - e.g. as 0/1)

# lets check a sample variable first

length(unique(houses$Order))

# here each value is unique

# using pipeline operator
# is more convenient in case
# of applying several nested functions

houses$Order %>% 
  unique() %>% 
  length()

# and we can easily check this for
# all numeric variables

sapply(houses[, houses_numeric_vars], 
        function(x) 
          unique(x) %>% 
          length()) %>% 
  # lets sort variables by increasing 
  # number of levels in the end
  sort()

# some of them have only 3-5 levels

table(houses$Bsmt_Full_Bath)
table(houses$Bsmt_Half_Bath)
table(houses$Full_Bath)
table(houses$Half_Bath)
table(houses$Kitchen_AbvGr)
table(houses$Fireplaces)
table(houses$Garage_Cars)

# some distributions are highly concentrated 
# in one / two values - maybe combining values
# is sensible?

# These variables are counts - maybe it would 
# be sensible to aggregate (some) levels -
# convert to values 0, 1, more than 1?

#---------------------------------
# qualitative variables

# we can also check the number of levels
# for categorical variables
# (still stored as character columns)

houses_character_vars <- 
  sapply(houses, is.character) %>% 
  which() %>% 
  names()

houses_character_vars

# lets check the number of unique
# values for each

sapply(houses[, houses_character_vars], 
        function(x) 
          unique(x) %>% 
          length()) %>% 
  sort()


# some only 2-3

table(houses$Street)
table(houses$Alley)
table(houses$Land_Slope)
table(houses$Central_Air)
table(houses$Paved_Drive)

table(houses$Utilities)
# this variable will be omitted 
# in the analysis, because it
# basically has one level