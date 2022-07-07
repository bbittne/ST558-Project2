ST558 - Project 2
================
Li Wang & Bryan Bittner
2022-07-06

  - [Load Packages](#load-packages)
  - [Introduction](#introduction)
  - [Data](#data)
      - [Summarizations](#summarizations)
  - [Modeling](#modeling)
      - [Linear Regression Model](#linear-regression-model)
      - [Random Forest Model](#random-forest-model)
      - [Boosted Tree Model and Explanation -
        Li](#boosted-tree-model-and-explanation---li)
  - [Comparison](#comparison)
  - [Automation](#automation)

# Load Packages

We will use the following packages:

``` r
library(httr)
library(jsonlite)
library(readr)
library(tidyverse)
library(lubridate)
library(knitr)
library(caret)
library(randomForest)
```

# Introduction

This data set summarizes a heterogeneous set of features about articles
published by Mashable in a period of two years.

Our target variable is the shares variable, and predict variables are
global\_rate\_positive\_words, global\_rate\_negative\_words,
n\_tokens\_title and so on.

The purpose of our analysis is to predict the number of shares in social
networks (popularity). In this project, we produce some basic (but
meaningful) summary statistics and plots about the training data, and
fit a linear regression model and an ensemble tree-based model for
predicting the number of shares.

# Data

Use a relative path to import the data.

``` r
newsData<-read_csv(file="../ST558-Project2/OnlineNewsPopularity.csv")
#newsData<-read_csv(file="../Datasets/OnlineNewsPopularity.csv")
head(newsData)
```

Subset the data to work on the data channel of interest

``` r
#Once we parameterize this file, part of the column name will be passed in as a parameter by the render code. I'm creating a separate field to handle this portion of the column name now and eventually we can just set the parameter to this field and the rest should work.

#Parameter Name will eventually go here instead of "lifestyle"
paramColumnNameType<-"lifestyle"
columnName<-paste("data_channel_is_",paramColumnNameType,sep="")

#According to dplyr help, to refer to column names stored as string, use the '.data' pronoun.
#https://dplyr.tidyverse.org/reference/filter.html
newsDataSubset <- filter(newsData,.data[[columnName]] == 1)

# remove data_channel_is_entertainment,data_channel_is_bus, data_channel_is_socmed ,data_channel_is_tech,data_channel_is_world columns.
newsDataSubset<-newsDataSubset%>%select(-c(15:19))
head(newsDataSubset)
```

## Summarizations

You should produce some basic (but meaningful) summary statistics and
plots about the training data you are working with (especially as it
relates to your response - aka the ‘shares’ field)

Start with the basic summary statistics for the ‘shares’ field.

``` r
summary(newsDataSubset$shares)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##      28    1100    1700    3682    3250  208300

Now lets show the Mean, Median, Variance, and Standard Deviation. Notice
the Variance and Standard Deviation are both extremely high. This might
be something we will have to investigate further

``` r
newsDataSubset %>% summarise(avg = mean(shares), med = median(shares), var = var(shares), sd = sd(shares))
```

Looking at the different columns in the dataset, there are two that
stand out. Generally speaking, people probably aren’t going to look at
articles that don’t have images or videos. Here are the summary stats
for the articles grouped on the number of images in the article.

``` r
newsDataSubset %>% group_by(num_imgs) %>%
summarise(avg = mean(shares), med = median(shares), var = var(shares), sd = sd(shares))
```

Here are the summary stats for articles with videos.

``` r
newsDataSubset %>% group_by(num_videos) %>%
summarise(avg = mean(shares), med = median(shares), var = var(shares), sd = sd(shares))
```

A scatter plot with the number of shares on the y-axis and
n\_tokens\_title on the x-axis is created:

``` r
g <- ggplot(newsDataSubset, aes(x = n_tokens_title, y = shares))
g + geom_point()
```

![](README_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

# Modeling

Before we do any modeling, lets set up our Train/Test split. This will
allow us to determine the model fit using a subset of data called
Training, while saving the remainder of the data called Test to test our
model predictions with.

``` r
set.seed(10)
train <- sample(1:nrow(newsDataSubset),size=nrow(newsDataSubset)*.7)
test <- dplyr::setdiff(1:nrow(newsDataSubset),train)

newsDataSubsetTrain <- newsDataSubset[train,]
newsDataSubsetTest <- newsDataSubset[test,]

head(newsDataSubsetTrain)
```

## Linear Regression Model

A Linear Regression Model is the first model type we will look at. These
models are an intuitive way to investigate the linear relation between
multiple variables. These models make the estimation procedure simple
and easy to understand. Linear Regression models can come in all
different shapes and sizes and can be used to model more than just a
straight linear relationship. Regression models can be modified with
interactive and or higher order terms that will conform to a more
complex relationship.

For the first linear model example, we can try a model using just the
“num\_imgs” and “num\_videos” as our predictors.

``` r
#Fit a  multiple linear regression model with Temperature and Season
mlrFit <- train(shares ~ num_imgs + num_videos, 
                data = newsDataSubsetTrain, 
                method="lm",
                trControl=trainControl(method="cv",number=5))
mlrFit
```

    ## Linear Regression 
    ## 
    ## 1469 samples
    ##    2 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 1176, 1174, 1176, 1174, 1176 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared     MAE     
    ##   8515.114  0.003603234  3287.381
    ## 
    ## Tuning parameter 'intercept' was held constant at a value of TRUE

Next we can try a linear model using all of the fields as a predictor
variables.

``` r
#Fit a  multiple linear regression model with Temperature and Season
mlrAllFit <- train(shares ~ ., 
                data = newsDataSubsetTrain, 
                method="lm",
                trControl=trainControl(method="cv",number=5))
mlrAllFit
```

    ## Linear Regression 
    ## 
    ## 1469 samples
    ##   55 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 1174, 1177, 1175, 1175, 1175 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared  MAE     
    ##   9004.098  NaN       3001.381
    ## 
    ## Tuning parameter 'intercept' was held constant at a value of TRUE

Add additional regression model(s) here.

## Random Forest Model

The Random Forest Model is an example of an ensemble based model.
Instead of traditional decision trees, ensemble methods average across
the tree. This will greatly increase our prediction power, but it will
come at the expense of the easy interpretation from traditional decision
trees. The Random Forest based model will not use all available
predictors. Instead it will take a random subset of the predictors for
each tree fit and calculate the model fit for that subset. It will
repeat the process a pre-determined number of times and automatically
pick the best predictors for the model. This will end up creating a
reduction in the overall model variance.

``` r
#Regression Tree so use mtry=# predictors/3
randomForestFit <- train(shares ~ ., 
                         data = newsDataSubsetTrain, 
                         method="rf",
                         preProcess=c("center","scale"),
                         trControl=trainControl(method="cv",number=5),
                         tuneGrid=data.frame(mtry=ncol(newsDataSubsetTrain)/3))
randomForestFit
```

    ## Random Forest 
    ## 
    ## 1469 samples
    ##   55 predictor
    ## 
    ## Pre-processing: centered (1522), scaled (1522) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 1174, 1176, 1175, 1175, 1176 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared    MAE     
    ##   7879.523  0.01318217  3160.918
    ## 
    ## Tuning parameter 'mtry' was held constant at a value of 18.66667

## Boosted Tree Model and Explanation - Li

# Comparison

# Automation
