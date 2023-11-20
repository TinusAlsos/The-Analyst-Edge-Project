#load libraries
library(dplyr)
library(tidyr)
library(Dict)
library(caret)
library(ROCR)

#import data
df <- read.csv("dataset.csv")

#display basic info on dataframe
str(df)

#get dataframe columns
colnames(df)

#exclude enrolled category in target variable
count(df)
df <- df[df$Target %in% c("Graduate", "Dropout"), ]
count(df)

#make mapping dictionary with Dropout as 1 and Graduate as 0
target_mapping <- c("Dropout" = 1, "Graduate" = 0)
new_target <- recode(df$Target, !!! target_mapping)
df$Target <- new_target

sum(df$Target == 1)/length(df$Target) * 100

#split data into train and test
set.seed(144)
split <- createDataPartition(df$Target, p = 0.7, list = FALSE)
df_train <- df[split,]
df_test <- df[-split,]

#logistic regression model
#accuracy calculation not coming out correctly
logistic_regression_model <- glm(Target ~., data=df_train, family="binomial")
logistic_regression_predicted_values <- predict(logistic_regression_model, newdata=df_test, type="response")

confusion_matrix <- table(df_test$Target, logistic_regression_predicted_values)
logistic_regression_accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)