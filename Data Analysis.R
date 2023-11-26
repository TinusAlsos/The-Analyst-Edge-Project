#load libraries
library(dplyr)
library(tidyr)
library(Dict)
library(caret)
library(ROCR)
library(e1071)
library(rpart)
library(rpart.plot)
library(randomForest)
library(neuralnet)

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
logistic_regression_model <- glm(Target ~., data=df_train, family="binomial")
logistic_regression_predicted_values_train <- predict(logistic_regression_model, newdata=df_train, type="response")
logistic_regression_predicted_values_test <- predict(logistic_regression_model, newdata=df_test, type="response")

logistic_regression_threshold_values_train <- (logistic_regression_predicted_values_train > 0.5)
logistic_regression_threshold_values_test <- (logistic_regression_predicted_values_test > 0.5)
logistic_regression_confusion_matrix_train <- table(df_train$Target, logistic_regression_threshold_values_train)
logistic_regression_confusion_matrix_test <- table(df_test$Target, logistic_regression_threshold_values_test)
logistic_regression_accuracy_train <- sum(diag(logistic_regression_confusion_matrix_train)) / sum(logistic_regression_confusion_matrix_train)
logistic_regression_accuracy_test <- sum(diag(logistic_regression_confusion_matrix_test)) / sum(logistic_regression_confusion_matrix_test)

#SVM model
SVM_model <- svm(Target ~., data=df_train, kernel = "linear", cost = 10, scale = FALSE)
SVM_predicted_values_train <- predict(SVM_model, newdata = df_train, type = "class")
SVM_predicted_values_test <- predict(SVM_model, newdata = df_test, type = "class")

SVM_threshold_values_train <- (SVM_predicted_values_train > 0.5)
SVM_threshold_values_test <- (SVM_predicted_values_test > 0.5)
SVM_confusion_matrix_train <- table(df_train$Target, SVM_threshold_values_train)
SVM_confusion_matrix_test <- table(df_test$Target, SVM_threshold_values_test)
SVM_accuracy_train <- sum(diag(SVM_confusion_matrix_train)) / sum(SVM_confusion_matrix_train)
SVM_accuracy_test <- sum(diag(SVM_confusion_matrix_test)) / sum(SVM_confusion_matrix_test)

#CART model
CART_model <- rpart(Target ~., data=df_train)
par(mar=c(1, 1, 1, 1))
prp(CART_model)
CART_predicted_values_train <- predict(CART_model, newdata = df_train)
CART_predicted_values_test <- predict(CART_model, newdata = df_test)

CART_threshold_values_train <- (CART_predicted_values_train > 0.5)
CART_threshold_values_test <- (CART_predicted_values_test > 0.5)
CART_confusion_matrix_train <- table(df_train$Target, CART_threshold_values_train)
CART_confusion_matrix_test <- table(df_test$Target, CART_threshold_values_test)
CART_accuracy_train <- sum(diag(CART_confusion_matrix_train)) / sum(CART_confusion_matrix_train)
CART_accuracy_test <- sum(diag(CART_confusion_matrix_test)) / sum(CART_confusion_matrix_test)

#random forest model
#warning message: are you certain about regression
df_train_as_factor <- df_train
df_test_as_factor <- df_test
df_train_as_factor$Target <- as.factor(df_train_as_factor$Target)
df_test_as_factor$Target <- as.factor(df_test_as_factor$Target)

random_forest_model <- randomForest(Target ~., data=df_train_as_factor, proximity=TRUE)
random_forest_predicted_values_train <- predict(random_forest_model, newdata = df_train_as_factor)
random_forest_predicted_values_test <- predict(random_forest_model, newdata = df_test_as_factor)

random_forest_confusion_matrix_train <- table(df_train_as_factor$Target, random_forest_predicted_values_train)
random_forest_confusion_matrix_test <- table(df_test_as_factor$Target, random_forest_predicted_values_test)
random_forest_accuracy_train <- sum(diag(random_forest_confusion_matrix_train)) / sum(random_forest_confusion_matrix_train)
random_forest_accuracy_test <- sum(diag(random_forest_confusion_matrix_test)) / sum(random_forest_confusion_matrix_test)

#neural network model
neural_network_model <- neuralnet(Target ~., data=df_train)

neural_network_predicted_values_train <- predict(neural_network_model, newdata = df_train)
neural_network_predicted_values_test <- predict(neural_network_model, newdata = df_test)

neural_network_threshold_values_train <- (neural_network_predicted_values_train > 0.5)
neural_network_threshold_values_test <- (neural_network_predicted_values_test > 0.5)
neural_network_confusion_matrix_train <- table(df_train$Target, neural_network_threshold_values_train)
neural_network_confusion_matrix_test <- table(df_test$Target, neural_network_threshold_values_test)
neural_network_accuracy_train <- sum(diag(neural_network_confusion_matrix_train)) / sum(neural_network_confusion_matrix_train)
neural_network_accuracy_test <- sum(diag(neural_network_confusion_matrix_test)) / sum(neural_network_confusion_matrix_test)

#make table with accuracy summary
accuracy_summary <- data.frame(logistic_regression <- c(logistic_regression_accuracy_train, logistic_regression_accuracy_test), 
                               SVM <- c(SVM_accuracy_train, SVM_accuracy_test), 
                               CART <- c(CART_accuracy_train, CART_accuracy_test), 
                               random_forest <- c(random_forest_accuracy_train, random_forest_accuracy_test), 
                               neural_network <- c(neural_network_accuracy_train, neural_network_accuracy_test))
row.names(accuracy_summary) <- c("Training Accuracy", "Test Accuracy")