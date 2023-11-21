## Project
# Install Packages
install.packages("readr")
install.packages("writexl")
install.packages("ggplot2")
install.packages("Hmisc")
install.packages("corrplot")
install.packages("rstatix")
install.packages("dplyr")
install.packages("dlookr")
install.packages("reshape2")
install.packages("lightgbm")
install.packages("caret")
install.packages("pROC")
install.packages("vip")
install.packages("ggplot2")
install.packages("DALEX")





# Load libraries
library(readr)
library(writexl)
library(psych)
library(nortest)
library(ggplot2)
library(Hmisc)
library(corrplot)
library(rstatix)
library(dplyr)
library(dlookr)
library(reshape2)
library(lightgbm)
library(caret)
library(pROC)
library(vip)
library(ggplot2)
library(DALEX)


#read data file
dataset <- read.csv("dataset.csv")

# Convert 'target' variable into a factor
dataset$Target <- factor(dataset$Target)
str(dataset)

#Redefine values for target variable
dataset <- dataset %>%
  mutate(Target = case_when(
    Target == 'Dropout' ~ 0,
    Target == 'Enrolled' ~ 2,
    Target == 'Graduate' ~ 1
  ))


#Note: imbalance in data set: Graduate (2209), Dropout (1421), Enrolled (794)

#Dataset with only enrolled students:
enrolled_data <- dataset[dataset$Target == 2, ]


# Dataset with drop and grad:
dropout_graduate_data <- dataset[dataset$Target %in% c(0, 1), ]


# Select only numeric columns from the drop & grad dataset
numeric_dropout_graduate_data <- dropout_graduate_data[sapply(dropout_graduate_data, is.numeric)]


# Calculate the correlation matrix
correlation_matrix <- cor(numeric_dropout_graduate_data)
print(correlation_matrix)


# For Spearman correlation
spearman_correlation_matrix <- cor(numeric_dropout_graduate_data, method = "spearman")

correlation_data_melted <- melt(correlation_matrix)

ggplot(data = correlation_data_melted, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), space = "Lab", 
                       name="Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "", y = "")


# Selected variables curriculum correlation
selected_vars_curriculum <- numeric_dropout_graduate_data[, c("Curricular.units.1st.sem..credited.", 
                             "Curricular.units.1st.sem..enrolled.", 
                             "Curricular.units.1st.sem..evaluations.", 
                             "Curricular.units.1st.sem..approved.", 
                             "Curricular.units.1st.sem..grade.", 
                             "Curricular.units.1st.sem..without.evaluations.")]

correlation_matrix_selected <- cor(selected_vars_curriculum)
print(correlation_matrix_selected)

melted_correlation_matrix_selected <- melt(correlation_matrix_selected)
ggplot(data = melted_correlation_matrix_selected, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", midpoint = 0, limit = c(-1, 1)) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1))


# Correlation matrix with only approved curriculum
dataset_approved_curriculum <- numeric_dropout_graduate_data[, !(names(numeric_data) %in% c("Curricular.units.1st.sem..credited.", 
                                                     "Curricular.units.1st.sem..enrolled.", 
                                                     "Curricular.units.1st.sem..evaluations.", 
                                                     "Curricular.units.1st.sem..grade.", 
                                                     "Curricular.units.1st.sem..without.evaluations.",
                                                     "Curricular.units.2nd.sem..credited.", 
                                                     "Curricular.units.2nd.sem..enrolled.", 
                                                     "Curricular.units.2nd.sem..evaluations.", 
                                                     "Curricular.units.2nd.sem..grade.", 
                                                     "Curricular.units.2nd.sem..without.evaluations."))]


correlation_matrix_approved_curriculum <- cor(dataset_approved_curriculum)
print(correlation_matrix_approved_curriculum)

# Melt the correlation matrix for ggplot for the approved curriculum features only
melted_correlation_matrix_approved_curriculum <- melt(correlation_matrix_approved_curriculum )

# Create heatmap for the approved curriculum features only
ggplot(data = melted_correlation_matrix_approved_curriculum, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", midpoint = 0, limit = c(-1, 1)) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1))








### Light GBM
## All variables

# Assign target and features
target <- numeric_dropout_graduate_data$Target
features <- numeric_dropout_graduate_data[, names(numeric_dropout_graduate_data) != "Target"]


# Split data into training and testing sets
set.seed(123)  # Setting seed for reproducibility
indices <- sample(1:nrow(features), size = 0.8 * nrow(features))
train_features <- features[indices, ]
train_target <- target[indices]
test_features <- features[-indices, ]
test_target <- target[-indices]


# Convert to LightGBM dataset
dtrain <- lgb.Dataset(data = as.matrix(train_features), label = train_target)


# Define parameters
params <- list(
  objective = "binary",   # Use "binary" for binary classification
  metric = "binary_logloss",  # Binary log loss for binary classification
  num_leaves = 31,
  learning_rate = 0.05
)
params$num_leaves <- 15  # Optionally adjust based on data

# Train the model
model <- lgb.train(
  params = params,
  data = dtrain,
  nrounds = 100
)

# Prepare test data as a matrix
test_matrix <- as.matrix(test_features)

# Make predictions using the model
predictions <- predict(model, test_matrix)

# Evaluate the model
predicted_labels <- ifelse(predictions > 0.5, 1, 0)  # Assuming binary classification
accuracy <- mean(predicted_labels == test_target)
print(paste("Accuracy:", accuracy))


# Convert test data to matrix
test_matrix <- as.matrix(test_features)

# Make predictions
predictions <- predict(model, test_matrix)

# Converting probabilities to binary class labels
predicted_labels <- ifelse(predictions > 0.5, 1, 0)

# Calculate accuracy, precision, recall, and F1-score
conf_matrix <- confusionMatrix(factor(predicted_labels), factor(test_target))
accuracy <- conf_matrix$overall['Accuracy']
precision <- conf_matrix$byClass['Precision']
recall <- conf_matrix$byClass['Recall']
F1 <- 2 * (precision * recall) / (precision + recall)

# Calculate AUC-ROC
roc_obj <- roc(test_target, predictions)
AUC_ROC <- auc(roc_obj)

# Print the metrics
print(paste("Accuracy:", accuracy))
print(paste("Precision:", precision))
print(paste("Recall:", recall))
print(paste("F1 Score:", F1))
print(paste("AUC-ROC:", AUC_ROC))




### Light GBM
## Exclude all curriculum variables except approved

# Excluded features list
excluded_features <- c("Curricular.units.1st.sem..credited.", 
                       "Curricular.units.1st.sem..enrolled.", 
                       "Curricular.units.1st.sem..evaluations.", 
                       "Curricular.units.1st.sem..grade.", 
                       "Curricular.units.1st.sem..without.evaluations.",
                       "Curricular.units.2nd.sem..credited.", 
                       "Curricular.units.2nd.sem..enrolled.", 
                       "Curricular.units.2nd.sem..evaluations.", 
                       "Curricular.units.2nd.sem..grade.", 
                       "Curricular.units.2nd.sem..without.evaluations.",
                       "Target")  # Add 'Target' to the list to be excluded

# Assuming 'numeric_dropout_graduate_data' contains the 'Target' column
target_excluded_curriculum_features <- numeric_dropout_graduate_data$Target

# Select features excluding the specified ones and the target variable
features_excluded_curriculum_features <- numeric_dropout_graduate_data[, 
                                                                       !(names(numeric_dropout_graduate_data) %in% excluded_features)]


# Split data into training and testing sets
set.seed(123)  # Setting seed for reproducibility
indices_excluded_curriculum_features <- sample(1:nrow(features_excluded_curriculum_features), 
                                               size = 0.8 * nrow(features_excluded_curriculum_features))

train_features_excluded_curriculum_features <- features_excluded_curriculum_features[indices_excluded_curriculum_features, ]
train_target_excluded_curriculum_features <- target_excluded_curriculum_features[indices_excluded_curriculum_features]

test_features_excluded_curriculum_features <- features_excluded_curriculum_features[-indices_excluded_curriculum_features, ]
test_target_excluded_curriculum_features <- target_excluded_curriculum_features[-indices_excluded_curriculum_features]

# Convert to LightGBM dataset
dtrain_excluded_curriculum_features <- lgb.Dataset(data = as.matrix(train_features_excluded_curriculum_features), 
                                                   label = train_target_excluded_curriculum_features)

# Define parameters for the new model
params_excluded_curriculum_features <- list(
  objective = "binary",   
  metric = "binary_logloss",
  num_leaves = 31,
  learning_rate = 0.05
)
params_excluded_curriculum_features$num_leaves <- 15

# Train the model with excluded features
model_excluded_curriculum_features <- lgb.train(
  params = params_excluded_curriculum_features,
  data = dtrain_excluded_curriculum_features,
  nrounds = 100
)

# Prepare test data as a matrix
test_matrix_excluded_curriculum_features <- as.matrix(test_features_excluded_curriculum_features)

# Make predictions with the new model
predictions_excluded_curriculum_features <- predict(model_excluded_curriculum_features, 
                                                    test_matrix_excluded_curriculum_features)

# Evaluate the new model
predicted_labels_excluded_curriculum_features <- ifelse(predictions_excluded_curriculum_features > 0.5, 1, 0)
accuracy_excluded_curriculum_features <- mean(predicted_labels_excluded_curriculum_features == test_target_excluded_curriculum_features)
print(paste("Accuracy with excluded curriculum features:", accuracy_excluded_curriculum_features))


# Convert test data to matrix for the excluded curriculum features model
test_matrix_excluded <- as.matrix(test_features_excluded_curriculum_features)

# Make predictions
predictions_excluded <- predict(model_excluded_curriculum_features, test_matrix_excluded)

# Converting probabilities to binary class labels
predicted_labels_excluded <- ifelse(predictions_excluded > 0.5, 1, 0)

# Calculate accuracy, precision, recall, and F1-score
conf_matrix_excluded <- confusionMatrix(factor(predicted_labels_excluded), factor(test_target_excluded_curriculum_features))
accuracy_excluded <- conf_matrix_excluded$overall['Accuracy']
precision_excluded <- conf_matrix_excluded$byClass['Precision']
recall_excluded <- conf_matrix_excluded$byClass['Recall']
F1_excluded <- 2 * (precision_excluded * recall_excluded) / (precision_excluded + recall_excluded)

# Calculate AUC-ROC
roc_obj_excluded <- roc(test_target_excluded_curriculum_features, predictions_excluded)
AUC_ROC_excluded <- auc(roc_obj_excluded)

# Print the metrics
print(paste("Accuracy with excluded features:", accuracy_excluded))
print(paste("Precision with excluded features:", precision_excluded))
print(paste("Recall with excluded features:", recall_excluded))
print(paste("F1 Score with excluded features:", F1_excluded))
print(paste("AUC-ROC with excluded features:", AUC_ROC_excluded))



## Plot feature importance Light GBM

PlotLightGBMFeatureImportance <- function(model, num_features = 10, title = "Feature Importance") {
  # Get feature importance
  importance <- lightgbm::lgb.importance(model, percentage = TRUE)
  importance_top_n <- head(importance[order(-importance$Gain), ], num_features)
  
  # Plotting
  ggplot(importance_top_n, aes(x = reorder(Feature, Gain), y = Gain, fill = Gain)) +
    geom_col() +
    scale_fill_gradient(low = "skyblue", high = "blue") +
    coord_flip() +
    labs(title = title, x = "Features", y = "Importance (Gain)") +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
      axis.title.x = element_text(size = 14, face = "bold"),
      axis.title.y = element_text(size = 14, face = "bold"),
      axis.text.x = element_text(size = 12),
      axis.text.y = element_text(size = 12),
      legend.title = element_blank(),
      legend.position = "none"
    ) +
    geom_text(aes(label = sprintf("%.2f", Gain)), hjust = -0.1, size = 4)
}

# Plot function with all variables
PlotLightGBMFeatureImportance(model, num_features = 10, title = "Top 10 Feature Importance - All Variables")


# Excluded variables
PlotLightGBMFeatureImportance(model_excluded_curriculum_features, num_features = 10, title = "Top 10 Feature Importance - Excluded Curriculum Variables")



