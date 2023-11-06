#import data
library(ggplot2)
library(Hmisc)
library(corrplot)
library(rstatix)
library(dplyr)
library(dlookr)

#read data file
dataset <- read.csv("dataset.csv")

#make non-int variable as factor
dataset$Target <- as.factor(dataset$Target)

#exclude enrolled students
dataset <- subset(dataset, dataset$Target != "Enrolled")

#basic statistics
summary <- summary(dataset)
description <- describe(dataset)

#test normality of variables
normality <- normality(dataset)

#correlation coefficients of variables
correlate(dataset)

#plot a few variables to check for biases in the data
ggplot(dataset, aes(x = Course)) + geom_bar()

ggplot(dataset, aes(x = Nacionality)) + geom_bar()

ggplot(dataset, aes(x = Gender)) + geom_bar()

ggplot(dataset, aes(x = Scholarship.holder)) + geom_bar()

ggplot(dataset, aes(x = Age.at.enrollment)) + geom_bar()

#check for multicollinearity using correlation table, threshold 0.70
dataset_without_target <- dataset[, !names(dataset) %in% c("Target")]
colnames(dataset_without_target) <- c("Mar.stat", "App.mode", "App.order", "Course",
                                      "Day.eve.att", "Prev.qual", "Nac", "M.s.qual",
                                      "F.s.qual", "M.s.occ", "F.s.occ", "Disp",
                                      "Edu", "Debtor", "Tuition", "Gender",
                                      "Schol.holder", "Age.at.en", "Curr.u.1.sem..cred",
                                      "Curr.u.1.sem..enro", "Curr.u.1.sem..eval", 
                                      "Curr.u.1.sem..appr", "Curr.u.1.sem..grad", 
                                      "Curr.u.1.sem..wout.eval", "Curr.u.2.sem..cred", 
                                      "Curr.u.2.sem..enro", "Curr.u.2.sem..eval", 
                                      "Curr.u.2.sem..appr", "Curr.u.2.sem..grad", 
                                      "Curr.u.2.sem..wout.eval", "Unemp.rate", 
                                      "Infl.rate", "GDP")
correlation_table <- cor_mat(dataset_without_target)
correlation_plot <- corrplot(cor(dataset_without_target), method = "color")