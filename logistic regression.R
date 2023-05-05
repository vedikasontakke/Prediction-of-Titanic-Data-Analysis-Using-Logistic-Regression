#  read the training and testing datasets from CSV files named 
traindata = read.csv("train.csv")
testdata = read.csv("test.csv")

View(traindata)
View(testdata)

# Emabarked : C for Cherbourg, Q for Queenstown and S for Southampton

library(dplyr)             #  for data manipulation and transformation  & used to select specific columns, remove specific columns
library(fastDummies)       # for creating dummy variables from categorical data.   used to convert the Pclass, Sex, and Embarked columns into dummy variables.
library(caret)             # provides functions for data splitting.  used to split the data into training and testing datasets, and to train a logistic regression model.
library(ggplot2)           # for data visualization for creating a wide range of graphics.
library(corrplot)          # A package for creating correlation matrices
library(gridExtra)         # Provides functions for arranging multiple grid-based plots on a single page.
library(grid)              #  Provides a low-level graphics system for creating and manipulating grid-based graphical objects.
library(tidyr)             # handling missing values during plotting

# display the first six rows of the training and testing datasets
head(traindata)
head(testdata)

#train data preprocessing
summary(traindata)


# generates a bar plot of the Survived variable in the complete_data dataset
# using the ggplot2 package, showing the distribution of the Survived variable
# between the two levels ("Died" and "Survived").

# Survived variable from a numeric variable to a factor variable.
# In the given context, the "Survived" variable was originally a numeric variable with two possible values: 0 and 1.
# However, it was converted into a factor variable, which means that it was transformed into a categorical variable with two possible levels or categories: "Died" and "Survived".
# This conversion is necessary for classification tasks, as many classification algorithms work with factor variables as the target variable.
# The conversion also makes it easier to plot and analyze the distribution of the variable using graphs and summary statistics.

ggplot(traindata, aes(x = factor(Survived))) +   #the variable Survived is being plotted on the x-axis.  This function converts the Survived variable from a numeric variable to a factor variable
  geom_bar() + #  a bar layer to the plot. This creates a histogram of the Survived variable, with one bar for each level of the factor variable.
  scale_x_discrete(labels = c("Died", "Survived")) +  #  sets the labels for the x-axis
  xlab("Survived") +
  ylab("Count") +
  ggtitle("Distribution of Survived")

# creating a subclass of the survived == "1"
survivors_df <- subset(traindata, Survived == "1")
print(survivors_df)

# count male and female count
survivor_counts = table(survivors_df$Sex)
print(survivor_counts)

survivor_counts_df <- data.frame(Gender = names(survivor_counts), Count = as.numeric(survivor_counts))


ggplot(survivor_counts_df, aes(x = Gender, y = Count, fill = Gender)) +
  geom_bar(stat = "identity") +
  ggtitle("Number of Titanic Survivors by Gender") +
  xlab("Gender") +
  ylab("Number of Survivors")

ggplot(traindata, aes(x = Sex, fill = Sex)) +
  geom_bar() +
  geom_text(stat='count', aes(label=..count..), vjust=2, size=4, color="black") +
  ggtitle("Gender Distribution in Titanic Dataset") +
  xlab("Gender") +
  ylab("Count") +
  scale_fill_manual(values = c("#1F77B4", "#FF7F0E"), labels = c("Female", "Male")) +
  theme_classic()

ggplot(survivors_df, aes(x = factor(Pclass))) +
  geom_bar(fill = "steelblue") +
  geom_text(stat='count', aes(label=..count..), vjust=2, size=4, color="black") +
  ggtitle("Survivors by Passenger Class in Titanic Dataset") +
  xlab("Passenger Class") +
  ylab("Count") +
  scale_x_discrete(labels = c("1st Class", "2nd Class", "3rd Class")) +
  theme_classic()

ggplot(traindata, aes(x = factor(Pclass))) +
  geom_bar(fill = "steelblue") +
  geom_text(stat='count', aes(label=..count..), vjust=2, size=4, color="black") +
  ggtitle("Passenger Class Distribution in Titanic Dataset") +
  xlab("Passenger Class") +
  ylab("Count")


ggplot(traindata, aes(x = Age)) +
  geom_histogram(fill = "tomato", color = "white", bins = 30) +
  ggtitle("Age Distribution in Titanic Dataset") +
  xlab("Age") +
  ylab("Count")




# the unique function to each variable in the complete_data dataset to count the number of unique values.
sapply(traindata, function(x) length(unique(x)))

unique_x <- unique(traindata)

length(traindata) == length(unique_x)

# replacing missing values
colSums(is.na(traindata))
colSums(traindata=='')


# These lines fill in missing values for the Embarked variable with the most common value ("S") and the Age variable with the median age.

traindata$Embarked[traindata$Embarked==""] <- "S"
traindata$Age[is.na(traindata$Age)] <- median(traindata$Age,na.rm=T)

colSums(is.na(traindata))
colSums(traindata=='')


# This line creates a new dataset called titanic_data by removing the Cabin, PassengerId, Ticket,
# and Name variables from the traindata dataset using the select function from the dplyr package.
titanic_data <- traindata %>% select(-c(Cabin, PassengerId, Ticket, Name))

tail(titanic_data)

# loop converts the Survived, Pclass, Sex, and Embarked variables in the titanic_data dataset 
# from numeric or character data types to factor data types using the as.factor function.

for (i in c("Survived","Pclass","Sex","Embarked")){
  titanic_data[,i]=as.factor(titanic_data[,i])
}

tail(titanic_data)


levels(titanic_data$Pclass)
levels(titanic_data$Sex)
levels(titanic_data$Embarked)

levels(titanic_data$Pclass)
levels(titanic_data$Sex)
levels(titanic_data$Embarked)
titanic_data <- dummy_cols(titanic_data, select_columns = c("Pclass","Sex","Embarked"))
titanic_data <- titanic_data %>% select(-c(Pclass, Sex, Embarked))
tail(titanic_data)

# check is numeric values
sapply(titanic_data, is.numeric)

# remove fare 
numeric_df=titanic_data%>% select(-c(Fare))
tail(numeric_df)

# function should be applied to rows (value of 1) or columns (value of 2) of the matrix or dataframe.
numeric_df <- apply(numeric_df, 2, as.numeric)    # This line converts all columns of the numeric_df dataframe to numeric data type
tail(numeric_df)  
apply(numeric_df, 2, is.numeric)    # whether each column is numeric or not.

#test data
summary(testdata)

colSums(is.na(testdata))
colSums(testdata=='')

sapply(testdata, function(x) length(unique(x)))

unique_x <- unique(testdata)

# Compare lengths to check for duplicates
length(x) != length(unique_x)

testdata$Embarked[testdata$Embarked==""] <- "S"
testdata$Age[is.na(testdata$Age)] <- median(testdata$Age,na.rm=T)

test_data <- testdata %>% select(-c(Cabin, PassengerId, Ticket, Name))

tail(test_data)

for (i in c("Survived","Pclass","Sex","Embarked")){
  test_data[,i]=as.factor(test_data[,i])
}

test_data <- dummy_cols(test_data, select_columns = c("Pclass","Sex","Embarked"))
test_data <- test_data %>% select(-c(Pclass, Sex, Embarked))
tail(test_data)

cor_data = cor(numeric_df)
corrplot(cor_data, method="circle")

# --------------------------------------------------------------------------
train<-titanic_data
test<-test_data

tail(train)
tail(test)


# Generalized linear model
#glm :  build a linear relationship between the response and predictors
# indicates that Survived is the response variable, and all other variables in the data frame (.) are the predictors.
# there are 3types of logistic regression model : 
# 1. binomial :  used when the outcome variable has only two categories
# 2. Multinomial : when the outcome variable has three or more unordered categories.
# 3. Ordinal : when the outcome variable has three or more ordered categories.
# ordered category : primary school , high school , graduate school
# unorded category : red , pink , green 

TitanicModel <- glm(Survived ~.,family=binomial(link='logit'),data=train)

# deviance residuals : difference between the observed and predicted values


## TitanicModel Summary : This function displays the summary statistics of the TitanicModel.
# summary of the statistical information of a model or data frame, allowing for a quick and easy assessment of its performance.
#summary(TitanicModel)

#  chi-square test should be used to compare the models.
anova(TitanicModel, test="Chisq")

#  a significant improvement in the fit of the logistic regression model. 
# 

# generates predicted probabilities for the test data using the predict function and assigns them to the result variable.
result <- predict(TitanicModel,newdata=test,type='response')

# converts the predicted probabilities to binary predictions 
result <- ifelse(result > 0.5,1,0)

print(result)         # predicated result
print(test$Survived)  # actual values

# checking if there is missing values
sum(is.na(test$Survived))
sum(is.na(result))

# This line creates a vector "missing_index" containing the indices of the missing values in the "result" vector 
missing_index <- which(is.na(result))
print(missing_index)
if (length(missing_index) > 0) {
  # "na.rm" argument set to "TRUE" to exclude missing values from the calculation.
  result[missing_index] <- mean(result, na.rm = TRUE)
  result[missing_index] <- ifelse(result > 0.5,1,0)
}

# again check missing values
sum(is.na(test$Survived))
sum(is.na(result))

print(result)

# create a factor variables of the result and test$survived 
result_factor <- factor(result, levels = c(0, 1))
test_factor <- factor(test$Survived, levels = c(0, 1))
conf_mat = confusionMatrix(data=result_factor, reference=test_factor)
print(conf_mat)
# ------------------------------------------------------------------------------
tp <- conf_mat$table[2,2]
tn <- conf_mat$table[1,1]
fp <- conf_mat$table[1,2]
fn <- conf_mat$table[2,1]

cat("True positive:", tp, "\n")
cat("True negative:", tn, "\n")
cat("False positive:", fp, "\n")
cat("False negative:", fn, "\n")

acc = (tp+tn)/(tp+tn+fp+fn)
cat(acc*100,"% \n")

#Crosschecking the accuracy by calculating the mean
accuracy <- mean(result_factor == test_factor)
print(accuracy)

accuracy <- conf_mat$overall[1]
precision <- conf_mat$byClass[1]
recall <- conf_mat$byClass[2]
f1_score <- conf_mat$byClass[3]

cat("Accuracy:", round(accuracy, 3), "\n")
cat("Precision:", round(precision, 3), "\n")
cat("Recall:", round(recall, 3), "\n")
cat("F1 score:", round(f1_score, 3), "\n")

matrix_data1 <- matrix(c(262, 4, 30, 122), nrow = 2, dimnames = list(Prediction = c(0, 1), Reference = c(0, 1)))
Model_cm <- confusionMatrix(matrix_data1)

plot_data1 <- as.data.frame.table(Model_cm$table)
plot1 <- ggplot(plot_data1, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Model", x = "Reference", y = "Prediction", fill = "Count") +
  geom_text(aes(label = Freq), size = 5)

grid.arrange(plot1, ncol = 1)

saveRDS(TitanicModel, "TitaicModel.RDS")
