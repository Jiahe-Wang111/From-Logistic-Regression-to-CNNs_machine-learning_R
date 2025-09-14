################################################################################

library(dplyr)
library(rpart)
library(randomForest)
library(nnet)
library(RSNNS)
library(data.table)
library(ggplot2)

################################################################################

# Part 1: Tabular Data (Adult Income)

adults <- readRDS("C:/Users/wjh/Desktop/ML/lab3/adults.rds") |> na.omit()

# Train/test split
set.seed(42)
n <- nrow(adults)
idx <- sample(seq_len(n), size = 0.8*n)
train <- adults[idx, ] |> as.data.frame()
test  <- adults[-idx, ] |> as.data.frame()

# Ensure target variable
train$y <- as.factor(train$y)
test$y  <- as.factor(test$y)

# Logistic Regression
glm_model <- glm(y ~ ., data = train, family = "binomial")
pred_glm <- ifelse(predict(glm_model, test, type = "response") > 0.5, 1, 0)
acc_glm <- mean(pred_glm == as.numeric(test$y)-1)

# Decision Tree
tree_model <- rpart(y ~ ., data = train, method = "class")
pred_tree <- predict(tree_model, test, type="class")
acc_tree <- mean(pred_tree == test$y)

# Random Forest
rf_model <- randomForest(y ~ ., data=train, ntree=100)
pred_rf <- predict(rf_model, test)
acc_rf <- mean(pred_rf == test$y)

# Simple Neural Network (nnet)
prepare_nn_data <- function(data){
  data <- as.data.frame(data)
  for(col in names(data)){
    if(is.factor(data[[col]])) data[[col]] <- as.numeric(data[[col]]) - 1
  }
  data
}

train_nn <- prepare_nn_data(train)
test_nn  <- prepare_nn_data(test)

set.seed(123)
nn_model <- nnet(y ~ ., data=train_nn, size=10, decay=0.01, maxit=200, trace=FALSE)
pred_nn <- ifelse(predict(nn_model, test_nn) > 0.5, 1, 0)
acc_nn <- mean(pred_nn == test_nn$y)

# Compare results
results_tabular <- c(
  Logistic=acc_glm,
  DecisionTree=acc_tree,
  RandomForest=acc_rf,
  NeuralNetwork=acc_nn
)
print("Adult dataset model accuracy:")
print(round(results_tabular, 4))

################################################################################

# Part 2: Image Data (Fashion MNIST)

train_2016 <- readRDS("C:/Users/wjh/Desktop/ML/lab3/fashion_2016_train.rds")
test_2016  <- readRDS("C:/Users/wjh/Desktop/ML/lab3/fashion_2016_test.rds")

x_train <- train_2016$images
y_train <- train_2016$labels
x_test  <- test_2016$images
y_test  <- test_2016$labels

# Flatten 28x28 images for MLP
x_train_flat <- matrix(x_train, nrow=dim(x_train)[1], ncol=28*28)
x_test_flat  <- matrix(x_test,  nrow=dim(x_test)[1],  ncol=28*28)

train_nn_img <- data.frame(x_train_flat, y=as.factor(y_train))
test_nn_img  <- data.frame(x_test_flat, y=as.factor(y_test))

# One-hidden-layer MLP with RSNNS
inputs  <- as.matrix(train_nn_img[, -which(names(train_nn_img)=="y")])
outputs <- decodeClassLabels(train_nn_img$y)  # one-hot

mlp_model <- mlp(inputs, outputs, size=c(64), learnFunc="Backpropagation", learnFuncParams=c(0.01), maxit=100)

# Predict
pred_mlp <- predict(mlp_model, as.matrix(test_nn_img[, -which(names(test_nn_img)=="y")]))
pred_classes <- apply(pred_mlp, 1, which.max) - 1
acc_mlp <- mean(pred_classes == test_nn_img$y)

print(paste("Fashion MNIST MLP accuracy:", round(acc_mlp,4)))
