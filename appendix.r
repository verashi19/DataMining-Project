setwd('/Users/RuochenWang/Documents/STATS 415/final')


SETUP

```{r}
library(glmnet)
library(grplasso)
library(randomForest)
library(leaps)
library(dplyr)
```

```{r}
expit = function(x) exp(x) / (1 + exp(x))
```


DATA LOADING

```{r}
interview = read.csv('interview_clean_v2.csv')
interview = na.omit(interview)
names(interview)[10:16] = c('Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7')
interview = interview[, -1]

set.seed(6942)
train = sample(1:nrow(interview), 0.8 * nrow(interview))

train_df = interview[train,]
train_df = train_df[complete.cases(train_df),]

test_df = interview[-train,]
test_df = test_df[complete.cases(test_df),]
```


MODEL SELECTION

## group lasso
```{r}
train_df.mod = train_df
train_df.mod$Observed.Attendance = as.numeric(train_df$Observed.Attendance) - 1 
train_X = model.matrix(Observed.Attendance ~ ., data = train_df.mod)[, -1]
train_Y = train_df.mod$Observed.Attendance

test_df.mod = test_df
test_df.mod$Observed.Attendance = as.numeric(test_df.mod$Observed.Attendance) - 1 
test_X = model.matrix(Observed.Attendance ~ ., data = test_df.mod)[, -1]
test_Y = test_df.mod$Observed.Attendance

grid = 10^seq(10, -2, length = 100)
lasso.mod = glmnet(train_X, train_Y, alpha = 1, lambda = grid, family = "binomial", type.multinomial = "grouped")

# cv
set.seed(6942)
lasso.best = cv.glmnet(train_X, train_Y, alpha = 1, type.measure = "auc")
lasso.bestlam = lasso.best$lambda.min

# evaluation
lasso.pred.train = predict(lasso.mod, s = lasso.bestlam, newx = train_X) > 0.5
lasso.accuracy.train = mean(lasso.pred.train == train_Y)

lasso.pred.test = predict(lasso.mod, s = lasso.bestlam, newx = test_X) > 0.5
lasso.accuracy.test = mean(lasso.pred.test == test_Y)
print(c(lasso.accuracy.train, lasso.accuracy.test))
```




## random forest + ridge
```{r}
rf = randomForest(Observed.Attendance ~ .,data = train_df, importance = TRUE, ntree = 1000)
imp = importance(rf, type = 1)
imp = data.frame(predictors = rownames(imp), imp)

# Order the predictor levels by importance
imp.sort = arrange(imp, desc(MeanDecreaseAccuracy))
imp.sort$predictors = factor(imp.sort$predictors, levels = imp.sort$predictors)

nfold = 5
source("remove_missing_levels.r")

cv.acc = c()
threshs = 2:17
for (thresh in threshs) {
	imp.select = imp.sort[1:thresh, ]
	print(imp.select)

	tmp = c()
	splits = split(1:nrow(train_df), 1:nfold)
	for (id in splits) {
		cv.train = train_df[-id,]
		cv.train.dimred = cbind(Observed.Attendance = cv.train$Observed.Attendance, cv.train[, c(imp.select$predictors)])
		cv.train.dimred$Observed.Attendance = as.numeric(cv.train.dimred$Observed.Attendance) - 1
		cv.train_X = model.matrix(Observed.Attendance ~ ., cv.train.dimred)[, -1]
		cv.train_Y = cv.train.dimred$Observed.Attendance

		cv.val   = train_df[id,]
		cv.val.dimred = cbind(Observed.Attendance = cv.val$Observed.Attendance, cv.val[, c(imp.select$predictors)])
		cv.val.dimred$Observed.Attendance = as.numeric(cv.val.dimred$Observed.Attendance) - 1
		cv.val_X = model.matrix(Observed.Attendance ~ ., cv.val.dimred)[, -1]
		cv.val_Y = cv.val.dimred$Observed.Attendance

		ridge.mod = cv.glmnet(cv.train_X, cv.train_Y, alpha = 0)
		lam = ridge.mod$lambda.min

		valPrediction = predict(ridge.mod, s = lam, newx = cv.val_X) > 0.5
		val.acc = mean(valPrediction == cv.val_Y)

		tmp = c(tmp, val.acc)
	}

	cv.acc = c(cv.acc, mean(tmp))
}


best.thresh = threshs[which.max(cv.acc)]
best.imp.select = imp.sort[1:best.thresh, ]

train_df.dimred = cbind(Observed.Attendance = train_df$Observed.Attendance, train_df[, c(best.imp.select$predictors)])
train_df.dimred$Observed.Attendance = as.numeric(train_df.dimred$Observed.Attendance) - 1
train_X = model.matrix(Observed.Attendance ~ ., train_df.dimred)[, -1]
train_Y = train_df.dimred$Observed.Attendance

test_df.dimred = cbind(Observed.Attendance = test_df$Observed.Attendance, test_df[, c(best.imp.select$predictors)])
test_df.dimred$Observed.Attendance = as.numeric(test_df.dimred$Observed.Attendance) - 1
test_X = model.matrix(Observed.Attendance ~ ., test_df.dimred)[, -1]
test_Y = test_df.dimred$Observed.Attendance

best.ridge.mod = cv.glmnet(train_X, train_Y, alpha = 0)
bestlam = best.ridge.mod$lambda.min

trainPrediction = predict(best.ridge.mod, s = bestlam, newx = train_X) > 0.5
train_acc = mean(trainPrediction == train_Y)

testPrediction = predict(best.ridge.mod, s = bestlam, newx = test_X) > 0.5
test_acc = mean(testPrediction == test_Y)

print(c(train_acc, test_acc))
```




## random forest + logistic regression

```{r}
rf = randomForest(Observed.Attendance ~ .,data = train_df, importance = TRUE, ntree = 1000)
imp = importance(rf, type = 1)
imp = data.frame(predictors = rownames(imp), imp)

# Order the predictor levels by importance
imp.sort = arrange(imp, desc(MeanDecreaseAccuracy))
imp.sort$predictors = factor(imp.sort$predictors, levels = imp.sort$predictors)

nfold = 5
source("remove_missing_levels.r")

cv.acc = c()
threshs = 2:17
for (thresh in threshs) {
	imp.select = imp.sort[1:thresh, ]
	print(imp.select)

	tmp = c()
	splits = split(1:nrow(train_df), 1:nfold)
	for (id in splits) {
		cv.train = train_df[-id,]
		cv.train.dimred = cbind(Observed.Attendance = cv.train$Observed.Attendance, cv.train[, c(imp.select$predictors)])
		cv.train.dimred$Observed.Attendance = as.numeric(cv.train.dimred$Observed.Attendance) - 1

		cv.val   = train_df[id,]
		cv.val.dimred = cbind(Observed.Attendance = cv.val$Observed.Attendance, cv.val[, c(imp.select$predictors)])
		cv.val.dimred$Observed.Attendance = as.numeric(cv.val.dimred$Observed.Attendance) - 1

		log.dim = glm(Observed.Attendance ~ ., data = cv.train.dimred, family = binomial)
		pred = predict(log.dim, remove_missing_levels(fit = log.dim, test_data = cv.val.dimred))
		predProbs = expit(pred)
		valPrediction = rep(0, nrow(cv.val.dimred))
		valPrediction[predProbs > .5] = 1
		val.acc = mean(valPrediction == cv.val.dimred$Observed.Attendance)

		tmp = c(tmp, val.acc)
	}

	cv.acc = c(cv.acc, mean(tmp))
}


best.thresh = threshs[which.max(cv.acc)]
best.imp.select = imp.sort[1:best.thresh, ]
train_df.dimred = cbind(Observed.Attendance = train_df$Observed.Attendance, train_df[, c(best.imp.select$predictors)])
train_df.dimred$Observed.Attendance = as.numeric(train_df.dimred$Observed.Attendance) - 1
test_df.dimred = cbind(Observed.Attendance = test_df$Observed.Attendance, test_df[, c(best.imp.select$predictors)])
test_df.dimred$Observed.Attendance = as.numeric(test_df.dimred$Observed.Attendance) - 1


best.log.mod = glm(Observed.Attendance ~ ., data = train_df.dimred, family = binomial)


pred = predict(best.log.mod, train_df.dimred)
predProbs = expit(pred)
trainPrediction = rep(0, nrow(train_df.dimred))
trainPrediction[predProbs > .5] = 1
#training error
train_acc = mean(trainPrediction == train_df.dimred$Observed.Attendance)


pred = predict(best.log.mod, newdata = remove_missing_levels(fit = best.log.mod, test_data = test_df.dimred))
predProbs = expit(pred)
testPrediction = rep(0, nrow(test_df.dimred))
testPrediction[predProbs > .5] = 1

#training error
test_acc = mean(testPrediction == test_df.dimred$Observed.Attendance)

print(c(train_acc, test_acc))
```


group lasso is the best method for this dataset.


ANALYSIS

```{r}
# data
train_df.mod = train_df
train_df.mod$Observed.Attendance = as.numeric(train_df$Observed.Attendance) - 1
train_df.mod.q = train_df.mod[, c(9:17)]
train_X.q = model.matrix(Observed.Attendance ~ ., data = train_df.mod.q)[, -1]
train_Y.q = train_df.mod.q$Observed.Attendance

test_df.mod = test_df
test_df.mod$Observed.Attendance = as.numeric(test_df.mod$Observed.Attendance) - 1
test_df.mod.q = test_df.mod[, c(9:17)]
test_X.q = model.matrix(Observed.Attendance ~ ., data = test_df.mod.q)[, -1]
test_Y.q = test_df.mod.q$Observed.Attendance


train_df.mod = train_df
train_df.mod$Observed.Attendance = as.numeric(train_df$Observed.Attendance) - 1
train_df.mod.nq = train_df.mod[, -c(9:16)]
train_X.nq = model.matrix(Observed.Attendance ~ ., data = train_df.mod.nq)[, -1]
train_Y.nq = train_df.mod.nq$Observed.Attendance

test_df.mod = test_df
test_df.mod$Observed.Attendance = as.numeric(test_df.mod$Observed.Attendance) - 1
test_df.mod.nq = test_df.mod[, -c(9:16)]
test_X.nq = model.matrix(Observed.Attendance ~ ., data = test_df.mod.nq)[, -1]
test_Y.nq = test_df.mod.nq$Observed.Attendance


## interview content only (questions + expected)
grid = 10^seq(10, -2, length = 100)
lasso.mod = glmnet(train_X.q, train_Y.q, alpha = 1, lambda = grid, family = "binomial", type.multinomial = "grouped")

set.seed(6942)
lasso.best = cv.glmnet(train_X.q, train_Y.q, alpha = 1, type.measure = "auc")
lasso.bestlam = lasso.best$lambda.min

lasso.pred.train = predict(lasso.mod, s = lasso.bestlam, newx = train_X.q) > 0.5
lasso.accuracy.train.q = mean(lasso.pred.train == train_Y.q)

lasso.pred.test = predict(lasso.mod, s = lasso.bestlam, newx = test_X.q) > 0.5
lasso.accuracy.test.q = mean(lasso.pred.test == test_Y.q)
print(c(lasso.accuracy.train.q, lasso.accuracy.test.q))


## non interview content only
grid = 10^seq(10, -2, length = 100)
lasso.mod = glmnet(train_X.nq, train_Y.nq, alpha = 1, lambda = grid, family = "binomial", type.multinomial = "grouped")

set.seed(6942)
lasso.best = cv.glmnet(train_X.nq, train_Y.nq, alpha = 1, type.measure = "auc")
lasso.bestlam = lasso.best$lambda.min

lasso.pred.train = predict(lasso.mod, s = lasso.bestlam, newx = train_X.nq) > 0.5
lasso.accuracy.train.nq = mean(lasso.pred.train == train_Y.nq)

lasso.pred.test = predict(lasso.mod, s = lasso.bestlam, newx = test_X.nq) > 0.5
lasso.accuracy.test.nq = mean(lasso.pred.test == test_Y.nq)
print(c(lasso.accuracy.train.nq, lasso.accuracy.test.nq))



```




