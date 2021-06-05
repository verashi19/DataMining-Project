library(randomForest)
library(leaps)
library(plyr)

set.seed(6942)

expit = function(x) exp(x) / (1 + exp(x))

interview = read.csv('C:/Users/ruifa/Dropbox/Stats 415/415project/interview_clean_v3.csv')
interview = na.omit(interview)
names(interview)[10:16] = c('Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7')
interview = interview[, -1]
#interview$Observed.Attendance

train = sample(1:nrow(interview), 0.8 * nrow(interview))

train_df = interview[train,]
#train_df = train_df[complete.cases(train_df),]

test_df = interview[-train,]
#test_df = test_df[complete.cases(test_df),]


#Train Random Forest
rf = randomForest(Observed.Attendance ~ .,data = train_df, importance = TRUE, ntree = 1000)

#Evaluate variable importance
imp = importance(rf, type = 1)
imp = data.frame(predictors = rownames(imp), imp)

# Order the predictor levels by importance
imp.sort = arrange(imp, desc(MeanDecreaseAccuracy))
imp.sort$predictors = factor(imp.sort$predictors, levels = imp.sort$predictors)

# Select the top 6 predictors
imp.6 = imp.sort[1:6,]
print(imp.6)

# Plot Important Variables
#varImpPlot(rf, type=1)

# Subset data with 20 independent and 1 dependent variables
train_df.dimred = cbind(Observed.Attendance = train_df$Observed.Attendance, train_df[, c(imp.6$predictors)])
train_df.dimred$Observed.Attendance = as.numeric(train_df.dimred$Observed.Attendance) - 1
test_df.dimred = cbind(Observed.Attendance = test_df$Observed.Attendance, test_df[, c(imp.6$predictors)])
test_df.dimred$Observed.Attendance = as.numeric(test_df.dimred$Observed.Attendance) - 1
########## log reg
log.dim = glm(Observed.Attendance ~ ., data = train_df.dimred, family = binomial)

## training data
pred = predict(log.dim, train_df.dimred)

predProbs = expit(pred)
trainPrediction = rep(0, nrow(train_df.dimred))
trainPrediction[predProbs > .5] = 1
#training error = 0.74968234
train_acc = mean(trainPrediction == train_df.dimred$Observed.Attendance)

source("C:/Users/ruifa/Dropbox/Stats 415/415project/remove_missing_levels.r")
pred = predict(log.dim, newdata = remove_missing_levels(fit = log.dim, test_data = test_df.dimred))

predProbs = expit(pred)
testPrediction = rep(0, nrow(test_df.dimred))
testPrediction[predProbs > .5] = 1

#training error
test_acc = mean(testPrediction == test_df.dimred$Observed.Attendance)

print(c(train_acc, test_acc))