---
author: "Slava Nikitin"
date: "2017-04-03"
draft: false
tags: ["lecture"]
title: "Model"
summary: "Predictive models, accuracy diagnostics, xgboost"
math: true
output: 
  html_document:
    self_contained: true
---


```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
knitr::opts_knit$set(root.dir = "~/Documents/training/class/")
```

```{r echo = FALSE}
library(xgboost)

```

Modeling of data patterns is a broad and deep topic that we can barely scratch in an intro class. We will concentrate on a couple types of models, aiming at prediction, on how to visualize models, and how to check predictive accuracy. The primary package we will use is 
**xgboost**. 

## Predictive task
We consider a situation when we have a data frame of data, with $n$ rows and $p$ columns. We identify one column as a response variable and treat other variables as predictors. In addition, we distinguish between quantitative variables, for example, time or salary or rate of return, and categorical variables, like political affiliation or gender or loan approval status. The later distinction is necessary to make good choices about what type of R data is appropriate for different types of information. This may require some data preparation, but ultimately all information will be stored as **double**.

Given data and our choices about response / predictor variables, our task is to find a predictive function that takes values of predictors and returns a value close to the observed response variable. Here is how we formulate it mathematically.

Lets consider a row $i \in \{1, 2, \dots, n\}$ of a data frame, where one value is a response variable $y_i \in \mathbb R$ and remaining values are predictors $x_{i1}, x_{i2}, \dots, x_{ip-1} \in \mathbb R ^ {p-1}$, hence accounting for the $p$ columns. We assume 

- that $(y_i, x_{i1}, \dots, x_{ip-1})$ have been sampled independently from a probability distribution $P(y, x_{1}, \dots, x_{p-1})$ that characterizes variation in each variable and dependencies among them, meaning data is representative,
- that there is a function $g(x_{1}, \dots, x_{p-1})$ that provides a true description of how $mean(y)$ dependends on predictors, meaning there is a dependency of response on predictors.

The first two assumptions are theoretical. To attempt to solve the problem, we also need practical assumptions that we are in control of and are primary components of what makes up a predictive model methodology:

- that we can define a set of possible models, $\{f(x_{1}, \dots, x_{p-1}, \theta) : \theta \in \Theta\ \subset \mathbb R\}$, such that there is at least one function in a set of possible functions that approximates the true relationship, $f(x_{1}, \dots, x_{p-1}, \theta ^ *)$, meaning we can find an adequate model of the true relationship
- that we can find a function to quantify overall predictive accuracy  $\sum_{i=1}^n L(y_i, f(x_{1}, \dots, x_{p-1}, \theta))$, called a loss function, meaning we can quantify discrepancy between truth and approximation
- that we can obtain an estimation algorithm that searches through the possible values of $\theta \in \Theta$, which modifies how response variable and predictors are related, so that we can maximize overall predictive accuracy, meaning we can solve $\mathrm{argmin}_{\theta \in \Theta}\ L(\theta)$ within computational, statistical, and organizational constraints
- that we can come up with an interpretable metric of accuracy to characterize our model, meaning we can summarise how well it does, compare it to other models and communicate it to someone else

The above points are very general, and we need to specify some details to actually make this doable. First, we are going to use a set of functions and an estimation algorithm offered by package `xgboost`, which iteratively builds step-wise functions, called trees, and combines them by averaging their outputs. We could consider other function sets like planes, quadratics, exponential curves, but all of these are very restrained and it becomes harder to manually put these together in a multivariate case, so we will just rely on a very general set of function from `xgboost` that can approximate any of those and more.

Secondly, we are going to use mean squared error to quantify mean predictive error of a model that predicts a quantitative response, defined as 

$$\sqrt{\frac{1}{n}\sum_{i = 1}^n(y_i - f(x_{1}, \dots, x_{p-1}, \theta))^2}$$
 and misclassification rate for a categorical response, defined as $\frac{1}{n}\sum_{i = 1}^n \delta(y_i \neq f(x_{1}, \dots, x_{p-1}, \theta))$, where $\delta(condition)$ is 1 if true and 0 if false.

## Examples

To follow along run these commands:

```
install.packages("xgboost")
library(xgboost)
```

Lets look at some examples. **mtcars** serves as a good example:
```{r}
head(mtcars)
```

**mpg**, miles per gallon, is a quantitative variable while **am**, status of automatic transmission, is categorical. Suppose **mpg** is the response variable $y$, then our task is to estimate an unknown mathematical function $g(x_{cyl}, x_{disp},...,x_{carb})$ that will take values of the other 10 variables in the table or similar values, and will output a predicted value of **mpg** $\hat y$ that is *close* to the makes of cars we see in the table, but also close to new, unseen car makes, so $\hat y \approx y$. Outputs of $f(\dots)$ will be quantitative, continuous, positive; this is called a regression model. If we are successful, then we can predict **mpg** given values of the other variables.

This expresses our choices about response and predictors:

```{r}
response <- "mpg"
preds <- setdiff(names(mtcars), response)
```
`xgboost` has a special data structure `xgb.DMatrix` to which we pass our variables:

```{r}
data <- xgb.DMatrix(data = as.matrix(mtcars[preds]), label = mtcars[[response]])
```
Next, we estimate a predictive model. First we need to specify what kind of predictive problem is it, regression or classification?
```{r}
params <- list(objective = "reg:linear", max.depth = 2)
```
Here `"reg:linear"` means regression, so predicting a quantitative response variable. We will see other *objective* settings when doing classification. Then we can run the estimation algorithm to obtain a candidate model:
```{r}
model <- xgboost(data, nrounds = 15, params = params)
```

So, we have a model. How well does it do? This is a problem of diagnostics. We can use quantitative and visual methods to establish the predictive accuracy of our model. The important thing is to do model building and checking on different parts of a dataset, sometimes called train and test data. This way information for modeling and evaluating its accuracy is distinct. This is important for reducing or eliminating the bias that arises when you build a solution and then test it against the same data. Given that the solution is obtained by searching for a function that closely resembles patterns in the data, it should not be surprising if such a function shows high predictive accuracy - it was meant to. 

Luckily, `xgboost` provides a special function to handle data partitions automatically. The technique is called **cross-validation**. Data gets partitioned into non-overlapping subsets. A model is found for all but one subset and its accuracy is quantified against the held-out subset. This allows to control for bias in results. Then we repeat the same process by holding out a different dataset and estimating a model on the remaning. This is repeated until each dataset has served as a test dataset.

Here is how you do it in `xgboost`:
```{r }
model_check <- xgb.cv(data = data, nrounds = 15, params = params, 
                      nfold = 5, prediction = TRUE, verbose = FALSE)
```

```{r echo = FALSE}
model_check

```
Output of `xgb.cv` contains summary statistics on train and test data, and also predictions on the hold-out data. In addition to summary statistics, we can also do a few visualizations of the full pattern of predictive accuracy.

```{r}
plot.ts(as.data.frame(model_check$evaluation_log)[,c(2, 4)], plot.type = "single")
```
The plot above shows mean predictive error in the model over training iterations. The gap between train and test shows the slight bias towards predicting train rather than test data. A good model should have both values low and not too far from each other.

We can also examine residuals - observed minus predicted values - by plotting their histogram:
```{r}
hist(mtcars$mpg - model_check$pred)
```
A good model should have this centered around 0.

We can also check for a relationship between observed and predicted values. For a good model, these should be highly correlated:
```{r}
plot(mtcars$mpg, model_check$pred)
```

All the summary statistics and plots look pretty good. We can use the model we build to generate predictions, for old or new inputs. We would only need to convert data with the predictors into a `xgb.DMatrix`. Here is an example using the predict function from base R on the full dataset:

```{r}
predict(model, data)
```


Similarly, if we used **am** as the response variable, which is categorical, the task is also to find a predictive function, with the only difference from **mpg** is that the output should be categorical, perhapse 1 for automatic and 0 manual, and no other possible values. By outputing only a few categorical values is the reason why such a predictive function is often called a classifier.

We can use `xgboost` to solve this problem, too. Lets first setup our variables and data.
```{r}
response <- "am"
preds <- setdiff(names(mtcars), response)

data <- xgb.DMatrix(data = as.matrix(mtcars[preds]), label = mtcars[[response]])
```

The classification problem is different and requires a different way to capture a discrepancy between a candidate model and the underlying true function. Here we use `"binary:logistic"`:

```{r}
params <- list(objective = "binary:logistic", max.depth = 2)
model <- xgboost(data, nrounds = 15, params = params)
```

Given our model, we once again ask how well it can do predict under new conditions.
```{r }
model_check <- xgb.cv(data = data, nrounds = 15, params = params, 
                      nfold = 5, prediction = TRUE, verbose = FALSE)
```

```{r echo = FALSE}
model_check

```
```{r}
plot.ts(as.data.frame(model_check$evaluation_log)[,c(2, 4)], plot.type = "single")
```

When dealing with a classification problem, instead of residuals, it is typicaly to see which classes a model gets well, and how often, using what is called a confusion matrix. Basically, we count how many correct and incorrect predictions are made and put them into a table. Note that predictions from the model are actually probabilities, and to get a class prediction we need to threshold a probability. It is typical to use .5.
```{r}
table(pred = as.integer(model_check$pred > .5), obs = mtcars$am)
```
We the model can discriminate between automatic and manual pretty well, only confusing the two sometimes. 

This example showed a two-class prediction problem, but `xgboost` can handle multi-class problem, too. We just need to adjust our parameters by setting objective argument by to `"multi:softmax"` and `num_class` to number of classes we are dealing with. Lastly, the classes should be represented with integers, starting from 0 and going to `num_class - 1`.


```{r eval = F, echo = F}
model <- xgboost(data, nrounds = 15, params = list(objective = "reg:linear"))
model = xgboost(xgb.DMatrix(mtcars[, -9], label = mtcars$am), nrounds = 50,params = list(objective = "binary:logistic"))
mean((predict(model, xgb.DMatrix(as.matrix(mtcars[, -1]))) - mtcars$mpg) ^ 2)
mean((predict(model, xgb.DMatrix(as.matrix(mtcars[, -9]))) > .5) != mtcars$am)
predict2 <- function(object, newdata) predict(object, newdata)

model_check <- xgb.cv(data = xgb.DMatrix(as.matrix(mtcars[preds]), label = mtcars$mpg),
nrounds = 15, params = list(objective = "reg:linear", max.depth = 2), nfold = 5,
prediction = TRUE)
plot.ts(as.data.frame(model_check$dt)[,c(1, 3)], plot.type = "single")
hist(mtcars$mpg - model_check$pred)
plot(mtcars$mpg, model_check$pred)
model_check$dt


model_check <- xgb.cv(data = xgb.DMatrix(as.matrix(mtcars[, -9]), label = mtcars$am),
nrounds = 15, params = list(objective = "binary:logistic", max.depth = 2), nfold = 5,
prediction = TRUE)
plot.ts(as.data.frame(model_check$dt)[,c(1, 3)], plot.type = "single")
hist(mtcars$mpg - model_check$pred)
plot(mtcars$mpg, model_check$pred)
model_check$dt
table(pred = factor(as.integer(model_check$pred > .5)), obs = factor(mtcars$am))

xgb.importance(model = model, names(mtcars)[-1])
plot(ICEbox:::ice(object = model, X = as.matrix(mtcars[, -1]), y = mtcars$mpg, predictor = 9, predictfcn = predict2))



hist(predict(model, xgb.DMatrix(as.matrix(mtcars[, -1]))) - mtcars$mpg)
hist(predict(model, xgb.DMatrix(as.matrix(mtcars[, -1]))) - mtcars$mpg)

plot(mtcars$mpg, predict(model, xgb.DMatrix(as.matrix(mtcars[, -1]))))
```




