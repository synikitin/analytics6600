---
author: "Slava Nikitin"
date: "2017-03-06"
draft: false
tags: ["lecture"]
title: "Program"
summary: "Pipes, functions, data structures, iteration"
math: false
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
knitr::opts_knit$set(root.dir = "~/Documents/training/class/")
library(stringr)
```

## Pipe

The purpose behind pipes is to help simplify code that contains a chain of function calls by avoiding saving intermediate results. Here is data prep as an example:

1. All tidyverse packages load the pipe by default, but **magrittr** is the official package that contains it, and other operators as well.
```
library(magrittr)
```  
Data example is based on the second assignment:
```
u.data <- read_tsv(
  "u.data",
  col_names = c("user_id", "movie_id", "rating", "timestamp")
)
u.item <- read_delim(
  "u.item",
  "|",
  col_names = c(
    "movie_id", "movie_title", "release_date",
    "video_release_date", "IMDb_URL", "unknown",
    "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western"
    )
  )
u.user <- read_delim(
  "u.user",
  "|",
  col_names = c("user_id", "age", "gender", "occupation", "zip_code")
)
```

2. Save each intermediate result
```
df1 <- inner_join(u.data, u.user, "user_id")
df2 <- inner_join(df1, u.item, "movie_id")
df3 <- mutate(df2,
    timestamp = as_datetime(timestamp), 
    release_date = dmy(release_date)
  ) 
df4 <- select(df3, -video_release_date)
```

3. Overwrite each result
```
df <- inner_join(u.data, u.user, "user_id")
df <- inner_join(df, u.item, "movie_id")
df <- mutate(df,
    timestamp = as_datetime(timestamp), 
    release_date = dmy(release_date)
  ) 
df <- select(df, -video_release_date)
```

4. Nest each result
```
df <- select(
  mutate(
    inner_join(
      inner_join(
        u.data,
        u.user,
        "user_id"
      ),
      u.item,
      "movie_id"
    ),
    timestamp = as_datetime(timestamp), 
    release_date = dmy(release_date)
  ),
  -video_release_date
)
```

5. Pipe each result
```
df <- inner_join(u.data, u.user, "user_id") %>% 
  inner_join(u.item, "movie_id") %>% 
  mutate(
    timestamp = as_datetime(timestamp), 
    release_date = dmy(release_date)
  ) %>% 
  select(-video_release_date)
```

6. Pipes rewrite your code as if you used pattern in 3., that is assigning results into the same name over and passing it forward. When you see pipes you should think that the left-handside is becoming the first argument in a function call:
    - `x %>% f()` is `f(x)`
    - `x %>% f(y)` is `f(x, y)`
    - `10 %>% rnorm(mean = 5, sd = 1)` is `rnorm(10, mean = 5, sd = 1)`

6. Pipes are great for several function calls that operate on the same or maybe a couple datasets and result in a single output. Long chains of function calls should be broken up, and when you are dealing with many datasets or many outputs, these cannot be or should not be combined as it may actually lead to more confusing code. It will also not work out of the box or at all with some special functions (book has examples).

7. Additional operator allowing side-effects, like printing structure to check intermediate results, in the middle of transformations
```
df <- inner_join(u.data, u.user, "user_id") %T>%
  str %>% 
  inner_join(u.item, "movie_id") %T>% 
  str %>% 
  mutate(
    timestamp = as_datetime(timestamp), 
    release_date = dmy(release_date)
  ) %T>% 
  str %>% 
  select(-video_release_date)
```


## Functions

One value of functions is to capture repeatedly written and used blocks of code, potentially with varying arguments, to enable reuse and code simplification. It is also about extending R to do things it cannot do out of the box or with any package. Lastly, it is a good way to wrap your complicated code into larger logical units and make it better communicable.

1. A function is an object that has several components: name, arguments, body, return value, environment. Here is a generic example of creating a new function
```
name <- function(argument) {
  var <- code(argument)
  return(var)
}
```
The generic example does not do anything, but here is a simple working example:
```
test_miss <- function(column) {
  missingness <- is.na(column)
  test <- any(missingness)
  return(test)
}
```
This example takes a column, checks for missing values and returns TRUE if there are any missing values and FALSE if there are no missing values. Here are its components:
    - `test_miss` is the name
    - `col` is the argument
    - `{missingness <- is.na(col); test <- any(missingness)}` is the body, or where your code goes
    - return value is `test`
    - environmment is global; this is where you have been creating your data objects and what you see in the *environment* tab in Rstudio. Do not worry about this component too much as it is a more advanced feature of R.

2. Name functions as any other object. Avoid reserved names
    - Start with letters, but can also contain numbers, as well as underscores `_` and points `.` to separate letters. `sum_var`, `sum.var`, `sum1` are all valid names. You can add special symbols by placing the name of your function into backsticks ``.

3. Arguments are also named. These can have default values.
    - Same naming conventions
    - Arguments can have default values by setting an argument equal to some value:
    ```
    mean_na <- function(var, na = TRUE) {
      x <- mean(var, na.rm = na)
      return(x)
    }
    ```

4. Body contains all the computations you want to perform repeatedly. This may be quite complex by adding conditional statements
    - The braces `{}` contain all your code as if it was in a script or Rmarkdown chunk
    - You can make your function do different things depending on some conditions
    
    ```
    check_categories <- function(var) {
      if (is.character(var) || is.factor(var)) {
        return(TRUE)
      } else {
        return(FALSE)
      }
    }
    
    discretise <- function(var) {
      if (var < 0) {
        value <- "negative"
        return(value)
      } else if (var > 0) {
        value <- "positive"
        return(value)
      } else if (var == 0) {
        value <- "neutral"
        return(value)
      }
    }
    ```

5. Return value may be required inside conditional statements, but otherwise is not needed because by default your funciton will return the last computed result in the body.

6. Environments are an advanced topic, but one thing that is important, is that functions can use variables not passed as arguments or created in the body, but rather created in the same place as the function. This is not recommended as it can make your functions unpredictable.

7. Write all these functions and test them with example inputs:
    - Implement a function, with no arguments or code in the body
    - Implement a function that has one argument and it just returns the argument value
    - Implement a function that takes a vector of numbers and a single number to check whether a number is in the vector. It should return TRUE or FALSE
    - Implement a function that takes a vector and a single value of the same type to find all missing values in a vector and replace them with a single value; use `if_else` from *dplyr* to help with this
    - Implement a function that takes a vector and returns how many unique elements it has
    - Implement a function that takes a vector and returns the shortest string if a vector is type character and smallest number if a vector is type double or integer. You can use function `which` to find position of the minimum value, e.g. `which(3:1 == min(1:3))`
    - Implement a function that calculates average squared difference between two vectors of numerical values
 

## Vectors
Data frames are a very useful data structure for data analysis, however when programming it may be an overkill or inadequate. This chapter introduces other ways of representing data in R. We concentrate on vectors - basic data structures in R out of which everything else is build, even data frames.

Load `purrr` library to get some of the functions.

1. Two basic types of vectors, atomic vectors that are usually just called vectors, and recursive vectors, usually called lists. There are two main differences: each element of an atomic vector has the same type and atomic vectors are flat, while recursive vectors can contain elements of different types and have hierarchical structure.

2. Vectors have types of logical, integer, double, complex, character, raw
```
c(TRUE, FALSE, NA)                 #logical
c(3L, 1L, 99L, NA)                 #integer
c(1.0, 0.008, NA, Inf, -Inf, NaN)  #double
c(1+4i, 44+1i, NA, Inf, -Inf, NaN) #complex
c("hello", "there", NA)            #character
c(as.raw(0x00), as.raw(0xf4))      #raw
```
3. You can create vectors with the `c()` function - which is mostly manual and shown above - or various functions that generate vectors with special values in them.

4. Logical vectors are typically created with logical comparisons or read from a file:
```{r}
c(3L, 1L, 99L, NA) == 1L
c(3L, 1L, 99L, NA) > 5L
```

5. Integer vectors are often read from a file, but are also used for subsetting other vectors, which requires pattern generation. If created by hand, numbers need *L* as a suffix because the default numbers are double type:
```{r}
1
1L

1:3

rep(1:3, times = 3)
rep(1:3, each = 3)

seq(0, 12, by = 4)
seq(0, 12, length.out = 3)

seq_along(letters)
```

6. Double vectors are often read from a file or may be generated with a random number generator:
```{r}
rnorm(10, mean = 5, sd = 1)
```

7. You will not see complex vectors unless you are doing pretty advanced math or physics.

8. Character vectors are often read from files, but also created, as you saw before with the Strings chapter:
```{r}
library(stringr)
str_c("file", 1:10, ".csv", sep = "")
```

9. Raw vectors are also pretty rare and mostly used in more intricate programming.

9. Another way to create a vector is to coerce one into another.
```{r}
as.logical(c(0, 1, 1, 0))
as.integer(c("1", "2", "3"))
as.double(c("1.001", "9.2842"))
as.complex(c("1.001", "9.2842"))
as.character(c(TRUE, FALSE))
as.raw(0xf4)
```

10. Coercision may be implicit by a function as a preparation for the downstream code
```{r}
typeof(c(TRUE, 1L)) # these conversions follow the hierarchy from logical to character
typeof(c(TRUE, 1))
typeof(c(TRUE, "1"))

sum(1:10 < 5)
```

11. You can use `typeof` or special test functions to determine types:
```
is.logical(TRUE)
is.integer(1L)
is.double(1.0)
is.complex(1+1i)
is.character("1")
is.raw(as.raw(0x00))
```

12. Another useful property of vectors is `length`:
```{r}
x <- c(TRUE, FALSE, NA)
length(x)
```

13. Working with multiple vectors of different length will invoke recycling which is repeating the elements of the shorter vector to match the length of the longer vector:
```{r}
1:3 + 3 == 1:3 + c(3, 3, 3)
```

14. It may be useful to name vectors
```{r results='hold'}
c(a = 1, b = 2, c = 3)
setNames(1:3, c("one", "two", "three"))
```

15. It is often needed to pull out specific elements of vectors. For this we use `[]` syntax, with either position or name:
```{r}
c(1, 2, 3)[3]
c("one", "two", "three")[2:3]
c("one", "two", "three")[-1]
c("one", "two", "three", "one", "two", "three")[c(1, 3, 5)]
c(1, 2, 3, 4)[-(2:3)]
letters[TRUE]
letters[c(TRUE, FALSE)]
1[0]

x <- c(one = 1, two = 2, three = 3)
x[c("one", "two")]
```

16. Lists share many features of atomic vectors, like naming, but are more complex than atomic vector because they can contain mixed types and hierarchies. To create a list you use 
```{r}
list(1, 2, 3)
list(TRUE, 1L, 1, 1+1i, "one")
list(one = 1, two = 2, three = 3)
list(one = 1, two = list(three = 3, four = 4))
str(list(one = 1, two = list(three = 3, four = 4)))
```

17. Lists also have the basic properties of vectors:
```{r results='hold'}
typeof(list(1))
length(list(1, 2, 3, 4))
```

18. You can also subset lists to pull various elements, either by position or name:
```{r}
list(1, 2, 3)[1:2]
list(1, 2, list(3, 4))[[3]]
x <- list(one = 1, two = 2)
x["one"]
x$one
```

19. Vectors and lists can be enhanced with `attributes`. The most important ones are *names*, *dims* and *class*.
```{r results='hold'}
attributes(mtcars) #to check
attr(mtcars, "something") <- 1:3 #to set

```

20. Several data structures you have seen are actually build out of vectors and lists combined with special attributes:

```{r results='hold'}
typeof(lubridate::as_date(Sys.time()))
attributes(lubridate::as_date(Sys.time()))
```

## Iteration
Recall that functions can extend functionality of R in novel directions, and also help you avoid duplication or rewriting of the same code. Another technique with managing duplication is iteration: doing the same operation on different inputs, be it single numbers stored in a vector, or columns of a data frame, or multiple data frames stored in a list.

1. Consider a situation where you loaded multiple data sets into a list and you want to check their dimensions to determine where they can be combined into a single data frame.
```{r results='hold'}
datasets <- list(
  data1 = mtcars,
  data2 = ToothGrowth,
  data3 = WorldPhones,
  data4 = anscombe
)
dim(datasets$data1)
dim(datasets$data2)
dim(datasets$data3)
dim(datasets$data4)
```
This code is repetitive and requires multiple copy-paste combinations and is not efficient, is error-prone and its verbosity competes with the logic of the code.

2. To remove repetitiveness of known, finite length, we can use for loops:
```{r}
results <- vector("integer", 4)       # 1. output
for (i in seq_along(datasets)) {      # 2. sequence
  results[i] <- ncol(datasets[[i]])   # 3. body 
}
results
```

  - output: usually a vector or a list to store results. `results <- vector("integer", 4)` uses `vector` function that takes type of data and length of the vector to create.
  - sequence: usually a vector or a list that holds varying inputs. `i in seq_along(df)` says that `i` will take values in a sequence of numbers going from 1 to the length of `datasets`.
  - body: code that is called repetitively with varying inputs. `results[i] <- ncol(datasets[[i]])` is the code that is executed with different values of i. `ncol` determines the number of columns for each data set in `datasets` and stores it inside `results`. This translates to 
```
results[1] <- ncol(datasets[[1]])
results[2] <- ncol(datasets[[2]])
results[3] <- ncol(datasets[[3]])
results[4] <- ncol(datasets[[4]])
```
Do these examples. Think carefully about how each problem relates to components of the loop:
  - Apply function `sd`, which calculates a standard deviation, to each column of mtcars
  - Determine type of each column in ToothGrowth data
  
3. The pattern you have seen so far is to loop over indices and create a new data object, however there are several useful variants on it:
  - Modifying an existing data object
```{r}
normalize <- function(x) {
  x_norm <- (x - mean(x)) / sd(x)
  x_norm2 <- round(x_norm, 2)
  return(x_norm2)
}
mtcars2 <- mtcars
head(mtcars)
for (i in seq_along(mtcars2)) {
  mtcars2[[i]] <- normalize(mtcars2[[i]])
}
head(mtcars2)
```
  - Looping over values or names instead of indices
  
```
image_names <- dir(pattern = ".jpg")
image_data <- vector("list", 4)
par(mfrow = c(2, 2))

for (name in image_names) {
  img <- jpeg::readJPEG(name)
  plot(1, 1, xlim = c(1, 640), ylim = c(1, 480), axes = FALSE, xlab = NA, ylab = NA)
  rasterImage(img, 1, 1, 640, 480)
}
```

![](https://synikitin.github.io/analytics6600/img/iteration.png)

  - Handling outputs of unknown length
  
```
means <- c(0, 1, 2)
out <- vector("list", length(means))
for (i in seq_along(means)) {
  n <- sample(100, 1)
  out[[i]] <- rnorm(n, means[[i]])
}
out <- unlist(out)

```
  - Handling inputs of unknown length
```
while (condition) {
  body
}
```

4. Loop exercises:
  - input: vector of logical values
    body: checks for TRUE and saves the index in the output vector
    output: vector of integers representing positions of TRUE values
  - Create a function that takes a data frame, loops through its columns, calculates mean of each column, and returns a vector of double numbers
  
  
  
  
