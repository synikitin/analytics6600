+++
abstract = ""
abstract_short = ""
authors = []
date = "2017-01-30"
image = ""
image_preview = ""
math = true
publication_types = []
publication = ""
publication_short = ""
selected = false
title = "Exploratory Data Analysis"
url_code = ""
url_dataset = "data/movies.csv"
url_pdf = ""
url_project = "https://fivethirtyeight.com/features/the-dollar-and-cents-case-against-hollywoods-exclusion-of-women/"
url_slides = ""
url_video = ""
+++

-----

## **Learning Objectives**
- Package loading
- Data import
- Data clean up
- Data manipulation
- Exploratory visualization

## **Context**
This assignment is about exploratory data analysis (EDA) and is based around a fivethirtyeight article that is a good example of it (under project button). EDA is about going through many question-answer cycles bridged by numerical summaries and visualizations of data. The first part of the assignment is to read the article while noting what kind of statistics and plots were chosen by the authors; the other part of the assignment will be to reproduce these and build on them.

## **Data Import**

Next, use dataset button to get data. I suggest creating a separate folder for this assignment and moving data there. Read [section of 8.4](http://r4ds.had.co.nz/workflow-projects.html) about setting up a project folder with RStudio which is a good practice in managing files.

Open a new R markdown file by going `File -> New File -> R markdown`. Once it opens, do `File -> Save As` to save it to your folder. Assuming you have created the project folder and data is in it, to get data into R you need to run

```
library(readr)
df <- read_csv("movie.csv")
```

Type **df** into console to test that data loaded properly. Also, use `str(df)` to see column names, what types of data you have in various columns and some example values. 

## **Data cleaning**

Next, we should clean up a few names and types using dplyr functions:

1. Use `rename` function to remove **$** and empty spaces from variables that have them. You will need to use backsticks, for example **x = \`budget_2013$\`**, to have proper syntax for rename. Save the data frame with corrected names as **df** or with some other name.
2. After correcting names, we need to also fix data types. Output of `str(df)` shows that, say **budget_2013$**, has type `chr` which represents text data, without much sense. Use `mutate` and `as.double` to convert all variables that are by their nature numerical, but are stored as text.

## **Test and Budget Statistics**

Now, lets get some basic information about movies and budget:


- Calculate percentage of movies passing the test in the period of 1970 - 1980 using **clean_test** variable. Use `count` function from dplyr to get you started and basic arithmetic for the rest. Is the number below 50? Do the same calculation but for 2000 - 2010. How did the number change?
- Switching to budget, first filter all movies from 1970 - 1989 from your data and save as an intermediate result. Then, calculate inflation-adjusted median budget for all movies and also separate medians for each unique value of **clean_test** variable. How do group-specific medians compare to the overall median? Do author's explanations provide a satisfying explanation of these numbers or do you have some additional factors you have in mind? Note you may get a warning about missing values; you can remove missing values during calculation by adding `na.rm = TRUE` to median.

## **Sales Statistics**

Next, we will look at returns:

- Lets add a few new variables to `df`. Note that **intgross** stands for worldwide gross sales. Add variables that represent international only gross sales, worldwide return, international return, and domestic return on investment. 
- Summarise returns by calculating median total return and median total returns grouped by outcome of Bechdel test. What pattern do you see?
- Next, select international only gross and clean_test variables followed by creating a new variable that only says "international". Save this as a separate data frame. Repeat the same computations to create a data frame for "domestic", and then combine the two into a new data frame using `bind_rows`. Calculate median gross sales for combination of clean_test result and origin of sales by using both variables for grouping. What can you say about relation of passing the test and market on gross sales?

## **Visualizations**

The last part of the assignment is to roughly reproduce the charts in the article.

- We start with the stacked bar chart. First, add a new variable to the data frame you made in part 2 when you filtered earlier years out. We need a variable representing 5 year periods. Load ggplot2 and use cut_width(year, width = 5, boundary = 1970) to create a new variable. Then use ggplot2 to create a rough plot; all I want to see is stacked bars for each year period where each bar represents one of the possible values of clean_test. Hint: play with fill aesthetic and position argument. Read over help file for geom_bar.
- Next, the median budget chart. Use summary data from part 2. Hint: remember about coordinate system to figure out the flipping of bars.
- Lastly, use data from part 5 to make the chart about gross sales by market and test result. Hint: faceting will be helpful here. 

Remember to place all your results in R markdown. Good luck!


