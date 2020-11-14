# ADM-Homework-2

This repository is structured in the following fashion:

- **Main.ipynb:** Jupyter notebook in which the results of all research questions can be found

- **functionality_new.py:** Code in which all functions are stored. The jupyter notebook feeds from this python file.

- **tests:** In this folder one can find a few basic tests performed over the compltete data sets (we have generated 
small data sets over which we were able to verify the correct funcioning of our code)

We have worked with all available data sets for most of the questions and justified the results obtained. Most of the 
research questions also generate a complementary data frame over which we could perform more detailed analysis if 
required.

We acknowledge some weak points of our code:

1. We could have generated a set of different tables to use recycle data frames throughout the notebook (in our 
current implementation we pre-process the data for every single research question)

2. We could have tried to implement the chunk_size loop as a wrapper function over each individual function

Despite these potential improvements we were able to properly work with a large set of data frames without the need of 
external help (e.g. aws)