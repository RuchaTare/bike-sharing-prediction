# src package

## Submodules

## src.logger module

Logging module for the application.

### src.logger.setup_logging()

Setup logging configurations for the application. Writes logs to a file and console.

## src.model_training module

Model training module

### src.model_training.linear_regression(data)

Train a linear regression model

### src.model_training.minmax_scale(data)

Minmax scale the data

### src.model_training.model_evaluation(data)

Evaluate the model

### src.model_training.rfe(data)

Perform Recursive Feature Elimination

### src.model_training.train_test_split(data, test_size)

Split the data into training and test set

## src.preprocessing module

Preprocess the data

### Functions

read_csv(file_path)
: Read csv file and return a pandas dataframe

drop_columns(data)
: Drop irrelevant columns

change_labels(data, config_data)
: Change labels of columns to more understandable labels as per the data dictionary

create_dummies(data)
: Convert the datatype of categorical columns and Create dummy variables for categorical columns

### src.preprocessing.change_labels(data, config_data)

Change labels of columns to more understandable labels as per the data dictionary

* **Parameters:**
  **data** (*pandas.DataFrame*) – The data to be processed

### src.preprocessing.create_dummies(data)

Convert the datatype of categorical columns and Create dummy variables for categorical columns

* **Parameters:**
  **data** (*pandas.DataFrame*) – The data to be processed
* **Returns:**
  The data with dummy variables created
* **Return type:**
  pandas.DataFrame

### src.preprocessing.drop_columns(data)

Drop irrelevant columns

* **Parameters:**
  **data** (*pandas.DataFrame*) – The data to be processed
* **Returns:**
  The data with irrelevant columns dropped
* **Return type:**
  pandas.DataFrame

### src.preprocessing.main()

Main function to preprocess the data

## src.utils module

This file contains utility functions that are used in the project.

### src.utils.read_csv(file_path)

Read csv file and return a pandas dataframe

* **Parameters:**
  **file_path** (*str*) – The path to the csv file
* **Returns:**
  The data from the csv file
* **Return type:**
  pandas.DataFrame

### src.utils.read_yaml(file_path)

Read yaml file and return the content

* **Parameters:**
  **file_path** (*str*) – The path to the yaml file
* **Returns:**
  The content of the yaml file
* **Return type:**
  dict

## Module contents
