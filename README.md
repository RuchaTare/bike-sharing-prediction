# Bike sharing demand prediction
## Author
[Rucha Tare](www.linkedin.com/in/ruchastare)

## Problem Statement:

A US bike-sharing provider BoomBikes has recently suffered considerable dips in their revenues due to the ongoing Corona pandemic. The company is finding it very difficult to sustain in the current market scenario. So, it has decided to come up with a mindful business plan to be able to accelerate its revenue as soon as the ongoing lockdown comes to an end, and the economy restores to a healthy state.

Essentially the company wants :

- To understand the demand for shared bikes among the people after this ongoing quarantine situation ends across the nation due to Covid-19
- To identify the variables affecting their revenues i.e. Which variables are significant in predicting the demand for shared bikes.
- To know the accuracy of the model, i.e. How well those variables describe the bike demands
- They have planned this to prepare themselves to cater to the people's needs once the situation gets better all around and stand out from other service providers and make huge profits.

## Data Description:

[Find the original dataset here](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) with its [Data Dictionary](data/data%20_dictionary.txt)

## Architecture:
- Monolith
## Repository Structure
## How to run
## Tech stack


## Key findings from EDA:
- Categorical Columns : season, yr, month, holiday, weekday, workingday, weathersit
- Numerical Columns : cnt, windspeed, hum, temp
- Column "cnt" is our target variable with mean 4508
- No null values in the any of the columns.
- No duplicate rows in the data set.
- No columns that has only one value overall for all records.
- Columns that could be dropped are 'instant','casual','registered','atemp','dteday'
    - instant : This columns is just a counter that we donot require for analysis
    - casual and regsitered : Both of these columns are to show the type of users , this is irrelevant to our analysis
    - atemp : atemp is adjusted temperature , since we already have temparature we will drop this column
    - dteday : Since we already have day, month and year we donot need date

- Season 3(fall) has the highest number of bikes rented about 5500
- High bike share from June to Sept and falls before and after those months. June is highest followed by september
- The year 2019 had a higher count of users as compared to the year 2018
- The bike demand is almost constant throughout the week.
- Non working days have more number of bikes rented
- The demand is less on holidays and more on non holidays
- Approximately 5000 bikes are rented on a weathersit 1 day being the highest followed by about 4000 weathersit 2 and about 2000 on weathersit3 we dont see any rentals on weathersit 4

## Preprocessing:
 - Dropped columns that seemed irrelevant from the EDA. i.e "instant", "dteday", "yr", "atemp", "windspeed", "casual", "registered"
 - For better human readability change the labels of "weekday", weathersit, mnth, season. For Example :   1: "january",   1: "spring", etc based on the data dictionary.
 - Scaled the numerical columns using min max scaler
 - All categorical columns were one hot encoded.

## Model Training:
- Used K vold cross validation to select the best model among a list of regression models based on Mean RMSE.
- Random Forest Regressor has slight edge with lower mean RMSE, although Extra tree Regressor has better Std RMSE. Since we want more accurate than more robust we chose Random Forest Regressor. It would be a good experiment to try Extra Trees Regressor to check how it performs on consistency and generalization.
- For Feature selection, RFECV was used, that gave Optimal features = 55.

### Model Performance:
- The trained model has R2 = 0.97 which is slightly overfitted compared to test RMSE = 0.83. Tried a PCA for dimensionality reduction over selected features from RFECV, but the model did not perform better infact R2 in test dropped to 0.80 while R2 in train stayed the same.
- It would be a future work to try other feature selection methods like VIF or use ridge to reduce overfitting.

## Future Work
- Implement Autodocs
- A Streamlit app
- Try extra trees regressor
- Reduce overfitting
## Contribution
If you want to contribute to enhancements and changes, please open an issue first to discuss what you would like to change or contribute.PRs welcome
## License