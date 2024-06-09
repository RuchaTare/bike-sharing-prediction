# Bike sharing demand prediction

## Problem Statement:

A US bike-sharing provider BoomBikes has recently suffered considerable dips in their revenues due to the ongoing Corona pandemic. The company is finding it very difficult to sustain in the current market scenario. So, it has decided to come up with a mindful business plan to be able to accelerate its revenue as soon as the ongoing lockdown comes to an end, and the economy restores to a healthy state.

Essentially the company wants :

- To understand the demand for shared bikes among the people after this ongoing quarantine situation ends across the nation due to Covid-19, by creating a linear model.
- To identify the variables affecting their revenues i.e. Which variables are significant in predicting the demand for shared bikes.
- To know the accuracy of the model, i.e. How well those variables describe the bike demands
- They have planned this to prepare themselves to cater to the people's needs once the situation gets better all around and stand out from other service providers and make huge profits.

## Data Description:

Original dataset : https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

Kaggle : https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset/data

Data Dictionary :
- instant: record index
- dteday : date
- season : season (1:springer, 2:summer, 3:fall, 4:winter)
- yr : year (0: 2018, 1:2019)
- mnth : month ( 1 to 12)
- holiday : weather day is holiday or not (extracted from [Web Link])
- weekday : day of the week
- workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
- weathersit :
- 1: Clear, Few clouds, Partly cloudy, Partly cloudy
- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
- temp : Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)
- atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
- hum: Normalized humidity. The values are divided to 100 (max)
- windspeed: Normalized wind speed. The values are divided to 67 (max)
- casual: count of casual users
- registered: count of registered users
- cnt: count of total rental bikes including both casual and registered

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

## Model Training:

### Design Decisions:

## Linear Regression Results:

### Model Performance:

## Future Work

