"""
Drop irrelevant columns
Change labels to numerical values
"""

weathersit_labels = {
    1 : 'clear',
    2 : 'cloudy',
    3 : 'light snow/rain',
    4 : 'heavy rain'
}
bike['weathersit'] = bike['weathersit'].replace(weathersit_labels)


weekday_labels = {
    0 : 'tuesday',
    1 : 'wednesday',
    2 : 'thursday',
    3 : 'friday',
    4 : 'saturday',
    5 : 'sunday',
    6 : 'monday'
}
bike['weekday'] = bike['weekday'].replace(weekday_labels)


mnth_labels = {
    1 : 'january',
    2 : 'february',
    3 : 'march',
    4 : 'april',
    5 : 'may',
    6 : 'june',
    7 : 'july',
    8 : 'august',
    9 : 'september',
    10 : 'october',
    11 : 'november',
    12 : 'december'
}
bike['mnth'] = bike['mnth'].replace(mnth_labels)


season_labels = {
    1 : 'spring',
    2 : 'summer',
    3 : 'fall',
    4 : 'winter'
}
bike['season'] = bike['season'].replace(season_labels)

bike['season']=bike['season'].astype('category')
bike['weekday']=bike['weekday'].astype('category')
bike['mnth']=bike['mnth'].astype('category')
bike['weathersit']=bike['weathersit'].astype('category')

bike=pd.get_dummies(bike,drop_first=True)

converting below columns into categorical variables
1. season
2. weekday
3. mnth
4. weathersit

Below columns are already mapped to 0 and 1 as we require
1. yr
2. holiday
3. workingday
