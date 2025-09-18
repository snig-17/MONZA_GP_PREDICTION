import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Enable FastF1 cache
fastf1.Cache.enable_cache('cache')

# Load 2024 monza Grand Prix data
session = fastf1.get_session(2024, 'Monza', 'R')
session.load()

# Extract lap data
laps = session.laps[['Driver', 'LapTime']].copy()
laps = laps.dropna(subset=['LapTime'])  # Drop laps with no time recorded
laps['LapTime'] = laps['LapTime'].dt.total_seconds()  # Convert LapTime to seconds

#2025 qualifying data
qualifying_session = pd.DataFrame({
    'Driver': ['VER', 'NOR', 'PIA', 'LEC', 'HAM', 'RUS', 'ANT', 'BOR', 'ALO', 'TSU', 
               'BEA', 'HUL', 'SAI', 'ALB', 'OCO', 'HAD', 'STR', 'COL', 'GAS', 'LAW'],
    'QualifyingTime (s)': [78.792, 78.869, 78.982, 79.007, 79.124, 79.157, 79.200, 79.390, 
                          79.424, 79.519, 79.446, 79.498, 79.528, 79.583, 79.707, 79.917, 
                          79.948, 79.992, 80.103, 80.279]
})
# Merge lap data with qualifying data
data = qualifying_session.merge(laps, on='Driver')

# qualifying time as feature, lap time as target
x = data[['QualifyingTime (s)']]
y = data['LapTime']

if x.shape[0] == 0:
    raise ValueError("No data available for training. Please check the data loading process.")

# Train gradient boosting regressor
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=39)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,random_state=39)
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(qualifying_session[['QualifyingTime (s)']])
qualifying_session['PredictedLapTime (s)'] = predictions

#rank drivers by predicted lap time
qualifying_session = qualifying_session.sort_values(by='PredictedLapTime (s)').reset_index(drop=True)
qualifying_session['PredictedPosition'] = qualifying_session.index + 1

# Display results
print(qualifying_session[['Driver', 'QualifyingTime (s)', 'PredictedLapTime (s)', 'PredictedPosition']])
mae = mean_absolute_error(y_test, model.predict(X_test))
print(f'Mean Absolute Error: {mae:.3f} seconds')