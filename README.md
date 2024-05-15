## Taxi Trip Duration Prediction

This project involves predicting the duration of taxi trips in New York City using linear regression. The dataset includes information about taxi trips, such as pickup and dropoff locations, times, distances, and various fare components. The project demonstrates data processing, feature extraction, model training, and evaluation.

### Table of Contents
1. [Installation](#installation)
2. [Data Processing](#data-processing)
3. [Model Training](#model-training)
4. [Validation](#validation)
5. [Usage](#usage)


### Installation
To run this project, you need Python and the following libraries installed:

- pandas
- numpy
- seaborn
- matplotlib
- pyarrow
- scikit-learn

You can install the required libraries using pip:

```bash
pip install pandas numpy seaborn matplotlib pyarrow scikit-learn
```

### Data Processing
The data processing steps include:
1. **Loading the Data**: Reading Parquet files containing taxi trip data.
2. **Converting to Datetime**: Converting pickup and dropoff times to datetime objects.
3. **Calculating Duration**: Calculating the duration of each trip in minutes.
4. **Filtering Outliers**: Removing trips with durations less than 1 minute or greater than 60 minutes.
5. **Converting Categorical Columns**: Converting pickup and dropoff location IDs to string type.

### Model Training
1. **Feature Extraction**: Using `DictVectorizer` to transform categorical features into a format suitable for model training.
2. **Training the Model**: Training a linear regression model to predict trip duration.
3. **Evaluating the Model**: Calculating the Root Mean Squared Error (RMSE) on the training data.

### Validation
1. **Processing Validation Data**: Similar steps as data processing to prepare the validation dataset.
2. **Making Predictions**: Using the trained model to predict durations on the validation set.
3. **Evaluating Predictions**: Calculating RMSE on the validation set to assess model performance.

### Usage
Below is an example of how to use the provided code.

#### Loading and Processing Data
```python
import pandas as pd

# Read and process data
df = pd.read_parquet('yellow_tripdata_2023-01.parquet')

# Convert datetime columns
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

# Calculate trip duration
df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).apply(lambda td: td.total_seconds() / 60)

# Filter outliers
df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]

# Convert categorical columns to string
categorical = ['PULocationID', 'DOLocationID']
df[categorical] = df[categorical].astype(str)
```

#### Training the Model
```python
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

# Prepare training data
train_dicts = df[categorical].to_dict(orient='records')
dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)
y_train = df['duration'].values

# Train linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Evaluate on training data
from sklearn.metrics import mean_squared_error
y_pred = lr.predict(X_train)
rmse_train = mean_squared_error(y_train, y_pred, squared=False)
print("Training RMSE:", rmse_train)
```

#### Validating the Model
```python
def read_data(filename):
    df = pd.read_parquet(filename)
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).apply(lambda td: td.total_seconds() / 60)
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype(str)
    return df

# Read validation data
df_val = read_data('yellow_tripdata_2023-02.parquet')

# Prepare validation data
val_dicts = df_val[categorical].to_dict(orient='records')
X_val = dv.transform(val_dicts)
y_val = df_val['duration'].values

# Make predictions and evaluate
y_pred_val = lr.predict(X_val)
rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
print("Validation RMSE:", rmse_val)
```

