#################### LINEAR MODELS ###########################

############################################ GENERALIZED LINEAR MODEL #########################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, explained_variance_score, max_error
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Gaussian, Poisson, Binomial
from statsmodels.genmod.families.links import Identity, Log
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import train_test_split

# Load the data from Excel
df = pd.read_excel('C:/Users/Patyc/OneDrive/Desktop/Dissertation/Data/Merged_File_v7_skin.xlsx')


############ DAYS #############

# Convert Screening_date to datetime
df['Screening_date'] = pd.to_datetime(df['Screening_date'], errors='coerce')

# Create a DataFrame with all dates in the range
date_range = pd.date_range(start=df['Screening_date'].min(), end=df['Screening_date'].max())
all_dates_df = pd.DataFrame(date_range, columns=['Screening_date'])

# Aggregate screening counts per day
daily_counts = df.groupby('Screening_date').size().reset_index(name='screening_count')

# Merge with all dates to ensure all days are accounted for
all_dates_df = all_dates_df.merge(daily_counts, on='Screening_date', how='left')

# Fill missing screening counts with 0
all_dates_df['screening_count'] = all_dates_df['screening_count'].fillna(0)

# Merge back with the original data on Screening_date
df = df.merge(all_dates_df, on='Screening_date', how='right')

# print(df.columns)

# Define target and features
y = df['screening_count']
X = df.drop(columns=['Location', 'Screening_type', 'Birth date', 'Profession', 'Education', 
                     'If so, which one?', 'If so, who?', 'If so, which one?2', 
                     'Skin_observations', 'Screening_date', 'screening_count'])


##### Preprocessing

# Impute numeric features
numeric_features = ['Age']  # Adjust this based on your actual numeric features
num_imputer = SimpleImputer(strategy='median')
X[numeric_features] = num_imputer.fit_transform(X[numeric_features])

# Impute categorical features
categorical_features = ['Gender', 'Personal_cancer_history', 'Family_cancer_history', 'Sun_exposure', 
                        'Body_signs', 'Phototype', 'Skin_diagnosis']  # Adjust based on your actual categorical features
cat_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_features] = cat_imputer.fit_transform(X[categorical_features])

# Scale numeric features
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Encode categorical features
encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
encoded_categories = encoder.fit_transform(X[categorical_features])
encoded_df = pd.DataFrame(encoded_categories.toarray(), columns=encoder.get_feature_names_out(categorical_features))

# Concatenate encoded features with the original DataFrame
X = pd.concat([X.drop(columns=categorical_features), encoded_df], axis=1)

# Split the data into train and test sets based on a specific date
split_date = '2024-03-01'
train_data = df[df['Screening_date'] < split_date]
test_data = df[df['Screening_date'] >= split_date]

X_train = X[df['Screening_date'] < split_date]
y_train = y[df['Screening_date'] < split_date]
X_test = X[df['Screening_date'] >= split_date]
y_test = y[df['Screening_date'] >= split_date]

# Add a constant to the model (intercept)
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Fit the GLM model with different family distributions and link functions
families = {
    'Gaussian': Gaussian,
    'Poisson': Poisson,
    'Binomial': Binomial
}
link_functions = {
    'identity': Identity(),
    'log': Log()
}

# Initialize an empty list to store the results
results = []

for family_name, family in families.items():
    for link_name, link in link_functions.items():
        print(f'Fitting GLM with {family_name} family and {link_name} link function...')
        model = sm.GLM(y_train, X_train, family=family(link=link))
        result = model.fit()
        
        y_pred = result.predict(X_test)
        
        # Calculate the metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Append the results to the list
        results.append({
            'Family': family_name,
            'Link': link_name,
            'MAE': mae,
            'MSE': mse,
            'R^2': r2
        })

# Create a DataFrame from the results list
results_df = pd.DataFrame(results)

# Display the results DataFrame
print(results_df)

# ## results

#      Family      Link         MAE           MSE       R^2
# 0  Gaussian  identity  137.590488  27656.408437 -1.643345
# 1  Gaussian       log  138.764798  27991.883062 -1.675409
# 2   Poisson  identity  136.369702  27328.086741 -1.611965
# 3   Poisson       log  137.927762  27767.972644 -1.654008
# 4  Binomial  identity  137.590488  27656.408437 -1.643345
# 5  Binomial       log  138.764798  27991.883085 -1.675409


#### ADDING DATE-RELATED VARIABLES #####

# Create date-related features
df['day_of_week'] = df['Screening_date'].dt.dayofweek
df['month'] = df['Screening_date'].dt.month
df['year'] = df['Screening_date'].dt.year
df['day_of_year'] = df['Screening_date'].dt.dayofyear
df['week_of_year'] = df['Screening_date'].dt.isocalendar().week
df['is_weekend'] = df['Screening_date'].dt.dayofweek >= 5
df['quarter'] = df['Screening_date'].dt.quarter
# Lag Features
df['lag_1'] = df['screening_count'].shift(1)
df['lag_7'] = df['screening_count'].shift(7)
df['lag_14'] = df['screening_count'].shift(14)
df['lag_21'] = df['screening_count'].shift(21)
df['lag_30'] = df['screening_count'].shift(30)
# Add rolling statistics
df['rolling_mean_7'] = df['screening_count'].rolling(window=7).mean()
df['rolling_sum_7'] = df['screening_count'].rolling(window=7).sum()
df['rolling_std_7'] = df['screening_count'].rolling(window=7).std()

df['rolling_mean_14'] = df['screening_count'].rolling(window=14).mean()
df['rolling_sum_14'] = df['screening_count'].rolling(window=14).sum()
df['rolling_std_14'] = df['screening_count'].rolling(window=14).std()

df['rolling_mean_30'] = df['screening_count'].rolling(window=30).mean()
df['rolling_sum_30'] = df['screening_count'].rolling(window=30).sum()
df['rolling_std_30'] = df['screening_count'].rolling(window=30).std()

# create an 'is_holiday' feature
holidays = ['2022-01-01', '2022-03-01', '2022-04-15', '2022-04-17', '2022-04-25', '2022-05-01', '2022-06-10', '2022-06-16',
            '2022-06-13', '2022-08-15', '2022-09-07', '2022-10-05', '2022-11-01', '2022-12-01', '2022-12-08', '2022-12-25',
            '2023-01-01', '2023-04-09', '2023-05-01', '2023-12-25', '2023-02-21', '2023-04-25', '2023-06-13', '2023-08-15',
            '2023-11-01', '2023-10-05', '2023-04-07', '2023-12-01', '2023-12-08', '2023-06-10', '2024-01-01', '2024-02-13',
            '2024-03-29', '2024-03-31', '2024-04-25', '2024-05-01', '2024-05-30', '2024-06-10']
holidays = pd.to_datetime(holidays)
df['is_holiday'] = df['Screening_date'].isin(holidays).astype(int)

print(df.columns)

# ### preprocessing

# Define numeric and categorical features
numeric_features = ['Age', 'day_of_week', 'month', 'year', 'day_of_year', 'week_of_year', 'quarter',
                    'lag_1', 'lag_7', 'lag_14', 'lag_21', 'lag_30', 'rolling_mean_7', 'rolling_sum_7', 'rolling_std_7',
                    'rolling_mean_14', 'rolling_sum_14', 'rolling_std_14',
                    'rolling_mean_30', 'rolling_sum_30', 'rolling_std_30']
categorical_features = ['Gender', 'Personal_cancer_history', 'Family_cancer_history', 'Sun_exposure', 
                        'Body_signs', 'Phototype', 'Skin_diagnosis', 'is_weekend', 'is_holiday']

# Impute missing values
num_imputer = SimpleImputer(strategy='median')
df[numeric_features] = num_imputer.fit_transform(df[numeric_features])

cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_features] = cat_imputer.fit_transform(df[categorical_features])

# Define target and features
y = df['screening_count']

# Define the columns to drop
columns_to_drop = ['Location', 'Screening_type', 'Birth date', 'Profession', 'Education', 
                   'If so, which one?', 'If so, who?', 'If so, which one?2', 
                   'Skin_observations', 'Screening_date', 'screening_count']

# Drop the specified columns
X = df.drop(columns=columns_to_drop, errors='ignore')  # Using errors='ignore' to handle any column not found

# Verify the columns after dropping
# print(X.columns)

# Encode categorical features
encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
encoded_categories = encoder.fit_transform(df[categorical_features])
encoded_df = pd.DataFrame(encoded_categories.toarray(), columns=encoder.get_feature_names_out(categorical_features))

# Concatenate encoded features with the original DataFrame
X = pd.concat([X.drop(columns=categorical_features), encoded_df], axis=1)

# Scale numeric features
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

print(X.columns)

# Split the data into train and test sets based on a specific date
split_date = '2024-03-01'
train_data = df[df['Screening_date'] < split_date]
test_data = df[df['Screening_date'] >= split_date]

X_train = X[df['Screening_date'] < split_date]
y_train = y[df['Screening_date'] < split_date]
X_test = X[df['Screening_date'] >= split_date]
y_test = y[df['Screening_date'] >= split_date]


# Split the data into training and testing sets, ensuring a representative split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Add a constant to the model (intercept)
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Ensure the test set has the same columns as the training set
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Fit the GLM model with different family distributions and link functions
families = {
    'Gaussian': Gaussian,
    'Poisson': Poisson,
    'Binomial': Binomial
}
link_functions = {
    'identity': Identity(),
    'log': Log()
}

# Initialize lists to store the results and predictions
results = []
predictions = {}

# Assuming 'families' and 'link_functions' dictionaries are already defined
for family_name, family in families.items():
    for link_name, link in link_functions.items():
        print(f'Fitting GLM with {family_name} family and {link_name} link function...')
        try:
            model = sm.GLM(y_train, X_train, family=family(link=link))
            result = model.fit()
            
            y_pred = result.predict(X_test)
            
            # Calculate the metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Append the results to the list
            results.append({
                'Family': family_name,
                'Link': link_name,
                'MAE': mae,
                'MSE': mse,
                'R^2': r2
            })
            
            # Store predictions
            predictions[f'{family_name}_{link_name}'] = (y_test, y_pred)

        except Exception as e:
            print(f'Error fitting {family_name} with {link_name}: {e}')

# Create a DataFrame from the results list
results_df = pd.DataFrame(results)
print(results_df)

# Plotting actual vs. predicted for each model in separate subplots
plt.figure(figsize=(18, 12))

# Plot actual vs. predicted for each model in separate subplots
for i, (key, (y_test, y_pred)) in enumerate(predictions.items(), 1):
    plt.subplot(2, 3, i)
    plt.plot(y_test.values, label='Actual', color='blue')
    plt.plot(y_pred, label='Predicted', alpha=0.7, color='orange')
    plt.title(key.replace('_', ' - '))
    plt.xlabel('Index')
    plt.ylabel('Screening Count')
    plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

# Plot 1: Distribution of Training and Testing Data
plt.subplot(1, 3, 1)
plt.hist(y_train, bins=20, alpha=0.7, label='Train')
plt.hist(y_test, bins=20, alpha=0.7, label='Test')
plt.title('Distribution of Training and Testing Data')
plt.legend()

# Plot 2: Distribution of Predicted Data
plt.subplot(1, 3, 2)
plt.hist(y_pred, bins=20, alpha=0.7, label='Predicted')
plt.title('Distribution of Predicted Data')
plt.legend()

plt.tight_layout()
plt.show()

# plot residuals

import matplotlib.pyplot as plt

# Plotting residuals for each family and link function combination
plt.figure(figsize=(18, 12))
for i, (key, (y_test, y_pred)) in enumerate(predictions.items(), 1):
    residuals = y_test.values - y_pred
    plt.subplot(2, 3, i)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.title(f'Residuals: {key.replace("_", " - ")}')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')

plt.tight_layout()
plt.show()

####### Gaussian identity check overfitting
# Define the GLM model
family = Gaussian
link = Identity()

# Fit the model on the training data
glm_model = sm.GLM(y_train, X_train, family=family(link=link))
result_train = glm_model.fit()

# Predict on the training data
y_pred_train = result_train.predict(X_train)

# Evaluate the model on the training data
mae_train = mean_absolute_error(y_train, y_pred_train)
mse_train = mean_squared_error(y_train, y_pred_train, squared=False)
r2_train = r2_score(y_train, y_pred_train)
print(f'GLM (Training) - MAE: {mae_train}, MSE: {mse_train}, R^2: {r2_train}')

# Plot actual vs. predicted for the training data
plt.figure(figsize=(8, 6))
plt.plot(y_train.values, label='Actual', color='blue')
plt.plot(y_pred_train, label='Predicted', alpha=0.7, color='orange')
plt.title('GLM (Training) - Actual vs. Predicted')
plt.xlabel('Index')
plt.ylabel('Screening Count')
plt.legend()
plt.show()

# Plot residuals for the training data
residuals_train = y_train.values - y_pred_train
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_train, residuals_train, alpha=0.5)
plt.axhline(0, color='r', linestyle='--')
plt.title('Residuals (Training Data)')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.show()

# ### results without lags and rolling statistics

#      Family      Link         MAE           MSE       R^2
# 0  Gaussian  identity  171.914888  38247.224745 -2.655595
# 1  Gaussian       log  242.429233  69157.247397 -5.609914
# 2   Poisson  identity  193.058322  46414.881519 -3.436243
# 3   Poisson       log  242.723620  69310.370959 -5.624549
# 4  Binomial  identity  171.933257  38253.880273 -2.656231

# ### results with 1, 7, 14, 21 and 30 lags and rolling statistics 7, 14 and 30

#      Family      Link        MAE          MSE       R^2
# 0  Gaussian  identity   4.507133   239.868686  0.970035
# 1  Gaussian       log  17.468334   886.795122  0.889218
# 2   Poisson  identity   7.420804   266.377984  0.966723
# 3   Poisson       log  19.702554  1192.307802  0.851052
# 4  Binomial  identity   4.508855   239.891855  0.970032



