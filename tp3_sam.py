# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:52:02 2023

@author: Samuel
"""

from pandas_datareader import data as pdr
import numpy as np
import yfinance as yf
yf.pdr_override() 
import pandas as pd
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import skew, kurtosis
import seaborn as sns
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

'''
iShares ESG Aware MSCI USA ETF - ESGU
iShares ESG Aware U.S. Aggregate Bond ETF - EAGG
iShares ESG Aware MSCI EM ETF - ESGE
iShares ESG Aware MSCI USA Small-Cap ETF - ESML
iShares ESG Aware 1-5 Year USD Corporate Bond ETF - SUSB
iShares ESG Aware MSCI EAFE ETF - ESGD
iShares 1-3 Year Treasury Bond ETF - SHY
iShares MSCI USA ESG Select ETF - SUSA
iShares U.S. Treasury Bond ETF - GOVT
iShares MBS ETF - MBB
iShares ESG Aware USD Corporate Bond ETF - SUSC
'''

api_key = '496662ff482d4f6b028d05fa48bfecbb'

ticker_list=['ESGU', 'EAGG', 'ESGE', 'ESML', 'SUSB', 'ESGD', 'SHY', 'SUSA' , 'GOVT', 'MBB', 'SUSC']


# Functions ----------------------------------------------------------------------------

def import_fred(tick_):
    # Set the start and end dates for the data
    start_date = datetime.datetime(2000, 1, 1)
    end_date = datetime.datetime(2023, 10, 31)   
    # Fetch data from FRED
    data = pdr.get_data_fred(tick_, start_date, end_date)
    return data

# Importing 1 month Risk free rate 
monthly_rf= import_fred("GS1M")
monthly_rf.index = monthly_rf.index.to_period('M')
monthly_rf = monthly_rf.apply(lambda x: pd.Series(x / 100))

# Tickers list
# We can add and delete any ticker from the list to get desired ticker live data

def getData(ticker):
    data = yf.download(ticker)["Close"] 
    # Compute daily returns
    data = data.pct_change()    
    # Compute monthly returns
    monthly_rets = data.resample('M').agg(lambda x: (x + 1).prod() - 1)
    monthly_rets.index = monthly_rets.index.to_period('M')

    return monthly_rets

def import_csv_with_date_index(csv_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Assuming the first column is the date column
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')  # Convert the first column to datetime

    # Set the first column as the index
    df.set_index(df.columns[0], inplace=True)

    return df

def TE_fct(df_1, df_2):
    # Calculate tracking errors
    tracking_errors = df_1 - df_2

    # Convert 'Period' values to strings
    tracking_errors.index = tracking_errors.index.strftime('%Y-%m')

    # Extract years for the x-axis labels
    years = tracking_errors.index.str[:4]

    # Create a grid of subplots
    num_plots = len(tracking_errors.columns)
    num_cols = 3
    num_rows = (num_plots - 1) // num_cols + 1

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 10), sharex=True)

    # Flatten the 2D array of subplots for easier indexing
    axes = axes.flatten()

    for i, ticker in enumerate(tracking_errors.columns):
        # Plot tracking error for each index
        axes[i].plot(tracking_errors.index, tracking_errors[ticker], label=f'Tracking error of index {ticker}', color='blue')
        axes[i].set_title(f'Tracking Error for {ticker}')
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel('Tracking Error')
        axes[i].axhline(y=0, color='red', linestyle='--', linewidth=1.5)  # Add a red dotted line at y=0


        # Set x-axis ticks and labels for every second year
        axes[i].set_xticks(tracking_errors.index[::24])  # Set x-axis ticks every 24 months (every second year)
        axes[i].set_xticklabels(years[::24], rotation=45)  # Set x-axis labels with every second year, rotated for better visibility

    # Hide empty subplots if the number of subplots is less than the total number of subplots in the grid
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Adjust layout
    plt.tight_layout()
    fig.suptitle('Tracking Errors Comparison', fontsize=16).set_y(1.01)

    # Adjust the layout to prevent clipping of labels
    plt.gcf().autofmt_xdate()
    plt.xticks(rotation=45, ha='right')
    plt.show()


def plot_time_series_grid(dataframe, title='Benchmark Indexes Time Series', y_axis_label='Return', 
                          num_cols=3, figsize=(15, 10), rotation=45, fontsize=10):
    # Convert 'Period' values to numeric
    dataframe.index = dataframe.index.to_timestamp()

    # Determine the number of rows and columns for the subplots
    num_plots = len(dataframe.columns)

    # Determine the number of rows and columns for the subplots
    num_cols = 3
    num_rows = (num_plots - 1) // num_cols + 1

    # Create subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize, sharex=True)

    # Flatten the 2D array of subplots for easier indexing
    axes = axes.flatten()

    # Iterate over each column in the DataFrame and each subplot
    for i, (column, ax) in enumerate(zip(dataframe.columns, axes)):
        # Plot each column on a separate subplot
        ax.plot(dataframe.index, dataframe[column], label=column)
        ax.set_title(column)
        ax.set_xlabel('Date')  # Set x-axis label on each subplot
        ax.set_ylabel(y_axis_label)  # Set y-axis label on each subplot

    # Hide empty subplots if the number of subplots is less than the total number of subplots in the grid
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Adjust layout
    plt.tight_layout()
    fig.suptitle(title, fontsize=16).set_y(1.01)

    plt.show()


# Code -------------------------------------------------------------------------------------

etf_monthly_rets = getData(ticker_list)
etf_monthly_rets.replace(0, np.nan, inplace=True)


df_index = pd.DataFrame()

for i in ticker_list:
    csv_file_path = (f'data/{i}_index.csv')
# Example usage
    stored_serie = import_csv_with_date_index(csv_file_path)
    
    # Ensure the indices are unique
    df_index = pd.concat([df_index, stored_serie.loc[~stored_serie.index.duplicated(keep='first')]], axis=1)


df_index.columns = ticker_list
df_index_rets = df_index.pct_change()   
df_index_rets = df_index_rets[df_index_rets.index.notna()]
df_index_rets = df_index_rets.resample('M').agg(lambda x: (x + 1).prod() - 1)
df_index_rets.index = df_index_rets.index.to_period('M')
df_index_rets.replace(0, np.nan, inplace=True)


common_index = df_index_rets.index.intersection(etf_monthly_rets.index)
common_index = common_index[
    common_index.isin(df_index_rets.index) &
    common_index.isin(etf_monthly_rets.index) &
    (pd.notna(df_index_rets.loc[common_index]).any(axis=1)) &
    (etf_monthly_rets.loc[common_index] != 0).any(axis=1)
]

# Select data for common non-zero and non-NaN rows
df1_common = df_index_rets.loc[common_index]
df2_common = etf_monthly_rets.loc[common_index]

# Initialize an empty DataFrame to store tracking errors
diff_df = pd.DataFrame(index=common_index, columns=df_index_rets.columns)
TE_list = pd.Series(index=ticker_list)

# Iterate over each column in the DataFrames
for column in diff_df.columns:
    # Identify non-zero and non-NaN values for the current column
    valid_indices = (df1_common[column] != 0) & pd.notna(df1_common[column]) & \
                    (df2_common[column] != 0) & pd.notna(df2_common[column])
    for rows in valid_indices.index:
    
    # Compute tracking error for the current column only for valid indices
        diff = (df1_common.loc[rows, column] - df2_common.loc[rows, column])
        # Fill in the tracking_errors_df DataFrame
        diff_df.loc[rows, column] = diff
        
    TE_list[column] = np.std(diff_df[column]) 
'''
Printing the TE for each ETF

    # Display the tracking_errors_df DataFrame
    print(f'{column} Tracking Error: {TE_list[column]}')
    print('', sep='\n')
'''

# Convert non-numeric values to NaN
tracking_errors_df_numeric = diff_df.apply(pd.to_numeric, errors='coerce')

'''
# Plots of the time series difference between ETF and Benchmark index


num_rows = 4
num_cols = 3
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 10), sharex=True)

# Flatten the 2D array of subplots for easier indexing
axes = axes.flatten()

for i, ticker in enumerate(tracking_errors_df_numeric.columns):
    # Convert Period index to datetime
    x_values = tracking_errors_df_numeric.index.to_timestamp()

    # Plot the actual and predicted time series
    axes[i].plot(x_values, tracking_errors_df_numeric[ticker], label=f'Tracking error of index {ticker}', color='blue')
    axes[i].set_title(f'Tracking Error for {ticker}')
    axes[i].set_xlabel('Date')
    axes[i].set_ylabel('Tracking Error')
    axes[i].axhline(y=0, color='red', linestyle='--', label='Zero Line')


# Adjust layout
plt.tight_layout()
fig.suptitle('ETF and benchmark Tracking error', fontsize=16).set_y(1.05)
plt.show()

'''






'''

TE_fct(df_index_rets, etf_monthly_rets)

'''






'''
knn = [5,10,25,50]

for k in knn: 
    imputer = KNNImputer(n_neighbors=k)  # You can adjust the number of neighbors as needed
    TE_fct(pd.DataFrame(imputer.fit_transform(df_index_rets), columns=df_index_rets.columns, index=df_index_rets.index),
           pd.DataFrame(imputer.fit_transform(etf_monthly_rets), columns=etf_monthly_rets.columns, index=etf_monthly_rets.index))
'''

# Create a KNNImputer instance
imputer = KNNImputer(n_neighbors=5)  # You can adjust the number of neighbors as needed
# Fit the imputer on the data and transform the DataFrame
df_imputed_index = pd.DataFrame(imputer.fit_transform(df_index_rets), columns=df_index_rets.columns, index=df_index_rets.index)
df_imputed_etf = pd.DataFrame(imputer.fit_transform(etf_monthly_rets), columns=etf_monthly_rets.columns, index=etf_monthly_rets.index)



'''
TE_fct(df_imputed_index, df_imputed_etf)
'''

#----------------------------------------- Analysis of the etfs benchmark indexes monthly returns -------------------------------------------------------


'''
                                 Input Data Analysis 
                                 
#==============================================================================================================================
# Clustermap of correlation of benchmark indexes returns
imputed_index_corr = pd.DataFrame(df_imputed_index).corr()

sns.clustermap(imputed_index_corr, annot=True, square=True)
plt.suptitle('Clustermap of the correlation between returns of the Benchmark Indexes').set_y(1.01)

plt.show();
#==============================================================================================================================



#==============================================================================================================================
# Grid of histograms of the Indexes monthly returns 

# Get the list of all column names
columns = df_imputed_index.columns

# Determine the number of rows and columns for the grid
num_rows = 3
num_cols = 3

# Create a figure and subplots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, 16))
fig.suptitle("Histograms of Indexes monthly returns", fontsize=18)

# Flatten the axs array for easier indexing
axs = axs.flatten()

# Loop through the columns and plot histograms
for i, column in enumerate(columns):
    if i >= num_rows * num_cols:
        break  # Exit the loop if you exceed the number of subplots
    
    ax = axs[i]
    df_imputed_index[column].hist(ax=ax, bins=50)  # You can adjust the number of bins as needed
    ax.set_title(column)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    print(df_imputed_index[column].describe())
    print(f'Average spread of the index for the etf {column}: {np.mean(df_imputed_index[column].max()-df_imputed_index[column].min())}')
    print(f'Maximum spread of the index for the etf {column}: {np.max(df_imputed_index[column].max()-df_imputed_index[column].min())}')
    print(f'Skewness of the index for the etf {column}: {np.mean(skew(df_imputed_index[column][:], nan_policy="omit"))} ')
    print(f'Kurtosis of the index for the etf {column}: {np.mean(kurtosis(df_imputed_index[column][:], nan_policy="omit"))} ')
    # Hide any empty subplots if there are more subplots than columns
    
    for i in range(len(columns), num_rows * num_cols):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()
#==============================================================================================================================
'''


#----------------------------------------- Machine Learning  -------------------------------------------------------

                                                
#============================================ Preparing the Data =====================================================================================================

# Features lists
features_list = ['BAMLCC0A0CMTRIV', 'BAMLC0A4CBBB', 'BAMLC0A3CA', 'HQMCB10YR', 'FEDFUNDS', 'T10Y2Y', 'TB3SMFFM',
                 'T5YFF', 'T1YFF', 'DLTIIT', 'NASDAQCOM', 'WILL5000PR', 'WILLLRGCAP', 'WILLSMLCAP', 'WILLLRGCAPGR', 
                 'WILLLRGCAPVAL', 'WILLMIDCAP', 'MSPUS', 'CCSA', 'BOPGSTB', 'VIXCLS', 'USSLIND', 'USALOLITONOSTSAM', 'UNRATE', 
                 'STICKCPIM157SFRBATL', 'EMVMACROBUS', 'MORTGAGE30US',  'WILLRESIPR', 'SBPREUE', 'MXEUMC', 'MXEULC', 'SPAXLCUP', 'SBPRAPU',
                 'MEMMG', 'MEMMG', 'MXEF', 'MXEFLC', 'MXEFMC', 'SML']


df_features = pd.DataFrame(columns=features_list)

features_from_fred_list = features_list[:-11]
df_features_from_fred = pd.DataFrame(columns=features_from_fred_list)


for i in features_from_fred_list:
    try:
        data= import_fred(i)
        data = data.resample('M').last()
        data.index = data.index.to_period("M")
        df_features_from_fred[i]  = data
    except Exception as e:
        print(f"Error fetching data for {i}: {e}")


features_from_files_list = features_list[-11:]
df_imported_feat = pd.DataFrame()


for i in features_from_files_list:
    csv_file_path = (f'data/{i}.csv')
    stored_serie = import_csv_with_date_index(csv_file_path)
    stored_serie.index = stored_serie.index.to_period('M')
    # Ensure the indices are unique
    df_imported_feat = pd.concat([df_imported_feat, stored_serie.loc[~stored_serie.index.duplicated(keep='first')]], axis=1)


df_imported_feat.columns = features_from_files_list
df_imported_feat=df_imported_feat.iloc[:-2,:]

df_features = pd.merge(df_features_from_fred, df_imported_feat,  left_index=True, right_index=True)


# Filling the missing values 
# Create a KNNImputer instance
imputer = KNNImputer(n_neighbors=5)  # You can adjust the number of neighbors as needed
# Fit the imputer on the data and transform the DataFrame
df_features = pd.DataFrame(imputer.fit_transform(df_features), columns=df_features.columns, index=df_features.index)
df_features = df_features.loc[df_imputed_index.head(1).index[0]:df_imputed_index.tail(1).index[0]]


merged_df = pd.merge(df_features, df_imputed_index, left_index=True, right_index=True)

# Specify columns to lag
columns_to_lag = features_list

# Number of periods to lag
lag_periods = 1

# Lag the specified columns
for col in columns_to_lag:
    merged_df[col] = merged_df[col].shift(lag_periods)
    merged_df.dropna(inplace=True, axis=0)

x = merged_df[features_list]
y = merged_df[ticker_list] 

# Standardize features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


time_serie_plot = merged_df.index.astype(str)




# Random Forest 
#==============================================================================================================================

# Initialize an empty DataFrame to store predicted returns
predicted_returns_df = pd.DataFrame(index=merged_df.index)
'''
# Number of folds for cross-validation
num_folds = 5

# Create a grid for subplots
num_rows = 4
num_cols = 3
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 10), sharex=True)

# Flatten the 2D array of subplots for easier indexing
axes = axes.flatten()
years = time_serie_plot.str[:4]

print(f'Average Mean Absolute Error using {num_folds}-fold cross-validation')
print('', sep='/n')

# Iterate over each index
for i, ticker in enumerate(ticker_list):
    # Extract features and target variable for the current stock
    x = merged_df[features_list]
    y = merged_df[ticker]
    
    # Standardize features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    # K-fold cross-validation
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    # Initialize model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Lists to store evaluation metrics
    mae_list = []
    
    # Lists to store actual and predicted returns for plotting
    actual_returns_list = []
    predicted_returns_list = []
    
    # Iterate over folds
    for train_index, test_index in kf.split(x_scaled):
        X_train, X_test = x_scaled[train_index], x_scaled[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Model Training
        rf_model.fit(X_train, y_train)
        
        # Model Prediction
        y_pred = rf_model.predict(X_test)
        
        # Evaluate Model Performance
        mae = mean_absolute_error(y_test, y_pred)
        mae_list.append(mae)
        
        # Store actual and predicted returns for plotting
        actual_returns_list.extend(y_test)
        predicted_returns_list.extend(y_pred)
    
    # Calculate average MAE across folds for the current stock
    average_mae = sum(mae_list) / len(mae_list)
    print(f'Average Mean Absolute Error for {ticker} - (Random Forest): {average_mae}')
    
    # Predict returns for the entire period
    predicted_returns = rf_model.predict(x_scaled)
    
    # Store predicted returns in the DataFrame
    predicted_returns_df[ticker] = predicted_returns

    # Plot the actual and predicted time series only if there is data
    if not y.empty:
        ax = axes[i]
        ax.plot(time_serie_plot, y, label=f'Actual - {ticker}', color='blue')
        ax.plot(time_serie_plot, predicted_returns, label=f'Predicted - {ticker}', linestyle='--', color='orange')
        ax.set_title(f'Actual vs Predicted Returns for {ticker}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Returns')
        ax.legend()

        # Adjust x-axis labels
        ax.set_xticks(time_serie_plot[::24])  # Set x-axis ticks every 24 months
        ax.set_xticklabels(years[::24], rotation=45, fontsize=8)  # Set x-axis labels with every second year, rotated for better visibility

# Hide empty subplots if the number of subplots is less than the total number of subplots in the grid
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

# Adjust layout
plt.tight_layout()
plt.show()
'''


# Regression Tree 
#==============================================================================================================================
#  Decision Tree 


# Standardize features for the entire dataset
scaler = StandardScaler()
x_scaled = scaler.fit_transform(merged_df[features_list])

predicted_returns_df_dt = pd.DataFrame(index=merged_df.index)

'''

# Create a 4x3 grid for subplots for Decision Tree
fig_dt, axes_dt = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 10), sharex=True)
axes_dt = axes_dt.flatten()

for i, ticker in enumerate(ticker_list):
    # Initialize Decision Tree model
    dt_model = DecisionTreeRegressor(random_state=42)
    
    # Lists to store evaluation metrics
    mae_list_dt = []
    
    # Lists to store actual and predicted returns for plotting
    actual_returns_list_dt = []
    predicted_returns_list_dt = []
    
    # K-fold cross-validation for Decision Tree
    for train_index, test_index in kf.split(x_scaled):
        X_train, X_test = x_scaled[train_index], x_scaled[test_index]
        y_train, y_test = merged_df[ticker].iloc[train_index], merged_df[ticker].iloc[test_index]
        
        # Model Training for Decision Tree
        dt_model.fit(X_train, y_train)
        
        # Model Prediction for Decision Tree
        y_pred_dt = dt_model.predict(X_test)
        
        # Evaluate Model Performance
        mae_dt = mean_absolute_error(y_test, y_pred_dt)
        mae_list_dt.append(mae_dt)
        
        # Store actual and predicted returns for Decision Tree
        actual_returns_list_dt.extend(y_test)
        predicted_returns_list_dt.extend(y_pred_dt)
    
    # Calculate average MAE across folds for the current stock
    average_mae_dt = sum(mae_list_dt) / len(mae_list_dt)
    print(f'Average Mean Absolute Error for {ticker} across {num_folds}-fold cross-validation (Decision Tree): {average_mae_dt}')
    
    # Predict returns for the entire period using Decision Tree
    predicted_returns_dt = dt_model.predict(x_scaled)
    
    # Store predicted returns in the DataFrame for Decision Tree
    predicted_returns_df_dt[ticker] = predicted_returns_dt
    
    # Plot the actual and predicted time series for Decision Tree
    axes_dt[i].plot(time_serie_plot, merged_df[ticker], label=f'Actual - {ticker}', color='blue')
    axes_dt[i].plot(time_serie_plot, predicted_returns_dt, label=f'Predicted (DT) - {ticker}', linestyle='--', color='yellow')
    axes_dt[i].set_title(f'Actual vs Predicted Returns for {ticker} (Decision Tree)')
    axes_dt[i].set_xlabel('Date')
    axes_dt[i].set_ylabel('Returns')
    axes_dt[i].legend()

# Adjust layout for Decision Tree
plt.tight_layout()
plt.show()

# Display the DataFrame with predicted returns for each stock using Decision Tree
print("\nPredicted Returns DataFrame (Decision Tree):")
print(predicted_returns_df_dt)

'''












































