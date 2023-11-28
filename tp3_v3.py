# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 17:35:29 2023

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
import riskfolio as rp
from riskfolio import factors_constraints


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
    
    # Function to get the index of the first non-NaN value in each column
def first_non_nan_index(column):
    return column.first_valid_index()


def compute_avg_turnover_and_exec_cost(df_turnover, df_execution, columns):
    for i in columns:
        # Create a masked array where values equal to 0 are masked
        df_turnover[i] = np.ma.masked_where(df_turnover[i] == 0, df_turnover[i])
        
        print(f'Portfolio {i}')
        print(f'Average yearly turnover: {np.mean(df_turnover[i])*100} %')
        
        # Assuming df_exec_cost is a column in your DataFrame
        # Create a masked array for execution cost
        df_execution[i]= np.ma.masked_where(df_execution[i] == 0, df_execution[i])
        
        print(f'Average execution cost estimation: {np.mean(df_execution[i])} $')
        print('', sep='\n')

# Code -------------------------------------------------------------------------------------

etf_monthly_rets = getData(ticker_list)
etf_monthly_rets.replace(0, np.nan, inplace=True)
etf_monthly_rets=etf_monthly_rets.loc["01-2005":]


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
df_index_rets=df_index_rets.loc["01-2005":]

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
imputer = KNNImputer(n_neighbors=50)  # You can adjust the number of neighbors as needed
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
                 'MEMMG', 'MXEF', 'MXEFLC', 'MXEFMC', 'SML']


df_features = pd.DataFrame(columns=features_list)

features_from_fred_list = features_list[:-10]
df_features_from_fred = pd.DataFrame(columns=features_from_fred_list)


for i in features_from_fred_list:
    try:
        data= import_fred(i)
        data = data.resample('M').last()
        data.index = data.index.to_period("M")
        df_features_from_fred[i]  = data
    except Exception as e:
        print(f"Error fetching data for {i}: {e}")


features_from_files_list = features_list[-10:]
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
merged_df = merged_df.loc["01-2005":]

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

# Number of folds for cross-validation
num_folds = 5
'''
# Create a grid for subplots
num_rows = 4
num_cols = 3
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 10), sharex=True)

# Flatten the 2D array of subplots for easier indexing
axes = axes.flatten()
years = time_serie_plot.str[:4]

print(f'Average Mean Absolute Error using {num_folds}-fold cross-validation')
print('', sep='/n')

'''
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
#   print(f'Average Mean Absolute Error for {ticker} - (Random Forest): {average_mae}')
    
    # Predict returns for the entire period
    predicted_returns = rf_model.predict(x_scaled)
    
    # Store predicted returns in the DataFrame
    predicted_returns_df[ticker] = predicted_returns
'''
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
        ax.set_xticks(time_serie_plot[::12])  # Set x-axis ticks every 24 months
        ax.set_xticklabels(years[::12], rotation=45, fontsize=8)  # Set x-axis labels with every second year, rotated for better visibility

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
print('Average Mean Absolute Error using {num_folds}-fold cross-validation - \033[1mRegression Tree\033[0m')
print('', sep='/n')

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
    print(f'Average Mean Absolute Error for {ticker}: {average_mae_dt}')
    
    # Predict returns for the entire period using Decision Tree
    predicted_returns_dt = dt_model.predict(x_scaled)
    
    # Store predicted returns in the DataFrame for Decision Tree
    predicted_returns_df_dt[ticker] = predicted_returns_dt
    
    # Plot the actual and predicted time series for Decision Tree
    ax = axes_dt[i]
    ax.plot(time_serie_plot, merged_df[ticker], label=f'Actual - {ticker}', color='blue')
    ax.plot(time_serie_plot, predicted_returns_dt, label=f'Predicted (DT) - {ticker}', linestyle='--', color='yellow')
    ax.set_title(f'Actual vs Predicted Returns for {ticker} (Decision Tree)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Returns')
    ax.legend()

    # Adjust x-axis labels
    ax.set_xticks(time_serie_plot[::12])  # Set x-axis ticks every 24 months
    ax.set_xticklabels(years[::12], rotation=45, fontsize=8)  # Set x-axis labels with every second year, rotated for better visibility

# Hide empty subplots if the number of subplots is less than the total number of subplots in the grid
for j in range(i + 1, len(axes_dt)):
    axes_dt[j].axis('off')

# Adjust layout for Decision Tree
plt.tight_layout()
fig_dt.suptitle('Regression Tree', fontsize=16).set_y(1.02)  # Adjust the y parameter for subtitle placement
plt.show()

'''
# ESG Portfolio linear constraint

# Your DataFrame
ESG_constraint_data = {
    'ESG Fund Rating': [6.6, 8.6, 7.1, 7.3, 6.5, 5.7, 6, 5.7, 8.2, 7.9, 7.6],
}

ESG_constraint = pd.DataFrame(ESG_constraint_data)

# Define the constraints based on your DataFrame
constraints = pd.DataFrame({
    'Disabled': [False],
    'Factor': ['ESG Fund Rating'],
    'Sign': ['>='],
    'Value': [7],
    'Relative Factor': '',
})

# Create the factors constraints matrices C and D
C, D = factors_constraints(constraints, loadings=ESG_constraint)


ETF_fees = [0.0015, 0.001, 0.0025, 0.0017, 0.0012, 0.002, 0.0015, 0.0025, 0.0005, 0.0004, 0.0018]


ranked_returns = predicted_returns_df.rank(axis=1, ascending=False, method='max')

# Min Variance ,Black Litterman and Efficient frontier optimization at each month


# Assuming you have initialized your DataFrames for minimum variance and Black Litterman results
columns_min_variance = ['Date', 'ESG Score'] + list(df_imputed_index.columns)
monthly_min_variance = pd.DataFrame(columns=columns_min_variance)
columns_bl_variance = ['Date', 'ESG Score'] + list(df_imputed_index.columns)
monthly_bl_variance = pd.DataFrame(columns=columns_bl_variance)
monthly_mu_cov_bl = []

# Define the start and end dates
start_date = '2010-01'
end_date = '2023-01'

# Convert the start and end dates to Timestamp objects
start_date_timestamp = pd.Timestamp(start_date)
end_date_timestamp = pd.Timestamp(end_date)

#If we want to backtest on imputed data for etf returns :
returns_period = df_imputed_index[(df_imputed_index.index >= start_date) & (df_imputed_index.index <= end_date)]

# Get relative ranking
relative_ranking = ranked_returns.loc[start_date:]

asset_classes = {'Assets': ["ESGU", "EAGG", "ESGE", "ESML", "SUSB", "ESGD", "SHY", "SUSA", "GOVT", "MBB", "SUSC"]}

asset_classes = pd.DataFrame(asset_classes)
asset_classes = asset_classes.sort_values(by=['Assets'])

# Create an empty list to store efficient frontier results for each date
efficient_frontier_list = []
efficient_frontier_list_bl = []  # Initialize the list for Black Litterman results
efficient_frontier_dict={}
# Iterate over the rolling windows
while start_date_timestamp < end_date_timestamp:
    # Filter the DataFrame for the current month for minimum variance optimization
    rets_df = df_imputed_index[
        (df_imputed_index.index.to_timestamp() < start_date_timestamp) &
        (df_imputed_index.index.to_timestamp() >= start_date_timestamp - pd.DateOffset(months=108))
    ]

    # Building the portfolio object for minimum variance optimization
    port_min_variance = rp.Portfolio(returns=rets_df, ainequality=C, binequality=D, nea=6, allowTO=True, turnover=0.0461)

    # Select method and estimate input parameters for minimum variance optimization
    method_mu = 'hist'
    method_cov = 'hist'
    port_min_variance.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

    model_min_variance = 'Classic'
    rm_min_variance = 'MV'
    obj_min_variance = 'MinRisk'
    hist_min_variance = True
    rf_min_variance = 0
    l_min_variance = 0

    # Perform minimum variance optimization
    w_min_variance = port_min_variance.optimization(model=model_min_variance, rm=rm_min_variance, obj=obj_min_variance,
                                                    rf=rf_min_variance, l=l_min_variance, hist=hist_min_variance)

    # Calculate the ESG Score for the minimum variance portfolio
    w_min_variance_weights = w_min_variance['weights'].values.reshape(1, -1)
    esg_ratings_min_variance = ESG_constraint['ESG Fund Rating'].values.reshape(-1, 1)
    ESG_Rating_portfolio_min_variance = w_min_variance_weights.dot(esg_ratings_min_variance).item()

    # Create a dictionary to store the results for the minimum variance portfolio
    result_dict_min_variance = {
        'Date': start_date_timestamp,
        'ESG Score': ESG_Rating_portfolio_min_variance
    }

    # Add weights for each ETF to the dictionary
    result_dict_min_variance.update(dict(zip(df_imputed_index.columns, w_min_variance['weights'].values)))

    # Append the results for the minimum variance portfolio to the list
    efficient_frontier_list.append(result_dict_min_variance)


#########################################   BLACK-LITTERMAN  ###########################################

    # Extract top 2 and bottom 2 assets based on rankings for the specific date
    top_assets = relative_ranking.loc[start_date_timestamp].nsmallest(2).index
    bottom_assets = relative_ranking.loc[start_date_timestamp].nlargest(2).index

    # Create views with dynamic positions and relatives
    views = {
        'Disabled': [False, False],
        'Type': ['Assets', 'Assets'],
        'Set': ['', ''],
        'Position': [top_assets[0], top_assets[1]],
        'Sign': ['>=', '>='],
        'Return': [0.05, 0.05],  # Annual terms
        'Type Relative': ['Assets', 'Assets'],
        'Relative Set': ['', ''],
        'Relative': [bottom_assets[0], bottom_assets[1]]
    }

    views_df = pd.DataFrame(views)
      # Get the P, Q matrices from the views
    P, Q = rp.assets_views(views_df, asset_classes)

    # Estimate Black Litterman inputs for minimum variance portfolio
    port_min_variance.blacklitterman_stats(P, Q/12, rf=rf_min_variance, w=w_min_variance['weights'].to_frame(), delta=None, eq=True)

    model_bl_min_variance = 'BL'  # Black Litterman
    rm_bl_min_variance = 'MV'  # Risk measure used, this time will be variance
    obj_bl_min_variance = 'Sharpe'  # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
    hist_bl_min_variance = False  # Use historical scenarios for risk measures that depend on scenarios

    # Perform Black Litterman optimization for minimum variance portfolio
    w_bl_min_variance = port_min_variance.optimization(model=model_bl_min_variance, rm=rm_bl_min_variance, obj=obj_bl_min_variance, rf=rf_min_variance, l=l_min_variance, hist=hist_bl_min_variance)

    # Calculate the ESG Score for the Black Litterman portfolio
    w_bl_weights_min_variance = w_bl_min_variance['weights'].values.reshape(1, -1)
    esg_ratings_bl_min_variance = ESG_constraint['ESG Fund Rating'].values.reshape(-1, 1)
    ESG_Rating_portfolio_bl_min_variance = w_bl_weights_min_variance.dot(esg_ratings_bl_min_variance).item()

    # Create a dictionary to store the results for the Black Litterman portfolio
    result_dict_bl_min_variance = {
        'Date': start_date_timestamp,
        'ESG Score': ESG_Rating_portfolio_bl_min_variance
    }

    # Add weights for each ETF to the dictionary
    result_dict_bl_min_variance.update(dict(zip(df_imputed_index.columns, w_bl_min_variance['weights'].values)))

    # Append the results for the Black Litterman portfolio to the list
    efficient_frontier_list_bl.append(result_dict_bl_min_variance)


    # Convert the list of dictionaries to a DataFrame for minimum variance
    monthly_min_variance = pd.DataFrame(efficient_frontier_list)

    # Ensure that monthly_bl_variance is initialized as an empty DataFrame
    monthly_bl_variance = pd.DataFrame(columns=columns_bl_variance)

    # Convert the list of dictionaries to a DataFrame for Black Litterman
    monthly_bl_variance = pd.DataFrame(efficient_frontier_list_bl)


    ######################## EFFICIENT FRONTIER ########################

    # Calculate and store efficient frontier results
    points = 5  # Number of points on the frontier
    frontier_weights = port_min_variance.efficient_frontier(model=model_bl_min_variance, rm=rm_bl_min_variance, points=points, rf=rf_min_variance, hist=hist_bl_min_variance).T

    # Create a dictionary to store the efficient frontier weights and ESG scores for the current date
    frontier_dict_for_date = {
        'Date': start_date_timestamp,
        'Points': []
    }

    # Iterate over each portfolio on the efficient frontier
    for i in range(points):
        # Calculate the ESG Score for the current portfolio
        w_weights = frontier_weights.iloc[i, :].values.reshape(1, -1)
        ESG_Rating_portfolio = w_weights.dot(esg_ratings_bl_min_variance).item()

        # Add the ESG Score and weights for each ETF to the dictionary
        portfolio_dict = {
            'Point': i + 1,  # Adding 1 to start indexing from 1
            'ESG Score': ESG_Rating_portfolio,
            'Weights': {ticker: weight for ticker, weight in zip(frontier_weights.columns[:], w_weights[0])}
        }

        frontier_dict_for_date['Points'].append(portfolio_dict)

    #  Now 'frontier_dict_for_date' contains the information for each point on the efficient frontier for the current date


        # Store mu_bl and cov_bl for later use
        mu_cov_bl = {
            'Date': start_date_timestamp,
            'mu_bl': port_min_variance.mu_bl.to_dict(orient = 'records')[0],
            'cov_bl': port_min_variance.cov_bl.to_dict(orient = 'records')
        }
        monthly_mu_cov_bl.append(mu_cov_bl)

    # Update the efficient_frontier_dict with the current date and efficient frontier information
    efficient_frontier_dict[start_date_timestamp] = frontier_dict_for_date
    # Increment the start_date_timestamp
    start_date_timestamp += pd.DateOffset(months=1)

    
    
#Calculate the cumulative return and the fees of our portfolio according to the risk rolerance

# Initialize lists to store data
portfolio_returns_bl_data = []
portfolio_returns_mv_data = []
portfolio_returns_1_data = []
portfolio_returns_2_data = []
portfolio_returns_3_data = []
portfolio_returns_4_data = []
portfolio_returns_5_data = []
portfolio_returns_eq_w_data = []

# Iterate over each date in the period
for index, row in monthly_bl_variance.iterrows():
    # Extract the date and weights for the current row
    date = row['Date']
    weights = row.drop(['Date', 'ESG Score']).values

    # Check if the date is present in the returns_period DataFrame
    if date in returns_period.index:
        # Extract the returns for the current date
        returns_at_date = returns_period.loc[date]

        # Calculate the portfolio return for the current date
        portfolio_return_bl = np.dot(returns_at_date, weights)

        # Append the data to the list
        portfolio_returns_bl_data.append({'Date': date, 'Portfolio_Return': portfolio_return_bl})

# Iterate over each date in the period
for index, row in monthly_min_variance.iloc[3:].iterrows():
    # Extract the date and weights for the current row
    date = row['Date']
    weights = row.drop(['Date', 'ESG Score']).values

    # Check if the date is present in the returns_period DataFrame
    if date in returns_period.index:
        # Extract the returns for the current date
        returns_at_date = returns_period.loc[date]

        # Calculate the portfolio return for the current date
        portfolio_return_mv = np.dot(returns_at_date, weights)
        portfolio_fees_mv = np.dot(ETF_fees, weights)

        # Append the data to the list
        portfolio_returns_mv_data.append({'Date': date, 'Portfolio_Return': portfolio_return_mv, 'Fees': portfolio_fees_mv})

        # Check if the date is present in the dictionary
        if date in efficient_frontier_dict:
            # Extract the points and create a DataFrame
            points = efficient_frontier_dict[date]['Points']

            weights_1 = np.array(list(points[0]['Weights'].values()))
            weights_2 = np.array(list(points[1]['Weights'].values()))
            weights_3 = np.array(list(points[2]['Weights'].values()))
            weights_4 = np.array(list(points[3]['Weights'].values()))
            weights_5 = np.array(list(points[4]['Weights'].values()))
        else:
            # Handle the case when the date is not present in the dictionary
            weights_1 = weights_2 = weights_3 = weights_4 = weights_5 = np.nan  # You can choose an appropriate way to handle this

        # Calculate the portfolio return for the current date
        portfolio_return_1 = np.dot(returns_at_date, weights_1)
        # Calculate the portfolio return for the current date
        portfolio_return_2 = np.dot(returns_at_date, weights_2)
        # Calculate the portfolio return for the current date
        portfolio_return_3 = np.dot(returns_at_date, weights_3)
        # Calculate the portfolio return for the current date
        portfolio_return_4 = np.dot(returns_at_date, weights_4)
        # Calculate the portfolio return for the current date
        portfolio_return_5 = np.dot(returns_at_date, weights_5)

        # Calculate the portfolio fee for the current date
        portfolio_fees_1 = np.dot(ETF_fees, weights_1)
        portfolio_fees_2 = np.dot(ETF_fees, weights_2)
        portfolio_fees_3 = np.dot(ETF_fees, weights_3)
        portfolio_fees_4 = np.dot(ETF_fees, weights_4)
        portfolio_fees_5 = np.dot(ETF_fees, weights_5)

        # Append the date and portfolio return to the DataFrame
        portfolio_returns_1_data.append({'Date': date, 'Portfolio_Return': portfolio_return_1, 'Fees': portfolio_fees_1, 'ESG Score': efficient_frontier_dict[date]['Points'][0]['ESG Score']})
        # Append the date and portfolio return to the DataFrame
        portfolio_returns_2_data.append({'Date': date, 'Portfolio_Return': portfolio_return_2, 'Fees': portfolio_fees_2, 'ESG Score': efficient_frontier_dict[date]['Points'][1]['ESG Score']})
        # Append the date and portfolio return to the DataFrame
        portfolio_returns_3_data.append({'Date': date, 'Portfolio_Return': portfolio_return_3, 'Fees': portfolio_fees_3, 'ESG Score': efficient_frontier_dict[date]['Points'][2]['ESG Score']})
        # Append the date and portfolio return to the DataFram
        portfolio_returns_4_data.append({'Date': date, 'Portfolio_Return': portfolio_return_4, 'Fees': portfolio_fees_4, 'ESG Score': efficient_frontier_dict[date]['Points'][3]['ESG Score']})
        # Append the date and portfolio return to the DataFrame
        portfolio_returns_5_data.append({'Date': date, 'Portfolio_Return': portfolio_return_5, 'Fees': portfolio_fees_5, 'ESG Score': efficient_frontier_dict[date]['Points'][4]['ESG Score']})
        weights_eq_w = np.full(11, 1/11)    
        portfolio_return_eq_w = np.dot(returns_at_date, weights_eq_w)

        # Append the data to the list
        portfolio_returns_eq_w_data.append({'Date': date, 'Portfolio_Return': portfolio_return_eq_w})

# Create DataFrames from the lists
portfolio_returns_bl = pd.DataFrame(portfolio_returns_bl_data)
portfolio_returns_mv = pd.DataFrame(portfolio_returns_mv_data)
portfolio_returns_1 = pd.DataFrame(portfolio_returns_1_data)
portfolio_returns_2 = pd.DataFrame(portfolio_returns_2_data)
portfolio_returns_3 = pd.DataFrame(portfolio_returns_3_data)
portfolio_returns_4 = pd.DataFrame(portfolio_returns_4_data)
portfolio_returns_5 = pd.DataFrame(portfolio_returns_5_data)
portfolio_returns_eq_w = pd.DataFrame(portfolio_returns_eq_w_data)


    
    
#---------------------- In notebook ---------------------------------------------------
combined_df = pd.DataFrame()

# Loop through each date in the dictionary
for date, data in efficient_frontier_dict.items():
    # Access the 'Points' data for each date
    df_date = pd.DataFrame(data['Points'])
    
    # Add a new column for the date
    df_date['Date'] = date
    
    # Concatenate the data for each date
    combined_df = pd.concat([combined_df, df_date])

# Reset the index of the combined DataFrame
combined_df.reset_index(drop=True, inplace=True)
combined_df.index = combined_df["Date"]
combined_df = combined_df.drop(["Date"], axis=1)

weight_dict = combined_df['Weights']

pf_index = ('1', '2', '3', '4', '5')
month_year = weight_dict.index.strftime('%Y-%m').unique()
year = weight_dict.index.strftime('%Y').unique()


dt_frame = pd.DataFrame(index=month_year, columns=ticker_list)
dict_pf = {'1': dt_frame.copy(), '2': dt_frame.copy(), '3': dt_frame.copy(), '4': dt_frame.copy(), '5': dt_frame.copy()}

for i in month_year:
    for tick in ticker_list:
        dict_pf['1'][tick].loc[i] = (weight_dict.loc[weight_dict.index.strftime('%Y-%m') == i][0][tick])
        dict_pf['2'][tick].loc[i] = float(weight_dict.loc[weight_dict.index.strftime('%Y-%m') == i][1][tick])
        dict_pf['3'][tick].loc[i] = float(weight_dict.loc[weight_dict.index.strftime('%Y-%m') == i][2][tick])
        dict_pf['4'][tick].loc[i] = float(weight_dict.loc[weight_dict.index.strftime('%Y-%m') == i][3][tick])
        dict_pf['5'][tick].loc[i] = float(weight_dict.loc[weight_dict.index.strftime('%Y-%m') == i][4][tick])


# Assuming df_weights is your DataFrame with assets weights
# and that the index is a datetime index with monthly frequency

df_yearly_turnover = pd.DataFrame(index= year,  columns=pf_index )
df_yearly_exec_cost = pd.DataFrame(index= year,  columns=pf_index)
bid_ask_spread=[0.02,0.02,0.03,0.09,0.04,0.02,0.01,0.02,0.05,0.01,0.05]

last_prices_list=pd.Series(index=ticker_list)
for tick in ticker_list:
    last_prices_list[tick] = yf.download(tick)["Close"] .tail(1)[0]
    

for col in df_yearly_turnover.columns:
# Calculate absolute monthly weight changes
    weight_changes = dict_pf[col].diff()
    weight_changes.dropna(inplace=True)
    weight_changes.index = pd.to_datetime(weight_changes.index)
    for yr in year:
        df_yearly_turnover[col].loc[yr] = np.sum(weight_changes.loc[yr]).abs().sum()
        df_yearly_exec_cost[col].loc[yr] = (np.sum(bid_ask_spread * weight_changes.loc[yr].abs())*last_prices_list).sum()
       
'''
# Assuming df_yearly_turnover and df_exec_cost are your DataFrames
compute_avg_turnover_and_exec_cost(df_yearly_turnover, df_yearly_exec_cost, pf_index)
'''





'''
efficient_frontier_dict

# Plotting
plt.figure(figsize=(12, 6))

# Plot cumulative returns for all portfolios
for i, (portfolio_returns, label) in enumerate(zip(
        [portfolio_returns_1, portfolio_returns_2, portfolio_returns_3, portfolio_returns_4, portfolio_returns_5, portfolio_returns_mv,portfolio_returns_eq_w],
        ['Portfolio 1', 'Portfolio 2', 'Portfolio 3', 'Portfolio 4', 'Portfolio 5', 'Minimum Variance', 'Equally Weighted'])):
    plt.plot(portfolio_returns['Date'], portfolio_returns['Cumulative_Return'], label=label)

plt.title('Cumulative Return Over Time for Multiple Portfolios')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.show()


# ------------------------------------- ESG Time Series plot -------------------------------------------------------------


# Assuming portfolio_returns_1, ..., portfolio_returns_5, monthly_min_variance contain 'Date' and 'ESG Score'
# Convert the 'Date' column to a datetime type
for portfolio_returns in [portfolio_returns_1, portfolio_returns_2, portfolio_returns_3, portfolio_returns_4, portfolio_returns_5, monthly_min_variance]:
    portfolio_returns['Date'] = pd.to_datetime(portfolio_returns['Date'])

# Plotting
plt.figure(figsize=(12, 6))


print(f'\033[1mMean ESG score on Backtest period:\033[0m')
print('', sep='\n')

# Plot ESG scores for portfolios 1 to 5
for i, portfolio_returns in enumerate([portfolio_returns_1, portfolio_returns_2, portfolio_returns_3, portfolio_returns_4, portfolio_returns_5]):
    plt.plot(portfolio_returns['Date'], portfolio_returns['ESG Score'], label=f'Portfolio {i + 1}')
    print(f'Portfolio {i+1}: {np.mean(portfolio_returns["ESG Score"])}')

# Plot ESG scores for monthly_min_variance
plt.plot(monthly_min_variance['Date'], monthly_min_variance['ESG Score'], label='Min Variance', linestyle='--', color='black')

plt.yticks(np.arange(6.8, 8, 0.1))

# Add a horizontal line at y=7
#plt.axhline(y=7, color='red', linestyle='--', alpha=0.5)

plt.title('ESG Score Over Time For Optimized Portfolios And Min Variance Portfolio')
plt.xlabel('Date')
plt.ylabel('ESG Score')
plt.legend()
plt.show()



#------------------------------------------------------------------------------------------------

print(np.mean(ETF_fees))
print(portfolio_returns_1['Fees'].mean())
print(portfolio_returns_2['Fees'].mean())
print(portfolio_returns_3['Fees'].mean())
print(portfolio_returns_4['Fees'].mean())
print(portfolio_returns_5['Fees'].mean())


#------------------------------------------------------------------------------------------------
'''





















