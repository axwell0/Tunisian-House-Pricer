import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder


def encode_top_n_cities(df: pd.DataFrame, city_column: str, n: int) -> pd.DataFrame:
    """
    Encodes the top N most frequent cities in a DataFrame column, grouping less frequent cities under 'Other'.

    This function performs the following steps:
    1. Identifies the top N most frequent city values in the city_column.
    2. Groups all other cities under the label 'Other'.
    3. Applies label encoding to the modified column, adding a new encoded column.

    Args:
        df (pd.DataFrame): The input DataFrame containing the city data.
        city_column (str): The name of the column containing city names.
        n (int): The number of top cities to retain. All other cities will be grouped as 'Other'.

    Returns:
        pd.DataFrame: A new DataFrame with the updated city column and an additional encoded column.
    """
    df_copy = df.copy()
    top_cities = df_copy[city_column].value_counts().nlargest(n).index
    df_copy[city_column] = df_copy[city_column].apply(lambda city: city if city in top_cities else 'Other')
    label_encoder = LabelEncoder()
    encoded_column_name = f'{city_column}_encoded'
    df_copy[encoded_column_name] = label_encoder.fit_transform(df_copy[city_column])

    return df_copy

def preprocess_data(train_df):
    temp = train_df.copy()
    # #
    # temp = remove_outliers_iqr(temp, 'area')
    # temp = remove_outliers_iqr(temp, 'n_bedrooms')
    # Encode top N cities and states
    temp = encode_top_n_cities(temp, 'city', 30)
    temp = encode_top_n_cities(temp, 'state', 10)
    temp['city'] = temp['city'].astype('category')
    temp['state'] = temp['state'].astype('category')
    return temp

def preprocss_test_data(test_set, cities,cities_encoding,states,states_encoding):
    temp = test_set.copy()
    temp['city'] = temp['city'].apply(lambda x: x if x in cities.value_counts() else 'Other')
    temp['city'] = temp['city'].astype('category')
    t = pd.concat([cities,cities_encoding],axis=1).drop_duplicates()

    temp['city_encoded'] = temp['city'].apply(lambda x: t.loc[t['city']==x,'city_encoded'].iloc[0])
    temp['state'] = temp['state'].apply(lambda x: x if x in states.value_counts() else 'Other')
    temp['state'] = temp['state'].astype('category')
    t = pd.concat([states,states_encoding],axis=1).drop_duplicates()

    temp['state_encoded'] = temp['state'].apply(lambda x: t.loc[t['state']==x,'state_encoded'].iloc[0])
    return temp
def train_xgb_with_hyperopt(train_set, test_set, numerical_features, target_column, categorical_features=None,
                            max_evals=100):
    """
    Train an XGBoost model with hyperparameter tuning using hyperopt.

    Parameters:
        train_set (pd.DataFrame): Pre-separated training set.
        test_set (pd.DataFrame): Pre-separated testing set.
        numerical_features (list): List of numerical feature column names.
        target_column (str): Target column name.
        categorical_features (list, optional): List of categorical feature column names. Default is None.
        max_evals (int): Maximum number of hyperparameter optimization iterations. Default is 100.

    Returns:
        dict: A dictionary containing the best hyperparameters and evaluation metrics.
    """

    X_train = train_set[numerical_features + categorical_features] if categorical_features else train_set[
        numerical_features]
    y_train = train_set[target_column]

    X_test = test_set[numerical_features + categorical_features] if categorical_features else test_set[
        numerical_features]
    y_test = test_set[target_column]

    def objective(params):
        params['max_depth'] = int(params['max_depth'])
        params['min_child_weight'] = int(params['min_child_weight'])
        model = xgb.XGBRegressor(enable_categorical=True, **params)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        r2_scores = []

        for train_idx, val_idx in kf.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_train_fold, y_train_fold)
            y_val_pred = model.predict(X_val_fold)
            r2 = r2_score(y_val_fold, y_val_pred)
            r2_scores.append(r2)

        mean_r2 = np.mean(r2_scores)
        return {'loss': -mean_r2, 'status': STATUS_OK}

    space = {
        'n_estimators': hp.choice('n_estimators', [100, 200, 300,]),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.15),
        'max_depth': hp.choice('max_depth', [3, 5, 7, 9, 12]),
        'min_child_weight': hp.choice('min_child_weight', [1, 3, 5, 7]),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
        'gamma': hp.uniform('gamma', 0, 0.4),
        'reg_alpha': hp.uniform('reg_alpha', 0, 0.05),
        'reg_lambda': hp.uniform('reg_lambda', 0, 0.05),
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'seed': 42
    }
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    best_params = space_eval(space, best)

    final_model = xgb.XGBRegressor(enable_categorical=True, **best_params)
    final_model.fit(X_train, y_train)
    y_test_pred = final_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores, rmse_scores, mae_scores = [], [], []

    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = xgb.XGBRegressor(enable_categorical=True, **best_params)
        model.fit(X_train_fold, y_train_fold)

        y_val_pred = model.predict(X_val_fold)
        r2_scores.append(r2_score(y_val_fold, y_val_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val_fold, y_val_pred)))
        mae_scores.append(mean_absolute_error(y_val_fold, y_val_pred))

    return {
        'final_model': final_model,
        'best_params': best_params,
        'test_rmse': rmse,
        'test_mae': mae,
        'test_r2': r2,
        'cv_mean_r2': np.mean(r2_scores),
        'cv_mean_rmse': np.mean(rmse_scores),
        'cv_mean_mae': np.mean(mae_scores)
    }


def convert_price_to_tnd(df, price_column='price', eur_to_tnd=3.34):
    h = df.copy()
    h[price_column] = (
            h[price_column]
            .str.extract(r'(\d[\d\s]*)')[0]
            .str.replace(r'\s+', '', regex=True).str.replace(',', '', regex=True)
            .astype('Int64')
            * h[price_column].apply(lambda x: eur_to_tnd if 'EUR' in x else 1)
    )
    h = h.reset_index(drop=True)
    return h.copy()
def remove_outliers_iqr(df: pd.DataFrame, column: str, quantiles: tuple = (0.25, 0.75)) -> pd.DataFrame:
    """
    Removes outliers from a DataFrame column based on the Interquartile Range (IQR) method.
    The IQR method defines outliers as values below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR,
    where Q1 and Q3 are the 25th and 75th percentiles, respectively.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        column (str): The name of the column from which outliers will be removed.
        quantiles (tuple): A tuple specifying the lower and upper quantiles to calculate IQR
                           (default is (0.25, 0.75)).
    Returns:
        pd.DataFrame: A new DataFrame with rows containing outliers in the specified column removed.
    """
    Q1 = df[column].quantile(quantiles[0])
    Q3 = df[column].quantile(quantiles[1])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_no_outliers = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed_rows = df.shape[0] - df_no_outliers.shape[0]
    print(f"IQR Method: Removed {removed_rows} rows.")

    return df_no_outliers









def get_collection(collection):
    return pd.DataFrame(collection.find({})).drop(['_id', 'title', 'url', 'description'], axis=1)


