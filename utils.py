import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import joblib
from config import *
import sklearn.calibration

from sklearn.metrics import *

def plot_time_categorical_column(df, label, column, order = False):
    df = df[df[column].notnull()]
    df[column] = df[column].astype(int)
    temp_df_app = df[df[label]==1][column].value_counts(normalize=True)
    temp_df_non_app = df[df[label]==0][column].value_counts(normalize=True)
    temp_df_merged = pd.merge(temp_df_app, temp_df_non_app, left_index = True, right_index = True).reset_index()
    temp_df_merged.rename({'{}_x'.format(column): '{}_1'.format(label),
                           '{}_y'.format(column): '{}_0'.format(label),
                           'index': column}, axis = 1, inplace=True)
    temp_df_melt = pd.melt(temp_df_merged, id_vars=column)
    temp_df_melt.rename({'variable':'label',
                         'value':'percentage'},axis = 1, inplace = True)
    
    if order:
        sns.barplot(temp_df_melt[column], temp_df_melt['percentage'], hue = temp_df_melt['label'],
                    order = df[column].value_counts().iloc[:10].index)
    else:
        sns.barplot(temp_df_melt[column], temp_df_melt['percentage'], hue = temp_df_melt['label'])
        
def clip_by_prop(df, col, proportion, value = 'other'):
    '''Clip a columns value so that anything less than proportion is swapped to "other"'''
    
    temp_df = df[col].value_counts()/len(df)
    df.loc[df[col].isin(temp_df[temp_df <= proportion].index), col] = 'other'
    return df

def clip_and_one_hot(df, col, proportion, value = 'other'):
    '''Runs clip_by_prop then one-hot-encodes the output. Returns one-hot of one column, still need to concatenate'''
    
    temp_df = clip_by_prop(df, col, proportion, value)
    temp_df = pd.get_dummies(temp_df[col], prefix = col) # Only have the one-hot of selected column
    return temp_df

def preprocessing(df):
    
    for col in ZERO_IMPUTE_FEATURES:
        df[col] = df[col].fillna(0)

    for col in UNKNOWN_IMPUTE_FEATURES:
        df[col] = df[col].fillna('unknown')

    clip_and_one_hot(df, 'nationality', 0.05)

    dummy_df = pd.get_dummies(df, columns = ONE_HOT_FEATURES)
    dummy_df.drop(DROP_FEATURES, axis = 1, inplace = True)
    return dummy_df

def preprocessing_with_filepaths(input_filepath, output_filepath):
    df = pd.read_csv(input_filepath)
    dummy_df = preprocessing(df)
    dummy_df.to_csv(output_filepath, index = False)
    
def fill_missing_cols(df, test_df, model = False):
    '''
    Add in missing columns from the training dataset to the testing dataset. Stops XGBoost from complaining that
    a feature is missing.
    
    Parameters:
    ___________
    df: training dataframe
    test_df: testing dataframe
    model: xgboost model
    
    '''
    if model:
        if (isinstance(model, xgb.sklearn.XGBRegressor)) or (isinstance(model, xgb.sklearn.XGBClassifier)):
            model = model.get_booster()
        if (isinstance(model, sklearn.calibration.CalibratedClassifierCV)):
            model = model.base_estimator.get_booster()
        missing_cols = set(model.feature_names).difference(test_df.columns)
    else:    
        missing_cols = set(df.columns).difference(test_df.columns)
    for col in missing_cols:
        test_df[col] = 0

def decile_plot(df, prediction_col, actual_col, what, q = 10):
    '''
    df: The dataframe with all relevant columns
    prediction_col: Column of predictions.
    actual_col: The truth value of the prediction column
    what: Column name of whatever you're plotting
    '''
    df['preds_decile'] = pd.qcut(x = df[prediction_col], q= q, labels = False, duplicates = 'drop')
    avg_actuals = []
    avg_preds = []
    
    for i in range(10):
        avg_actual = df[df['preds_decile']==i][actual_col].mean()
        avg_pred = df[df['preds_decile']==i][prediction_col].mean()
        avg_actuals.append(avg_actual)
        avg_preds.append(avg_pred)
        
    plt.plot(np.arange(1,11), avg_preds, label = 'predicted')
    plt.plot(np.arange(1,11), avg_actuals, label = 'actual')
    plt.xlabel('Predicted {} decile'.format(what))
    plt.ylabel('Average {}'.format(what))
    plt.legend()

def add_predictions(df, col_name, model):
    '''
    Convenience function that adds a prediction column from the booster object to a dataframe. Does not return anything.
    
    Parameters:
    ___________
    df: dataframe that has requisite features for the model. Column is added to this dataframe
    col_name: name of prediction column
    model: model that does prediction
    '''
    fill_missing_cols(df, df, model)
    
    if (isinstance(model, xgb.sklearn.XGBRegressor)) or (isinstance(model, xgb.sklearn.XGBClassifier)):
        model = model.get_booster()
        df[col_name] = model.predict(xgb.DMatrix(df[model.feature_names]))
    
    if (isinstance(model, sklearn.calibration.CalibratedClassifierCV)):
        df[col_name] = model.predict_proba(df[model.base_estimator.get_booster().feature_names])[:,1]
        
    if (isinstance(model, xgb.core.Booster)):
        df[col_name] = model.predict(xgb.DMatrix(df[model.feature_names]))
    
def add_predictions_freq(df, col_name, model, label):
    '''
    Convenience function that adds a prediction column from the booster object to a dataframe. Does not return anything.
    
    Parameters:
    ___________
    df: dataframe that has requisite features for the model. Column is added to this dataframe
    col_name: name of prediction column
    model: model that does prediction
    label: True label, required for including exposure. Used in the function as df[label]
    '''
    fill_missing_cols(df, df, model)
    
    if (isinstance(model, xgb.sklearn.XGBRegressor)) or (isinstance(model, xgb.sklearn.XGBClassifier)):
        model = model.get_booster()
        df[col_name] = model.predict(xgb.DMatrix(df[model.feature_names], base_margin = np.log(df[label])))
    
    if (isinstance(model, sklearn.calibration.CalibratedClassifierCV)):
        df[col_name] = model.predict_proba(df[model.base_estimator.get_booster().feature_names])[:,1]
        
    if (isinstance(model, xgb.core.Booster)):
        df[col_name] = model.predict(xgb.DMatrix(df[model.feature_names], base_margin = np.log(df[label])))
        
def add_predictions_sev(df, col_name, model):
    '''
    Convenience function that adds a prediction column from the booster object to a dataframe. Does not return anything.
    
    Parameters:
    ___________
    df: dataframe that has requisite features for the model. Column is added to this dataframe
    col_name: name of prediction column
    model: model that does prediction
    '''
    fill_missing_cols(df, df, model)
    
    if (isinstance(model, xgb.sklearn.XGBRegressor)) or (isinstance(model, xgb.sklearn.XGBClassifier)):
        model = model.get_booster()
        df[col_name] = model.predict(xgb.DMatrix(df[model.feature_names]))
    
    if (isinstance(model, sklearn.calibration.CalibratedClassifierCV)):
        df[col_name] = model.predict_proba(df[model.base_estimator.get_booster().feature_names])[:,1]
        
    if (isinstance(model, xgb.core.Booster)):
        df[col_name] = model.predict(xgb.DMatrix(df[model.feature_names]))
        
    
    
    
def add_preds_at_top(df, preds_col, top = 1000):
    '''
    Returns dataframe with new column {}_label.format(preds_col) that
    can be used to calculate recall/precision at top n.
    
    Arguments:
    __________
    df: dataframe that you want to add new column to
    preds_col: Name of prediction column to sort on
    top: Integer, top n considered as 1
    
    '''
    
    df_sorted = df.sort_values(preds_col, ascending=False).reset_index(drop=True)
    df_sorted['{}_label'.format(preds_col)] = 0
    df_sorted.iloc[:top, df_sorted.columns.get_loc('{}_label'.format(preds_col))] = 1
    
    return df_sorted

# def recall_at_top(df, true_col, pred_col):
#     '''Calculates recall at top, assuming that the predictions are already binary (not continuous)
#     Does not make sense. Don't use
#     '''
    
#     small_df = df[df[pred_col]==1]
#     recall = recall_score(small_df[true_col], small_df[pred_col])
#     return recall

def precision_at_top(df, true_col, pred_col):
    small_df = df[df[pred_col]==1]
    precision = precision_score(small_df[true_col], small_df[pred_col])
    return precision


def pipeline_of_dfs(model_open, model_click, model_ci, model_tl, df):
    '''
    All the models should be an xgb booster object
    '''
    
    fill_missing_cols(df, df, model_open)
    fill_missing_cols(df, df, model_click)
#     fill_missing_cols(df, df, model_final)
    fill_missing_cols(df, df, model_ci)
    fill_missing_cols(df, df, model_tl)
    
    add_predictions(df, 'preds_open', model_open)
    add_predictions(df, 'preds_click', model_click)
#     add_predictions(df, 'preds_final', model_final)
    add_predictions(df, 'preds_ci', model_ci)
    add_predictions(df, 'preds_tl', model_tl)
    
    df['preds_sum'] = df['preds_open'] + df['preds_click']
    df['preds_mult'] = df['preds_open'] * df['preds_click']
    
    df.loc[df[LABEL_CI] > 1, LABEL_CI] = 1
    df.loc[df[LABEL_TL] > 1, LABEL_TL] = 1

    # Need a way to find out who has been marketed to, and whether they said yes or not!
    df = add_preds_at_top(df,'preds_sum')
    df = add_preds_at_top(df,'preds_mult')
#     df = add_preds_at_top(df,'preds_final')
    df = add_preds_at_top(df,'preds_ci')
    df = add_preds_at_top(df,'preds_tl')
    
#     df['preds_ult'] = df['preds_mult'] * df['preds_final']
    df['preds_ult_ci'] = df['preds_mult'] * df['preds_ci']
    df['preds_ult_tl'] = df['preds_mult'] * df['preds_tl']

#     df = add_preds_at_top(df,'preds_ult')
    df = add_preds_at_top(df,'preds_ult_ci')
    df = add_preds_at_top(df,'preds_ult_tl')

    recall_score(df[LABEL_CI], df['preds_ult_ci_label']),\
    recall_score(df[LABEL_TL], df['preds_ult_tl_label'])

    return df

def model_performance(df, top = 1000):
    '''
    Prints the model performance (recall) on a df. Assumed that the df arg has LABEL_CI, LABEL_TL,
    preds_ult_ci_label, preds_ult_tl_label (obtained from pipeline_of_dfs_function)
    '''
    if df[LABEL_CI].sum() < top:
        max_recall_for_ci = 1
    else:
        max_recall_for_ci = top/df[LABEL_CI].sum()
        
    if df[LABEL_TL].sum() < top:
        max_recall_for_tl = 1
    else:
        max_recall_for_tl = top/df[LABEL_TL].sum()
        
    print('Max recall for CI: {}'.format(max_recall_for_ci))
    print('Attained recall for CI: {}'.format(recall_score(df[LABEL_CI], df['preds_ult_ci_label'])))
    print('Max recall for TL: {}'.format(max_recall_for_tl))
    print('Attained recall for TL: {}'.format(recall_score(df[LABEL_TL], df['preds_ult_tl_label'])))

    