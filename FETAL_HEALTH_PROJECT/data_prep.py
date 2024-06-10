
from utils import standard_scaling, smote_df
import pandas as pd
import numpy as np


def feature_engineering(df):
    df['N_total_decelerations'] = df['light_decelerations'] + df['severe_decelerations'] + df['prolonged_decelerations']
    df['N_short_term_variability_rate'] = df['abnormal_short_term_variability'] / (df['abnormal_short_term_variability'] + df['mean_value_of_short_term_variability'])
    df['N_long_term_variability_rate'] = df['abnormal_long_term_variability'] / (df['abnormal_long_term_variability'] + df['mean_value_of_long_term_variability'])
    df['N_abnormal_variability_score'] = df['abnormal_short_term_variability'] + df['abnormal_long_term_variability']
    df['N_mean_variability_score'] = df['mean_value_of_short_term_variability'] + df['mean_value_of_long_term_variability'] 
    df['N_total_variability'] = df['N_abnormal_variability_score'] + df['N_mean_variability_score']
    df['N_acceleration_ratio'] = df['accelerations'] / (df['fetal_movement'] + df['accelerations'] + df['uterine_contractions'])
    df['N_deceleration_ratio'] = df['N_total_decelerations'] / (df['fetal_movement'] + df['accelerations'] + df['uterine_contractions'])                                                                
    df["N_fetal_activity_ratio"] = df["fetal_movement"] / (df["fetal_movement"] + df["uterine_contractions"])
    df["histogram_symmetry"] = np.abs(df["histogram_mean"] - df["histogram_median"])
    return df
   

def data_prep_for_all(df):
  df = df.rename(columns = {'baseline value':'baseline_value', 
                          'percentage_of_time_with_abnormal_long_term_variability': 'abnormal_long_term_variability',
                          'prolongued_decelerations': 'prolonged_decelerations'}) 
  return df


def data_prep_for_modelling(df):
    df = data_prep_for_all(df)
    target_col = "fetal_health"
    df = df.drop_duplicates()
    #num_cols = [col for col in df.columns if col not in + [target_col]]
    #df[num_cols] = knn_imputer(df, num_cols, n_neighbors=5)
    
    df.loc[df['fetal_health']==1, 'fetal_health'] = 0
    df.loc[df['fetal_health']==2, 'fetal_health'] = 1
    df.loc[df['fetal_health']==3, 'fetal_health'] = 2

    num_cols = [col for col in df.columns if col not in [target_col]]
    df[num_cols] = df[num_cols] + 0.000001
    feature_engineering(df)

    X = df.drop("fetal_health", axis=1)
    y = df["fetal_health"]

    X_scaled = standard_scaling(X)
    X_smote, y_smote = smote_df(X_scaled, y)

    return X_smote, y_smote


def data_prep_for_prediction(X):

    X = data_prep_for_all(X)
    X = X + 0.0001
    X = feature_engineering(X)
    X = standard_scaling(X)

    return X

def data_prep_for_one_obv(X):
    X = X + 0.0001
    X = feature_engineering(X)
    return X