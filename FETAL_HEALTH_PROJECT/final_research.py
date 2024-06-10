from utils import *
from config import *
import joblib
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

df_ = pd.read_csv("data/fetal_health.csv")
df = df_.copy()

df = df.rename(columns = {'baseline value':'baseline_value', 
                          'percentage_of_time_with_abnormal_long_term_variability': 'abnormal_long_term_variability',
                          'prolongued_decelerations': 'prolonged_decelerations'})

df.loc[df['fetal_health']==1, 'fetal_health'] = 0
df.loc[df['fetal_health']==2, 'fetal_health'] = 1
df.loc[df['fetal_health']==3, 'fetal_health'] = 2


target_col = "fetal_health"
num_cols = [col for col in df.columns if col not in [target_col]]

df = df.drop_duplicates()

df[num_cols] = df[num_cols] + 0.000001

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

X = df.drop("fetal_health", axis=1)
y = df["fetal_health"]

X_scaled = standard_scaling(X)
X_smote, y_smote = smote_df(X_scaled, y)

best_model = hyperparameter_optimization(X_smote, y_smote, all_scoring, classifiers, cv=5)
lr = best_model["LR"].fit(X_smote, y_smote)

joblib.dump(lr, "lr_model.pkl")








