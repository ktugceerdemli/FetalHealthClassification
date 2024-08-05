#FETAL HEALTH CLASSIFICATION


#IMPORT LIBRARIES
#--------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore')

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import streamlit as st
# import missingno as msno
from datetime import date
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

from imblearn.over_sampling import SMOTE

from matplotlib.figure import Figure

# Modeling
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import  learning_curve

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression


pd.set_option('display.float_format', lambda x: '%.2f' % x)


#Data Information

def data_eda(df):
    df = df.rename(columns={'baseline value': 'baseline_value',
                            'percentage_of_time_with_abnormal_long_term_variability': 'abnormal_long_term_variability',
                            'prolongued_decelerations': 'prolonged_decelerations'})

    target_col = "fetal_health"
    num_cols = [col for col in df.columns if col not in [target_col]]

    map_fetal_health = {1: "Normal", 2: "Suspect", 3: "Pathological"}
    df["fetal_health"] = df["fetal_health"].replace(map_fetal_health)

    df = df.drop_duplicates()
    return df, target_col, num_cols
def check_df(dataframe, head=5):
    st.subheader("Data Analysis")

    st.markdown("First 10 value of dataset")
    dataframe1 = dataframe.head(10)
    st.dataframe(dataframe1)

    st.markdown("Last 10 value of dataset")
    dataframe2 = dataframe.tail(10)
    st.dataframe(dataframe2)

    st.markdown("Missing Values")
    dataframe3 = dataframe.isnull().sum()
    st.write(dataframe3)

    st.markdown("Statistical Analysis")
    dataframe4 = dataframe.describe([0.01, 0.05, 0.10, 0.50, 0.75, 0.90, 0.95, 0.99]).T
    st.write(dataframe4)

    st.write("Data Shape: ", dataframe.shape)

#Data Visualization

def box_plot(df, col):
    fig = Figure()
    fig.set_size_inches(16,6)
    ax = fig.subplots()

    sns.set_style("dark")

    sns.boxplot(x=df[col], data=df, ax=ax)
    ax.set_xlabel(col)
    st.pyplot(fig)

def hist_plot(df, col1, col2):
    fig = Figure()
    fig.set_size_inches(10,6)
    ax = fig.subplots()
    sns.histplot(data=df, x=df[col1], hue=df[col2], ax=ax)

    st.pyplot(fig)


def class_dist_plot(df, target_col):
    fig = Figure()
    fig.set_size_inches(5,3)
    ax = fig.subplots()

    ax.pie(
        df[target_col].value_counts(),
        autopct = '%.2f%%',
        labels = ["Normal", "Suspect", "Pathological"]
    )
    st.pyplot(fig)

def countpl(dataframe,col):
    fig = Figure()
    fig.set_size_inches(16, 6)
    ax = fig.subplots()
    sns.countplot(
        y=dataframe[col], data=dataframe,
        ax=ax,
    )
    ax.set_xlabel("Count")
    st.pyplot(fig)

def bar_plot(df, target_col, num_cols):
    fig = Figure()
    fig.set_size_inches(5, 3)
    ax = fig.subplots()

    sns.barplot(x=target_col, y=num_cols, data=df, ax=ax)

    ax.set_title(f'{num_cols} and {target_col}')
    ax.set_xlabel(target_col)
    ax.set_ylabel(num_cols)
    st.pyplot(fig)

def corelation_matrix(df, num_cols):
    corr = df[num_cols].corr()

    sns.heatmap(corr, annot=True)

    fig = Figure()
    fig.set_size_inches(40, 20)
    ax = fig.subplots()
    sns.heatmap(corr, cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
                annot=True, annot_kws={"size": 10}, square=True, ax=ax)
    st.pyplot(fig)

def smote_prep(df):
    hist_cols = [col for col in num_cols if "hist" in col]

    int_cols = ["baseline_value", "abnormal_short_term_variability", "abnormal_long_term_variability"]

    int_cols = int_cols + hist_cols

    for col in int_cols:
        df[col] = df[col].astype(int)

    # Drop highly corr columns
    df = df.drop(["histogram_mean", "histogram_median"], axis=1)
    return df

def smote_df(X, y):
    smote = SMOTE(random_state=42)
    smote_X, smote_y = smote.fit_resample(X, y)
    return smote_X, smote_y

def data_prep(df, num_cols):
    # Score of abnormal and normal variability

    df['N_abnormal_variability_score'] = df['abnormal_short_term_variability'] + df[
        'abnormal_long_term_variability']

    df['N_mean_variability_score'] = df['mean_value_of_short_term_variability'] + df[
        'mean_value_of_long_term_variability']

    # To avoid getting zero division error
    df[num_cols] = df[num_cols] + 0.0001

    df['N_total_decelerations'] = df['light_decelerations'] + df['severe_decelerations'] + df[
        'prolonged_decelerations']

    # Percentage of abnormal short and long variability rate
    df['N_short_term_variability_rate'] = df['abnormal_short_term_variability'] / (
                df['abnormal_short_term_variability'] + df['mean_value_of_short_term_variability'])
    df['N_long_term_variability_rate'] = df['abnormal_long_term_variability'] / (
                df['abnormal_long_term_variability'] + df[
            'mean_value_of_long_term_variability'])

    # Total Variability: Kısa vadeli ve uzun vadeli değişkenliğin toplamı.
    df['N_total_variability'] = df['abnormal_short_term_variability'] + df['mean_value_of_long_term_variability']

    # Acceleration Ratio: Accelerations sayısının toplam fetal hareket sayısına oranı.
    df['N_acceleration_ratio'] = df['accelerations'] / (
                df['fetal_movement'] + df['accelerations'] + df['uterine_contractions'])

    # Deceleration Ratio: Decelerations sayısının toplam fetal hareket sayısına oranı.
    df['N_deceleration_ratio'] = df['N_total_decelerations'] / (
                df['fetal_movement'] + df['accelerations'] + df['uterine_contractions'])

    # Fetal activity ratio
    df["N_fetal_activity_ratio"] = df["fetal_movement"] / (df["fetal_movement"] + df["uterine_contractions"])

    df["N_histogram_symmetry"] = np.abs(df["histogram_mean"] - df["histogram_median"])
    return df
def plot_confusion_matrix(y, y_pred):
  acc = round(accuracy_score(y,y_pred), 2)
  cm = confusion_matrix(y, y_pred)

  fig = Figure()
  fig.set_size_inches(40, 20)
  ax = fig.subplots()
  sns.heatmap(cm, annot=True, fmt=".0f", cbar=False, ax=ax)

  st.pyplot(fig)

  ax.set_xlabel("y_pred")
  ax.set_ylabel("y")
  ax.set_title("Accuracy Score: {0}".format(acc), size=10)



def lr_curve(X, y, estimator, cv=5, scoring="accuracy"):
    train_sizes, train_scores, test_scores = learning_curve(estimator=estimator, X=X, y=y,
                                                            train_sizes=np.linspace(0.1, 1.0, 10), cv=cv,
                                                            scoring=scoring)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig = Figure()
    fig.set_size_inches(40,20)
    ax = fig.subplots()
    sns.lineplot(x=train_sizes, y=train_mean, color='blue', marker='o', markersize=5, label='Train', ax= ax)
    ax.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    sns.lineplot(x=train_sizes, y=test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Test', ax=ax)
    ax.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

    ax.set_ylabel(scoring)
    ax.legend(loc='lower right')
    ax.set_ylim([0.5, 1.0])
    st.pyplot(fig)


#Define Sidebar Sections
sections = ['Welcome Page','Explatory Data Analysis', 'Prediction']

#Define the sidebar options for each section
section_options = {
    'Welcome Page': ['About Project', 'About Us'],
    'Explatory Data Analysis': ['Check data', 'Visualization of Data', 'Workflow'],
    'Prediction': ['Test Results', 'Prediction']

}

default_section = 'BHealthPrediction'
default_option = 'About Project'

sidebar = st.sidebar
with sidebar:
    st.title(":baby: Baby Health Prediction")
    st.write("Welcome to BHealthPrediction")
    st.write(":round_pushpin: Choose an option from below:")
    selected_section = sidebar.selectbox('Select section', sections, index=0)
    selected_option = sidebar.selectbox('Select option', section_options[selected_section], index=0)

#Read a dataset
df = pd.read_csv("datasets/fetal_health.csv")
df, target_col, num_cols = data_eda(df)

if selected_option == 'About Project':
    st.title("Welcome to BHealthPrediction")
    st.write("Our mission is to predict babies' health conditions using cardiogram (CTG) results." 
    " CTG is a crucial tool in obstetric care and provides valuable information for both mother and baby during or before labour."
    " Our platform is specifically designed to indicate fetal health providing predictions with ML methods.")


    st.header("Fetal Health Prediction")
    st.write("One of the most important indicators of a country's level of development is fetal mortality rate."
" Predicting infant mortality and identifying risk factors are crucial for developing health policies and directing resources.")

    st.image("PicturesfStreamlit/motherbaby.webp", caption="Fetal Health Prediction", use_column_width = True)

if selected_option == 'About Us':
    st.title("Who we are?")

    st.subheader("Kısmet Tugce Erdemli Al")
    st.image("PicturesfStreamlit/WhatsApp Image 2024-03-19 at 17.10.53.jpeg")

    st.subheader("Açelya Uslu")
    st.image("PicturesfStreamlit/açelya.png")

    st.subheader("Hatice Sönmez")
    st.image("PicturesfStreamlit/hatice.png")

    st.subheader("Sevdegül Güven")
    st.image("PicturesfStreamlit/sevdegül.png")

if selected_option == 'Check data':
    st.header("Exploratory Data Analysis")
    st.header("Data")
    st.markdown("Data from Kaggle: https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification/data")
    #Check our data set

    check = check_df(df)

    #Variables Types Info
    st.subheader("Variable Analysis")

    st.markdown("Target column:")
    st.info(target_col)
    st.write("Unique values of target column")
    st.dataframe(df["fetal_health"].value_counts())

    class_dist_plot(df, target_col)

    var1, var2 = st.columns(2)

    var1.markdown("Categorical Columns")
    var1.write("Only target column is categoical")
    var1.info(target_col)
    var1.text("Number of Variables")
    var1.write(len(target_col))

    var2.markdown("Numerical Columns")
    var2.info(num_cols)
    var2.text("Number of Variables")
    var2.write(len(num_cols))

if selected_option == 'Visualization of Data':

    st.subheader("Visualization of Data")


    st.header("Target Analysis")
    chart_choice = st.selectbox("Plots", ["Plot Type", "Box Plot", "Histogram Plot", "Bar Plot"])
    if chart_choice == "Plot Type":
        st.write("Please select plot type")

    if chart_choice == "Box Plot":

        for col in num_cols:
            box_plot(df, col=col)

    if chart_choice == "Histogram Plot":

        for col in num_cols:
            hist_plot(df, col, "fetal_health")

    if chart_choice == "Bar Plot":
        for col in num_cols:
            bar_plot(df, "fetal_health", col)


    st.header("Correlation Matrix")
    st.markdown("Correlation between numeric variables.")
    if st.button("Correlation Matrix"):
        corelation_matrix(df, num_cols)

if selected_option == 'Workflow':

    st.header("Workflow of Data")

    df = data_prep(df, num_cols)

    X = df.drop("fetal_health", axis=1)
    y = df["fetal_health"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_smote, y_smote = smote_df(X, y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    st.subheader("Modelling of Data")

    st.header("ML modelling")
    chart_choice = st.selectbox("Select model to run",
                                ["Models", "Logistic Regression", "SVC", "KNN"])

    if chart_choice == "Models":
        st.write("Please select model")
    if chart_choice == "Logistic Regression":
        lr_final = LogisticRegression(C=0.01, class_weight=None, max_iter=200, penalty=None, tol=0.001,
                                      random_state=42).fit(X_smote, y_smote)
        y_pred = lr_final.predict(X_smote)
        score_f1 = f1_score(y_smote, y_pred, average="weighted")
        score_accuracy = accuracy_score(y_smote, y_pred)

        st.write("To calculate the test scores press SCORE button")
        if st.button("SCORE"):
            st.write(f"f1 score :\n \n \n {score_f1}")
            st.write(f"Accuracy score :\n \n \n {score_accuracy}")

    if chart_choice == "SVC":
        svc_final = SVC(random_state=42).fit(X_smote, y_smote)

        y_pred_svc = svc_final.predict(X_smote)
        score_f1 = f1_score(y_smote, y_pred_svc, average="weighted")
        score_accuracy = accuracy_score(y_smote, y_pred_svc)

        st.write("To calculate the test scores press SCORE button")
        if st.button("SCORE"):
            st.write(f"f1 score :\n \n \n {score_f1}")
            st.write(f"Accuracy score :\n \n \n {score_accuracy}")

    if chart_choice == "KNN":
        knn_final = KNeighborsClassifier().fit(X_smote, y_smote)

        y_pred_knn = knn_final.predict(X_smote)

        score_f1 = f1_score(y_smote, y_pred_knn, average="weighted")
        score_accuracy = accuracy_score(y_smote, y_pred_knn)

        st.write("To calculate the test scores press SCORE button")
        if st.button("SCORE"):
            st.write(f"f1 score :\n \n \n {score_f1}")
            st.write(f"Accuracy score :\n \n \n {score_accuracy}")

if selected_option == "Test Results":
    st.header("Results")
    st.write("Selected model is KNN ")
    df = data_prep(df, num_cols)

    X = df.drop("fetal_health", axis=1)
    y = df["fetal_health"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_smote, y_smote = smote_df(X, y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lr_final = LogisticRegression(C=0.01, class_weight=None, max_iter=200, penalty=None, tol=0.001,
                                  random_state=42).fit(X_smote, y_smote)
    y_pred = lr_final.predict(X_smote)

    test_acc = accuracy_score(y_smote, y_pred)
    test_f1 = f1_score(y_smote, y_pred, average="weighted")
    test_precision = precision_score(y_smote, y_pred, average="weighted")
    test_recall = recall_score(y_smote, y_pred, average="weighted")

    st.write(f"Accuracy score :\n \n \n {test_acc}")
    st.write(f"F1 score :\n \n \n {test_f1}")
    st.write(f"Precision score :\n \n \n {test_precision}")
    st.write(f"Recall score :\n \n \n {test_recall}")

    st.subheader("Confusion Matrix")
    if st.button("Confusion Matrix Plot"):
        plot_confusion_matrix(y_smote, y_pred)

    st.subheader("Learning Curve")
    if st.button("Learning Curve"):
        lr_curve(X_smote, y_smote, lr_final, cv=5, scoring="f1_weighted")

elif selected_option == "Prediction":
    st.header("Predict your data")
    content = "You can insert your data below the slider."
    st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div></br>", unsafe_allow_html=True)


    col1, col2, col3 = st.columns(3)


    with col1:
        baseline_value = st.slider('Baseline Value', 100, 160)
        accelerations = st.slider('accelerations', 0.0,0.019)
        fetal_movement = st.slider('fetal_movement', 0.0, 0.481)
        uterine_contractions = st.slider('uterine_contractions', 0.0,0.015)
        light_decelerations = st.slider('light_decelerations', 0.0,0.015)
        severe_decelerations = st.slider('severe_decelerations', 0.0, 0.001)
        prolonged_decelerations = st.slider('prolonged_decelerations',0.0, 0.005)

    with col2:
        abnormal_short_term_variability = st.slider('abnormal_short_term_variability', 12, 87)
        mean_value_of_short_term_variability = st.slider('mean_value_of_short_term_variability', 0.2,7.0)
        abnormal_long_term_variability= st.slider('abnormal_long_term_variability', 0,91)
        mean_value_of_long_term_variability = st.slider('mean_value_of_long_term_variability', 0,50)
        histogram_width = st.slider('histogram_width', 3,180)
        histogram_min = st.slider('histogram_min', 50,159)
        histogram_max = st.slider('histogram_max', 122,238)

    with col3:
        histogram_number_of_peaks = st.slider('histogram_number_of_peaks', 0,18)
        histogram_number_of_zeroes = st.slider('histogram_number_of_zeroes', 0,10)
        histogram_mode = st.slider('histogram_mode', 60,187)
        histogram_mean = st.slider('histogram_mean', 73,182)
        histogram_median = st.slider('histogram_median', 77, 186)
        histogram_variance = st.slider('histogram_variance',0, 269)
        histogram_tendency = st.slider('histogram_tendency',-1,0,1)

    data = {'baseline_value': baseline_value , 'accelerations':accelerations , 'fetal_movement' : fetal_movement , 'uterine_contractions': uterine_contractions,
                                     'light_decelerations': light_decelerations, 'severe_decelerations' : severe_decelerations, 'prolonged_decelerations': prolonged_decelerations,
                                     'abnormal_short_term_variability': abnormal_short_term_variability, 'mean_value_of_short_term_variability': mean_value_of_short_term_variability,
                                     'abnormal_long_term_variability': abnormal_long_term_variability,
                                     'mean_value_of_long_term_variability': mean_value_of_long_term_variability, 'histogram_width': histogram_width, 'histogram_min': histogram_min, 'histogram_max': histogram_max,
                                     'histogram_number_of_peaks': histogram_number_of_peaks, 'histogram_number_of_zeroes': histogram_number_of_zeroes, 'histogram_mode': histogram_mode,
                                     'histogram_mean': histogram_mean, 'histogram_median': histogram_median, 'histogram_variance': histogram_variance, 'histogram_tendency': histogram_tendency,
                                     'N_abnormal_variability_score': abnormal_short_term_variability + abnormal_short_term_variability, 'N_mean_variability_score': mean_value_of_short_term_variability + mean_value_of_long_term_variability,
                                    'N_total_decelerations': light_decelerations + severe_decelerations + prolonged_decelerations,
                                    'N_short_term_variability_rate': abnormal_short_term_variability / (abnormal_short_term_variability + mean_value_of_short_term_variability),
                                    'N_long_term_variability_rate': abnormal_long_term_variability / (abnormal_long_term_variability + mean_value_of_long_term_variability + 0.0001),
                                    'N_total_variability': abnormal_short_term_variability + mean_value_of_long_term_variability,
                                    'N_acceleration_ratio': accelerations / (fetal_movement + accelerations + uterine_contractions + 0.0001),
                                    'N_deceleration_ratio': (light_decelerations + severe_decelerations + prolonged_decelerations) / (fetal_movement + accelerations + uterine_contractions + 0.001),
                                    'N_fetal_activity_ratio': fetal_movement / (fetal_movement + uterine_contractions + 0.001),
                                    'N_histogram_symmetry': np.abs(histogram_mean - histogram_median)
                                        }

    data = pd.DataFrame(data, index=[0])
    df = data_prep(df, num_cols)

    X = df.drop("fetal_health", axis=1)
    y = df["fetal_health"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_smote, y_smote = smote_df(X, y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lr_final = LogisticRegression(C=0.01, class_weight=None, max_iter=200, penalty=None, tol=0.001,
                                  random_state=42).fit(X_smote, y_smote)
    predict = lr_final.predict(data)

    if st.button("Predict"):
        st.write(f"Our prediction is :\n \n \n {predict}")

    if st.button("Clear"):
        st.rerun()






