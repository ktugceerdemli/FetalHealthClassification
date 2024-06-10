import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.neighbors import LocalOutlierFactor,  KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
import missingno as msn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate, GridSearchCV, learning_curve, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, make_scorer, roc_auc_score,
                             accuracy_score, precision_score, recall_score, f1_score)
from imblearn.over_sampling import SMOTE
import logging
logger = logging.getLogger('lightgbm')
logger.setLevel(logging.ERROR)


def check_df(dataframe, head=5):
    print("##################### Info #####################")
    print(dataframe.info())
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### Missing Values #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def num_summary(df, num_cols):
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    for col in num_cols:
        print(col.center(30), "NA:", df[col].isnull().sum(), "\n")
        print(df[col].describe(quantiles).T)
        print("-" * 20)


def num_plot(df, num_cols):
    sns.set_style("whitegrid")
    for col in num_cols:
        sns.histplot(df[col], bins = 50, kde = True)
        plt.title(col)
        plt.ylabel('Frequency')
        plt.show(block=True)


def box_plot(df, num_cols):
    sns.set_style("white")
    for col in num_cols:
        sns.boxplot(x=df[col], whis=4, color="pink")
        plt.title(col)
        plt.show(block = True)

        
def box_plot_with_target(df, target_col, num_cols):
    class_colors = {'Normal': '#366ECA', 'Suspect': '#E7992F', 'Pathological': '#FF4E4E'}
    sns.set_style("white")
    for col in num_cols:
        sns.boxplot(x=target_col, y=col, data=df, palette=class_colors, showmeans=True, showfliers=False)
        sns.despine()
        plt.title(col)
        plt.show(block = True)
        

def lineplot_with_target(df, target_col, num_cols):
    sns.set_palette("RdPu")
    sns.set_style("white")
    map_ = {'Normal':0, 'Suspect':1, 'Pathological':2}
    y = df[target_col].replace(map_)
    for col in num_cols:
        sns.lineplot(x=col, y=y, data=df, color="brown", ci=None)
        y_ax = [0,1,2]
        labels = ['Normal', 'Suspect', 'Pathological']
        plt.yticks(y_ax, labels)
        plt.title(f'{col} and {target_col}')
        plt.show(block=True)         


def cat_summary(df, cat_cols):
    for col in cat_cols:
        print(col.center(30), "NA:", df[col].isnull().sum(), "\n")
        print(pd.DataFrame({"Count": df[col].value_counts(), "Ratio": df[col].value_counts(normalize = True)}))
        print("-" * 20)


def cat_plot(df, cat_cols):
    sns.set_style("darkgrid")
    for col in cat_cols:
        df[col].value_counts().sort_values(ascending = False).plot(kind = "bar", color="lightblue")
        plt.title(col)
        plt.xticks(rotation = 45)
        plt.show(block = True)


def target_summary_with_cat(df, target_col, cat_cols):
    for col in cat_cols:
        if col != target_col:
            print(pd.DataFrame({target_col + " Mean": df.groupby(col)[target_col].mean()}))
            print("-" * 20, "\n")


def target_summary_with_num(df, target_col, num_cols):
    for col in num_cols:
        if col != target_col:
            print(df.groupby(target_col).agg({col: ["mean", "median"]}))
            print("-" * 20, "\n")


def plot_target_summary_with_num(df, target_col, num_cols):
    class_colors = {'Normal': '#366ECA', 'Suspect': '#E7992F', 'Pathological': '#FF4E4E'}
    sns.set_style("white")
    for col in num_cols:
        sns.barplot(x=target_col, y=col, data=df, palette=class_colors)
        plt.margins(0.2)
        plt.title(f'{col} and {target_col}')
        plt.show(block=True)



def plot_num_summary_with_target(df, target_col, num_cols):
    class_colors = {'Normal': '#366ECA', 'Suspect': '#E7992F', 'Pathological': '#FF4E4E'}
    sns.set_style("white")
    for col in num_cols:
        sns.kdeplot(data=df, x=col, hue=target_col, palette=class_colors,
                    fill=True, common_norm=False, alpha=.5, linewidth=0)
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.title(f"Density of {col}")
        plt.legend(title=target_col, labels=['Pathological','Normal','Suspect'])
        sns.despine()
        plt.show(block=True)

        
               
def high_correlated_cols(df, num_cols, corr_th = 0.90, plot = False):
    corr = df[num_cols].corr()
    corr_matrix = corr.abs()
    upper_triangle_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype("bool"))
    mask = np.triu(np.ones_like(num_cols, dtype=bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        sns.set(rc = {"figure.figsize": (15,15)})
        sns.heatmap(corr, mask = mask,  cmap = "RdBu", annot = True, fmt=".2f")
        plt.show()
    return drop_list


def plot_importance(model, features, num, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 15))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred),2)
    f1 = round(f1_score(y, y_pred, average="weighted"),2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="BrBG")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy: {0}, F1 {1}'.format(acc, f1), size=10)
    plt.show()


def val_curve_params(model, X, y, param_name, param_range, cv=5, scoring="f1"):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')
    plt.plot(param_range, mean_test_score,
             label="Test Score", color='g')
    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)

def lr_curve(X, y, estimator, cv=5, scoring="f1_weighted"):
        
    train_sizes, train_scores, test_scores = learning_curve(estimator=estimator, X=X, y=y, 
                                                            train_sizes=np.linspace(0.1, 1.0, 10), cv=cv, scoring=scoring)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Train')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Test')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.grid()
    #plt.xlabel('Eğitim seti büyüklüğü')
    plt.ylabel(scoring)
    plt.legend(loc='lower right')
    plt.ylim([0.5, 1.0])
    plt.show()


def base_models(X, y, scoring):

    print("Base Models....")
    
    classifiers = [('LR', LogisticRegression(random_state=42)),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC(random_state=42)),
                   ("CART", DecisionTreeClassifier(random_state=42)),
                   ("RF", RandomForestClassifier(random_state=42)),
                   ('Adaboost', AdaBoostClassifier(random_state=42)),
                   ('GBM', GradientBoostingClassifier(random_state=42)),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric = 'logloss', random_state=42)),
                   ('LightGBM', LGBMClassifier(verbose= -1, random_state=42)),
                   ('CatBoost', CatBoostClassifier(verbose=False, random_state=42)),
                   ('GNB', GaussianNB())
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=5, scoring=scoring, error_score="raise")
        print(name)
        print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
        print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
        print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}") 
        print(f"F1 Score: {round(cv_results['test_f1_score'].mean(), 4)}")
        print("-" * 20, "\n")
    
    
def smote_df(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote


def standard_scaling(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def hyperparameter_optimization(X, y, scoring, classifiers, cv=5):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cvresults = cross_validate(final_model, X, y, cv=cv, scoring=scoring, error_score="raise")
        print()
        print(f"Accuracy: {round(cvresults['test_accuracy'].mean(), 4)}")
        print(f"F1 Score: {round(cvresults['test_f1_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models


def voting_classifier(best_models, estimators, scoring, X, y):
    print("Voting Classifier...")
    voting_clf = VotingClassifier(estimators=estimators, voting='hard').fit(X, y)
    cv_results = cross_validate(voting_clf, X, y, cv=5, scoring=scoring)
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"Precision Score: {cv_results['test_precision'].mean()}")
    print(f"Recall Score: {cv_results['test_recall'].mean()}")
    print(f"F1 Score: {cv_results['test_f1_score'].mean()}")
    return voting_clf



