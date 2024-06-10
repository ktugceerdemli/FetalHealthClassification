from utils import *
from  data_prep import data_prep_for_modelling
from config import all_scoring, skf, classifiers
import joblib


def main():
    df = pd.read_csv("data/fetal_health.csv")
    X, y = data_prep_for_modelling(df)
    best_model = hyperparameter_optimization(X, y, all_scoring, classifiers, cv=skf)
    lr = best_model["LR"].fit(X, y)
    joblib.dump(lr, "lr_model.pkl")
    #return voting_clf

if __name__ == "__main__":
    print("İşlem başladı")
    main()