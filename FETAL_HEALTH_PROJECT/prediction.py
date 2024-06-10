
import joblib
import data_prep as dp

model = joblib.load("lr_model.pkl")

def predict(X):
   X =  dp.data_prep_for_prediction(X)
   preds = model.predict(X)
   return preds

def predict_for_one_obv(X):
   X =  dp.data_prep_for_one_obv(X)
   preds = model.predict(X)
   return preds