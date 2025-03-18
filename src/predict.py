import pandas as pd
import joblib
from feature_engineering import feat_transformer

# Carregar modelo
model = joblib.load('logistic_regression_model.pkl')

# Carregar dataset de predição
df_predict = pd.read_csv('data/processed/test.csv')
df_predict = df_predict[df_predict['split'] != 'train']
X_predict = df_predict.drop(columns=['account_id', 'offer_id', 'split', 'amount'])

# Fazer previsões
predictions = model.predict(X_predict)
probabilities = model.predict_proba(X_predict)[:, 1]

# Criar dataframe de saída
df_output = df_predict[['account_id', 'offer_id']].copy()
df_output['prediction'] = predictions
df_output['probability'] = probabilities

df_output.to_csv('predictions.csv', index=False)