import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE
from hyperopt import fmin, tpe, hp
from sklearn.base import clone
import joblib
from feature_engineering import feat_transformer

# Carregar dataset
df_train = pd.read_csv('data/processed/train.csv')
df_validation = pd.read_csv('data/processed/validation.csv')

X_train = df_train.drop(columns=['offer_accepted', 'split'])
y_train = df_train['offer_accepted']

X_validation = df_validation.drop(columns=['offer_accepted', 'split'])
y_validation = df_validation['offer_accepted']

# Pipeline de regressão logística
lr_pipe = Pipeline([
    ('feat_trans', feat_transformer),
    ('over', SMOTE()),
    ('logreg', LogisticRegression(random_state=0))
])

# Espaço de otimização
lr_opt_space = {
    'logreg__warm_start': hp.choice('logreg__warm_start', [True, False]),
    'logreg__fit_intercept': hp.choice('logreg__fit_intercept', [True, False]),
    'logreg__tol': hp.uniform('logreg__tol', 0.00001, 0.0001),
    'logreg__C': hp.uniform('logreg__C', 0.05, 3),
    'logreg__solver': hp.choice('logreg__solver', ['newton-cg', 'lbfgs', 'liblinear']),
    'logreg__class_weight': 'balanced'
}

# Função de otimização
def optimize_hyperparameters(opt_space, pipe, X, y, max_evals=100):
    def obj(params):
        model = clone(pipe).set_params(**params)
        preds = cross_val_predict(model, X, y, cv=3, n_jobs=-1)
        return -f1_score(y, preds, average='macro')
    
    best_hypers = fmin(obj, space=opt_space, algo=tpe.suggest, max_evals=max_evals, return_argmin=False)
    return best_hypers

# Otimizar hiperparâmetros
best_hypers = optimize_hyperparameters(lr_opt_space, lr_pipe, X_validation, y_validation)

# Treinar modelo final
model = clone(lr_pipe).set_params(**best_hypers).fit(X_train, y_train)

# Salvar modelo
joblib.dump(model, 'logistic_regression_model.pkl')
