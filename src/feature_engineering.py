import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.impute import MissingIndicator

# Carregar dataset
df = pd.read_csv('data/processed/df_customers_offers.csv')

# Definir features numéricas e categóricas
target = 'offer_accepted'
key_feats = ['account_id', 'offer_id', 'split', 'amount']
feats = df_features_pd.drop([target] + key_feats, axis=1).columns
num_feats = [feat for feat in feats if df_features_pd[feat].dtype != 'O']
cat_feats = [feat for feat in feats if feat not in num_feats]

# Definir transformações
num_transformer = FeatureUnion([
    ('num_pipe', Pipeline([
        ('scaler', StandardScaler()),
        ('imputer', SimpleImputer())
    ])),
    ('nan_flag', MissingIndicator(error_on_new=False))
])

feat_transformer = ColumnTransformer([
    ('num_trans', num_transformer, num_feats),
    ('cat_trans', OneHotEncoder(handle_unknown='ignore'), cat_feats)
], remainder='passthrough', sparse_threshold=0)

# Dividir dataset
df_train = df[df['split'] == 'train']
df_validation = df[df['split'] == 'validation']
df_test = df[df['split'] == 'test']

df_train.to_csv('data/processed/train.csv', index=False)
df_validation.to_csv('data/processed/validation.csv', index=False)
df_test.to_csv('data/processed/test.csv', index=False)
