# Predição de Aceitação de Ofertas

Este repositório contém um pipeline para prever a aceitação de ofertas usando métodos de Machine Learning, incluindo regressão logística e XGBoost. O objetivo é treinar modelos preditivos e gerar probabilidades de conversão para ofertas personalizadas.

## Estrutura do Projeto

- `data/`: Contém os datasets utilizados no treinamento e teste.
- `notebooks/`: Contém notebooks exploratórios e de modelagem.
- `src/`: Implementação dos modelos e pipeline.
- `README.md`: Documentação do projeto.

## Dataset
O dataset principal possui as seguintes colunas:

- **account_id**: Identificação do cliente.
- **split**: Indica se a amostra pertence ao conjunto de treino (`train`), validação (`validation`) ou teste (`test`).
- **offer_id**: Identificação da oferta.
- **offer_accepted**: Indica se a oferta foi aceita (0 ou 1).
- **Variáveis adicionais**: Incluem informações demográficas, histórico de compras e características da oferta.

## Treinamento e Otimização

1. **Divisão dos Dados**:
   - Treinamento: `X_train`, `y_train`
   - Validação: `X_validation`, `y_validation`
   - Teste: `X_test`, `y_test`

2. **Otimização de Hiperparâmetros**:
   - Realizada usando `X_validation` e `y_validation`.
   - Utiliza `hyperopt` para encontrar a melhor configuração do modelo.

3. **Treinamento Final**:
   - O modelo é treinado com `X_train` e `y_train` usando os hiperparâmetros otimizados.

4. **Avaliação**:
   - O modelo é avaliado no conjunto de teste (`X_test`, `y_test`).

## Geração do Dataset de Predição

Após treinar o modelo, é possível gerar um dataset com as previsões:

```python
import pandas as pd

df_pred = df_test[['account_id', 'offer_id']].copy()
df_pred['prediction'] = y_pred
df_pred['probability'] = y_pred_prob[:, 1]  # Probabilidade de aceitação da oferta

df_pred.to_csv('predictions.csv', index=False)
```

### Estrutura do Arquivo `predictions.csv`

|              account_id               |               offer_id              | prediction | probability |
|---------------------------------------|-------------------------------------|------------|-------------|
| abc4359eb34e4e2ca2349da2ddf771b6      | 0b1e1539f2cc45b7b9fa7c272da2e1d7    | 1          | 0.85        |
| f6c178ca2b1847d6b91fb46123b44981      | f19421c1d4aa40978ebb69ca19b0e20d    | 0          | 0.30        |

## Contato
Para mais informações ou dúvidas, entre em contato:
email: mclara.castro@hotmail.com