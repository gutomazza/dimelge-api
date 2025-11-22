import sys
import os
import pandas as pd
from joblib import load

def main():
    # Verifica argumentos de linha de comando
    if len(sys.argv) < 3:
        print("Uso: python prever.py <arquivo_entrada.xlsx> <arquivo_saida.xlsx>")
        sys.exit(1)

    caminho_entrada = sys.argv[1]
    caminho_saida = sys.argv[2]

    # Verifica se o arquivo de entrada existe
    if not os.path.exists(caminho_entrada):
        print(f"Erro: arquivo de entrada '{caminho_entrada}' não encontrado.")
        sys.exit(1)

    # 1) Carrega o modelo treinado (pipeline ou modelo simples)
    modelo = load("modelo_dimel.pkl")  # ajuste se o nome for diferente

    # 2) Lê os novos dados
    df_novos = pd.read_excel(caminho_entrada)

    # 3) Seleciona as colunas usadas no treinamento
    colunas_features = [
        "QT_APTOS",
        "QT_IDOSOS",
        "QT_DEFICIENTES",
        "QT_BAIXA_ESCOLARIDADE",
        "QT_BIOMETRIA"
    ]  # ajuste para as suas colunas reais

    # Verificação das colunas obrigatórias
    for col in colunas_features:
        if col not in df_novos.columns:
            print(f"Erro: coluna obrigatória '{col}' não encontrada no arquivo de entrada.")
            sys.exit(1)

    # Copia apenas as features
    X_novos = df_novos[colunas_features].copy()

    # 3a) Marca linhas que têm NaN nas features
    df_novos["TEM_NAN_FEATURES"] = X_novos.isna().any(axis=1)

    # 3b) Trata valores faltantes (aqui substituo por 0; adapte se quiser outra regra)
    if X_novos.isna().any().any():
        print("Aviso: foram encontrados valores NaN nas features. Substituindo por 0.")
        X_novos = X_novos.fillna(0)

    # 4) Faz as previsões
    df_novos["PREVISAO"] = modelo.predict(X_novos)

    # (Opcional) se o modelo for de classificação e tiver predict_proba:
    # df_novos["PROB_CLASSE_1"] = modelo.predict_proba(X_novos)[:, 1]

    # 5) Salva o resultado em um novo Excel
    df_novos.to_excel(caminho_saida, index=False)

    print(f"Previsões geradas com sucesso em: {caminho_saida}")

if __name__ == "__main__":
    main()
