# modelo_dimel.py
# Projeto DIMEL – Classificação de CLUSTER por seção eleitoral
#
# VERSÃO MAIS ROBUSTA:
# Modelo base: GradientBoostingClassifier
# Variáveis explicativas:
#   - QT_APTOS
#   - QT_IDOSOS
#   - QT_DEFICIENTE
#   - QT_BAIXA_ESCOLARIDADE
#   - QT_BIOMETRIA

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path
from joblib import dump

# -----------------------------
# Configurações gerais
# -----------------------------

ARQUIVO_DADOS = "dados_analise03.xlsx"
ARQUIVO_MODELO = "modelo_dimel.pkl"

FEATURES = ["QT_APTOS", "QT_IDOSOS", "QT_DEFICIENTE", "QT_BAIXA_ESCOLARIDADE","QT_BIOMETRIA"]
TARGET = "CLUSTER"


# -----------------------------
# Função de treino e salvamento
# -----------------------------

def treinar_e_salvar_modelo(
    caminho_dados: str = ARQUIVO_DADOS,
    caminho_modelo: str = ARQUIVO_MODELO,
):
    """
    VERSÃO MAIS ROBUSTA – GradientBoosting
    """

    # 1) Carregar base
    df = pd.read_excel(caminho_dados)

    # 2) Selecionar colunas e remover linhas com NA nessas colunas
    df_modelo = df[FEATURES + [TARGET]].dropna()

    X = df_modelo[FEATURES]
    y = df_modelo[TARGET].astype(int)

    # 3) Separar treino e teste (30% para teste, estratificado)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    # 4) Definir o modelo GradientBoosting (VERSÃO MAIS ROBUSTA)
    modelo = GradientBoostingClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    # 5) Treinar
    modelo.fit(X_train, y_train)
    dump(modelo, "modelo_dimel_v2.pkl")

    # 6) Avaliar
    y_pred = modelo.predict(X_test)

    print("\n=== RELATÓRIO DE CLASSIFICAÇÃO (DIMEL – GradientBoosting) ===")
    print(classification_report(y_test, y_pred, digits=4))

    print("\n=== MATRIZ DE CONFUSÃO (linhas = real, colunas = predito) ===")
    print(confusion_matrix(y_test, y_pred))

    # 7) Salvar modelo treinado
    joblib.dump(modelo, caminho_modelo)
    print(f"\nModelo salvo em: {Path(caminho_modelo).resolve()}")


# -----------------------------
# Funções de uso pelo agente
# -----------------------------

def carregar_modelo(caminho_modelo: str = ARQUIVO_MODELO):
    """
    Carrega o modelo DIMEL salvo em disco (VERSÃO MAIS ROBUSTA – GradientBoosting).
    """
    modelo = joblib.load(caminho_modelo)
    return modelo


def prever_cluster(modelo, registro: dict) -> dict:
    """
    Usa o modelo treinado para prever o CLUSTER de um novo registro.
    """

    # Garantir que todas as features necessárias existem
    for f in FEATURES:
        if f not in registro:
            raise ValueError(f"Campo obrigatório ausente: {f}")

    # Montar DataFrame na ordem correta das features
    x = pd.DataFrame(
        [[registro[f] for f in FEATURES]],
        columns=FEATURES,
    )

    # Probabilidades e classe
    proba = modelo.predict_proba(x)[0]
    classe = int(modelo.predict(x)[0])

    # Mapeamento classe ↔ índice
    classes_modelo = modelo.classes_.tolist()
    idx1 = classes_modelo.index(1)
    idx2 = classes_modelo.index(2)
    idx3 = classes_modelo.index(3)

    saida = {
        "CLUSTER_PREDITO": classe,
        "PROB_CLUSTER_1": float(proba[idx1]),
        "PROB_CLUSTER_2": float(proba[idx2]),
        "PROB_CLUSTER_3": float(proba[idx3]),
    }

    return saida


# -----------------------------
# Gerador de laudo textual DIMEL
# -----------------------------

def _fmt_num(x, casas=1):
    """Formata número com vírgula decimal (estilo PT-BR)."""
    if x is None:
        return ""
    return f"{x:.{casas}f}".replace(".", ",")


def gerar_laudo_dimel(registro: dict, resultado: dict) -> str:
    """
    Gera um laudo textual padrão DIMEL.
    """

    qt_aptos = int(registro.get("QT_APTOS", 0))
    qt_idosos = int(registro.get("QT_IDOSOS", 0))
    qt_def = int(registro.get("QT_DEFICIENTE", 0))
    qt_baixa = int(registro.get("QT_BAIXA_ESCOLARIDADE", 0))
    qt_biometria = int(registro.get("QT_BIOMETRIA", 0))

    if qt_aptos > 0:
        pct_idosos = 100 * qt_idosos / qt_aptos
        pct_def = 100 * qt_def / qt_aptos
        pct_baixa = 100 * qt_baixa / qt_aptos
    else:
        pct_idosos = pct_def = pct_baixa = 0.0

    cluster = int(resultado.get("CLUSTER_PREDITO"))
    p1 = float(resultado.get("PROB_CLUSTER_1", 0.0))
    p2 = float(resultado.get("PROB_CLUSTER_2", 0.0))
    p3 = float(resultado.get("PROB_CLUSTER_3", 0.0))

    municipio = registro.get("NM_MUNICIPIO")
    zona = registro.get("NR_ZONA")
    secao = registro.get("NR_SECAO")

    identificacao_partes = []
    if municipio:
        identificacao_partes.append(f"Município de {municipio}")
    if zona is not None:
        identificacao_partes.append(f"Zona {zona}")
    if secao is not None:
        identificacao_partes.append(f"Seção {secao}")

    if identificacao_partes:
        identificacao = " – ".join(identificacao_partes)
    else:
        identificacao = "Seção eleitoral analisada"

    laudo = f"""LAUDO DE CLASSIFICAÇÃO DIMEL (VERSÃO MAIS ROBUSTA – GradientBoosting)

1. Identificação da unidade analisada
{identificacao}.

2. Dados utilizados no modelo
- Eleitores aptos (QT_APTOS): {qt_aptos}
- Eleitores idosos (QT_IDOSOS): {qt_idosos} ({_fmt_num(pct_idosos)}% dos aptos)
- Eleitores com deficiência (QT_DEFICIENTE): {qt_def} ({_fmt_num(pct_def)}% dos aptos)
- Eleitores com baixa escolaridade (QT_BAIXA_ESCOLARIDADE): {qt_baixa} ({_fmt_num(pct_baixa)}% dos aptos)
- Eleitores com biometria(QT_BIOMETRIA): {qt_biometria} ({_fmt_num(pct_biometria)}% dos aptos)

3. Resultado da classificação
- CLUSTER predito: {cluster}

Probabilidades:
- P(CLUSTER = 1): {_fmt_num(100 * p1)}%
- P(CLUSTER = 2): {_fmt_num(100 * p2)}%
- P(CLUSTER = 3): {_fmt_num(100 * p3)}%

4. Interpretação
Este resultado reflete a similaridade estatística da seção em relação às seções históricas usadas no treinamento do modelo DIMEL.

5. Observações
O laudo baseia-se exclusivamente nas variáveis informadas e no modelo vigente na data de sua execução.
"""

    return laudo


# -----------------------------
# Exemplo de uso em linha de comando
# -----------------------------

if __name__ == "__main__":
    treinar_e_salvar_modelo()
    
