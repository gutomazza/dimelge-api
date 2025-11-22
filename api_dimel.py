# api_dimel.py
# API para expor o modelo DIMEL (GradientBoosting) via HTTP

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from modelo_dimel import carregar_modelo, prever_cluster, gerar_laudo_dimel

app = FastAPI(
    title="API DIMEL",
    description="API para classificação de seções eleitorais usando o modelo DIMEL (GradientBoosting).",
    version="1.0.0",
)

# Carrega o modelo na inicialização da API
modelo = carregar_modelo()


class EntradaDIMEL(BaseModel):
    QT_APTOS: int
    QT_IDOSOS: int
    QT_DEFICIENTES: int
    QT_BAIXA_ESCOLARIDADE: int
    QT_BIOMETRIA: int
    NM_MUNICIPIO: Optional[str] 
    NR_ZONA: Optional[int] 
    NR_SECAO: Optional[int]


class SaidaDIMEL(BaseModel):
    CLUSTER_PREDITO: int
    PROB_CLUSTER_1: float
    PROB_CLUSTER_2: float
    PROB_CLUSTER_3: float
    LAUDO: str


@app.post("/dimel/prever", response_model=SaidaDIMEL)
def dimel_prever(dados: EntradaDIMEL):
    """
    Endpoint principal da API DIMEL.
    """
    registro = dados.dict()
    resultado = prever_cluster(modelo, registro)
    laudo = gerar_laudo_dimel(registro, resultado)

    saida = SaidaDIMEL(
        CLUSTER_PREDITO=resultado["CLUSTER_PREDITO"],
        PROB_CLUSTER_1=resultado["PROB_CLUSTER_1"],
        PROB_CLUSTER_2=resultado["PROB_CLUSTER_2"],
        PROB_CLUSTER_3=resultado["PROB_CLUSTER_3"],
        LAUDO=laudo,
    )
    return saida
