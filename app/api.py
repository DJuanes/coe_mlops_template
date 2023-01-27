from datetime import datetime
from functools import wraps
from http import HTTPStatus
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, Request

from app.schemas import PredictPayload
from coe_template import main, predict
from config import config
from config.config import logger

# Definir aplicación
app = FastAPI(
    title="CoE MLOps Template",
    description="Clasificación de proyectos de machine learning.",
    version="0.1",
)


@app.on_event("startup")
def load_artifacts():
    global artifacts
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = main.load_artifacts(run_id=run_id)
    logger.info("Listo para la inferencia!")


def construct_response(f):
    """Construir una respuesta JSON para un endpoint."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs) -> Dict:
        results = f(request, *args, **kwargs)
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }
        if "data" in results:
            response["data"] = results["data"]
        return response

    return wrap


@app.get("/", tags=["General"])
@construct_response
def _index(request: Request) -> Dict:
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.get("/performance", tags=["Performance"])
@construct_response
def _performance(request: Request, filter: str = None) -> Dict:
    """Obtener las métricas de performance."""
    performance = artifacts["performance"]
    data = {"performance": performance.get(filter, performance)}
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": data,
    }
    return response


@app.get("/args/{arg}", tags=["Arguments"])
@construct_response
def _arg(request: Request, arg: str) -> Dict:
    """Obtener el valor de un parámetro específico utilizado para la ejecución."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            arg: vars(artifacts["args"]).get(arg, ""),
        },
    }
    return response


@app.get("/args", tags=["Arguments"])
@construct_response
def _args(request: Request) -> Dict:
    """Obtener todos los argumentos utilizados para la ejecución."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "args": vars(artifacts["args"]),
        },
    }
    return response


@app.post("/predict", tags=["Prediction"])
@construct_response
def _predict(request: Request, payload: PredictPayload) -> Dict:
    """Predecir etiquetas para una lista de textos."""
    texts = [item.text for item in payload.texts]
    predictions = predict.predict(texts=texts, artifacts=artifacts)
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"predictions": predictions},
    }
    return response
