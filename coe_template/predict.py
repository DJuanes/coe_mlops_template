from typing import Dict, List

import numpy as np


def custom_predict(y_prob: np.ndarray, threshold: float, index: int) -> np.ndarray:
    """Función de predicción personalizada que por defecto
    es un índice si no se cumplen las condiciones.
    Args:
        y_prob (np.ndarray): probabilidades pronosticadas.
        threshold (float): puntaje softmax mínimo para predecir la clase mayoritaria.
        index (int): índice de etiqueta que se utilizará si no se cumplen
                     las condiciones personalizadas.
    Returns:
        np.ndarray: índices de etiquetas pronosticadas.
    """
    y_pred = [np.argmax(p) if max(p) > threshold else index for p in y_prob]
    return np.array(y_pred)


def predict(texts: List, artifacts: Dict) -> List:
    """Predecir etiquetas para textos dados.
    Args:
        texts (List): textos de entrada sin procesar para clasificar.
        artifacts (Dict): artefactos de una ejecución.
    Returns:
        List: predicciones para textos de entrada.
    """
    x = artifacts["vectorizer"].transform(texts)
    y_pred = custom_predict(
        y_prob=artifacts["model"].predict_proba(x),
        threshold=artifacts["args"].threshold,
        index=artifacts["label_encoder"].class_to_index["other"],
    )
    tags = artifacts["label_encoder"].decode(y_pred)
    predictions = [
        {
            "input_text": texts[i],
            "predicted_tag": tags[i],
        }
        for i in range(len(tags))
    ]
    return predictions
