from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from snorkel.slicing import PandasSFApplier, slicing_function


@slicing_function()
def nlp_cnn(x):
    """Proyectos de NLP que utilizan convolución."""
    nlp_projects = "natural-language-processing" in x.tag
    convolution_projects = "CNN" in x.text or "convolution" in x.text
    return nlp_projects and convolution_projects


@slicing_function()
def short_text(x):
    """Proyectos con títulos y descripciones cortos."""
    return len(x.text.split()) < 8  # menos de 8 palabras


def get_slice_metrics(y_true: np.ndarray, y_pred: np.ndarray, slices: np.recarray) -> Dict:
    """Generar métricas para segmentos de datos.
    Args:
        y_true (np.ndarray): etiquetas reales.
        y_pred (np.ndarray): etiquetas de predicción.
        slices (np.recarray): segmentos generados.
    Returns:
        Dict: métricas de segmentos.
    """
    metrics = {}
    for slice_name in slices.dtype.names:
        mask = slices[slice_name].astype(bool)
        if sum(mask):
            slice_metrics = precision_recall_fscore_support(
                y_true[mask], y_pred[mask], average="micro"
            )
            metrics[slice_name] = {}
            metrics[slice_name]["precision"] = slice_metrics[0]
            metrics[slice_name]["recall"] = slice_metrics[1]
            metrics[slice_name]["f1"] = slice_metrics[2]
            metrics[slice_name]["num_samples"] = len(y_true[mask])
    return metrics


def get_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, classes: List, df: pd.DataFrame = None
) -> Dict:
    """Métricas de rendimiento utilizando verdades y predicciones.
    Args:
        y_true (np.ndarray): etiquetas reales.
        y_pred (np.ndarray): etiquetas de predicción.
        classes (List): lista de etiquetas de clase.
        df (pd.DataFrame, optional): dataframe para generar métricas de segmento.
                                     El valor default es None.
    Returns:
        Dict: métricas de performance.
    """
    # Performance
    metrics = {"overall": {}, "class": {}}

    # Métricas generales
    overall_metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    metrics["overall"]["precision"] = overall_metrics[0]
    metrics["overall"]["recall"] = overall_metrics[1]
    metrics["overall"]["f1"] = overall_metrics[2]
    metrics["overall"]["num_samples"] = np.float64(len(y_true))

    # Métricas por clase
    class_metrics = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i, _class in enumerate(classes):
        metrics["class"][_class] = {
            "precision": class_metrics[0][i],
            "recall": class_metrics[1][i],
            "f1": class_metrics[2][i],
            "num_samples": np.float64(class_metrics[3][i]),
        }

    # Métricas de secciones
    if df is not None:
        slices = PandasSFApplier([nlp_cnn, short_text]).apply(df)
        metrics["slices"] = get_slice_metrics(y_true=y_true, y_pred=y_pred, slices=slices)

    return metrics
