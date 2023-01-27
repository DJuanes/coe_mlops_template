import json
import random
from typing import Dict

import numpy as np


def load_dict(filepath: str) -> Dict:
    """Cargar un diccionario desde la ruta de archivo de un JSON.
    Args:
        filepath (str): ubicación del archivo.
    Returns:
        Dict: datos JSON cargados.
    """
    with open(filepath) as fp:
        d = json.load(fp)
    return d


def save_dict(d: Dict, filepath: str, cls=None, sortkeys: bool = False) -> None:
    """Guardar un diccionario en una ubicación específica.
    Args:
        d (Dict): datos a guardar.
        filepath (str): ubicación de dónde guardar los datos.
        cls (optional): codificador para usar en datos. Valor default es None.
        sortkeys (bool, optional): si ordenar las claves alfabéticamente. Valor default es False.
    """
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)


def set_seeds(seed: int = 42) -> None:
    """Setear seeds.
    Args:
        seed (int, optional): número que se utilizará como seed. Valor default es 42.
    """
    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
