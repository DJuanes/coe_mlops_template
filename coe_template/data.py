import json
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

from config import config


def replace_oos_labels(
    df: pd.DataFrame, labels: List, label_col: str, oos_label: str = "other"
) -> pd.DataFrame:
    """Reemplazar las etiquetas fuera de alcance (oos).
    Args:
        df (pd.DataFrame): DataFrame Pandas con los datos.
        labels (List): lista de etiquetas aceptadas.
        label_col (str): nombre de la columna del dataframe que tiene las etiquetas.
        oos_label (str, optional): nombre de la nueva etiqueta para etiquetas OOS.
                                   El valor default es "other".
    Returns:
        pd.DataFrame: Dataframe con etiquetas OOS reemplazadas.
    """
    oos_tags = [item for item in df[label_col].unique() if item not in labels]
    df[label_col] = df[label_col].apply(lambda x: oos_label if x in oos_tags else x)
    return df


def replace_minority_labels(
    df: pd.DataFrame, label_col: str, min_freq: int, new_label: str = "other"
) -> pd.DataFrame:
    """Reemplazar las etiquetas minoritarias con otra etiqueta.
    Args:
        df (pd.DataFrame): DataFrame Pandas con los datos.
        label_col (str): nombre de la columna del dataframe que tiene las etiquetas.
        min_freq (int): número mínimo de puntos de datos que debe tener una etiqueta.
        new_label (str, optional): nombre de la nueva etiqueta para reemplazar
                                  las etiquetas minoritarias. El valor default es "other".
    Returns:
        pd.DataFrame: Dataframe con etiquetas minoritarias reemplazadas.
    """
    labels = Counter(df[label_col].values)
    labels_above_freq = Counter(label for label in labels.elements() if (labels[label] >= min_freq))
    df[label_col] = df[label_col].apply(lambda label: label if label in labels_above_freq else None)
    df[label_col] = df[label_col].fillna(new_label)
    return df


def clean_text(text: str, lower: bool, stem: bool, stopwords=config.STOPWORDS) -> str:
    """Limpiar texto sin procesar.
    Args:
        text (str): texto sin procesar que se va a limpiar.
        lower (bool): si se pone en minúsculas el texto.
        stem (bool): si se debe derivar el texto.
    Returns:
        str: texto limpio.
    """
    # Lower
    if lower:
        text = text.lower()

    # Remover stopwords
    if len(stopwords):
        pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
        text = pattern.sub("", text)

    # Espaciado y filtros
    text = re.sub(
        r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text
    )  # añadir espacio entre los objetos a filtrar
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # eliminar caracteres no alfanuméricos
    text = re.sub(" +", " ", text)  # eliminar múltiples espacios
    text = text.strip()  # eliminar espacios en blanco en los extremos

    # Quitar links
    text = re.sub(r"http\S+", "", text)

    # Stemming
    if stem:
        stemmer = PorterStemmer()
        text = " ".join([stemmer.stem(word, to_lowercase=lower) for word in text.split(" ")])

    return text


def preprocess(df: pd.DataFrame, lower: bool, stem: bool, min_freq: int) -> pd.DataFrame:
    """Preprocesar los datos.
    Args:
        df (pd.DataFrame): DataFrame Pandas con los datos.
        lower (bool): si se pone en minúsculas el texto.
        stem (bool): si se debe derivar el texto.
        min_freq (int): número mínimo de puntos de datos que debe tener una etiqueta.
    Returns:
        pd.DataFrame: Dataframe con datos preprocesados.
    """
    df["text"] = df.title + " " + df.description  # feature engineering
    df.text = df.text.apply(clean_text, lower=lower, stem=stem)  # limpiar texto
    df = replace_oos_labels(
        df=df, labels=config.ACCEPTED_TAGS, label_col="tag", oos_label="other"
    )  # reemplazar etiquetas OOS
    df = replace_minority_labels(
        df=df, label_col="tag", min_freq=min_freq, new_label="other"
    )  # reemplazar las etiquetas por debajo de la frecuencia mínima

    return df


class LabelEncoder:
    """Codificar las etiquetas en índices únicos.

    ```python
    # Codificar etiquetas
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    y = label_encoder.encode(labels)
    ```
    """

    def __init__(self, class_to_index: Dict = {}) -> None:
        """Inicializar el codificador de etiquetas.
        Args:
            class_to_index (Dict, optional): mapeo entre clases e índices únicos.
                                             El valor default es {}.
        """
        self.class_to_index = class_to_index or {}  # mutable defaults ;)
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())

    def __len__(self):
        return len(self.class_to_index)

    def __str__(self):
        return f"<LabelEncoder(num_classes={len(self)})>"

    def fit(self, y: List):
        """Ajustar una lista de etiquetas al codificador.
        Args:
            y (List): etiquetas sin procesar.
        Returns:
            Instancia de LabelEncoder ajustada.
        """
        classes = np.unique(y)
        for i, class_ in enumerate(classes):
            self.class_to_index[class_] = i
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())
        return self

    def encode(self, y: List) -> np.ndarray:
        """Codificar una lista de etiquetas sin procesar.
        Args:
            y (List): etiquetas sin procesar.
        Returns:
            np.ndarray: etiquetas codificadas como índices.
        """
        encoded = np.zeros((len(y)), dtype=int)
        for i, item in enumerate(y):
            encoded[i] = self.class_to_index[item]
        return encoded

    def decode(self, y: List) -> List:
        """Decodificar una lista de índices.
        Args:
            y (List): índices.
        Returns:
            List: etiquetas.
        """
        classes = []
        for i, item in enumerate(y):
            classes.append(self.index_to_class[item])
        return classes

    def save(self, fp: str) -> None:
        """Guardar instancia de clase en archivo JSON.
        Args:
            fp (str): ruta de archivo para guardar.
        """
        with open(fp, "w") as fp:
            contents = {"class_to_index": self.class_to_index}
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp: str):
        """Cargar instancia de LabelEncoder desde archivo.
        Args:
            fp (str): ruta del archivo JSON para cargar.
        Returns:
            Instancia de LabelEncoder.
        """
        with open(fp) as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)


def get_data_splits(X: pd.Series, y: np.ndarray, train_size: float = 0.7) -> Tuple:
    """Generar divisiones de datos equilibradas.
    Args:
        X (pd.Series): features de entrada.
        y (np.ndarray): etiquetas codificadas.
        train_size (float, optional): proporción de datos a usar para el entrenamiento.
                                      El valor default es 0.7.
    Returns:
        Tuple: datos divididos como arrays Numpy.
    """
    X_train, X_, y_train, y_ = train_test_split(X, y, train_size=train_size, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_, y_, train_size=0.5, stratify=y_)
    return X_train, X_val, X_test, y_train, y_val, y_test
