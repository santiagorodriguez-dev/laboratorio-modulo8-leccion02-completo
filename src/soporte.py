
# Para gestionar el feature scaling
# -----------------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, RobustScaler

# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import math

import seaborn as sns
import matplotlib.pyplot as plt


def analisis_numericos (df):

    escalador_robust = RobustScaler()
    datos_transf_robust = escalador_robust.fit_transform(df[["powerCV_knn", "kilometer_knn","yearOfRegistration_knn"]])
    df[["powerCV_robust", "kilometer_Ratesrobust", "yearOfRegistration_Ratesrobust"]] = datos_transf_robust

    escalador_min_max = MinMaxScaler()
    datos_transf_min_max = escalador_min_max.fit_transform(df[["powerCV_knn", "kilometer_knn","yearOfRegistration_knn"]])
    df[["powerCV_min_max", "kilometer_min_max", "yearOfRegistration_min_max"]]  = datos_transf_min_max

    escalador_norm = Normalizer()
    datos_transf_norm = escalador_norm.fit_transform(df[["powerCV_knn", "kilometer_knn","yearOfRegistration_knn"]])
    df[["powerCV_norm", "kilometer_norm", "yearOfRegistration_norm"]]  = datos_transf_norm

    escalador_estandar = StandardScaler()
    datos_transf_estandar = escalador_estandar.fit_transform(df[["powerCV_knn", "kilometer_knn","yearOfRegistration_knn"]])
    df[["powerCV_estandar", "kilometer_estandar", "yearOfRegistration_estandar"]]  = datos_transf_estandar

    return df

def visualizar_tablas(dataframe, lista_col):

    num_filas = math.ceil(len(lista_col) / 5)

    fig, axes = plt.subplots(nrows=num_filas, ncols=5, figsize=(25, 15))
    axes = axes.flat

    for indice, columna in enumerate(lista_col):
        sns.boxplot(x=columna, data=dataframe, ax=axes[indice])
        axes[indice].set_title(f"{columna}")
        axes[indice].set_xlabel("")

    # if len(lista_col) % 2 != 0:
    #     fig.delaxes(axes[-1])

    fig.suptitle("")
    plt.tight_layout()
    plt.show()