#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erf

plt.rcParams.update({'font.size': 13.5})  # actualizar tamaño de fuente en las figura

### CÁLCULO DE MAGNITUDES ###

def momento_transversal(p):
    '''
    Cálculo del valor del momento transversal para una partícula

    Parámetros:
        p (array):
            p[0]: componente x del momento
            p[1]: componente y del momento

    '''
    return np.sqrt(p[0]**2 + p[1]**2)


def pseudorapidez(p):
    '''
    Cálculo de la pseudorapidez de una partícula

    Parámetros:
        p(array):
            p[0]: componente x del momento
            p[1]: componente y del momento
            p[2]: componente y del momento

    '''
    theta = np.arctan(np.sqrt(p[0]**2 + p[1]**2)/abs(p[2]))
    return - np.log(np.tan(theta/2))


def masa_boson(p):
    '''
    Cálculo de la masa del bosón Z a partir de los momentos y las energías de las dos partículas
    hijas de su desnitegracion

    Parámetros:
        p(array):
            p[0]: componente x del momento de la primera partícula
            p[1]: componente y del momento de la primera partícula
            p[2]: componente z del momento de la primera partícula
            p[3]: energía de la primera partícula
            p[4]: componente x del momento de la segunda partícula
            p[5]: componente y del momento de la segunda partícula
            p[6]: componente z del momento de la segunda partícula
            p[7]: energía de la segunda partícula
    '''
    m = np.sqrt((p[3] + p[7])**2 - (p[0] + p[4])**2 - (p[1] + p[5])**2 - (p[2] + p[6])**2)
    return m

### FUNCIONES DE AJUSTE ###

def gaussian(x, a, x0, sigma):
    '''
    Función gaussiana.

    Parámetros:
        a (float): amplitud
        x0 (float): centro de la gaussiana
        sigma (float): dispersión
    '''
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def BW(x, a, x0, gamma):
    '''
    Función BW.

    Parámetros:
        a (float): amplitud
        x0 (float): valor central de la función 
        gamma (float): FWHM

    '''

    return a*1/np.pi*0.5*gamma/((gamma**2/4)+(x-x0)**2)


def skew_gaussian(x, a, x0, sigma, alfa):
    '''
    Función gaussiana asimétrica.

    Parámetros:
        a (float): amplitud
        x0 (float): centro de la gaussiana
        sigma (float): dispersión
        alfa (float): parametrización de la asimetría

    '''
    gauss = np.exp((-(x - x0)**2)/(2*sigma**2))
    skew = (1 + erf((alfa*(x - x0))/(sigma*np.sqrt(2))))
    return a * gauss * skew


### FUNCIONES PROCESADO DE DATOS ###

def load_data(archivo_datos):
    '''
    Lee un archivo de datos y genera un pd.dataframe donde calcula los momentos
    transversales y la pseudorapidez de los dos leptones.

    Parámetros:
        archivo_datos (str): nombre del archivo de datos a leer. El formato de los archivos ha
                             de ser 'tipodedato_tipodepartícula.txt'
    Devuelve:
        df (pd.dataframe): dataframe con los datos del archivo de datos + los momentos y
                           pseudorapideces

    '''
    df = pd.read_csv(archivo_datos, delim_whitespace=True)  # leer el archivo y generar df

    # nombrar al dataframe para identificar el archivo con el que se está trabajando
    df.name = str(archivo_datos.replace('.txt', ''))

    # calcular magnitudes
    df['PT1'] = df[['PX1', 'PY1']].apply(momento_transversal, axis=1)
    df['PT2'] = df[['PX2', 'PY2']].apply(momento_transversal, axis=1)
    df['ETA1'] = df[['PX1', 'PY1', 'PZ1']].apply(pseudorapidez, axis=1)
    df['ETA2'] = df[['PX2', 'PY2', 'PZ2']].apply(pseudorapidez, axis=1)

    return df


def calc_masa(df):
    '''
    Calcula la masa del Bosón Z y la añade en una columna al dataframe

    Parámetros:
        df: dataframe con los valores del momento transcersal y pseudorapidez calculados

    '''
    df['MBZ'] = df[['PX1', 'PY1', 'PZ1', 'E1', 'PX2', 'PY2', 'PZ2', 'E2']].apply(masa_boson, axis=1)
    return df

def filter_data(df):
    '''
    Descarta las filas tales que la pseudorapidez sea mayor a 1.2 o el momento
    transversal sea menor a 20 GeV para cualquiera de las dos partículas.

    Parámetros:
        df: dataframe con los valores del momento transcersal y pseudorapidez calculados

    '''
    name = df.name
    df = df.drop(df[(abs(df.PT1) < 20)].index)
    df = df.drop(df[(abs(df.PT2) < 20)].index)
    df = df.drop(df[(abs(df.ETA1) > 1.2)].index)
    df = df.drop(df[(abs(df.ETA2) > 1.2)].index)
    df.name = f'{name}_filtered'  # actualizar el nombre del df para identificar que ha sido filtrado
    return df


def hist(df, var, bins=100, rng=None, save=False):
    '''
    Representa las variables momento transversal o pseudorapidez en un histograma

    Parámetros
        df: dataframe con los valores del momento transcersal y pseudorapidez calculados
        var(tuple): lista que contiene los nombres de las columnas de df a representar.
                    todas las columnas deben equivaler al mismo tipo de variable (PT o ETA)
        bins (int): número de intervalos en el histograma
        rng (list): rango en el eje x para el cual representar el histograma
        save (bool): if True guarda la figura
    '''

    variables = df[var]     # tomar un nuevo df con una columna para cada variable a representar

    # comprobar que variable está siendo representada para ajustar la unidades del histograma
    tipo_variable = var[0][:-1]
    if tipo_variable == 'PT':
        unidades = '(GeV/c)'
    elif tipo_variable == 'ETA':
        unidades = ''

    # generar el histograma
    fig, ax = plt.subplots(figsize=(8, 6))

    for column in variables:
        n, bins = np.histogram(variables[column], bins=bins, range=rng)
        x = 0.5*(bins[:-1] + bins[1:])
        ax.scatter(x, n, s=16, label=variables[column].name)

    ax.set_ylabel('Eventos')
    ax.set_xlabel('{:s} {:s}'.format(tipo_variable, unidades))
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(f'{directorio_datos}{df.name}_hist_{tipo_variable}.png')
    plt.show()


###


def hist_fit(df, fit_function, bins=100, save=False):
    '''
    Representa la masa del bosón z y a ajusta a fit_function. Imprime por pantalla el
    valor de la masa obtenido del ajuste con su incertidumbre

    Parámetros
        df (pd.dataframe): dataframe con todos los datos y la masa del bosoón calculada
                           Se pasa el data frame completo y no unicamente la variable 'MBZ'
                           para poder conocer el archivo del que proceden los datos, según
                           el nombre del df, y poder guardar las fuguras acordemente.
        fit_function (function): función para realizar el ajuste
        bins (int): número de intervalos en el histograma
        save (bool): if True guarda la figura
    '''
    masa = df['MBZ']   # tomar la masa del bosón de df

    n, bins = np.histogram(masa, range=(70, 110), bins=bins)
    x = 0.5*(bins[:-1] + bins[1:])  # tomar el punto medio de los bins para representar como scatter

    # ajuste gaussiano
    if fit_function == gaussian:
        p0 = (4000, 90, 1)      # valores iniciales para el ajuste
        popt, pcov = curve_fit(gaussian, x, n, p0=p0)
        err = np.sqrt(np.diag(pcov))  # calcular errores en el ajuste
        ajuste = 'Ajuste gaussiano'
        print('{:s}. Masa bosón Z = {:.2f} +- {:.2f} GeV/c^2'.format(ajuste, popt[1], err[1]))

    # ajuste BW
    elif fit_function == BW:
        p0 = (4000, 90, 10)
        popt, pcov = curve_fit(BW, x, n, p0=p0)
        err = np.sqrt(np.diag(pcov))
        ajuste = 'Ajuste función BW'
        print('{:s}.Masa bosón Z = {:.2f} +- {:.2f} GeV/c^2'.format(ajuste, popt[1], err[1]))

    # ajuste gaussiano asim
    elif fit_function == skew_gaussian:
        p0 = [1000, 90, 1, 0.5]
        popt, pcov = curve_fit(skew_gaussian, x, n, p0=p0)
        err = np.sqrt(np.diag(pcov))
        ajuste = 'Ajuste gaussiano asimétrico'
        print('{:s}. Masa bosón Z = {:.2f} +- {:.2f} GeV/c^2'.format(ajuste, popt[1], err[1]))

    xx = np.linspace(70, 110, 300)  # array en eje x para representar el ajuste

    # representar histograma + ajuste
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, n, s=16)
    ax.plot(xx, fit_function(xx, *popt))
    ax.set_title(f'{ajuste}')
    ax.set_ylabel('Eventos')
    ax.set_xlabel('MBZ (GeV/c$^2$)')
    ax.text(95, max(n)*9/10, 'MBZ = {:.2f} +- {:.2f}' '\n' r'GeV/c$^2$'.format(popt[1], err[1]))
    fig.tight_layout()
    if save:
        ax.set_title('')
        fig.tight_layout()
        fig.savefig(f'{directorio_datos}{df.name}_{fit_function.__name__}.png')
    plt.show()

def complete_analysis2(archivo_datos, bins_masa=100, save=False):
    '''
    Análisis completo de los datos. Representa los histogramas de los momentos transversales
    antes y después del filtrado. Representa y ajusta la masa del bosón Z a todas las funciones
    de ajuste antes y después del filtrado

    Parámetros:
        archivo_datos (str): archivo de datos a importar
        bins_masa (int): número de intervalos en los histogramas de la masa del bosón Z
        save (bool): if True guarda la figura
    '''

    df = load_data(archivo_datos)
    calc_masa(df)

    print('------ Datos sin procesar -------')
    hist(df, ['PT1', 'PT2'], rng=(0, 100), save=save)
    hist(df, ['ETA1', 'ETA2'], rng=(0, 2.5), save=save)
    hist_fit(df, gaussian, bins=bins_masa, save=save)
    hist_fit(df, BW, bins=bins_masa, save=save)
    hist_fit(df, skew_gaussian, bins=bins_masa, save=save)

    print('------- Datos procesados --------')
    df = filter_data(df)
    hist(df, ['PT1', 'PT2'], rng=(0, 100), save=save)
    hist(df, ['ETA1', 'ETA2'], rng=(0, 2.5), save=save)
    hist_fit(df, gaussian, bins=bins_masa, save=save)
    hist_fit(df, BW, bins=bins_masa, save=save)
    hist_fit(df, skew_gaussian, bins=bins_masa, save=save)

directorio_datos = ''
complete_analysis2('reconstruccion_electron.txt', bins_masa=100, save=False)
