import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np


def deriv_SIR(y, t, N, beta, gamma):
    """ Le equazioni differenziali
        del modello SIR
        
        params:
        ------
        y: il vettore contenenti le tre
            categorie della popolazione
        t: arco di tempo dell'analisi 
        N: numero totale della popolazione
        beta: tasso di infezione
        gamma: tasso medio di guarigione
        
        returns:
        ------
        le equazioni        
    """
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def resolve_SIR(y0, t, N, beta, gamma):
    """ Integra il sistema di equazioni 
        differenziali ordinarie
        
        params:
        ------
        y0: condizione iniziale
        t: arco di tempo dell'analisi 
        N: numero totale della popolazione
        beta: tasso di infezione
        gamma: tasso medio di guarigione
        
        returns:
        ------
        la soluzione alle equazioni    
    
    """
    # risoluzione delle eq. diff.
    ret = odeint(deriv_SIR, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    return S, I, R


def deriv_SEIR(y, t, N, beta, gamma, alpha):
    """ Le equazioni differenziali
        del modello SEIR
        
        params:
        ------
        y: il vettore contenenti le tre
            categorie della popolazione
        t: arco di tempo dell'analisi 
        N: numero totale della popolazione
        beta: tasso di infezione
        gamma: tasso medio di guarigione
        alpha: inverso del tempo di incubazione
        
        returns:
        ------
        le equazioni
    """
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - alpha * E
    dIdt = alpha * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


def resolve_SEIR(y0, t, N, beta, gamma, alpha):
    """ Integra il sistema di equazioni 
        differenziali ordinarie
    """
    ret = odeint(deriv_SEIR, y0, t, args=(N, beta, gamma, alpha))
    S, E, I, R = ret.T
    return S, E, I, R


def deriv_SEIR_dist(y, t, N, beta, gamma, alpha, rho):
    """ Modello SEIR con distanziamento sociale
    
        params:
        ------
        y: il vettore contenenti le tre
            categorie della popolazione
        t: arco di tempo dell'analisi 
        N: numero totale della popolazione
        beta: tasso di infezione
        gamma: tasso medio di guarigione
        rho: indice di distanziamento sociale

        returns:
        ------
        le equazioni del modello
    """
    S, E, I, R = y
    dSdt = -rho * beta * S * I / N
    dEdt = rho * beta * S * I / N - alpha * E
    dIdt = alpha * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

def resolve_SEIR_dist(y0, t, N, beta, gamma, alpha, rho):
    """ Integra il sistema di equazioni 
        differenziali ordinarie
    """
    ret = odeint(deriv_SEIR_dist, y0, t, args=(N, beta, gamma, alpha, rho))
    S, E, I, R = ret.T
    return S, E, I, R
    

