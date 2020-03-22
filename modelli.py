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


def resolve(y0, t, N, beta, gamma):
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
    S, E, I, R = y
    dSdt = -beta * S * I 
    dIdt = alpha *E - gamma * I
    dEdt = beta * S * I - alpha * E
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


def resolve_SEIR(y0, t, N, beta, gamma, alpha):
    ret = odeint(deriv_SEIR, y0, t, args=(N, beta, gamma, alpha))
    S, E, I, R = ret.T
    return S, E, I, R
    
    

def plot(S, I, R, t, title='curve epidemiologiche modello SIR'):
    """ Visualizza le tre curve di soggetti
        S(t), I(t) and R(t) nel tempo
        
        params:
        ------
        S, I, R: soluzioni alle eq. diff.
    """ 
    fig = plt.figure()
    plt.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='suscettibili')
    plt.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='infetti')
    plt.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='rimossi')
    plt.xlabel('tempo [giorni]')
    plt.ylabel('soggetti [1000s]')
    plt.legend()
    plt.title(title)

