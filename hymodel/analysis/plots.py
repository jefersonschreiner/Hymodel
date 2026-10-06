# Standardized Plots for Water Resources

import numpy as np
import matplotlib.pyplot as plt

def plot_hidrograma(t, Q, titulo="Hidrograma", xlabel="Tempo", ylabel="Vazão (m³/s)"):
    # Plots a simple hydrograph 
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, Q, 'b-', linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_curva_permanencia(Q, titulo="Curva de Permanência"):
    # Plots the permanence curve of flows
    Q_sorted = np.sort(Q)[::-1]
    prob = np.arange(1, len(Q_sorted) + 1) / (len(Q_sorted) + 1) * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(prob, Q_sorted, 'b-', linewidth=1.5)
    ax.set_xlabel("Probabilidade de Excedência (%)")
    ax.set_ylabel("Vazão (m³/s)")
    ax.set_title(titulo)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_secao_transversal(forma, **kwargs):
    # Plots the cross-section of a channel
    fig, ax = plt.subplots(figsize=(6, 4))
    
    if forma == 'retangular':
        b, h = kwargs['b'], kwargs['h']
        ax.plot([0, 0, b, b], [0, h, h, 0], 'k-', linewidth=2)
    elif forma == 'trapezoidal':
        b, h, z = kwargs['b'], kwargs['h'], kwargs['z']
        ax.plot([0, 0, b, b], [0, h, h, 0], 'k-', linewidth=2)
    
    ax.set_aspect('equal')
    ax.set_xlabel("Largura (m)")
    ax.set_ylabel("Profundidade (m)")
    ax.set_title(f"Seção {forma}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax