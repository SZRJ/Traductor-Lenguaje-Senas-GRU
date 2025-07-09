import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_sequences():
    # Cargar secuencias de I y J
    i_seq = np.load('data/sequences/I/0.npy')
    j_seq = np.load('data/sequences/J/0.npy')
    
    print('=== ANÁLISIS DE SECUENCIAS I vs J ===')
    print(f'Forma de secuencia I: {i_seq.shape}')
    print(f'Forma de secuencia J: {j_seq.shape}')
    print()
    
    # Analizar varianza (movimiento)
    i_variance = np.var(i_seq, axis=0).mean()
    j_variance = np.var(j_seq, axis=0).mean()
    print(f'Varianza total en I: {i_variance:.6f}')
    print(f'Varianza total en J: {j_variance:.6f}')
    print(f'Ratio J/I: {j_variance/i_variance:.2f}x más movimiento')
    print()
    
    # Analizar diferencias entre frames consecutivos
    i_movement = np.mean([np.mean(np.abs(i_seq[i+1] - i_seq[i])) for i in range(len(i_seq)-1)])
    j_movement = np.mean([np.mean(np.abs(j_seq[i+1] - j_seq[i])) for i in range(len(j_seq)-1)])
    print(f'Movimiento promedio entre frames en I: {i_movement:.6f}')
    print(f'Movimiento promedio entre frames en J: {j_movement:.6f}')
    print(f'Ratio J/I: {j_movement/i_movement:.2f}x más movimiento')
    print()
    
    # Analizar múltiples secuencias
    print('=== ANÁLISIS DE MÚLTIPLES SECUENCIAS ===')
    i_variances = []
    j_variances = []
    
    for i in range(min(10, len(os.listdir('data/sequences/I')))):
        i_seq = np.load(f'data/sequences/I/{i}.npy')
        j_seq = np.load(f'data/sequences/J/{i}.npy')
        
        i_var = np.var(i_seq, axis=0).mean()
        j_var = np.var(j_seq, axis=0).mean()
        
        i_variances.append(i_var)
        j_variances.append(j_var)
    
    print(f'Varianza promedio I (10 secuencias): {np.mean(i_variances):.6f}')
    print(f'Varianza promedio J (10 secuencias): {np.mean(j_variances):.6f}')
    print(f'Ratio promedio J/I: {np.mean(j_variances)/np.mean(i_variances):.2f}x')
    
    # Analizar landmark específicos (mano)
    print()
    print('=== ANÁLISIS DE LANDMARKS ESPECÍFICOS ===')
    
    # Los landmarks de la mano están en las últimas 21 posiciones
    hand_landmarks = slice(-21*3, None)  # Últimos 21 puntos x 3 coordenadas
    
    i_hand_var = np.var(i_seq[:, hand_landmarks], axis=0).mean()
    j_hand_var = np.var(j_seq[:, hand_landmarks], axis=0).mean()
    
    print(f'Varianza en landmarks de mano - I: {i_hand_var:.6f}')
    print(f'Varianza en landmarks de mano - J: {j_hand_var:.6f}')
    print(f'Ratio J/I en mano: {j_hand_var/i_hand_var:.2f}x')

if __name__ == "__main__":
    analyze_sequences()
