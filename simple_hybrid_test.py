# simple_hybrid_test.py
# Prueba simple del modelo híbrido vs original para I/J

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import os

def test_models_on_i_j():
    """Prueba específica para I y J"""
    print("🔍 Probando modelos en señas I y J específicamente...")
    
    # Cargar datos de I y J
    i_sequences = []
    j_sequences = []
    
    # Cargar secuencias de I
    i_path = 'data/sequences/I'
    for file in os.listdir(i_path):
        if file.endswith('.npy'):
            seq = np.load(os.path.join(i_path, file))
            if len(seq) > 0:
                i_sequences.append(seq)
    
    # Cargar secuencias de J
    j_path = 'data/sequences/J'
    for file in os.listdir(j_path):
        if file.endswith('.npy'):
            seq = np.load(os.path.join(j_path, file))
            if len(seq) > 0:
                j_sequences.append(seq)
    
    print(f"📊 Secuencias I: {len(i_sequences)}, J: {len(j_sequences)}")
    
    # Preparar datos
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    # Combinar secuencias
    all_sequences = i_sequences + j_sequences
    labels = [0] * len(i_sequences) + [1] * len(j_sequences)  # 0=I, 1=J
    
    # Padding
    X = pad_sequences(all_sequences, maxlen=50, padding='post', truncating='post', dtype='float32')
    y = np.array(labels)
    
    print(f"📊 Forma de datos: {X.shape}")
    
    # Función para calcular características de movimiento
    def calculate_motion_features(sequence):
        if len(sequence) < 2:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        from scipy.spatial.distance import euclidean
        import statistics
        
        # Calcular movimiento frame a frame
        frame_movements = []
        for i in range(1, len(sequence)):
            movement = euclidean(sequence[i-1], sequence[i])
            frame_movements.append(movement)
        
        # Calcular aceleración
        accelerations = []
        for i in range(1, len(frame_movements)):
            acceleration = abs(frame_movements[i] - frame_movements[i-1])
            accelerations.append(acceleration)
        
        # Varianza de posición
        position_variance = np.var(sequence, axis=0).mean()
        
        # Deriva
        drift = euclidean(sequence[0], sequence[-1])
        
        return np.array([
            sum(frame_movements),
            statistics.mean(frame_movements),
            max(frame_movements),
            statistics.variance(frame_movements) if len(frame_movements) > 1 else 0,
            sum(accelerations) if accelerations else 0,
            position_variance
        ])
    
    # Preparar características de movimiento para modelo híbrido
    motion_features = []
    for seq in all_sequences:
        motion_feat = calculate_motion_features(seq)
        motion_features.append(motion_feat)
    
    motion_features = np.array(motion_features)
    
    print("📊 Características de movimiento calculadas")
    
    # Probar modelo original
    print("\n🔍 Probando modelo original...")
    try:
        model_original = tf.keras.models.load_model('data/sign_model_gru.h5')
        
        # Necesitamos mapear I y J a los índices correctos del modelo original
        signs = np.load('data/label_encoder.npy')
        i_idx = np.where(signs == 'I')[0][0]
        j_idx = np.where(signs == 'J')[0][0]
        
        # Predecir con modelo original
        predictions_orig = model_original.predict(X, verbose=0)
        predicted_classes_orig = np.argmax(predictions_orig, axis=1)
        
        # Extraer solo las predicciones para I y J
        i_predictions = predictions_orig[:, i_idx]
        j_predictions = predictions_orig[:, j_idx]
        
        # Determinar clase predicha entre I y J
        ij_predictions = np.where(i_predictions > j_predictions, 0, 1)
        
        # Calcular métricas
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        acc_orig = accuracy_score(y, ij_predictions)
        prec_i_orig = precision_score(y, ij_predictions, pos_label=0)
        prec_j_orig = precision_score(y, ij_predictions, pos_label=1)
        rec_i_orig = recall_score(y, ij_predictions, pos_label=0)
        rec_j_orig = recall_score(y, ij_predictions, pos_label=1)
        
        print(f"📈 Modelo Original:")
        print(f"   Precisión general: {acc_orig:.3f}")
        print(f"   Precisión I: {prec_i_orig:.3f}")
        print(f"   Precisión J: {prec_j_orig:.3f}")
        print(f"   Recall I: {rec_i_orig:.3f}")
        print(f"   Recall J: {rec_j_orig:.3f}")
        
        # Matriz de confusión
        cm_orig = confusion_matrix(y, ij_predictions)
        print(f"   Matriz de confusión:")
        print(f"   I clasificadas como I: {cm_orig[0,0]}, como J: {cm_orig[0,1]}")
        print(f"   J clasificadas como I: {cm_orig[1,0]}, como J: {cm_orig[1,1]}")
        
    except Exception as e:
        print(f"❌ Error con modelo original: {e}")
        return
    
    # Probar modelo híbrido
    print("\n🔍 Probando modelo híbrido...")
    try:
        model_hybrid = tf.keras.models.load_model('data/sign_model_hybrid.h5')
        
        # Predecir con modelo híbrido
        predictions_hyb = model_hybrid.predict([X, motion_features], verbose=0)
        predicted_classes_hyb = np.argmax(predictions_hyb, axis=1)
        
        # Extraer solo las predicciones para I y J
        i_predictions_hyb = predictions_hyb[:, i_idx]
        j_predictions_hyb = predictions_hyb[:, j_idx]
        
        # Determinar clase predicha entre I y J
        ij_predictions_hyb = np.where(i_predictions_hyb > j_predictions_hyb, 0, 1)
        
        # Calcular métricas
        acc_hyb = accuracy_score(y, ij_predictions_hyb)
        prec_i_hyb = precision_score(y, ij_predictions_hyb, pos_label=0)
        prec_j_hyb = precision_score(y, ij_predictions_hyb, pos_label=1)
        rec_i_hyb = recall_score(y, ij_predictions_hyb, pos_label=0)
        rec_j_hyb = recall_score(y, ij_predictions_hyb, pos_label=1)
        
        print(f"📈 Modelo Híbrido:")
        print(f"   Precisión general: {acc_hyb:.3f}")
        print(f"   Precisión I: {prec_i_hyb:.3f}")
        print(f"   Precisión J: {prec_j_hyb:.3f}")
        print(f"   Recall I: {rec_i_hyb:.3f}")
        print(f"   Recall J: {rec_j_hyb:.3f}")
        
        # Matriz de confusión
        cm_hyb = confusion_matrix(y, ij_predictions_hyb)
        print(f"   Matriz de confusión:")
        print(f"   I clasificadas como I: {cm_hyb[0,0]}, como J: {cm_hyb[0,1]}")
        print(f"   J clasificadas como I: {cm_hyb[1,0]}, como J: {cm_hyb[1,1]}")
        
    except Exception as e:
        print(f"❌ Error con modelo híbrido: {e}")
        return
    
    # Comparación
    print("\n" + "="*60)
    print("COMPARACIÓN DE RESULTADOS")
    print("="*60)
    print(f"{'Métrica':<20} {'Original':<12} {'Híbrido':<12} {'Mejora':<12}")
    print("-"*60)
    print(f"{'Precisión General':<20} {acc_orig:<12.3f} {acc_hyb:<12.3f} {acc_hyb-acc_orig:<12.3f}")
    print(f"{'Precisión I':<20} {prec_i_orig:<12.3f} {prec_i_hyb:<12.3f} {prec_i_hyb-prec_i_orig:<12.3f}")
    print(f"{'Precisión J':<20} {prec_j_orig:<12.3f} {prec_j_hyb:<12.3f} {prec_j_hyb-prec_j_orig:<12.3f}")
    print(f"{'Recall I':<20} {rec_i_orig:<12.3f} {rec_i_hyb:<12.3f} {rec_i_hyb-rec_i_orig:<12.3f}")
    print(f"{'Recall J':<20} {rec_j_orig:<12.3f} {rec_j_hyb:<12.3f} {rec_j_hyb-rec_j_orig:<12.3f}")
    
    # Análisis de confusión
    i_as_j_orig = cm_orig[0,1]
    j_as_i_orig = cm_orig[1,0]
    i_as_j_hyb = cm_hyb[0,1]
    j_as_i_hyb = cm_hyb[1,0]
    
    print(f"\nCONFUSIONES I/J:")
    print(f"{'Confusión':<20} {'Original':<12} {'Híbrido':<12} {'Mejora':<12}")
    print("-"*60)
    print(f"{'I → J':<20} {i_as_j_orig:<12} {i_as_j_hyb:<12} {i_as_j_orig-i_as_j_hyb:<12}")
    print(f"{'J → I':<20} {j_as_i_orig:<12} {j_as_i_hyb:<12} {j_as_i_orig-j_as_i_hyb:<12}")
    print(f"{'Total confusión':<20} {i_as_j_orig+j_as_i_orig:<12} {i_as_j_hyb+j_as_i_hyb:<12} {(i_as_j_orig+j_as_i_orig)-(i_as_j_hyb+j_as_i_hyb):<12}")
    
    # Conclusión
    print("\n" + "="*60)
    print("CONCLUSIÓN")
    print("="*60)
    
    if acc_hyb > acc_orig:
        print("✅ El modelo híbrido MEJORA la precisión general")
    else:
        print("❌ El modelo híbrido NO mejora la precisión general")
    
    if (i_as_j_orig + j_as_i_orig) > (i_as_j_hyb + j_as_i_hyb):
        print("✅ El modelo híbrido REDUCE la confusión entre I y J")
    else:
        print("❌ El modelo híbrido NO reduce la confusión entre I y J")
    
    confusion_reduction = (i_as_j_orig + j_as_i_orig) - (i_as_j_hyb + j_as_i_hyb)
    print(f"📊 Reducción de confusión: {confusion_reduction} casos")
    
    # Análisis de características de movimiento
    print("\n📊 ANÁLISIS DE CARACTERÍSTICAS DE MOVIMIENTO:")
    i_motion = motion_features[:len(i_sequences)]
    j_motion = motion_features[len(i_sequences):]
    
    print(f"Movimiento promedio I: {np.mean(i_motion[:, 1]):.4f}")
    print(f"Movimiento promedio J: {np.mean(j_motion[:, 1]):.4f}")
    print(f"Ratio J/I: {np.mean(j_motion[:, 1])/np.mean(i_motion[:, 1]):.2f}x")

if __name__ == "__main__":
    test_models_on_i_j()
