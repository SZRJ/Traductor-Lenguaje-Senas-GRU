# advanced_model_trainer.py
# Arquitectura avanzada para distinción señas estáticas/dinámicas

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, 
    BatchNormalization, Attention, MultiHeadAttention, 
    GlobalAveragePooling1D, Concatenate, Add, LayerNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import json
from datetime import datetime

class AdvancedModelTrainer:
    def __init__(self, data_path='data/sequences'):
        self.data_path = data_path
        self.sequence_length = 50
        self.num_features = 126  # 21 landmarks * 3 coords * 2 hands
        
        # Clasificación de señas por tipo
        self.static_signs = {
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
            'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
            'V', 'W', 'X', 'Y'
        }
        
        self.dynamic_signs = {
            'J', 'Z', 'Ñ', 'RR', 'LL'
        }
        
        # Cargar datos
        self.X, self.y, self.motion_features, self.signs, self.sign_types = self._load_and_prepare_data()
        print(f"📊 Datos cargados: {self.X.shape[0]} muestras, {len(self.signs)} señas")

    def _calculate_motion_features(self, sequence):
        """Calcula características de movimiento para una secuencia"""
        if len(sequence) < 2:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Calcular movimiento frame a frame
        frame_movements = []
        for i in range(1, len(sequence)):
            movement = np.linalg.norm(sequence[i] - sequence[i-1])
            frame_movements.append(movement)
        
        # Calcular aceleración
        accelerations = []
        for i in range(1, len(frame_movements)):
            acceleration = abs(frame_movements[i] - frame_movements[i-1])
            accelerations.append(acceleration)
        
        # Métricas adicionales
        position_variance = np.var(sequence, axis=0).mean()
        drift = np.linalg.norm(sequence[-1] - sequence[0])
        movement_variance = np.var(frame_movements) if len(frame_movements) > 1 else 0
        peak_movement = max(frame_movements) if frame_movements else 0
        
        return np.array([
            sum(frame_movements),                    # Movimiento total
            np.mean(frame_movements),               # Movimiento promedio  
            peak_movement,                          # Movimiento máximo
            movement_variance,                      # Varianza del movimiento
            sum(accelerations) if accelerations else 0,  # Cambios de aceleración
            position_variance,                      # Varianza de posición
            drift,                                  # Deriva total
            len([m for m in frame_movements if m > np.mean(frame_movements)]),  # Picos de movimiento
        ])

    def _classify_sign_type(self, sign):
        """Clasifica una seña como estática, dinámica o frase"""
        if sign in self.static_signs:
            return 0  # estática
        elif sign in self.dynamic_signs:
            return 1  # dinámica
        else:
            return 2  # frase/palabra

    def _load_and_prepare_data(self):
        """Carga y prepara los datos con características de movimiento"""
        sequences, labels, motion_features, sign_types = [], [], [], []
        
        # Obtener lista de señas
        signs = [name for name in os.listdir(self.data_path) 
                if os.path.isdir(os.path.join(self.data_path, name))]
        
        label_encoder = LabelEncoder()
        label_encoder.fit(signs)
        
        for sign in signs:
            sign_path = os.path.join(self.data_path, sign)
            sign_type = self._classify_sign_type(sign)
            
            for seq_file in os.listdir(sign_path):
                if seq_file.endswith('.npy'):
                    # Cargar secuencia
                    seq = np.load(os.path.join(sign_path, seq_file))
                    
                    # Calcular características de movimiento
                    motion_feat = self._calculate_motion_features(seq)
                    
                    sequences.append(seq)
                    labels.append(label_encoder.transform([sign])[0])
                    motion_features.append(motion_feat)
                    sign_types.append(sign_type)
        
        # Padding de secuencias
        X = pad_sequences(sequences, maxlen=self.sequence_length, 
                         padding='post', truncating='post', dtype='float32')
        
        # Convertir a arrays
        y = to_categorical(labels).astype(int)
        motion_features = np.array(motion_features)
        sign_types = np.array(sign_types)
        
        return X, y, motion_features, signs, sign_types

    def build_cnn_lstm_attention_model(self):
        """Construye modelo CNN+LSTM+Attention avanzado"""
        # Entrada de secuencia
        sequence_input = Input(shape=(self.sequence_length, self.num_features), name='sequence_input')
        
        # Bloque CNN para extracción de características locales
        x = Conv1D(64, 3, activation='relu', padding='same')(sequence_input)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Conv1D(128, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.2)(x)
        
        x = Conv1D(256, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Bloque LSTM bidireccional para patrones temporales
        lstm_out = tf.keras.layers.Bidirectional(LSTM(128, return_sequences=True, dropout=0.3))(x)
        lstm_out = tf.keras.layers.Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))(lstm_out)
        
        # Mecanismo de atención multi-cabeza
        attention_out = MultiHeadAttention(
            num_heads=8, 
            key_dim=64,
            dropout=0.2
        )(lstm_out, lstm_out)
        
        # Conexión residual
        attention_out = Add()([lstm_out, attention_out])
        attention_out = LayerNormalization()(attention_out)
        
        # Pooling global
        sequence_features = GlobalAveragePooling1D()(attention_out)
        sequence_features = Dropout(0.4)(sequence_features)
        
        # Entrada de características de movimiento
        motion_input = Input(shape=(8,), name='motion_input')
        motion_dense = Dense(64, activation='relu')(motion_input)
        motion_dense = BatchNormalization()(motion_dense)
        motion_dense = Dropout(0.3)(motion_dense)
        motion_dense = Dense(32, activation='relu')(motion_dense)
        motion_dense = Dropout(0.2)(motion_dense)
        
        # Fusión de características
        combined = Concatenate()([sequence_features, motion_dense])
        combined = Dense(256, activation='relu')(combined)
        combined = BatchNormalization()(combined)
        combined = Dropout(0.5)(combined)
        
        combined = Dense(128, activation='relu')(combined)
        combined = BatchNormalization()(combined)
        combined = Dropout(0.4)(combined)
        
        combined = Dense(64, activation='relu')(combined)
        combined = Dropout(0.3)(combined)
        
        # Salida final
        output = Dense(len(self.signs), activation='softmax', name='classification_output')(combined)
        
        # Crear modelo
        model = Model(inputs=[sequence_input, motion_input], outputs=output)
        
        return model

    def build_transformer_model(self):
        """Construye modelo basado en Transformer"""
        # Entrada de secuencia
        sequence_input = Input(shape=(self.sequence_length, self.num_features), name='sequence_input')
        
        # Embedding posicional
        x = Dense(256)(sequence_input)
        
        # Bloques Transformer
        for i in range(4):
            # Multi-head attention
            attention_out = MultiHeadAttention(
                num_heads=8,
                key_dim=32,
                dropout=0.1
            )(x, x)
            
            # Add & Norm
            x = Add()([x, attention_out])
            x = LayerNormalization()(x)
            
            # Feed Forward
            ff_out = Dense(512, activation='relu')(x)
            ff_out = Dropout(0.2)(ff_out)
            ff_out = Dense(256)(ff_out)
            
            # Add & Norm
            x = Add()([x, ff_out])
            x = LayerNormalization()(x)
        
        # Global pooling
        sequence_features = GlobalAveragePooling1D()(x)
        sequence_features = Dropout(0.4)(sequence_features)
        
        # Características de movimiento
        motion_input = Input(shape=(8,), name='motion_input')
        motion_dense = Dense(64, activation='relu')(motion_input)
        motion_dense = BatchNormalization()(motion_dense)
        motion_dense = Dropout(0.3)(motion_dense)
        
        # Fusión
        combined = Concatenate()([sequence_features, motion_dense])
        combined = Dense(256, activation='relu')(combined)
        combined = BatchNormalization()(combined)
        combined = Dropout(0.5)(combined)
        
        combined = Dense(128, activation='relu')(combined)
        combined = Dropout(0.4)(combined)
        
        # Salida
        output = Dense(len(self.signs), activation='softmax')(combined)
        
        model = Model(inputs=[sequence_input, motion_input], outputs=output)
        return model

    def train_with_cross_validation(self, model_type='cnn_lstm_attention', n_folds=5):
        """Entrena con validación cruzada para mayor robustez"""
        print(f"🚀 Entrenando modelo {model_type} con validación cruzada...")
        
        # Configurar validación cruzada estratificada
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_scores = []
        
        # Obtener etiquetas para estratificación
        y_labels = np.argmax(self.y, axis=1)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, y_labels)):
            print(f"\n📊 Fold {fold + 1}/{n_folds}")
            
            # Dividir datos
            X_train_fold = self.X[train_idx]
            X_val_fold = self.X[val_idx]
            y_train_fold = self.y[train_idx]
            y_val_fold = self.y[val_idx]
            motion_train_fold = self.motion_features[train_idx]
            motion_val_fold = self.motion_features[val_idx]
            
            # Construir modelo
            if model_type == 'cnn_lstm_attention':
                model = self.build_cnn_lstm_attention_model()
            elif model_type == 'transformer':
                model = self.build_transformer_model()
            else:
                raise ValueError(f"Tipo de modelo no soportado: {model_type}")
            
            # Calcular pesos de clase para dataset desbalanceado
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_labels),
                y=y_labels[train_idx]
            )
            class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
            
            # Compilar modelo
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['categorical_accuracy']
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_categorical_accuracy',
                    patience=15,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=8,
                    min_lr=1e-6
                )
            ]
            
            # Entrenar
            history = model.fit(
                [X_train_fold, motion_train_fold],
                y_train_fold,
                validation_data=([X_val_fold, motion_val_fold], y_val_fold),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=1
            )
            
            # Evaluar
            val_score = model.evaluate([X_val_fold, motion_val_fold], y_val_fold, verbose=0)
            fold_scores.append(val_score[1])  # accuracy
            
            print(f"Fold {fold + 1} - Accuracy: {val_score[1]:.4f}")
        
        # Estadísticas finales de CV
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        print(f"\n📊 RESULTADOS VALIDACIÓN CRUZADA:")
        print(f"Accuracy promedio: {mean_score:.4f} ± {std_score:.4f}")
        print(f"Scores por fold: {fold_scores}")
        
        # Entrenar modelo final con todos los datos
        print(f"\n🎯 Entrenando modelo final con todos los datos...")
        
        if model_type == 'cnn_lstm_attention':
            final_model = self.build_cnn_lstm_attention_model()
        else:
            final_model = self.build_transformer_model()
        
        # Compilar
        final_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy']
        )
        
        # Callbacks para modelo final
        callbacks = [
            EarlyStopping(
                monitor='loss',
                patience=20,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            ),
            ModelCheckpoint(
                f'data/sign_model_{model_type}_best.h5',
                monitor='loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Entrenar modelo final
        final_history = final_model.fit(
            [self.X, self.motion_features],
            self.y,
            epochs=150,
            batch_size=32,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Guardar modelo final
        final_model.save(f'data/sign_model_{model_type}_final.h5')
        
        # Guardar métricas
        results = {
            'model_type': model_type,
            'cv_mean_accuracy': float(mean_score),
            'cv_std_accuracy': float(std_score),
            'cv_fold_scores': [float(s) for s in fold_scores],
            'final_training_history': {
                'loss': [float(x) for x in final_history.history['loss']],
                'categorical_accuracy': [float(x) for x in final_history.history['categorical_accuracy']]
            },
            'dataset_info': {
                'total_samples': int(self.X.shape[0]),
                'num_signs': len(self.signs),
                'static_signs': list(self.static_signs),
                'dynamic_signs': list(self.dynamic_signs)
            },
            'training_date': datetime.now().isoformat()
        }
        
        with open(f'training_results_{model_type}.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Entrenamiento completado!")
        print(f"📄 Resultados guardados en: training_results_{model_type}.json")
        print(f"🤖 Modelo guardado en: data/sign_model_{model_type}_final.h5")
        
        return final_model, results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenar modelos avanzados')
    parser.add_argument('--model-type', choices=['cnn_lstm_attention', 'transformer'], 
                       default='cnn_lstm_attention', help='Tipo de modelo a entrenar')
    parser.add_argument('--data-path', default='data/sequences', help='Ruta a los datos')
    parser.add_argument('--folds', type=int, default=5, help='Número de folds para CV')
    
    args = parser.parse_args()
    
    try:
        trainer = AdvancedModelTrainer(data_path=args.data_path)
        model, results = trainer.train_with_cross_validation(
            model_type=args.model_type,
            n_folds=args.folds
        )
        
        print(f"\n🎉 ¡Entrenamiento exitoso!")
        print(f"📊 Accuracy final: {results['cv_mean_accuracy']:.4f} ± {results['cv_std_accuracy']:.4f}")
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
