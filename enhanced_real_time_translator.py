# enhanced_real_time_translator.py
# Traductor mejorado para distinguir señas estáticas vs dinámicas

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import time

class EnhancedRealTimeTranslator:
    def __init__(self, model_path='data/sign_model_hybrid.h5', model_type='hybrid'):
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Configuración del modelo
        self.model_type = model_type
        self.model = tf.keras.models.load_model(model_path)
        self.signs = np.load('data/label_encoder.npy')
        
        # Configuración de predicción
        self.sequence_length = 50
        self.sequence_buffer = deque(maxlen=self.sequence_length)
        self.prediction_threshold = 0.6  # Umbral más bajo para híbrido
        self.static_threshold = 0.8      # Umbral más alto para señas estáticas
        self.dynamic_threshold = 0.7     # Umbral medio para señas dinámicas
        
        # Historial y análisis de movimiento
        self.prediction_history = deque(maxlen=10)
        self.confidence_history = deque(maxlen=10)
        self.motion_history = deque(maxlen=50)  # Para análisis de movimiento
        
        # Estado actual
        self.current_prediction = ""
        self.current_confidence = 0.0
        self.current_motion_level = 0.0
        self.last_prediction_time = time.time()
        self.prediction_flash_time = 0
        
        # Señas conocidas estáticas vs dinámicas (para LSP - Lenguaje de Señas Peruano)
        self.static_signs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
        self.dynamic_signs = ['J', 'Z', 'HOLA', 'GRACIAS', 'POR FAVOR']
        
        print(f"🚀 Traductor mejorado inicializado")
        print(f"🧠 Modelo: {model_type}")
        print(f"📝 Señas disponibles: {len(self.signs)}")
        print(f"🔄 Señas estáticas: {len(self.static_signs)}")
        print(f"⚡ Señas dinámicas: {len(self.dynamic_signs)}")

    def extract_landmarks(self, frame):
        """Extrae landmarks de pose y manos"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        pose_results = self.pose.process(frame_rgb)
        hands_results = self.hands.process(frame_rgb)
        
        frame_rgb.flags.writeable = True
        
        landmarks = []
        
        # Landmarks de pose (primeros 33)
        if pose_results.pose_landmarks:
            for landmark in pose_results.pose_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
        else:
            landmarks.extend([0.0] * 99)  # 33 landmarks * 3 coordenadas
        
        # Landmarks de manos (21 puntos por mano)
        hand_landmarks = [0.0] * 126  # 2 manos * 21 puntos * 3 coordenadas
        
        if hands_results.multi_hand_landmarks:
            for i, hand_landmark in enumerate(hands_results.multi_hand_landmarks):
                if i < 2:  # Máximo 2 manos
                    start_idx = i * 63  # 21 puntos * 3 coordenadas por mano
                    for j, landmark in enumerate(hand_landmark.landmark):
                        if j < 21:
                            idx = start_idx + j * 3
                            hand_landmarks[idx:idx+3] = [landmark.x, landmark.y, landmark.z]
        
        landmarks.extend(hand_landmarks)
        
        return np.array(landmarks), hands_results, pose_results

    def calculate_motion_features(self, sequence):
        """Calcula características de movimiento para el modelo híbrido"""
        if len(sequence) < 2:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        seq_array = np.array(sequence)
        
        # 1. Varianza temporal
        temporal_variance = np.var(seq_array, axis=0).mean()
        
        # 2. Movimiento entre frames
        frame_diffs = np.mean([np.mean(np.abs(seq_array[i+1] - seq_array[i])) 
                              for i in range(len(seq_array)-1)])
        
        # 3. Velocidad de manos (últimos 126 features)
        hand_landmarks = seq_array[:, -126:]
        hand_velocity = np.mean([np.mean(np.abs(hand_landmarks[i+1] - hand_landmarks[i])) 
                               for i in range(len(hand_landmarks)-1)])
        
        # 4. Aceleración
        velocities = [np.mean(np.abs(seq_array[i+1] - seq_array[i])) 
                     for i in range(len(seq_array)-1)]
        acceleration = np.mean([abs(velocities[i+1] - velocities[i]) 
                              for i in range(len(velocities)-1)]) if len(velocities) > 1 else 0
        
        # 5. Frecuencia dominante
        fft_magnitude = np.abs(np.fft.fft(seq_array.flatten()))
        dominant_freq = np.argmax(fft_magnitude[1:len(fft_magnitude)//2]) + 1
        
        # 6. Entropía de movimiento
        movement_entropy = -np.sum(temporal_variance * np.log(temporal_variance + 1e-8))
        
        return np.array([temporal_variance, frame_diffs, hand_velocity, 
                        acceleration, dominant_freq, movement_entropy])

    def adaptive_threshold(self, predicted_sign, motion_level):
        """Umbral adaptativo basado en el tipo de seña y nivel de movimiento"""
        if predicted_sign in self.static_signs:
            # Para señas estáticas, requerir alta confianza y bajo movimiento
            if motion_level < 0.001:  # Muy poco movimiento
                return self.static_threshold * 0.9  # Umbral un poco más bajo
            else:
                return self.static_threshold
        elif predicted_sign in self.dynamic_signs:
            # Para señas dinámicas, requerir movimiento significativo
            if motion_level > 0.01:  # Movimiento considerable
                return self.dynamic_threshold * 0.8  # Umbral más bajo si hay movimiento
            else:
                return self.dynamic_threshold * 1.2  # Umbral más alto sin movimiento
        else:
            return self.prediction_threshold

    def draw_enhanced_overlay(self, frame, hands_results, pose_results):
        """Dibuja overlay mejorado con información de movimiento"""
        height, width = frame.shape[:2]
        
        # Panel principal de predicción
        if self.current_prediction:
            # Determinar color basado en tipo de seña
            if self.current_prediction in self.static_signs:
                color = (0, 255, 255)  # Amarillo para estáticas
                sign_type = "ESTÁTICA"
            elif self.current_prediction in self.dynamic_signs:
                color = (255, 0, 255)  # Magenta para dinámicas
                sign_type = "DINÁMICA"
            else:
                color = (0, 255, 0)    # Verde para otras
                sign_type = "GENERAL"
            
            # Fondo del panel principal
            cv2.rectangle(frame, (10, 10), (width-10, 120), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (width-10, 120), color, 3)
            
            # Texto principal
            cv2.putText(frame, self.current_prediction, (30, 60), 
                       cv2.FONT_HERSHEY_DUPLEX, 2.0, color, 3)
            
            # Información adicional
            conf_text = f"Confianza: {self.current_confidence:.1%}"
            motion_text = f"Movimiento: {self.current_motion_level:.4f}"
            type_text = f"Tipo: {sign_type}"
            
            cv2.putText(frame, conf_text, (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, motion_text, (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, type_text, (width-200, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Panel de historial
        if self.prediction_history:
            panel_y = 140
            cv2.rectangle(frame, (10, panel_y), (300, panel_y + len(self.prediction_history) * 25 + 20), (40, 40, 40), -1)
            cv2.putText(frame, "Historial:", (20, panel_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            for i, (pred, conf) in enumerate(zip(self.prediction_history, self.confidence_history)):
                y = panel_y + 40 + i * 25
                alpha = max(0.3, 1.0 - i * 0.15)  # Fade effect
                
                color = (0, 255, 255) if pred in self.static_signs else (255, 0, 255) if pred in self.dynamic_signs else (0, 255, 0)
                color = tuple(int(c * alpha) for c in color)
                
                cv2.putText(frame, f"{pred} ({conf:.1%})", (25, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Indicador de movimiento en tiempo real
        motion_bar_width = int(min(self.current_motion_level * 50000, 200))
        cv2.rectangle(frame, (width-220, 20), (width-20, 40), (50, 50, 50), -1)
        cv2.rectangle(frame, (width-220, 20), (width-220 + motion_bar_width, 40), (0, 255, 0), -1)
        cv2.putText(frame, "Movimiento", (width-220, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Dibujar landmarks
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

    def run(self):
        """Ejecuta el traductor mejorado"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("🎥 Cámara iniciada. Presiona 'q' para salir")
        print("💡 Realiza señas frente a la cámara")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Extraer landmarks
            current_landmarks, hands_results, pose_results = self.extract_landmarks(frame)
            
            if current_landmarks is not None:
                self.sequence_buffer.append(current_landmarks)
                
                # Calcular nivel de movimiento actual
                if len(self.sequence_buffer) >= 2:
                    motion = np.mean(np.abs(current_landmarks - self.sequence_buffer[-2]))
                    self.motion_history.append(motion)
                    self.current_motion_level = np.mean(list(self.motion_history)[-10:])  # Promedio de últimos 10 frames
                
                # Realizar predicción
                if len(self.sequence_buffer) == self.sequence_length:
                    try:
                        sequence_array = np.expand_dims(np.array(self.sequence_buffer), axis=0)
                        
                        if self.model_type == 'hybrid':
                            # Modelo híbrido: usar secuencias + características de movimiento
                            motion_features = self.calculate_motion_features(self.sequence_buffer)
                            motion_features = np.expand_dims(motion_features, axis=0)
                            
                            prediction = self.model.predict([sequence_array, motion_features], verbose=0)[0]
                        else:
                            # Modelo estándar
                            prediction = self.model.predict(sequence_array, verbose=0)[0]
                        
                        predicted_idx = np.argmax(prediction)
                        confidence = prediction[predicted_idx]
                        predicted_sign = self.signs[predicted_idx]
                        
                        # Umbral adaptativo
                        threshold = self.adaptive_threshold(predicted_sign, self.current_motion_level)
                        
                        # Actualizar predicción
                        if confidence > threshold:
                            if (predicted_sign != self.current_prediction or 
                                time.time() - self.last_prediction_time > 3.0):
                                
                                self.current_prediction = predicted_sign
                                self.current_confidence = confidence
                                self.last_prediction_time = time.time()
                                self.prediction_flash_time = time.time()
                                
                                # Agregar al historial
                                self.prediction_history.appendleft(predicted_sign)
                                self.confidence_history.appendleft(confidence)
                                
                                # Log detallado
                                motion_type = "ESTÁTICA" if predicted_sign in self.static_signs else "DINÁMICA"
                                print(f"✨ {motion_type}: {predicted_sign} | Conf: {confidence:.2f} | Mov: {self.current_motion_level:.4f} | Umbral: {threshold:.2f}")
                        
                        else:
                            # Reducir confianza gradualmente
                            self.current_confidence = max(0, self.current_confidence - 0.01)
                    
                    except Exception as e:
                        print(f"Error en predicción: {e}")
            
            # Dibujar overlay mejorado
            self.draw_enhanced_overlay(frame, hands_results, pose_results)
            
            # Instrucciones
            instructions = [
                "CONTROLES:",
                "Q - Salir",
                "Amarillo - Señas estáticas",
                "Magenta - Señas dinámicas"
            ]
            
            for i, instruction in enumerate(instructions):
                cv2.putText(frame, instruction, (10, frame.shape[0] - 80 + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Traductor LSP Mejorado', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Traductor cerrado")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Traductor de señas en tiempo real mejorado')
    parser.add_argument('--model', default='data/sign_model_hybrid.h5', help='Ruta al modelo entrenado')
    parser.add_argument('--model-type', default='hybrid', choices=['hybrid', 'standard'], help='Tipo de modelo')
    
    args = parser.parse_args()
    
    try:
        translator = EnhancedRealTimeTranslator(model_path=args.model, model_type=args.model_type)
        translator.run()
    except FileNotFoundError:
        print("❌ Modelo no encontrado. Entrena primero el modelo con:")
        print("   python enhanced_model_trainer.py --model-type hybrid")
    except Exception as e:
        print(f"❌ Error: {e}")
