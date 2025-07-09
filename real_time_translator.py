# real_time_translator_sequence.py

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import os
import time
from datetime import datetime
from tensorflow.keras.models import load_model

class RealTimeSequenceTranslator:
    def __init__(self, model_path='data/sign_model_gru.h5', signs_path='data/label_encoder.npy'):
        self.model = load_model(model_path)
        self.signs = np.load(signs_path)
        self.sequence_length = 50
        self.sequence_buffer = deque(maxlen=self.sequence_length)
        self.prediction_threshold = 0.7
        
        # Configuración visual mejorada
        self.prediction_history = deque(maxlen=5)  # Historial de predicciones
        self.confidence_history = deque(maxlen=5)  # Historial de confianza
        self.last_prediction_time = time.time()
        self.current_prediction = ""
        self.current_confidence = 0.0
        self.prediction_flash_time = 0  # Para efecto de flash en nueva predicción
        self.ui_colors = {
            'primary': (64, 128, 255),      # Azul moderno
            'success': (46, 204, 113),      # Verde éxito
            'warning': (255, 193, 7),       # Amarillo advertencia
            'danger': (231, 76, 60),        # Rojo peligro
            'dark': (52, 73, 94),           # Gris oscuro
            'light': (236, 240, 241),       # Gris claro
            'background': (44, 62, 80),     # Fondo oscuro
            'text': (255, 255, 255),        # Texto blanco
            'accent': (155, 89, 182)        # Púrpura acento
        }
        
        # Configuración de MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2, 
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        
        # Configurar ventana para mejor visualización
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def _extract_landmarks(self, hand_landmarks):
        """Extrae landmarks normalizados de una mano"""
        base_x, base_y, base_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
        return landmarks

    def _draw_overlay_panel(self, frame, x, y, width, height, color, alpha=0.7):
        """Dibuja un panel translúcido para overlays"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def _draw_text_with_background(self, frame, text, position, font_scale=1.0, 
                                  color=(255, 255, 255), bg_color=(0, 0, 0), 
                                  thickness=2, padding=10):
        """Dibuja texto con fondo para mejor legibilidad"""
        font = cv2.FONT_HERSHEY_DUPLEX
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Calcular posiciones
        text_x, text_y = position
        bg_x1 = text_x - padding
        bg_y1 = text_y - text_size[1] - padding
        bg_x2 = text_x + text_size[0] + padding
        bg_y2 = text_y + padding
        
        # Dibujar fondo
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 2)
        
        # Dibujar texto
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

    def _draw_confidence_bar(self, frame, confidence, x, y, width=200, height=20):
        """Dibuja una barra de confianza visual"""
        # Fondo de la barra
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.ui_colors['dark'], -1)
        
        # Barra de progreso
        fill_width = int(width * confidence)
        color = self.ui_colors['success'] if confidence > 0.8 else \
                self.ui_colors['warning'] if confidence > 0.6 else \
                self.ui_colors['danger']
        
        cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)
        
        # Borde
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.ui_colors['light'], 2)
        
        # Texto de porcentaje
        percentage_text = f"{confidence*100:.1f}%"
        text_x = x + width + 10
        text_y = y + height - 5
        cv2.putText(frame, percentage_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ui_colors['text'], 2)

    def _draw_prediction_history(self, frame, x, y):
        """Dibuja el historial de predicciones"""
        if not self.prediction_history:
            return
        
        # Panel de fondo
        panel_height = len(self.prediction_history) * 35 + 40
        self._draw_overlay_panel(frame, x, y, 300, panel_height, self.ui_colors['background'], 0.8)
        
        # Título
        cv2.putText(frame, "HISTORIAL", (x + 10, y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.ui_colors['primary'], 2)
        
        # Mostrar historial
        for i, (prediction, confidence) in enumerate(zip(self.prediction_history, self.confidence_history)):
            text_y = y + 50 + i * 35
            alpha = 1.0 - (i * 0.2)  # Desvanecimiento gradual
            
            # Texto de predicción
            color = tuple(int(c * alpha) for c in self.ui_colors['text'])
            cv2.putText(frame, f"{prediction}", (x + 10, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Mini barra de confianza
            bar_x = x + 150
            bar_y = text_y - 12
            bar_width = 100
            bar_height = 8
            
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         self.ui_colors['dark'], -1)
            
            fill_width = int(bar_width * confidence * alpha)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                         color, -1)

    def _draw_status_indicators(self, frame):
        """Dibuja indicadores de estado del sistema"""
        frame_height, frame_width = frame.shape[:2]
        
        # Panel superior derecho
        panel_x = frame_width - 350
        panel_y = 10
        self._draw_overlay_panel(frame, panel_x, panel_y, 330, 120, self.ui_colors['background'], 0.8)
        
        # Título del panel
        cv2.putText(frame, "ESTADO DEL SISTEMA", (panel_x + 10, panel_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ui_colors['accent'], 2)
        
        # Estado del buffer con barra visual
        buffer_percentage = len(self.sequence_buffer) / self.sequence_length
        buffer_status = f"Buffer: {len(self.sequence_buffer)}/{self.sequence_length}"
        cv2.putText(frame, buffer_status, (panel_x + 10, panel_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ui_colors['primary'], 1)
        
        # Mini barra de buffer
        bar_x = panel_x + 10
        bar_y = panel_y + 45
        bar_width = 200
        bar_height = 6
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     self.ui_colors['dark'], -1)
        fill_width = int(bar_width * buffer_percentage)
        buffer_color = self.ui_colors['success'] if buffer_percentage == 1.0 else self.ui_colors['warning']
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                     buffer_color, -1)
        
        # Estado de actividad
        fps_text = "🔴 ACTIVO"
        cv2.putText(frame, f"Estado: {fps_text}", (panel_x + 10, panel_y + 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ui_colors['success'], 1)
        
        # Umbral de confianza
        threshold_text = f"Umbral: {self.prediction_threshold*100:.0f}%"
        cv2.putText(frame, threshold_text, (panel_x + 10, panel_y + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ui_colors['light'], 1)
        
        # Timestamp
        current_time = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"Hora: {current_time}", (panel_x + 10, panel_y + 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.ui_colors['light'], 1)

    def _draw_main_prediction(self, frame):
        """Dibuja la predicción principal con estilo mejorado"""
        frame_height, frame_width = frame.shape[:2]
        
        # Panel principal centrado
        panel_width = 600
        panel_height = 140
        panel_x = (frame_width - panel_width) // 2
        panel_y = 20
        
        # Efecto de glow/resplandor para predicciones con alta confianza
        flash_active = time.time() - self.prediction_flash_time < 1.0  # Flash durante 1 segundo
        
        if self.current_confidence > 0.8 or flash_active:
            # Dibujar varios paneles con transparencia decreciente para efecto glow
            glow_intensity = 3 if flash_active else 2
            for i in range(glow_intensity):
                alpha = 0.15 - (i * 0.04) if flash_active else 0.1 - (i * 0.03)
                offset = i * (7 if flash_active else 5)
                glow_color = self.ui_colors['success'] if flash_active else self.ui_colors['primary']
                self._draw_overlay_panel(frame, panel_x - offset, panel_y - offset, 
                                       panel_width + (offset * 2), panel_height + (offset * 2), 
                                       glow_color, alpha)
        
        # Panel principal
        self._draw_overlay_panel(frame, panel_x, panel_y, panel_width, panel_height, 
                               self.ui_colors['background'], 0.9)
        
        # Borde decorativo
        border_color = self.ui_colors['primary'] if self.current_confidence > 0.7 else self.ui_colors['warning']
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                     border_color, 3)
        
        # Título
        cv2.putText(frame, "PREDICCION ACTUAL", (panel_x + 20, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ui_colors['accent'], 2)
        
        if self.current_prediction:
            # Predicción principal con sombra
            text_size = cv2.getTextSize(self.current_prediction, cv2.FONT_HERSHEY_DUPLEX, 2.2, 3)[0]
            text_x = panel_x + (panel_width - text_size[0]) // 2
            text_y = panel_y + 70
            
            # Sombra del texto
            cv2.putText(frame, self.current_prediction, (text_x + 3, text_y + 3), 
                       cv2.FONT_HERSHEY_DUPLEX, 2.2, (0, 0, 0), 3)
            # Texto principal
            text_color = self.ui_colors['success'] if self.current_confidence > 0.8 else self.ui_colors['primary']
            cv2.putText(frame, self.current_prediction, (text_x, text_y), 
                       cv2.FONT_HERSHEY_DUPLEX, 2.2, text_color, 3)
            
            # Barra de confianza mejorada
            confidence_x = panel_x + 50
            confidence_y = panel_y + 100
            self._draw_confidence_bar(frame, self.current_confidence, confidence_x, confidence_y, 400, 25)
            
        else:
            # Mensaje cuando no hay predicción
            no_pred_text = "Esperando gesto..."
            text_size = cv2.getTextSize(no_pred_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            text_x = panel_x + (panel_width - text_size[0]) // 2
            text_y = panel_y + 80
            cv2.putText(frame, no_pred_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.ui_colors['warning'], 2)

    def _draw_hand_connections_styled(self, frame, hand_landmarks):
        """Dibuja las conexiones de las manos con estilo mejorado"""
        # Conexiones personalizadas con colores
        mp_drawing_styles = mp.solutions.drawing_styles
        
        self.mp_draw.draw_landmarks(
            frame, 
            hand_landmarks, 
            self.mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

    def _draw_help_panel(self, frame):
        """Dibuja panel de ayuda y consejos"""
        frame_height, frame_width = frame.shape[:2]
        
        # Panel inferior izquierdo
        panel_x = 20
        panel_y = frame_height - 200
        panel_width = 400
        panel_height = 180
        
        self._draw_overlay_panel(frame, panel_x, panel_y, panel_width, panel_height, 
                               self.ui_colors['background'], 0.85)
        
        # Título
        cv2.putText(frame, "💡 CONSEJOS DE USO", (panel_x + 10, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ui_colors['accent'], 2)
        
        # Consejos
        tips = [
            "• Manten las manos visibles",
            "• Gestos claros y lentos", 
            "• Buena iluminacion",
            "• Espera que se llene el buffer",
            "• Confianza >70% es mejor"
        ]
        
        for i, tip in enumerate(tips):
            y_pos = panel_y + 50 + (i * 25)
            cv2.putText(frame, tip, (panel_x + 15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ui_colors['light'], 1)
        
        # Indicador de calidad de detección
        buffer_quality = len(self.sequence_buffer) / self.sequence_length
        quality_text = "Calidad de deteccion:"
        cv2.putText(frame, quality_text, (panel_x + 15, panel_y + 170), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.ui_colors['light'], 1)
        
        # Barra de calidad
        quality_x = panel_x + 200
        quality_y = panel_y + 165
        quality_width = 150
        quality_height = 8
        
        cv2.rectangle(frame, (quality_x, quality_y), 
                     (quality_x + quality_width, quality_y + quality_height), 
                     self.ui_colors['dark'], -1)
        
        if buffer_quality > 0:
            fill_width = int(quality_width * buffer_quality)
            quality_color = self.ui_colors['success'] if buffer_quality > 0.8 else \
                           self.ui_colors['warning'] if buffer_quality > 0.5 else \
                           self.ui_colors['danger']
            cv2.rectangle(frame, (quality_x, quality_y), 
                         (quality_x + fill_width, quality_y + quality_height), 
                         quality_color, -1)

    def run(self):
        """Bucle principal mejorado con interfaz visual moderna"""
        print("=" * 60)
        print("🤟 TRADUCTOR DE LENGUAJE DE SEÑAS - VERSIÓN MEJORADA")
        print("=" * 60)
        print("🚀 Iniciando sistema de traducción en tiempo real...")
        print("💡 CONTROLES:")
        print("   • Presiona 'Q' para salir")
        print("   • Presiona 'R' para reiniciar buffer")
        print("   • Presiona 'ESPACIO' para info de estado")
        print("📋 CONSEJOS:")
        print("   • Mantén las manos visibles en la cámara")
        print("   • Realiza gestos claros y lentos")
        print("   • Asegúrate de tener buena iluminación")
        print("=" * 60)
        
        # Variables para la interfaz
        frame_count = 0
        last_time = time.time()
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                print("❌ Error al capturar frame de la cámara")
                continue

            # Voltear frame horizontalmente para efecto espejo
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            # Inicializar landmarks para este frame
            current_landmarks = np.zeros(2 * 21 * 3)
            hands_detected = 0
            
            # Procesar detecciones de manos
            if results.multi_hand_landmarks:
                handedness_labels = []
                
                # Obtener etiquetas de lateralidad
                if results.multi_handedness:
                    for h in results.multi_handedness:
                        handedness_labels.append(h.classification[0].label)
                
                # Procesar cada mano detectada
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    hands_detected += 1
                    
                    # Dibujar landmarks con estilo mejorado
                    self._draw_hand_connections_styled(frame, hand_landmarks)
                    
                    # Extraer y asignar landmarks
                    lm = self._extract_landmarks(hand_landmarks)
                    
                    if handedness_labels and idx < len(handedness_labels):
                        if handedness_labels[idx] == 'Left':
                            current_landmarks[0:63] = lm
                        else:
                            current_landmarks[63:126] = lm
                    else:
                        # Fallback sin etiquetas de lateralidad
                        if idx == 0:
                            current_landmarks[0:63] = lm
                        elif idx == 1:
                            current_landmarks[63:126] = lm

            # Agregar frame actual al buffer de secuencia
            self.sequence_buffer.append(current_landmarks)
            
            # Realizar predicción si el buffer está completo
            if len(self.sequence_buffer) == self.sequence_length:
                try:
                    sequence_array = np.expand_dims(np.array(self.sequence_buffer), axis=0)
                    prediction = self.model.predict(sequence_array, verbose=0)[0]
                    predicted_idx = np.argmax(prediction)
                    confidence = prediction[predicted_idx]

                    # Actualizar predicción si supera el umbral
                    if confidence > self.prediction_threshold:
                        new_prediction = self.signs[predicted_idx]
                        
                        # Solo actualizar si es diferente o ha pasado tiempo suficiente
                        if (new_prediction != self.current_prediction or 
                            time.time() - self.last_prediction_time > 2.0):
                            
                            self.current_prediction = new_prediction
                            self.current_confidence = confidence
                            self.last_prediction_time = time.time()
                            self.prediction_flash_time = time.time()  # Activar efecto flash
                            
                            # Agregar al historial
                            self.prediction_history.appendleft(new_prediction)
                            self.confidence_history.appendleft(confidence)
                            
                            # Mensaje en consola para nueva predicción
                            print(f"✨ Nueva predicción: {new_prediction} (Confianza: {confidence:.2f})")
                    
                    else:
                        # Reducir confianza gradualmente si no hay predicción clara
                        self.current_confidence = max(0, self.current_confidence - 0.02)
                        
                except Exception as e:
                    print(f"⚠️ Error en predicción: {e}")

            # === DIBUJAR INTERFAZ VISUAL ===
            
            # Predicción principal
            self._draw_main_prediction(frame)
            
            # Historial de predicciones
            self._draw_prediction_history(frame, 20, 180)
            
            # Indicadores de estado
            self._draw_status_indicators(frame)
            
            # Panel de ayuda
            self._draw_help_panel(frame)
            
            # Indicador de manos detectadas (mejorado)
            hands_indicator_x = frame.shape[1] - 300
            hands_indicator_y = frame.shape[0] - 60
            hands_text = f"👐 Manos: {hands_detected}/2"
            color = self.ui_colors['success'] if hands_detected > 0 else self.ui_colors['warning']
            
            # Fondo para el indicador
            text_size = cv2.getTextSize(hands_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (hands_indicator_x - 10, hands_indicator_y - 25), 
                         (hands_indicator_x + text_size[0] + 10, hands_indicator_y + 5), 
                         self.ui_colors['dark'], -1)
            
            cv2.putText(frame, hands_text, (hands_indicator_x, hands_indicator_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Instrucciones mejoradas
            instructions = "⌨️  Q=Salir | R=Reiniciar Buffer | ESPACIO=Info Estado"
            cv2.putText(frame, instructions, (20, frame.shape[0] - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ui_colors['primary'], 1)
            
            # Mostrar frame
            cv2.imshow('🤟 TRADUCTOR LSE - Versión Pro | MediaLengS', frame)
            
            # Manejar teclas
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                print("👋 Cerrando traductor...")
                break
            elif key == ord('r'):
                print("🔄 Reiniciando buffer de secuencias...")
                self.sequence_buffer.clear()
                self.current_prediction = ""
                self.current_confidence = 0.0
                self.prediction_history.clear()
                self.confidence_history.clear()
                print("✅ Buffer reiniciado correctamente")
            elif key == ord(' '):
                print("� === ESTADO ACTUAL DEL SISTEMA ===")
                print(f"   Buffer: {len(self.sequence_buffer)}/{self.sequence_length}")
                print(f"   Predicción actual: {self.current_prediction if self.current_prediction else 'Ninguna'}")
                print(f"   Confianza: {self.current_confidence:.2f}")
                print(f"   Manos detectadas: {hands_detected}")
                print(f"   Historial: {len(self.prediction_history)} predicciones")
                if self.prediction_history:
                    print(f"   Últimas predicciones: {list(self.prediction_history)[:3]}")
                print("=" * 45)
            
            frame_count += 1
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Traductor cerrado correctamente")

# =================================================================
# !! AÑADE O VERIFICA QUE ESTE BLOQUE EXISTA AL FINAL DEL ARCHIVO !!
# =================================================================
if __name__ == '__main__':
    MODEL_PATH = 'data/sign_model_gru.h5'
    SIGNS_PATH = 'data/label_encoder.npy'

    # --- Verificación de archivos ---
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: No se encontró el archivo del modelo en la ruta: {MODEL_PATH}")
        print("Asegúrate de haber ejecutado 'model_trainer_sequence.py' primero.")
    elif not os.path.exists(SIGNS_PATH):
        print(f"ERROR: No se encontró el archivo de etiquetas en la ruta: {SIGNS_PATH}")
        print("Asegúrate de que el archivo 'label_encoder.npy' exista en la carpeta 'data'.")
    else:
        print("Modelo y etiquetas encontrados. Iniciando traductor...")
        translator = RealTimeSequenceTranslator(model_path=MODEL_PATH, signs_path=SIGNS_PATH)
        translator.run()
