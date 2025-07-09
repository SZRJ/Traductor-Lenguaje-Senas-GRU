# analyze_i_j_sequences.py
# Análisis específico de las diferencias entre secuencias I y J

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import statistics

class IJSequenceAnalyzer:
    def __init__(self, data_path='data/sequences'):
        self.data_path = data_path
        self.sequence_length = 50
        
        # Cargar datos de I y J
        self.sequences_i = self._load_sign_sequences('I')
        self.sequences_j = self._load_sign_sequences('J')
        
        print(f"📊 Cargadas {len(self.sequences_i)} secuencias de I")
        print(f"📊 Cargadas {len(self.sequences_j)} secuencias de J")

    def _load_sign_sequences(self, sign):
        """Carga todas las secuencias de una seña específica"""
        sequences = []
        sign_path = os.path.join(self.data_path, sign)
        
        if not os.path.exists(sign_path):
            print(f"❌ No se encontró la carpeta para la seña: {sign}")
            return sequences
        
        for seq_file in os.listdir(sign_path):
            if seq_file.endswith('.npy'):
                seq_data = np.load(os.path.join(sign_path, seq_file))
                sequences.append(seq_data)
        
        return sequences

    def calculate_movement_metrics(self, sequence):
        """Calcula métricas de movimiento para una secuencia"""
        if len(sequence) < 2:
            return {
                'total_movement': 0,
                'average_movement': 0,
                'max_movement': 0,
                'movement_variance': 0,
                'acceleration_changes': 0
            }
        
        # Calcular movimiento frame a frame
        frame_movements = []
        for i in range(1, len(sequence)):
            movement = euclidean(sequence[i-1], sequence[i])
            frame_movements.append(movement)
        
        # Calcular aceleración (cambios en el movimiento)
        accelerations = []
        for i in range(1, len(frame_movements)):
            acceleration = abs(frame_movements[i] - frame_movements[i-1])
            accelerations.append(acceleration)
        
        return {
            'total_movement': sum(frame_movements),
            'average_movement': statistics.mean(frame_movements),
            'max_movement': max(frame_movements),
            'movement_variance': statistics.variance(frame_movements) if len(frame_movements) > 1 else 0,
            'acceleration_changes': sum(accelerations),
            'movement_std': statistics.stdev(frame_movements) if len(frame_movements) > 1 else 0
        }

    def calculate_stability_metrics(self, sequence):
        """Calcula métricas de estabilidad para una secuencia"""
        if len(sequence) < 5:
            return {
                'stability_score': 0,
                'position_variance': 0,
                'drift_magnitude': 0
            }
        
        # Calcular posición promedio
        avg_position = np.mean(sequence, axis=0)
        
        # Calcular varianza de posición
        position_variances = []
        for i in range(len(sequence[0])):  # Para cada característica
            feature_values = [frame[i] for frame in sequence]
            if len(set(feature_values)) > 1:  # Evitar división por cero
                position_variances.append(statistics.variance(feature_values))
        
        avg_position_variance = statistics.mean(position_variances) if position_variances else 0
        
        # Calcular deriva (diferencia entre primer y último frame)
        drift_magnitude = euclidean(sequence[0], sequence[-1])
        
        # Puntuación de estabilidad (menor es más estable)
        stability_score = avg_position_variance + drift_magnitude
        
        return {
            'stability_score': stability_score,
            'position_variance': avg_position_variance,
            'drift_magnitude': drift_magnitude
        }

    def analyze_temporal_patterns(self, sequence):
        """Analiza patrones temporales en la secuencia"""
        if len(sequence) < 3:
            return {
                'has_clear_phases': False,
                'phase_count': 0,
                'rhythmic_pattern': False
            }
        
        # Calcular movimiento por frame
        movements = []
        for i in range(1, len(sequence)):
            movement = euclidean(sequence[i-1], sequence[i])
            movements.append(movement)
        
        # Detectar fases de movimiento (picos y valles)
        movement_threshold = statistics.mean(movements) + statistics.stdev(movements) if len(movements) > 1 else 0
        
        phases = []
        current_phase = 'low'
        for movement in movements:
            if movement > movement_threshold and current_phase == 'low':
                phases.append('high')
                current_phase = 'high'
            elif movement <= movement_threshold and current_phase == 'high':
                phases.append('low')
                current_phase = 'low'
        
        # Detectar patrones rítmicos
        rhythmic_pattern = len(phases) > 2 and len(set(phases)) > 1
        
        return {
            'has_clear_phases': len(phases) > 1,
            'phase_count': len(phases),
            'rhythmic_pattern': rhythmic_pattern,
            'movement_peaks': len([p for p in phases if p == 'high'])
        }

    def analyze_sign_sequences(self, sequences, sign_name):
        """Analiza todas las secuencias de una seña"""
        print(f"\n🔍 Analizando secuencias de {sign_name}...")
        
        movement_metrics = []
        stability_metrics = []
        temporal_metrics = []
        
        for i, seq in enumerate(sequences):
            # Normalizar secuencia si es necesario
            if len(seq) > self.sequence_length:
                seq = seq[:self.sequence_length]
            elif len(seq) < self.sequence_length:
                # Padding con el último frame
                last_frame = seq[-1] if len(seq) > 0 else np.zeros(126)
                while len(seq) < self.sequence_length:
                    seq = np.vstack([seq, last_frame])
            
            # Calcular métricas
            movement = self.calculate_movement_metrics(seq)
            stability = self.calculate_stability_metrics(seq)
            temporal = self.analyze_temporal_patterns(seq)
            
            movement_metrics.append(movement)
            stability_metrics.append(stability)
            temporal_metrics.append(temporal)
        
        # Agregar métricas
        avg_movement = {
            'total_movement': statistics.mean([m['total_movement'] for m in movement_metrics]),
            'average_movement': statistics.mean([m['average_movement'] for m in movement_metrics]),
            'max_movement': statistics.mean([m['max_movement'] for m in movement_metrics]),
            'movement_variance': statistics.mean([m['movement_variance'] for m in movement_metrics]),
            'acceleration_changes': statistics.mean([m['acceleration_changes'] for m in movement_metrics]),
            'movement_std': statistics.mean([m['movement_std'] for m in movement_metrics])
        }
        
        avg_stability = {
            'stability_score': statistics.mean([s['stability_score'] for s in stability_metrics]),
            'position_variance': statistics.mean([s['position_variance'] for s in stability_metrics]),
            'drift_magnitude': statistics.mean([s['drift_magnitude'] for s in stability_metrics])
        }
        
        temporal_summary = {
            'avg_phases': statistics.mean([t['phase_count'] for t in temporal_metrics]),
            'rhythmic_sequences': sum([1 for t in temporal_metrics if t['rhythmic_pattern']]),
            'avg_movement_peaks': statistics.mean([t['movement_peaks'] for t in temporal_metrics])
        }
        
        return {
            'movement': avg_movement,
            'stability': avg_stability,
            'temporal': temporal_summary,
            'individual_movement': movement_metrics,
            'individual_stability': stability_metrics,
            'individual_temporal': temporal_metrics
        }

    def compare_signs(self):
        """Compara las características de I y J"""
        print("\n🔍 Comparando características de I y J...")
        
        # Analizar ambas señas
        i_analysis = self.analyze_sign_sequences(self.sequences_i, 'I')
        j_analysis = self.analyze_sign_sequences(self.sequences_j, 'J')
        
        # Mostrar comparación
        print("\n" + "="*60)
        print("COMPARACIÓN DE CARACTERÍSTICAS I vs J")
        print("="*60)
        
        print("\n📊 MÉTRICAS DE MOVIMIENTO:")
        print("-" * 40)
        print(f"{'Métrica':<20} {'I':<12} {'J':<12} {'Diferencia':<12}")
        print("-" * 40)
        
        movement_metrics = ['total_movement', 'average_movement', 'max_movement', 
                           'movement_variance', 'acceleration_changes', 'movement_std']
        
        for metric in movement_metrics:
            i_val = i_analysis['movement'][metric]
            j_val = j_analysis['movement'][metric]
            diff = j_val - i_val
            print(f"{metric:<20} {i_val:<12.4f} {j_val:<12.4f} {diff:<12.4f}")
        
        print("\n📊 MÉTRICAS DE ESTABILIDAD:")
        print("-" * 40)
        print(f"{'Métrica':<20} {'I':<12} {'J':<12} {'Diferencia':<12}")
        print("-" * 40)
        
        stability_metrics = ['stability_score', 'position_variance', 'drift_magnitude']
        
        for metric in stability_metrics:
            i_val = i_analysis['stability'][metric]
            j_val = j_analysis['stability'][metric]
            diff = j_val - i_val
            print(f"{metric:<20} {i_val:<12.4f} {j_val:<12.4f} {diff:<12.4f}")
        
        print("\n📊 MÉTRICAS TEMPORALES:")
        print("-" * 40)
        print(f"{'Métrica':<20} {'I':<12} {'J':<12} {'Diferencia':<12}")
        print("-" * 40)
        
        temporal_metrics = ['avg_phases', 'rhythmic_sequences', 'avg_movement_peaks']
        
        for metric in temporal_metrics:
            i_val = i_analysis['temporal'][metric]
            j_val = j_analysis['temporal'][metric]
            diff = j_val - i_val
            print(f"{metric:<20} {i_val:<12.4f} {j_val:<12.4f} {diff:<12.4f}")
        
        return i_analysis, j_analysis

    def generate_recommendations(self, i_analysis, j_analysis):
        """Genera recomendaciones basadas en el análisis"""
        print("\n💡 RECOMENDACIONES PARA MEJORAR LA DISTINCIÓN I/J:")
        print("="*60)
        
        # Analizar diferencias clave
        movement_diff = j_analysis['movement']['average_movement'] - i_analysis['movement']['average_movement']
        stability_diff = j_analysis['stability']['stability_score'] - i_analysis['stability']['stability_score']
        temporal_diff = j_analysis['temporal']['avg_phases'] - i_analysis['temporal']['avg_phases']
        
        recommendations = []
        
        if movement_diff > 0.01:
            recommendations.append("✅ J tiene significativamente más movimiento que I")
            recommendations.append(f"   → Usar umbral de movimiento: {movement_diff/2:.4f}")
        
        if stability_diff > 0.01:
            recommendations.append("✅ I es más estable que J")
            recommendations.append(f"   → Usar puntuación de estabilidad como característica")
        
        if temporal_diff > 0.5:
            recommendations.append("✅ J tiene más fases temporales que I")
            recommendations.append(f"   → Detectar patrones temporales para distinguir")
        
        if abs(movement_diff) < 0.005:
            recommendations.append("⚠️  Movimiento promedio muy similar")
            recommendations.append("   → Considerar usar aceleración o varianza de movimiento")
        
        # Recomendaciones específicas para el modelo
        recommendations.append("\n🔧 CONFIGURACIONES RECOMENDADAS:")
        
        # Umbral de movimiento
        i_avg_movement = i_analysis['movement']['average_movement']
        j_avg_movement = j_analysis['movement']['average_movement']
        recommended_threshold = (i_avg_movement + j_avg_movement) / 2
        
        recommendations.append(f"• Umbral de movimiento: {recommended_threshold:.4f}")
        recommendations.append(f"• I (estática): movimiento < {recommended_threshold:.4f}")
        recommendations.append(f"• J (dinámica): movimiento > {recommended_threshold:.4f}")
        
        # Ventana temporal
        j_phases = j_analysis['temporal']['avg_phases']
        recommended_window = max(10, int(j_phases * 3))
        recommendations.append(f"• Ventana de análisis temporal: {recommended_window} frames")
        
        # Umbral de estabilidad
        stability_threshold = (i_analysis['stability']['stability_score'] + 
                             j_analysis['stability']['stability_score']) / 2
        recommendations.append(f"• Umbral de estabilidad: {stability_threshold:.4f}")
        
        for rec in recommendations:
            print(rec)
        
        return {
            'movement_threshold': recommended_threshold,
            'temporal_window': recommended_window,
            'stability_threshold': stability_threshold
        }

    def visualize_differences(self, i_analysis, j_analysis):
        """Crea visualizaciones de las diferencias"""
        print("\n📊 Generando visualizaciones...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Análisis Comparativo: Secuencias I vs J', fontsize=16, fontweight='bold')
        
        # 1. Comparación de movimiento promedio
        movement_metrics = ['total_movement', 'average_movement', 'max_movement', 'movement_variance']
        i_movement = [i_analysis['movement'][m] for m in movement_metrics]
        j_movement = [j_analysis['movement'][m] for m in movement_metrics]
        
        x = np.arange(len(movement_metrics))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, i_movement, width, label='I (Estática)', color='#3498db')
        axes[0, 0].bar(x + width/2, j_movement, width, label='J (Dinámica)', color='#e74c3c')
        axes[0, 0].set_title('Métricas de Movimiento')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(['Total', 'Promedio', 'Máximo', 'Varianza'], rotation=45)
        axes[0, 0].legend()
        
        # 2. Distribución de movimiento individual
        i_movements = [m['average_movement'] for m in i_analysis['individual_movement']]
        j_movements = [m['average_movement'] for m in j_analysis['individual_movement']]
        
        axes[0, 1].hist(i_movements, bins=15, alpha=0.7, label='I (Estática)', color='#3498db')
        axes[0, 1].hist(j_movements, bins=15, alpha=0.7, label='J (Dinámica)', color='#e74c3c')
        axes[0, 1].set_title('Distribución de Movimiento Promedio')
        axes[0, 1].set_xlabel('Movimiento Promedio')
        axes[0, 1].set_ylabel('Frecuencia')
        axes[0, 1].legend()
        
        # 3. Comparación de estabilidad
        stability_metrics = ['stability_score', 'position_variance', 'drift_magnitude']
        i_stability = [i_analysis['stability'][m] for m in stability_metrics]
        j_stability = [j_analysis['stability'][m] for m in stability_metrics]
        
        x = np.arange(len(stability_metrics))
        
        axes[1, 0].bar(x - width/2, i_stability, width, label='I (Estática)', color='#3498db')
        axes[1, 0].bar(x + width/2, j_stability, width, label='J (Dinámica)', color='#e74c3c')
        axes[1, 0].set_title('Métricas de Estabilidad')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(['Puntuación', 'Varianza Pos.', 'Deriva'], rotation=45)
        axes[1, 0].legend()
        
        # 4. Análisis temporal
        temporal_metrics = ['avg_phases', 'rhythmic_sequences', 'avg_movement_peaks']
        i_temporal = [i_analysis['temporal'][m] for m in temporal_metrics]
        j_temporal = [j_analysis['temporal'][m] for m in temporal_metrics]
        
        x = np.arange(len(temporal_metrics))
        
        axes[1, 1].bar(x - width/2, i_temporal, width, label='I (Estática)', color='#3498db')
        axes[1, 1].bar(x + width/2, j_temporal, width, label='J (Dinámica)', color='#e74c3c')
        axes[1, 1].set_title('Análisis Temporal')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(['Fases Prom.', 'Seq. Rítmicas', 'Picos Mov.'], rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('i_j_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Visualización guardada en: i_j_analysis.png")
        plt.show()

    def run_complete_analysis(self):
        """Ejecuta el análisis completo"""
        print("🚀 Iniciando análisis completo de secuencias I/J...")
        
        if len(self.sequences_i) == 0 or len(self.sequences_j) == 0:
            print("❌ No se encontraron secuencias suficientes para I o J")
            return
        
        # Realizar comparación
        i_analysis, j_analysis = self.compare_signs()
        
        # Generar recomendaciones
        recommendations = self.generate_recommendations(i_analysis, j_analysis)
        
        # Crear visualizaciones
        self.visualize_differences(i_analysis, j_analysis)
        
        # Guardar resultados
        results = {
            'i_analysis': i_analysis,
            'j_analysis': j_analysis,
            'recommendations': recommendations
        }
        
        np.save('i_j_analysis_results.npy', results)
        print("\n📄 Resultados guardados en: i_j_analysis_results.npy")
        
        print("\n✅ Análisis completo finalizado!")
        return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analizar diferencias entre secuencias I y J')
    parser.add_argument('--data-path', default='data/sequences', help='Ruta a los datos')
    
    args = parser.parse_args()
    
    try:
        analyzer = IJSequenceAnalyzer(data_path=args.data_path)
        analyzer.run_complete_analysis()
    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()
