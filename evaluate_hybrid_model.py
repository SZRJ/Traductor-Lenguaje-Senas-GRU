# evaluate_hybrid_model.py
# Script para evaluar el rendimiento del modelo híbrido vs modelo original

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class ModelComparator:
    def __init__(self, data_path='data/sequences'):
        self.data_path = data_path
        self.sequence_length = 50
        self.num_features = 21 * 3 * 2  # 126 características
        
        # Cargar datos
        self.X, self.y, self.signs = self._load_data()
        
        # Dividir en conjuntos de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"📊 Datos cargados: {self.X.shape[0]} muestras, {len(self.signs)} señas")
        print(f"📊 Entrenamiento: {self.X_train.shape[0]}, Prueba: {self.X_test.shape[0]}")

    def _load_data(self):
        """Carga los datos de secuencias"""
        sequences, labels = [], []
        signs = np.array([name for name in os.listdir(self.data_path) 
                         if os.path.isdir(os.path.join(self.data_path, name))])
        
        label_encoder = LabelEncoder()
        label_encoder.fit(signs)
        
        for sign in signs:
            sign_path = os.path.join(self.data_path, sign)
            for seq_file in os.listdir(sign_path):
                res = np.load(os.path.join(sign_path, seq_file))
                sequences.append(res)
                labels.append(label_encoder.transform([sign])[0])
        
        # Padding y normalización
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.utils import to_categorical
        
        X = pad_sequences(sequences, maxlen=self.sequence_length, 
                         padding='post', truncating='post', dtype='float32')
        y = to_categorical(labels).astype(int)
        
        return X, y, signs

    def evaluate_model(self, model_path, model_name):
        """Evalúa un modelo específico"""
        print(f"\n🔍 Evaluando {model_name}...")
        
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"❌ Error cargando {model_path}: {e}")
            return None
        
        # Realizar predicciones
        y_pred = model.predict(self.X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.y_test, axis=1)
        
        # Métricas generales
        test_loss, test_accuracy = model.evaluate(self.X_test, self.y_test, verbose=0)
        
        print(f"📈 Precisión: {test_accuracy:.4f}")
        print(f"📈 Pérdida: {test_loss:.4f}")
        
        # Reporte de clasificación
        report = classification_report(
            y_true_classes, y_pred_classes, 
            target_names=self.signs, 
            output_dict=True, 
            zero_division=0
        )
        
        # Matriz de confusión
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        return {
            'accuracy': test_accuracy,
            'loss': test_loss,
            'predictions': y_pred,
            'predicted_classes': y_pred_classes,
            'true_classes': y_true_classes,
            'classification_report': report,
            'confusion_matrix': cm,
            'model_name': model_name
        }

    def analyze_i_j_confusion(self, results):
        """Analiza específicamente la confusión entre I y J"""
        print(f"\n🔍 Análisis de confusión I/J para {results['model_name']}...")
        
        # Encontrar índices de I y J
        try:
            i_idx = np.where(self.signs == 'I')[0][0]
            j_idx = np.where(self.signs == 'J')[0][0]
        except IndexError:
            print("❌ No se encontraron las señas I o J en el dataset")
            return None
        
        cm = results['confusion_matrix']
        
        # Extraer confusiones específicas
        i_as_j = cm[i_idx, j_idx]  # I clasificada como J
        j_as_i = cm[j_idx, i_idx]  # J clasificada como I
        i_correct = cm[i_idx, i_idx]  # I clasificada correctamente
        j_correct = cm[j_idx, j_idx]  # J clasificada correctamente
        
        # Calcular totales
        total_i = np.sum(cm[i_idx, :])
        total_j = np.sum(cm[j_idx, :])
        
        # Métricas de confusión
        i_accuracy = i_correct / total_i if total_i > 0 else 0
        j_accuracy = j_correct / total_j if total_j > 0 else 0
        i_j_confusion_rate = (i_as_j + j_as_i) / (total_i + total_j) if (total_i + total_j) > 0 else 0
        
        results_ij = {
            'I_accuracy': i_accuracy,
            'J_accuracy': j_accuracy,
            'I_as_J_count': i_as_j,
            'J_as_I_count': j_as_i,
            'I_correct_count': i_correct,
            'J_correct_count': j_correct,
            'total_I_samples': total_i,
            'total_J_samples': total_j,
            'confusion_rate': i_j_confusion_rate
        }
        
        print(f"📊 Señas I en test: {total_i}, J en test: {total_j}")
        print(f"✅ I correctas: {i_correct}/{total_i} ({i_accuracy:.3f})")
        print(f"✅ J correctas: {j_correct}/{total_j} ({j_accuracy:.3f})")
        print(f"❌ I clasificadas como J: {i_as_j}")
        print(f"❌ J clasificadas como I: {j_as_i}")
        print(f"⚠️  Tasa de confusión I/J: {i_j_confusion_rate:.3f}")
        
        return results_ij

    def plot_comparison(self, results_original, results_hybrid, save_path='model_comparison.png'):
        """Crea gráficos comparativos"""
        print("\n📊 Generando gráficos comparativos...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comparación: Modelo Original vs Híbrido', fontsize=16, fontweight='bold')
        
        # 1. Comparación de precisión general
        models = ['Original', 'Híbrido']
        accuracies = [results_original['accuracy'], results_hybrid['accuracy']]
        
        axes[0, 0].bar(models, accuracies, color=['#3498db', '#e74c3c'])
        axes[0, 0].set_title('Precisión General')
        axes[0, 0].set_ylabel('Precisión')
        axes[0, 0].set_ylim(0, 1)
        
        # Agregar valores en las barras
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Matriz de confusión para I y J - Modelo Original
        i_idx = np.where(self.signs == 'I')[0][0]
        j_idx = np.where(self.signs == 'J')[0][0]
        
        cm_original_ij = results_original['confusion_matrix'][[i_idx, j_idx]][:, [i_idx, j_idx]]
        sns.heatmap(cm_original_ij, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['I', 'J'], yticklabels=['I', 'J'], ax=axes[0, 1])
        axes[0, 1].set_title('Confusión I/J - Modelo Original')
        axes[0, 1].set_ylabel('Verdadero')
        axes[0, 1].set_xlabel('Predicho')
        
        # 3. Matriz de confusión para I y J - Modelo Híbrido
        cm_hybrid_ij = results_hybrid['confusion_matrix'][[i_idx, j_idx]][:, [i_idx, j_idx]]
        sns.heatmap(cm_hybrid_ij, annot=True, fmt='d', cmap='Greens',
                   xticklabels=['I', 'J'], yticklabels=['I', 'J'], ax=axes[1, 0])
        axes[1, 0].set_title('Confusión I/J - Modelo Híbrido')
        axes[1, 0].set_ylabel('Verdadero')
        axes[1, 0].set_xlabel('Predicho')
        
        # 4. Comparación de métricas I/J
        ij_original = self.analyze_i_j_confusion(results_original)
        ij_hybrid = self.analyze_i_j_confusion(results_hybrid)
        
        metrics = ['I_accuracy', 'J_accuracy', 'confusion_rate']
        original_values = [ij_original[m] for m in metrics]
        hybrid_values = [ij_hybrid[m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, original_values, width, label='Original', color='#3498db')
        axes[1, 1].bar(x + width/2, hybrid_values, width, label='Híbrido', color='#e74c3c')
        
        axes[1, 1].set_title('Métricas I/J Específicas')
        axes[1, 1].set_ylabel('Valor')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(['Precisión I', 'Precisión J', 'Tasa Confusión'])
        axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 1)
        
        # Agregar valores en las barras
        for i, (orig, hyb) in enumerate(zip(original_values, hybrid_values)):
            axes[1, 1].text(i - width/2, orig + 0.01, f'{orig:.3f}', 
                           ha='center', va='bottom', fontsize=8)
            axes[1, 1].text(i + width/2, hyb + 0.01, f'{hyb:.3f}', 
                           ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico guardado en: {save_path}")
        plt.show()

    def generate_detailed_report(self, results_original, results_hybrid):
        """Genera un reporte detallado"""
        print("\n📋 Generando reporte detallado...")
        
        report_content = []
        report_content.append("=" * 80)
        report_content.append("REPORTE COMPARATIVO: MODELO ORIGINAL VS HÍBRIDO")
        report_content.append("=" * 80)
        report_content.append("")
        
        # Métricas generales
        report_content.append("MÉTRICAS GENERALES:")
        report_content.append("-" * 50)
        report_content.append(f"Modelo Original - Precisión: {results_original['accuracy']:.4f}")
        report_content.append(f"Modelo Híbrido  - Precisión: {results_hybrid['accuracy']:.4f}")
        report_content.append(f"Mejora: {results_hybrid['accuracy'] - results_original['accuracy']:.4f}")
        report_content.append("")
        
        # Análisis I/J
        ij_original = self.analyze_i_j_confusion(results_original)
        ij_hybrid = self.analyze_i_j_confusion(results_hybrid)
        
        report_content.append("ANÁLISIS ESPECÍFICO I/J:")
        report_content.append("-" * 50)
        report_content.append("MODELO ORIGINAL:")
        report_content.append(f"  - Precisión I: {ij_original['I_accuracy']:.4f}")
        report_content.append(f"  - Precisión J: {ij_original['J_accuracy']:.4f}")
        report_content.append(f"  - I → J: {ij_original['I_as_J_count']} casos")
        report_content.append(f"  - J → I: {ij_original['J_as_I_count']} casos")
        report_content.append(f"  - Tasa confusión: {ij_original['confusion_rate']:.4f}")
        report_content.append("")
        
        report_content.append("MODELO HÍBRIDO:")
        report_content.append(f"  - Precisión I: {ij_hybrid['I_accuracy']:.4f}")
        report_content.append(f"  - Precisión J: {ij_hybrid['J_accuracy']:.4f}")
        report_content.append(f"  - I → J: {ij_hybrid['I_as_J_count']} casos")
        report_content.append(f"  - J → I: {ij_hybrid['J_as_I_count']} casos")
        report_content.append(f"  - Tasa confusión: {ij_hybrid['confusion_rate']:.4f}")
        report_content.append("")
        
        # Mejoras
        i_improvement = ij_hybrid['I_accuracy'] - ij_original['I_accuracy']
        j_improvement = ij_hybrid['J_accuracy'] - ij_original['J_accuracy']
        confusion_improvement = ij_original['confusion_rate'] - ij_hybrid['confusion_rate']
        
        report_content.append("MEJORAS CONSEGUIDAS:")
        report_content.append("-" * 50)
        report_content.append(f"  - Mejora precisión I: {i_improvement:+.4f}")
        report_content.append(f"  - Mejora precisión J: {j_improvement:+.4f}")
        report_content.append(f"  - Reducción confusión: {confusion_improvement:+.4f}")
        report_content.append("")
        
        # Conclusiones
        report_content.append("CONCLUSIONES:")
        report_content.append("-" * 50)
        if confusion_improvement > 0:
            report_content.append("✅ El modelo híbrido REDUCE exitosamente la confusión I/J")
        else:
            report_content.append("❌ El modelo híbrido NO mejora la confusión I/J")
            
        if i_improvement > 0 and j_improvement > 0:
            report_content.append("✅ Mejora la precisión de AMBAS señas I y J")
        elif i_improvement > 0:
            report_content.append("⚠️  Mejora la precisión de I pero no de J")
        elif j_improvement > 0:
            report_content.append("⚠️  Mejora la precisión de J pero no de I")
        else:
            report_content.append("❌ No mejora la precisión de ninguna seña")
        
        report_content.append("")
        report_content.append("=" * 80)
        
        # Guardar reporte
        report_text = "\n".join(report_content)
        with open('model_comparison_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        print("\n📄 Reporte guardado en: model_comparison_report.txt")

    def run_full_evaluation(self):
        """Ejecuta la evaluación completa"""
        print("🚀 Iniciando evaluación comparativa completa...")
        
        # Evaluar modelo original
        results_original = self.evaluate_model('data/sign_model_gru.h5', 'Original GRU')
        if results_original is None:
            print("❌ No se pudo evaluar el modelo original")
            return
        
        # Evaluar modelo híbrido
        results_hybrid = self.evaluate_model('data/sign_model_hybrid.h5', 'Híbrido')
        if results_hybrid is None:
            print("❌ No se pudo evaluar el modelo híbrido")
            return
        
        # Generar comparaciones
        self.plot_comparison(results_original, results_hybrid)
        self.generate_detailed_report(results_original, results_hybrid)
        
        print("\n✅ Evaluación completa finalizada!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluar y comparar modelos')
    parser.add_argument('--data-path', default='data/sequences', help='Ruta a los datos')
    
    args = parser.parse_args()
    
    try:
        comparator = ModelComparator(data_path=args.data_path)
        comparator.run_full_evaluation()
    except Exception as e:
        print(f"❌ Error durante la evaluación: {e}")
        import traceback
        traceback.print_exc()
