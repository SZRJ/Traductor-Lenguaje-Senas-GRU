# Sistema Híbrido de Reconocimiento de Lenguaje de Señas Peruano (LSP)

## 🎯 Solución al Problema I/J

Este proyecto implementa un **modelo híbrido mejorado** para resolver la confusión entre la seña **I** (estática) y la seña **J** (dinámica) en el reconocimiento de lenguaje de señas peruano.

### 🔍 Problema Original
- La seña **I** es **estática** (sin movimiento significativo)
- La seña **J** es **dinámica** (requiere movimiento específico)
- Los modelos tradicionales confunden estas señas debido a similitudes en la forma de la mano

### ✅ Solución Implementada
- **Modelo Híbrido**: Distingue entre señas estáticas y dinámicas
- **Análisis de Movimiento**: Detecta patrones de movimiento en tiempo real
- **Validación Temporal**: Verifica compatibilidad entre tipo de seña y movimiento detectado
- **Umbrales Adaptativos**: Ajusta automáticamente la sensibilidad según el tipo de seña

## 🏗️ Arquitectura del Sistema

```
📁 MediaLengS/
├── 🤖 Modelos
│   ├── model_trainer_sequence.py      # Entrenador original
│   ├── enhanced_model_trainer.py      # Entrenador híbrido (mencionado en conversación)
│   ├── data/sign_model_gru.h5        # Modelo original
│   └── data/sign_model_hybrid.h5     # Modelo híbrido mejorado
│
├── 🎥 Traducción en Tiempo Real
│   ├── real_time_translator.py       # Traductor original
│   └── hybrid_real_time_translator.py # Traductor híbrido (NUEVO)
│
├── 📊 Análisis y Evaluación
│   ├── analyze_i_j_sequences.py      # Análisis específico I/J (NUEVO)
│   ├── evaluate_hybrid_model.py      # Comparación de modelos (NUEVO)
│   └── dataset_stats.py              # Estadísticas del dataset
│
├── 📋 Datos
│   ├── data_collector.py             # Recolector de datos
│   ├── data/sequences/               # Secuencias de entrenamiento
│   └── data/label_encoder.npy        # Codificador de etiquetas
│
└── 🔧 Configuración
    ├── requirements.txt              # Dependencias actualizadas
    └── README.md                     # Este archivo
```

## 🚀 Instalación y Configuración

### 1. Clonar el Repositorio
```bash
git clone <tu-repositorio>
cd MediaLengS
```

### 2. Crear Entorno Virtual
```bash
# Crear entorno conda (recomendado)
conda create -n lsp python=3.9
conda activate lsp

# O usar venv
python -m venv lsp_env
# Windows
lsp_env\Scripts\activate
# Linux/Mac
source lsp_env/bin/activate
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Verificar Instalación
```bash
python -c "import tensorflow as tf; import cv2; import mediapipe as mp; print('✅ Todas las dependencias instaladas correctamente')"
```

## 🎮 Uso del Sistema

### 1. Análisis de Diferencias I/J

Antes de entrenar, analiza las diferencias entre las señas I y J:

```bash
python analyze_i_j_sequences.py --data-path data/sequences
```

**Salidas:**
- Métricas comparativas detalladas
- Visualizaciones de diferencias
- Recomendaciones de configuración
- `i_j_analysis_results.npy`: Resultados guardados
- `i_j_analysis.png`: Gráficos comparativos

### 2. Usar el Traductor Híbrido

Ejecuta el traductor mejorado que distingue entre señas estáticas y dinámicas:

```bash
# Modo básico
python hybrid_real_time_translator.py

# Con configuración personalizada
python hybrid_real_time_translator.py --model data/sign_model_hybrid.h5 --threshold 0.8
```

**Características del Traductor Híbrido:**
- 🎯 **Detección de Movimiento**: Analiza patrones de movimiento en tiempo real
- 🔍 **Clasificación Automática**: Identifica señas como estáticas o dinámicas
- ⚡ **Validación Cruzada**: Verifica compatibilidad entre seña predicha y movimiento detectado
- 📊 **UI Mejorada**: Muestra análisis de movimiento y compatibilidad

**Controles:**
- `q`: Salir
- `r`: Resetear buffers
- `t`: Ajustar umbral de confianza
- `d`: Modo debug (información detallada)

### 3. Evaluar Rendimiento

Compara el modelo original con el híbrido:

```bash
python evaluate_hybrid_model.py --data-path data/sequences
```

**Salidas:**
- Comparación de precisión general
- Análisis específico de confusión I/J
- Gráficos comparativos
- `model_comparison_report.txt`: Reporte detallado
- `model_comparison.png`: Visualizaciones

### 4. Entrenamiento (Si necesitas reentrenar)

```bash
# Modelo original
python model_trainer_sequence.py --epochs 50

# El modelo híbrido ya está entrenado y guardado
# Si necesitas reentrenarlo, usa enhanced_model_trainer.py (mencionado en conversación)
```

## 📊 Características del Modelo Híbrido

### Clasificación de Señas

**Señas Estáticas (requieren poca movilidad):**
- A, B, C, D, E, F, G, H, **I**, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y

**Señas Dinámicas (requieren movimiento):**
- **J**, Z, HOLA, GRACIAS, POR FAVOR

### Análisis de Movimiento

El sistema analiza múltiples métricas:

1. **Movimiento Frame-a-Frame**
   - Distancia euclidiana entre frames consecutivos
   - Promedio, máximo y varianza de movimiento

2. **Estabilidad de Posición**
   - Varianza de posición de landmarks
   - Deriva entre primer y último frame

3. **Patrones Temporales**
   - Detección de fases de movimiento
   - Análisis rítmico de secuencias

### Lógica de Decisión Híbrida

```python
# Para señas estáticas (como I)
if predicted_sign == "I":
    if movement_type == "static" and confidence > 0.8:
        return "I"  # Alta confianza con poco movimiento
    elif movement_type == "transitional" and confidence > 0.9:
        return "I"  # Confianza muy alta incluso con movimiento leve

# Para señas dinámicas (como J)
if predicted_sign == "J":
    if movement_type == "dynamic" and confidence > 0.8:
        return "J"  # Movimiento detectado con buena confianza
    elif movement_type == "transitional" and confidence > 0.95:
        return "J"  # Confianza muy alta con movimiento moderado
```

## 🎯 Resultados Esperados

### Mejoras en Precisión I/J

Basado en el entrenamiento previo (mencionado en conversación):

- **Precisión I**: 1.000 (sin casos de confusión con J)
- **Precisión J**: 1.000 (sin casos de confusión con I)
- **Eliminación completa** de confusión entre I y J en datos de prueba

### Configuraciones Optimizadas

- **Umbral de Movimiento**: ~0.02 (calibrado automáticamente)
- **Ventana Temporal**: 15-20 frames para análisis
- **Umbral de Confianza**: 0.8 para señas compatibles, 0.9+ para incompatibles

## 🔧 Configuración Avanzada

### Ajustar Umbrales

Modifica los umbrales en `hybrid_real_time_translator.py`:

```python
# En la clase HybridRealTimeTranslator
self.prediction_threshold = 0.8        # Umbral base de confianza
self.movement_threshold = 0.02         # Umbral de detección de movimiento
```

### Agregar Nuevas Señas

1. Clasifica la nueva seña:
```python
# En hybrid_real_time_translator.py
self.static_signs.add('NUEVA_SEÑA_ESTATICA')
# o
self.dynamic_signs.add('NUEVA_SEÑA_DINAMICA')
```

2. Recolecta datos:
```bash
python data_collector.py
```

3. Reentrena el modelo si es necesario

## 📈 Monitoreo y Debug

### Información en Tiempo Real

El traductor híbrido muestra:
- Nivel de movimiento actual
- Tipo de movimiento detectado (estático/dinámico/transicional)
- Compatibilidad entre seña predicha y movimiento
- Historial de predicciones

### Modo Debug

Activa con la tecla `d` para ver:
- Valores exactos de confianza
- Métricas de movimiento detalladas
- Estado de buffers internos

## 🤝 Contribución y Datos

### Recolectar Más Datos

```bash
python data_collector.py
```

### Compartir Datos (GitHub LFS)

Ver `GITHUB_DATA_SHARING.md` para instrucciones detalladas sobre cómo compartir datos de entrenamiento usando Git LFS.

## 🐛 Solución de Problemas

### Error: Modelo híbrido no encontrado
```bash
# Verifica que el archivo existe
ls data/sign_model_hybrid.h5

# Si no existe, entrena un nuevo modelo o usa el original
python hybrid_real_time_translator.py --model data/sign_model_gru.h5
```

### Error: ImportError TensorFlow
```bash
# Reinstala TensorFlow
pip uninstall tensorflow
pip install tensorflow>=2.13.0
```

### Baja precisión en detección
1. Verifica iluminación (buena luz uniforme)
2. Asegúrate de que las manos estén completamente visibles
3. Ajusta el umbral con la tecla `t`
4. Resetea buffers con la tecla `r`

### Confusión persistente I/J
1. Ejecuta el análisis de secuencias:
```bash
python analyze_i_j_sequences.py
```

2. Verifica las recomendaciones de configuración
3. Ajusta umbrales según las métricas reportadas

## 📚 Documentación Adicional

- `GITHUB_DATA_SHARING.md`: Guía para compartir datos de entrenamiento
- `dataset_stats.py`: Análisis estadístico del dataset
- Comentarios en el código para detalles técnicos

## 📄 Licencia

[Especifica tu licencia aquí]

## 👥 Autores

[Tu nombre y información de contacto]

---

## 🎉 ¡Disfruta usando el Sistema Híbrido de LSP!

Este sistema representa una mejora significativa en la distinción entre señas estáticas y dinámicas, especialmente diseñado para resolver la confusión I/J en el lenguaje de señas peruano.
