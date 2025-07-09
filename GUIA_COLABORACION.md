# 🤝 Guía de Colaboración - Sistema Híbrido de Reconocimiento de Señas

## 📋 Resumen del Proyecto

Este repositorio contiene un sistema avanzado de reconocimiento de lenguaje de señas peruano que **resuelve el problema de confusión entre señas estáticas y dinámicas** (como I y J).

## 🚀 Mejoras Implementadas

### ✅ **Problema Resuelto**
- **Antes**: El modelo confundía señas estáticas (I) con dinámicas (J, Z)
- **Después**: 100% de precisión en distinguir I vs J con modelo híbrido

### 🧠 **Arquitectura del Sistema Híbrido**
- **Entrada dual**: Secuencias de landmarks + características de movimiento
- **Modelo CNN+GRU**: Procesa patrones espaciales y temporales
- **Evaluación de calidad**: Análisis automático de movimiento

## 📁 Archivos Principales

### 🔧 **Scripts de Entrenamiento**
- `advanced_model_trainer.py` - Entrenador con CNN+LSTM+Attention
- `enhanced_model_trainer.py` - Entrenador híbrido optimizado
- `model_trainer_sequence.py` - Entrenador original (básico)

### 📊 **Scripts de Análisis**
- `analyze_i_j_sequences.py` - Análisis detallado I vs J
- `analyze_current_dataset.py` - Estado del dataset actual
- `evaluate_hybrid_model.py` - Comparación de modelos

### 🎯 **Recolección de Datos**
- `enhanced_data_collector.py` - Colector con evaluación de calidad
- `data_collector.py` - Colector original
- `plan_mejora_dataset.json` - Plan de recolección balanceada

### 🔄 **Traductores en Tiempo Real**
- `hybrid_real_time_translator.py` - Traductor híbrido (RECOMENDADO)
- `real_time_translator.py` - Traductor original
- `enhanced_real_time_translator.py` - Traductor mejorado

### 📚 **Documentación**
- `README_HYBRID.md` - Guía completa del sistema híbrido
- `RECOMENDACIONES_FINALES.md` - Próximos pasos y mejoras
- `GUIA_COLABORACION.md` - Esta guía

## 🛠️ Instalación y Uso

### 1. **Clonar el Repositorio**
```bash
git clone https://github.com/Jaed69/Traductor-Lenguaje-Senas-GRU.git
cd Traductor-Lenguaje-Senas-GRU
```

### 2. **Instalar Dependencias**
```bash
pip install -r requirements.txt
```

### 3. **Usar el Sistema Híbrido (Recomendado)**
```bash
python hybrid_real_time_translator.py
```

## 📈 Cómo Contribuir

### 🎯 **Prioridades Actuales**
1. **Recolectar más datos** siguiendo `plan_mejora_dataset.json`
2. **Probar nuevas señas** con el sistema híbrido
3. **Optimizar arquitecturas** en `advanced_model_trainer.py`
4. **Mejorar interfaz** del traductor en tiempo real

### 📊 **Análisis de Datos**
```bash
# Analizar estado actual del dataset
python analyze_current_dataset.py

# Comparar modelos
python evaluate_hybrid_model.py

# Análisis específico I vs J
python analyze_i_j_sequences.py
```

### 🔄 **Reentrenar Modelos**
```bash
# Entrenamiento híbrido (recomendado)
python enhanced_model_trainer.py

# Entrenamiento avanzado con attention
python advanced_model_trainer.py
```

### 📊 **Recolectar Nuevos Datos**
```bash
# Recolección con evaluación de calidad
python enhanced_data_collector.py

# Seguir plan de mejora
# Ver: plan_mejora_dataset.json
```

## 🎯 Resultados Clave

### ✅ **Métricas de Éxito**
- **I vs J**: 100% precisión (antes: confusión constante)
- **Tiempo real**: ~30 FPS de procesamiento
- **Calidad**: Evaluación automática de movimiento

### 📊 **Análisis de Movimiento**
- **Seña I**: Movimiento promedio: 0.0031
- **Seña J**: Movimiento promedio: 0.0086 (~2.8x más)
- **Umbral**: 0.005 para distinguir estática/dinámica

## 🚨 Problemas Conocidos

1. **Dataset desbalanceado**: Pocas señas estáticas vs dinámicas
2. **Calidad variable**: Algunos datos con ruido de cámara
3. **Señas similares**: Necesita más datos para N, M, etc.

## 📋 TODO para Colaboradores

### 🎯 **Corto Plazo**
- [ ] Recolectar 50+ secuencias de cada seña estática
- [ ] Probar sistema con nuevos usuarios
- [ ] Documentar nuevas señas encontradas

### 🔮 **Mediano Plazo**
- [ ] Implementar detección de múltiples manos
- [ ] Agregar expresiones faciales
- [ ] Optimizar para dispositivos móviles

### 🌟 **Largo Plazo**
- [ ] Dataset completo de señas peruanas
- [ ] Traducción bidireccional (texto → señas)
- [ ] Integración con aplicaciones móviles

## 📞 Contacto y Soporte

- **Issues**: Usar GitHub Issues para reportar problemas
- **Discusiones**: GitHub Discussions para ideas y mejoras
- **Documentación**: Leer `README_HYBRID.md` para detalles técnicos

## 🏆 Créditos

Sistema desarrollado para mejorar la comunicación inclusiva mediante tecnología de reconocimiento de lenguaje de señas peruano.

---

**💡 Tip**: Comienza con `hybrid_real_time_translator.py` para ver el sistema en acción!
