# Traductor de Lenguaje de Señas Dinámico en Tiempo Real 🤟

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Este proyecto es una aplicación de Visión por Computadora que traduce señas dinámicas del Lenguaje de Señas Americano (ASL) a texto en tiempo real. Utiliza la librería MediaPipe de Google para la detección precisa de los puntos de la mano y una Red Neuronal Recurrente (GRU) para interpretar secuencias de movimiento, permitiendo el reconocimiento de gestos complejos más allá de letras estáticas.

## ✨ Características Principales

- **🎯 Detección en Tiempo Real:** Captura y procesa el video de la cámara web con baja latencia
- **🔄 Reconocimiento de Señas Dinámicas:** Analiza secuencias de movimientos para interpretar gestos complejos
- **🎨 Interfaz Visual Moderna:** Nueva interfaz profesional con overlays, historial y barras de confianza
- **📊 Monitoreo en Tiempo Real:** Panel de estado, historial de predicciones y consejos visuales
- **⚡ Efectos Visuales:** Animaciones de glow, flash en nuevas predicciones y indicadores de calidad
- **🎮 Controles Intuitivos:** Teclado para reiniciar, capturar estado y salir
- **📈 Extensible:** Sistema diseñado para añadir nuevas señas fácilmente

## 🎨 Nuevas Características Visuales (v2.0)

### 🖥️ Interfaz Moderna
- **Panel Principal:** Predicción actual con efectos de glow y sombras
- **Historial Visual:** Últimas 5 predicciones con desvanecimiento gradual
- **Barras de Confianza:** Indicadores visuales de la certeza de las predicciones
- **Panel de Estado:** Monitoreo del buffer, umbral y timestamp
- **Panel de Consejos:** Tips de uso y calidad de detección en tiempo real

### ✨ Efectos Visuales
- **Flash de Predicción:** Efecto luminoso cuando se detecta una nueva seña
- **Glow Dinámico:** Resplandor que se intensifica con alta confianza
- **Degradado de Historial:** Las predicciones pasadas se desvanecen gradualmente
- **Indicadores de Color:** Verde para éxito, amarillo para advertencia, rojo para error

### 🎮 Controles Mejorados
- **Q:** Salir del programa
- **R:** Reiniciar buffer de secuencias
- **ESPACIO:** Mostrar información detallada del estado actual

## 🛠️ Tecnologías Utilizadas

- **Lenguaje:** Python 3.9
- **Framework de Deep Learning:** TensorFlow / Keras
- **Visión por Computadora:** OpenCV, MediaPipe
- **Librerías de Datos:** NumPy, Scikit-learn, Pandas

## 🚀 Instalación y Configuración

Sigue estos pasos para poner el proyecto en funcionamiento en tu máquina local.

1.  **Clona el repositorio:**
    ```bash
    git clone [https://github.com/tu-usuario/tu-repositorio.git](https://github.com/tu-usuario/tu-repositorio.git)
    cd tu-repositorio
    ```

2.  **Crea y activa un entorno virtual** (se recomienda Conda):
    ```bash
    conda create -n traductor_senas python=3.9
    conda activate traductor_senas
    ```

3.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

## 📋 Modo de Uso

El proyecto funciona en tres etapas principales:

### 1. Recolección de Datos

Ejecuta el script para capturar tus propias secuencias de señas. El programa te guiará para grabar múltiples ejemplos para cada seña que definas.

```bash
python data_collector.py
```

### 2. Entrenamiento del Modelo

Una vez que tengas los datos, entrena el modelo GRU. Este script procesará las secuencias, construirá la red neuronal y la entrenará, guardando el modelo final en la carpeta `data`.

```bash
python model_trainer_sequence.py
```

### 3. Ejecutar el Traductor

¡La parte divertida! Ejecuta este script para iniciar la aplicación. Usará el modelo que acabas de entrenar para traducir las señas que hagas frente a la cámara.

```bash
python real_time_translator.py
```

También puedes ejecutar directamente:

```bash
python main.py
```

## 🤝 Colaboración y Compartición de Datos

Para proyectos en equipo, puedes compartir datos de entrenamiento para crear un dataset más grande y diverso:

### Exportar tus datos
```bash
python data_exporter.py --contributor-id user_001 --contributor-name "Tu Nombre"
```

### Importar datos de otros colaboradores
```bash
python data_importer.py --import shared_data/user_002/
```

### Entrenar con dataset combinado
```bash
python model_trainer_sequence.py --use-merged-data --epochs 100
```

### Analizar estadísticas del dataset
```bash
python dataset_stats.py --visualizations --report dataset_report.txt
```

📖 **Consulta `DATA_SHARING.md` para más detalles sobre colaboración.**

## 🔮 Futuras Mejoras

- [ ] Añadir un vocabulario más extenso de señas, incluyendo palabras completas.
- [ ] Desarrollar una interfaz gráfica de usuario (GUI) con Tkinter o PyQt para mostrar el texto traducido de forma más amigable.
- [ ] Implementar la capacidad de construir frases completas.
- [ ] Optimizar el modelo para un rendimiento aún más rápido en dispositivos con menos recursos.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

## 👤 Autor

* **[Tu Nombre]** - [tu-usuario-de-github](https://github.com/tu-usuario)

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para contribuir:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## ⭐ ¿Te gusta el proyecto?

Si este proyecto te ha sido útil, ¡no olvides darle una estrella en GitHub!