import tensorflow as tf

# Cargar y examinar el modelo híbrido
model = tf.keras.models.load_model('data/sign_model_hybrid.h5')

print('=== ESTRUCTURA DEL MODELO HÍBRIDO ===')
model.summary()

print('\n=== ENTRADAS DEL MODELO ===')
for i, input_layer in enumerate(model.inputs):
    print(f'Entrada {i}: {input_layer.name}, Forma: {input_layer.shape}')

print('\n=== SALIDAS DEL MODELO ===')
for i, output_layer in enumerate(model.outputs):
    print(f'Salida {i}: {output_layer.name}, Forma: {output_layer.shape}')
