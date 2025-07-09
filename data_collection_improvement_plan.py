# PLAN DE MEJORA PARA DISTINCIÓN SEÑAS ESTÁTICAS/DINÁMICAS
# Lenguaje de Señas Peruano (LSP)

"""
PROBLEMA IDENTIFICADO:
- Dataset muy desbalanceado: 80% estáticas vs 6.7% dinámicas
- Solo 80 secuencias dinámicas vs 960 estáticas (ratio 1:12)
- Faltan señas dinámicas importantes como Ñ
"""

import os
import numpy as np
import json
from datetime import datetime

class DataCollectionStrategy:
    def __init__(self):
        self.static_signs = {
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
            'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
            'V', 'W', 'X', 'Y'
        }
        
        # Señas dinámicas en LSP
        self.dynamic_signs = {
            'J': 'Movimiento en J con dedo índice',
            'Z': 'Movimiento zigzag',
            'Ñ': 'Movimiento ondulatorio',
            'RR': 'Vibración de la R fuerte',
            'LL': 'Movimiento lateral de doble L'
        }
        
        # Palabras dinámicas comunes en LSP
        self.dynamic_words = {
            'HOLA': 'Saludo con movimiento de mano',
            'ADIÓS': 'Despedida con movimiento',
            'GRACIAS': 'Movimiento de agradecimiento',
            'POR FAVOR': 'Gesto de petición',
            'SÍ': 'Afirmación con movimiento de cabeza/mano',
            'NO': 'Negación con movimiento',
            'CÓMO': 'Pregunta con movimiento',
            'QUÉ': 'Interrogación dinámica',
            'DÓNDE': 'Pregunta direccional',
            'CUÁNDO': 'Pregunta temporal'
        }
        
        # Números dinámicos (algunos tienen movimiento)
        self.dynamic_numbers = {
            '100': 'Número con movimiento',
            '1000': 'Número con movimiento amplio'
        }

    def generate_collection_plan(self):
        """Genera plan detallado de recolección"""
        plan = {
            'objetivo': 'Balancear dataset para mejor distinción estáticas/dinámicas',
            'meta_total': 2000,  # secuencias objetivo
            'balance_objetivo': {
                'estaticas': '50%',
                'dinamicas': '30%', 
                'frases': '20%'
            },
            'recoleccion_necesaria': {},
            'prioridades': [],
            'estrategias': []
        }
        
        # Calcular necesidades actuales
        current_static = 960
        current_dynamic = 80
        current_phrases = 160
        
        # Objetivos
        target_static = 1000   # 50%
        target_dynamic = 600   # 30%
        target_phrases = 400   # 20%
        
        # Necesidades
        need_static = max(0, target_static - current_static)
        need_dynamic = target_dynamic - current_dynamic
        need_phrases = target_phrases - current_phrases
        
        plan['recoleccion_necesaria'] = {
            'estaticas': need_static,
            'dinamicas': need_dynamic,
            'frases': need_phrases
        }
        
        # Prioridades (orden de importancia)
        plan['prioridades'] = [
            {
                'tipo': 'CRÍTICO',
                'items': ['J', 'Z', 'Ñ', 'RR'],
                'objetivo_por_item': 100,
                'razon': 'Señas dinámicas básicas del alfabeto'
            },
            {
                'tipo': 'ALTO',
                'items': ['ADIÓS', 'SÍ', 'NO', 'CÓMO'],
                'objetivo_por_item': 80,
                'razon': 'Palabras dinámicas de uso común'
            },
            {
                'tipo': 'MEDIO',
                'items': ['QUÉ', 'DÓNDE', 'CUÁNDO', 'LL'],
                'objetivo_por_item': 60,
                'razon': 'Expansión de vocabulario dinámico'
            },
            {
                'tipo': 'BAJO',
                'items': ['100', '1000'],
                'objetivo_por_item': 40,
                'razon': 'Números con componente dinámico'
            }
        ]
        
        return plan

    def create_collection_protocol(self):
        """Protocolo específico para recolección de señas dinámicas"""
        protocol = {
            'configuracion_camara': {
                'fps': 30,
                'resolucion': '1280x720',
                'iluminacion': 'Uniforme, sin sombras',
                'fondo': 'Contrastante, preferiblemente oscuro',
                'distancia': '1-1.5 metros de la cámara'
            },
            'duracion_secuencias': {
                'estaticas': '2-3 segundos (hold estable)',
                'dinamicas': '3-5 segundos (movimiento completo)',
                'frases': '4-6 segundos (expresión natural)'
            },
            'variaciones_requeridas': {
                'personas': 'Mínimo 3 personas diferentes por seña',
                'velocidades': 'Lenta, normal, rápida',
                'amplitudes': 'Movimiento pequeño, normal, amplio',
                'angulos': 'Frontal, semi-lateral (±30°)',
                'iluminacion': 'Buena, regular, tenue'
            },
            'criterios_calidad': {
                'manos_visibles': '100% del tiempo',
                'landmarks_detectados': '>95% de frames',
                'movimiento_completo': 'Inicio, desarrollo, fin',
                'estabilidad_inicial': '0.5s inicial estable para dinámicas',
                'estabilidad_final': '0.5s final estable para dinámicas'
            }
        }
        return protocol

    def save_plan(self, plan, protocol):
        """Guarda el plan en archivo JSON"""
        full_plan = {
            'fecha_creacion': datetime.now().isoformat(),
            'plan_recoleccion': plan,
            'protocolo': protocol,
            'progreso': {
                'completado': False,
                'items_recolectados': [],
                'ultima_actualizacion': None
            }
        }
        
        with open('plan_mejora_dataset.json', 'w', encoding='utf-8') as f:
            json.dump(full_plan, f, indent=2, ensure_ascii=False)
        
        print("📄 Plan guardado en: plan_mejora_dataset.json")

def main():
    strategy = DataCollectionStrategy()
    
    print("🎯 GENERANDO PLAN DE MEJORA DEL DATASET")
    print("=" * 50)
    
    plan = strategy.generate_collection_plan()
    protocol = strategy.create_collection_protocol()
    
    # Mostrar resumen
    print(f"\n📊 RESUMEN DEL PLAN:")
    print(f"Meta total: {plan['meta_total']} secuencias")
    print(f"Balance objetivo: {plan['balance_objetivo']}")
    
    print(f"\n📋 RECOLECCIÓN NECESARIA:")
    for tipo, cantidad in plan['recoleccion_necesaria'].items():
        print(f"  {tipo.capitalize()}: {cantidad} secuencias")
    
    print(f"\n🚨 PRIORIDADES DE RECOLECCIÓN:")
    for prioridad in plan['prioridades']:
        total_needed = len(prioridad['items']) * prioridad['objetivo_por_item']
        print(f"\n  {prioridad['tipo']} ({total_needed} secuencias):")
        print(f"    Items: {', '.join(prioridad['items'])}")
        print(f"    Por item: {prioridad['objetivo_por_item']} secuencias")
        print(f"    Razón: {prioridad['razon']}")
    
    # Guardar plan
    strategy.save_plan(plan, protocol)
    
    print(f"\n✅ Plan de mejora generado exitosamente!")

if __name__ == "__main__":
    main()
