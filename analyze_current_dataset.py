import os

def analyze_current_dataset():
    """Analiza el dataset actual"""
    sequences_path = 'data/sequences'
    
    if not os.path.exists(sequences_path):
        print("❌ No se encontró el directorio de secuencias")
        return
    
    print("📊 ANÁLISIS DEL DATASET ACTUAL")
    print("=" * 50)
    
    # Clasificar señas por tipo
    static_signs = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'}
    dynamic_signs = {'J', 'Z'}
    phrase_signs = {'HOLA', 'GRACIAS', 'POR FAVOR'}
    
    total_static = 0
    total_dynamic = 0
    total_phrases = 0
    
    print("\n🤚 SEÑAS ESTÁTICAS:")
    for sign in sorted(os.listdir(sequences_path)):
        if os.path.isdir(os.path.join(sequences_path, sign)) and sign in static_signs:
            count = len(os.listdir(os.path.join(sequences_path, sign)))
            print(f"  {sign}: {count} secuencias")
            total_static += count
    
    print(f"\n  Total estáticas: {total_static} secuencias")
    
    print("\n👋 SEÑAS DINÁMICAS:")
    for sign in sorted(os.listdir(sequences_path)):
        if os.path.isdir(os.path.join(sequences_path, sign)) and sign in dynamic_signs:
            count = len(os.listdir(os.path.join(sequences_path, sign)))
            print(f"  {sign}: {count} secuencias")
            total_dynamic += count
    
    print(f"\n  Total dinámicas: {total_dynamic} secuencias")
    
    print("\n💬 FRASES/PALABRAS:")
    for sign in sorted(os.listdir(sequences_path)):
        if os.path.isdir(os.path.join(sequences_path, sign)) and sign in phrase_signs:
            count = len(os.listdir(os.path.join(sequences_path, sign)))
            print(f"  {sign}: {count} secuencias")
            total_phrases += count
    
    print(f"\n  Total frases: {total_phrases} secuencias")
    
    print("\n📊 BALANCE DEL DATASET:")
    total = total_static + total_dynamic + total_phrases
    print(f"Total secuencias: {total}")
    print(f"Estáticas: {total_static} ({total_static/total*100:.1f}%)")
    print(f"Dinámicas: {total_dynamic} ({total_dynamic/total*100:.1f}%)")
    print(f"Frases: {total_phrases} ({total_phrases/total*100:.1f}%)")
    
    # Identificar desbalances
    print("\n⚠️  PROBLEMAS IDENTIFICADOS:")
    if total_dynamic < total_static * 0.5:
        print(f"- Dataset muy desbalanceado hacia señas estáticas")
        print(f"  Recomendación: Recolectar más señas dinámicas")
    
    if total_dynamic < 100:
        print(f"- Pocas señas dinámicas ({total_dynamic})")
        print(f"  Recomendación: Mínimo 200-300 secuencias dinámicas")
    
    # Señas dinámicas faltantes
    all_dynamic = {'J', 'Z', 'Ñ'}  # Ñ también puede ser dinámica en LSP
    missing_dynamic = all_dynamic - set(os.listdir(sequences_path))
    if missing_dynamic:
        print(f"- Señas dinámicas faltantes: {', '.join(missing_dynamic)}")

if __name__ == "__main__":
    analyze_current_dataset()
