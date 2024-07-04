# Importamos las bibliotecas necesarias
import chromadb  # Para interactuar con nuestra base de datos vectorial
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer  # Para procesar las preguntas

# Inicializamos el cliente de ChromaDB
# Usamos PersistentClient para que los datos se guarden en disco en la carpeta "./cromadb"
chroma_client = chromadb.PersistentClient(path="./cromadb")
# Obtenemos la colección "documents" que creamos anteriormente
collection = chroma_client.get_collection("documents")

# Cargamos el modelo y tokenizador para codificar preguntas
# Estos modelos convierten las preguntas en embeddings (representaciones numéricas)
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

def search_experiments(query, top_k=3):
    """
    Función para buscar experimentos basados en una consulta.
    
    :param query: La pregunta o consulta del usuario
    :param top_k: Número de resultados a devolver (por defecto 3)
    :return: Los resultados más relevantes de la búsqueda
    """
    # Codificamos la pregunta en un embedding
    inputs = question_tokenizer(query, return_tensors="pt")
    question_embedding = question_encoder(**inputs).pooler_output.detach().numpy()[0].tolist()
    
    # Buscamos en ChromaDB los documentos más similares al embedding de la pregunta
    results = collection.query(query_embeddings=[question_embedding], n_results=top_k)
    
    return results

def main():
    """
    Función principal que maneja la interacción con el usuario.
    """
    print("Bienvenido al buscador de experimentos secretos de ColdF. Escribe 'salir' para terminar.")
    while True:
        # Solicitamos una consulta al usuario
        query = input("\nIngresa tu consulta (o 'salir' para terminar): ")
        if query.lower() == 'salir':
            print("¡Hasta luego!")
            break
        
        # Realizamos la búsqueda
        results = search_experiments(query)
        
        # Mostramos los resultados
        print("\nResultados encontrados:")
        for i, (doc, score) in enumerate(zip(results['documents'][0], results['distances'][0]), 1):
            print(f"\n{i}. Puntuación: {score:.4f}")
            print(doc)

# Este bloque asegura que main() solo se ejecute si este script se ejecuta directamente
if __name__ == "__main__":
    main()