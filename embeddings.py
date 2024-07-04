# Importamos las bibliotecas necesarias para nuestro proyecto
import requests  # Para hacer peticiones HTTP y obtener datos de una URL
import chromadb  # Para trabajar con nuestra base de datos vectorial
import re        # Para usar expresiones regulares
import uuid      # Para generar identificadores únicos
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer  # Modelos de procesamiento de lenguaje

# Función para obtener y procesar los experimentos secretos de ColdF
def parse_experiments():
    # URL donde se encuentran los datos de los experimentos
    url = "https://raw.githubusercontent.com/cgrodrigues/rag-intro/main/coldf_secret_experiments.txt"
    
    # Hacemos una petición GET a la URL
    response = requests.get(url)
    if response.status_code == 200:  # Si la petición es exitosa (código 200)
        text = response.text  # Obtenemos el texto de la respuesta
        
        # Dividimos el texto en experimentos individuales
        # Usamos '# Experiment' como delimitador porque cada experimento comienza con esta cadena
        experiments = text.split('# Experiment')
        
        # Procesamos cada experimento:
        # - Eliminamos espacios en blanco al inicio y final
        # - Añadimos de nuevo '# Experiment' al inicio de cada uno
        # - Eliminamos experimentos vacíos
        experiments = ['# Experiment ' + exp.strip() for exp in experiments if exp.strip()]
        
        # Informamos cuántos experimentos se han obtenido
        print(f"Se han obtenido {len(text)} experimentos.")
        return experiments
    else:
        # Si la petición falla, lanzamos una excepción
        raise Exception(f"Failed to fetch the file: {response.status_code}")

# Función para inicializar y obtener el cliente de ChromaDB
def init_chroma_db(store_name:str="documents"):
    # Creamos un cliente persistente de ChromaDB (los datos se guardarán en disco)
    chroma_client = chromadb.PersistentClient(path="./cromadb")
    # Obtenemos o creamos una colección en la base de datos
    vector_store = chroma_client.get_or_create_collection(store_name)
    return chroma_client, vector_store

# Función para dividir el texto en chunks y generar embeddings
def chunk_embed_text(input, context_encoder, context_tokenizer, chunk_size:int=0, overlap_size:int=0):
    chunks = []     # Lista para almacenar los chunks de texto
    ids = []        # Lista para almacenar los IDs de cada chunk
    embeddings = [] # Lista para almacenar los embeddings de cada chunk
    pattern = r"^# Experiment \d+.*"  # Patrón para identificar el inicio de un experimento
    
    for text in input:
        start = 0
        
        # Determinamos el tamaño del chunk
        if chunk_size == 0:
            _chunk_size = len(text) + overlap_size
        else:
            _chunk_size = chunk_size
        
        # Buscamos el ID del experimento (o generamos uno si no se encuentra)
        match = re.findall(pattern, text)
        if match:
            id = match[0]
        else:
            id = uuid.uuid4()  # Generamos un ID aleatorio
        
        ct = 0  # Contador para el número de chunks en este experimento
        while start < len(text):
            # Obtenemos el chunk
            end = start + _chunk_size
            chunk = f"{text[start:end]}"
            chunks.append(chunk)
            
            # Movemos el inicio para el siguiente chunk, considerando el solapamiento
            start += _chunk_size - overlap_size
            
            # Generamos el embedding para este chunk
            inputs = context_tokenizer(chunk, return_tensors='pt')
            embedding = context_encoder(**inputs).pooler_output.detach().numpy()[0].tolist()
            embeddings.append(embedding)
            
            # Generamos un ID único para este chunk
            ids.append(f"{id}_{str(ct)}")
            ct += 1
        
    return chunks, ids, embeddings

# Función principal para procesar el texto y almacenarlo en ChromaDB
def preprocess_text_to_chroma(text, vector_store, chunk_size:int=0, overlap_size:int=0):
    # Cargamos el modelo y tokenizador para generar embeddings
    context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    
    # Generamos chunks, IDs y embeddings a partir del texto de los experimentos
    chunks, ids, embeddings = chunk_embed_text(input=text, 
                                               context_encoder=context_encoder, 
                                               context_tokenizer=context_tokenizer, 
                                               chunk_size=chunk_size, 
                                               overlap_size=overlap_size)
    
    # Añadimos los chunks, IDs y embeddings a la base de datos vectorial
    vector_store.add(documents=chunks, embeddings=embeddings, ids=ids)
    print("Los experimentos se han procesado y almacenado con éxito en ChromaDB.")

# Ejecución principal del script

# Obtenemos el texto de los experimentos secretos
text = parse_experiments()

# Inicializamos el cliente de la base de datos vectorial
chroma_client, vector_store = init_chroma_db("documents")

# Procesamos y almacenamos los experimentos en la base de datos vectorial
preprocess_text_to_chroma(text=text, 
                          vector_store=vector_store, 
                          chunk_size=0, 
                          overlap_size=0) 