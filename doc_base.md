# Mejore sus asistentes de IA con Generación Aumentada por Recuperación (RAG)

## Una guía para implementar RAG con LLaMA 3 en su infraestructura

Por Carlos Jose Garces Rodrigues

Publicado en AI Advances el 15 de junio de 2024 - 11 minutos de lectura

A muchos de nosotros nos preocupa la privacidad y la seguridad de los datos, y la verdad es que en el momento en que ponemos información en Internet, perdemos el control sobre quién tendrá acceso a ella. Otros necesitarán garantizar que se implementen las medidas necesarias para evitar cualquier fuga, otros protegerán nuestra información y otros decidirán qué se debe hacer con nuestra información.

En los últimos meses, una tecnología que ha capturado mucha información son los Asistentes de IA. Cada vez que pide ayuda para reescribir su carta personal, mejorar su plan de negocios o corregir cualquier error en su ensayo, está compartiendo información personal, confidencial y sensible con otros, perdiendo el control sobre quién puede usar esta información y cómo.

En mi [artículo anterior](enlace_al_articulo_anterior), expliqué cómo configurar un Asistente de IA privado utilizando el modelo LLaMA 3, evitando compartir información y manteniendo todo bajo su control. Sin embargo, los modelos públicos como Llama 3 se entrenan utilizando datos públicos de un período específico, por lo que la información de los últimos días y su información personal, confidencial y sensible no está ahí. Esto significa que las respuestas que recibe de su Asistente de IA personal pueden no ser la respuesta más correcta. ¿Cómo puedo incluir esta información faltante en mi solicitud a mi Asistente de IA personal? Hay tres enfoques posibles:

- Ajuste fino de un modelo público con su información privada.
- Dar toda la información necesaria en el prompt inicial.
- Usar un método de Generación Aumentada por Recuperación (RAG).

El ajuste fino consiste en continuar entrenando un modelo de lenguaje pre-entrenado con sus propios datos. Este enfoque tiene varias ventajas y limitaciones que dependen del modelo y del caso de uso.

La ingeniería de prompts consiste en escribir cuidadosamente el texto de entrada con las plantillas y la información que queremos que use el modelo. La forma en que se estructura la información en el prompt inicial influye en la salida del modelo. Encontrar el prompt óptimo es un proceso iterativo y que consume mucho tiempo, pero por otro lado, no es necesario dedicar esfuerzo para reentrenar un modelo.

Este artículo se centrará en el último enfoque, explorando cómo se puede usar RAG para mejorar sus modelos con sus datos de manera eficiente y segura.

## Idea principal de RAG

La idea de RAG fue introducida en un artículo de 2020 por investigadores de Facebook AI Research, University College London y New York University. El artículo, titulado "[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](enlace_al_articulo)"¹, propuso el uso de memoria no paramétrica para resolver la limitación de conocimiento contenida en los parámetros del modelo.

Primero, comencemos explicando la idea principal detrás de RAG.

Como escribí, los modelos se entrenan utilizando datos públicos de un período específico. Durante el entrenamiento del modelo, estos datos se utilizan para cambiar los parámetros del modelo (por ejemplo, los modelos de la familia LLaMA 3 pueden tener 8B o 70B parámetros). Todo el conocimiento contenido en estos parámetros se puede llamar "memoria paramétrica entrenada" y, como dije antes, está limitado a la información contenida en los datos utilizados durante el entrenamiento.

Para generar o entender texto, los modelos de lenguaje miran un texto de entrada llamado contexto. El tamaño de este contexto depende del modelo. LLaMA 3 tiene un tamaño de contexto de 8k palabras o trozos de palabras (es decir, tokens). Esto significa que usando este contexto, podemos agregar información adicional a nuestro modelo que impactará en el resultado final. Este es el enfoque utilizado por la Ingeniería de Prompts, pasando toda la información e instrucciones necesarias en el prompt inicial para generar la salida. Sin embargo, como expliqué, este contexto es limitado en tamaño y la información que debe pasarse debe seleccionarse correctamente. Aquí es donde entra la parte "Aumentada por Recuperación" de RAG.

Imagine que da un prompt inicial (por ejemplo, un correo electrónico de un cliente, una factura para analizar, un posible escenario de fraude, etc.), puede buscar en diferentes fuentes información relacionada con este prompt inicial (por ejemplo, servicios y productos mencionados en el correo electrónico del cliente, contratos e historial de pagos asociados con la factura, uso histórico del posible defraudador, etc.), y luego usar toda esta información recuperada para aumentar el prompt inicial y usarlo como el contexto inicial del modelo. Usando este enfoque, el modelo será alimentado con la información correcta, obteniendo una salida no limitada por la memoria paramétrica entrenada.

## Los Pipelines

Se pueden considerar dos procesos diferentes:

- El primero está asociado con el preprocesamiento de datos de diferentes fuentes en un formato estándar que permite una búsqueda fácil de información relacionada. La complejidad de este proceso depende del formato original de los datos (por ejemplo, si es un PDF para transformar en texto, con imágenes que deben describirse, o si son datos tabulares, etc.) y también de la frecuencia de actualización de estos datos (por ejemplo, usos de clientes, pagos, cambios en productos, nuevas ofertas, etc.). En nuestro ejemplo, este proceso será el más fácil, recopilando datos en texto y la base para cualquier otro formato. Llamemos a este proceso el "**Proceso de Recolección de Datos**".

- El segundo está asociado con la inferencia del modelo. Obtiene el prompt inicial, recupera la información relacionada de la salida del Proceso de Recolección de Datos, crea una memoria no paramétrica explícita con el prompt inicial y los datos recuperados, alimenta el modelo con esta memoria como contexto y obtiene la salida del modelo. Llamemos a este proceso el "**Proceso de Inferencia**".

El punto en común de estos dos pipelines es la salida del Proceso de Recolección de Datos, que el Proceso de Inferencia utiliza para enriquecer la entrada del modelo. Una forma de gestionar y recuperar información relevante es utilizando una Base de Datos Vectorial como [Chroma](https://www.trychroma.com/).

Una Base de Datos Vectorial se puede usar para detectar la similitud entre el prompt de entrada y los textos almacenados. Básicamente, usando un modelo de lenguaje, los datos de texto se convierten en un formato numérico, conocido como embeddings. Los embeddings son representaciones vectoriales densas de texto, que capturan el significado semántico de palabras y frases. Usando embeddings, tanto el prompt de entrada como los textos almacenados se pueden comparar, y la información más relevante se puede recuperar en función de su similitud.

En este ejemplo, se utiliza un modelo [Dense Passage Retrieval (DPR)](https://github.com/facebookresearch/DPR) "***facebook/dpr-ctx_encoder-single-nq-base***" para transformar un texto en embeddings:

```python
In [1]: from transformers import DPRContextEncoder, DPRContextEncoderTokenizer  
text = "apple"  
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")  
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")  
inputs = context_tokenizer(text, return_tensors='pt')  
embeddings = context_encoder(**inputs).pooler_output.detach().numpy()[0].tolist()  
print(f"Embeddings: {embeddings}")  
print(f"Length embeddings: {len(embeddings)}")  
  
  
Out[1]:  
Embeddings: [-0.44394582509994507, 0.40732043981552124, … , -0.0779857337474823]  
Length embeddings: 768
```

Como se puede ver, el embedding de la palabra "apple" es un vector denso en un espacio con 768 dimensiones.

Cada palabra, frase o documento se puede representar como un punto en un espacio de alta dimensión donde los textos semánticamente similares se posicionan más cerca. Esto permite búsquedas y recuperaciones de similitud efectivas, ya que las representaciones vectoriales facilitan la comparación de grandes volúmenes de datos de texto.

La unidad básica de texto utilizada por los modelos de lenguaje se llama "Token". Un token puede ser una palabra, una parte de una palabra o un carácter, dependiendo del método de tokenización utilizado. Esto significa que el proceso comienza transformando el texto en su representación de tokens, y luego cada token se transforma en su embedding. Al final, es posible capturar el significado semántico de un bloque de texto (por ejemplo, párrafo, documento, etc.) usando todos los embeddings asociados con los tokens en este bloque de texto.

Sin embargo, dependiendo del tamaño del texto, el significado semántico puede cambiar. Al dividir el texto en piezas más pequeñas, coherentes y semánticamente relevantes, cada pieza puede preservar mejor el significado contextual y evitar diluir ideas individuales, haciendo que la detección de similitud sea más precisa. Además, como se explicó anteriormente, los modelos de lenguaje tienen una ventana de contexto fija (por ejemplo, el modelo LLaMA 3 tiene un tamaño de contexto de 8k tokens como límite). Dividir el texto en pequeñas piezas asegura que cada pieza se ajuste a la ventana de contexto del modelo. En el contexto de RAG, estas pequeñas piezas se llaman chunks. Para obtener más información sobre el chunking, puede leer [este notebook](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb) de [Greg Kamradt](https://www.youtube.com/@DataIndependent).

## Hora de programar

Si ha leído hasta este punto, debería entender la teoría básica detrás de RAG. Ahora, es hora de pasar de la teoría a la práctica. En nuestro ejemplo de código, usaremos Choma como base de datos vectorial, Ollama sirviendo un modelo LLaMA 3, y datos *privados y secretos* de una empresa ficticia llamada **ColdF**, que se enfoca en experimentos de reacción nuclear de fusión fría. Los datos de ColdF se dividen por Experimentos en pequeñas piezas de texto, lo que facilita nuestro proceso de chunking. No es el objetivo de este artículo evaluar diferentes estrategias de chunking.

Además, para ejecutar el código, necesitamos Python instalado (versión 3.11.9) y las siguientes bibliotecas:

- ollama==0.2.1
- chromadb==0.5.0  
- transformers==4.41.2
- torch==2.3.1

## Proceso de Recolección de Datos

El Proceso de Recolección de Datos implica recopilar datos de texto de fuentes de datos y almacenarlos en nuestra base de datos vectorial. Los pasos involucrados son los siguientes:

1. **Extracción de datos**: Recopilar datos de texto de varias fuentes, como páginas de intranet, registros de experimentos, artículos de investigación, documentos, informes, etc. Esto puede implicar transformar PDFs en texto, describir imágenes o convertir datos tabulares en un formato legible. También puede incluir eliminar información irrelevante y corregir errores. No incluiremos esta parte en nuestro código de ejemplo, y asumimos que los datos están en formato de texto [aquí](https://raw.githubusercontent.com/cgrodrigues/rag-intro/main/coldf_secret_experiments.txt).

2. **Generación de embeddings**: Aquí tenemos dos opciones: La primera es usar un modelo como "***facebook/dpr-ctx_encoder-single-nq-base***" o similar para tokenizar el texto y generar representaciones vectoriales densas (embeddings) para cada pieza de texto. La segunda es delegar esta tarea a la base de datos Chroma, que genera automáticamente los embeddings utilizando el modelo [Sentence Transformers](https://www.sbert.net/) "***all-MiniLM-L6-v2***". Nuestro enfoque será el primero.

3. **Almacenamiento en la Base de Datos Vectorial**: Almacenar los embeddings en una Base de Datos Vectorial. Esta base de datos permitirá búsquedas de similitud eficientes para recuperar información relevante basada en los prompts de entrada. Por defecto, Chroma utiliza la distancia L2 (Euclidiana) para los cálculos de similitud. Sin embargo, puede configurarlo para usar otras métricas, como la similitud coseno. En nuestro ejemplo, mantendremos la configuración predeterminada, pero esto se puede optimizar según la naturaleza de sus datos.

Aquí hay un fragmento de código para ilustrar la generación y el almacenamiento de embeddings:

```python
import requests import chromadb  
import re  
import uuid  
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer  
  
def  parse_experiments():  
  """ Get a list of secret experiment from ColdF  """  
 url = "https://raw.githubusercontent.com/cgrodrigues/rag-intro/main/coldf_secret_experiments.txt"  
  
 response = requests.get(url)  
  if response.status_code == 200:  
 text = response.text  
  
  # Split the text using the experiment identifier as a delimiter  
 experiments = text.split('# Experiment')  
    
  # Remove empty strings and reformat each experiment  
 experiments = ['# Experiment ' + exp.strip() for exp in experiments if exp.strip()]  
    
  return experiments  
  else:  
  raise Exception(f"Failed to fetch the file: {response.status_code}")  
  
def  init_chroma_db(store_name:str="documents"):  
  """ Initialize ChromaDB client. """  
 chroma_client = chromadb.PersistentClient(path="./cromadb")  
 vector_store = chroma_client.get_or_create_collection(store_name)  
  return chroma_client, vector_store  
  
def  chunk_embed_text(input, context_encoder, context_tokenizer, chunk_size:int=0, overlap_size:int=0  ):  
  """Generate chunks and id from the list of texts."""  
  
 chunks = []  
 ids = []  
 embeddings = []  
 pattern = r"^# Experiment \d+.*"  
  for text in  input:  
 start = 0  
    
  if chunk_size == 0:  
 _chunk_size = len(text) + overlap_size  
  else:  
 _chunk_size = chunk_size  
  match = re.findall(pattern, text)  
  if  match:  
  id = match[0]  
  else: # some random id  
  id = uuid.uuid4()  
 ct = 0  
  while start < len(text):  
  # get the chunk  
 end = start + _chunk_size  
 chunk = f"{text[start:end]}"  
 chunks.append(chunk)  
 start += _chunk_size - overlap_size  
  
  # get the embeddings  
 inputs = context_tokenizer(chunk, return_tensors='pt')  
 embedding = context_encoder(**inputs).pooler_output.detach().numpy()[0].tolist()  
 embeddings.append(embedding)  
  
  # get the id  
 ids.append(f"{id}_{str(ct)}")  
 ct += 1  
    
  return chunks, ids, embeddings  
  
  
def  preprocess_text_to_chroma(text, vector_store, chunk_size:int=0, overlap_size:int=0):   
  """Process text and store chunks in ChromaDB."""  
  
  # Get the encoder and tokenizer   
 context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")  
 context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")  
    
  # Create the chunks, ids and embeddings from the experiment text to put in the database  
 chunks, ids, embeddings = chunk_embed_text(input=text,   
 context_encoder=context_encoder,   
 context_tokenizer=context_tokenizer,   
 chunk_size=chunk_size,   
 overlap_size=overlap_size)  
    
  # Add to the database  
 vector_store.add(documents=chunks, embeddings=embeddings, ids=ids)  
    
# Get the secret experiment text  
text = parse_experiments()  
  
# Get the Vector Database Client  
chroma_client, vector_store = init_chroma_db("documents")  
  
# Put the secret experiments in the vector database  
preprocess_text_to_chroma(text=text,   
 vector_store=vector_store,   
 chunk_size=0,   
 overlap_size=0)
```

Después de ejecutar este código, tendremos una base de datos con todos nuestros documentos indexados y listos para comenzar la parte de inferencia.

## Proceso de Inferencia

El Proceso de Inferencia utiliza los datos recopilados y procesados, explicados en el paso anterior, para enriquecer la entrada de nuestro modelo LLaMA 3 y obtener una mejor salida.

Este proceso se basa en la inferencia del modelo utilizando Ollama. Antes de ejecutar el código, debe tener un servidor Ollama ejecutándose localmente. Para obtener detalles sobre cómo ejecutar un servidor Ollama localmente, consulte este [artículo](https://medium.com/the-beginners-guide/running-llama-3-on-your-own-infrastructure-with-ollama-f0f51d86d357).

Los pasos de este proceso son los siguientes:

1.  **Prompt inicial**: Aceptar el prompt inicial del usuario, que podría ser una pregunta, un documento o cualquier texto que requiera análisis.
2.  **Recuperar información relevante**: Buscar en la Base de Datos Vectorial los embeddings más similares al prompt de entrada. Recuperar los datos de texto correspondientes para aumentar el prompt inicial.
3.  **Generar contexto**: Crear un contexto que incluya el prompt inicial y la información recuperada. En el mundo real, se debe asegurar que se ajuste a la ventana de contexto del modelo. Esto significa que necesitamos obtener el número de tokens de nuestro contexto inicial y verificar que no exceda el contexto máximo de nuestro modelo.
4.  **Inferencia del modelo**: Alimentar el contexto en el modelo de lenguaje para generar la salida deseada.

Aquí hay un fragmento de código para ilustrar este proceso:

```python
<![endif]-->

from chromadb.api.types import QueryResult from ollama import Client  
  
def  get_inference_prompt(question:str, context_encoder, context_tokenizer) -> tuple[str, QueryResult]:  
  """ Based on the question get the most relevants chunks from teh database and create the prompt to feed the model Return the prompt and the result of the search in the database"""  
    
 inputs = context_tokenizer(question, return_tensors='pt')  
 query_embeddings = context_encoder(**inputs).pooler_output.detach().numpy()[0].tolist()  
    
 results = vector_store.query(query_embeddings, n_results=3)  
  # results = vector_store.query(query_texts=question, n_results=10)  
  
 documents = "\n".join(results['documents'][0])  
  
 prompt = f"""DOCUMENT:  {documents} QUESTION: {question} INSTRUCTIONS:  
Answer the users QUESTION using the DOCUMENT text above.  
Give short and concise answers.  
Keep your answer ground in the facts of the DOCUMENT.  
If the DOCUMENT doesn’t contain the facts to answer the QUESTION return 'NONE'"""  
  
  return prompt, results  
  
  
def  get_inference(question, context_encoder, context_tokenizer):  
  """ Inference in the LLaMA 3 model serve by Ollama """  
    
 host = ""  
 model = "llama3"  
 prompt, db_results = get_inference_prompt(question, context_encoder, context_tokenizer)  
  
 system_message = {"role": "system", "content": prompt}  
 messages = [system_message]  
  
 response = Client(host=host).chat(model=model, messages=messages, options= {"seed": 42, "top_p": 0.9, "temperature": 0 })  
  
  return response, prompt, db_results  
  
# Get the query from the end user, search in the vector database.  
question = input("Please enter question: ")  
  
# Get the encoder and tokenizer   
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")  
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")  
  
# Prepara the data and get the answer  
response, prompt, db_results = get_inference(question, context_encoder, context_tokenizer)  
  
print("\n================================\n")  
print(f"Prompt: {prompt}")  
print("\n================================\n")  
print(f"Database Results: {db_results}")  
print("\n================================\n")  
print(f"Response: {response['message']['content']}")  
print("\n================================\n")
```

Este simple ejemplo demuestra el ciclo completo de implementación de un pipeline de Generación Aumentada por Recuperación (RAG). Al utilizar una base de datos vectorial para almacenar y recuperar embeddings, podemos aumentar dinámicamente el contexto de nuestro modelo con información relevante, asegurando que las salidas del modelo se basen en los datos más pertinentes disponibles. Este enfoque nos permite superar las limitaciones de la memoria paramétrica del modelo, proporcionando respuestas más precisas y contextualmente relevantes.

## Personalización y desarrollo adicional

Ahora que tiene un pipeline básico para implementar RAG en su propia infraestructura con sus datos, puede personalizarlo aún más. Aquí hay algunas ideas:

-   ¿Qué tipo de documentos desea procesar? Tal vez crear agentes que transformen datos de PDF, Confluence, correo electrónico, base de datos, etc. a base de datos vectorial. ¿Y cuál debería ser el enfoque de chunking correcto para cada caso?
-   Calidad de la parte de recuperación del proceso. ¿Son correctos los chunks que estamos seleccionando?
-   ¿Imágenes a texto? Incluya un modelo LLaVa para transformar diagramas, fotos, etc., en texto.

Estos y muchos otros son material para otros artículos.

## Conclusión

Al combinar las fortalezas de los modelos de lenguaje pre-entrenados y los sistemas dinámicos de recuperación de información, puede asegurarse de que su IA brinde respuestas precisas, actualizadas y contextualmente relevantes. Además, al usar su propio Asistente de IA privado, mantiene un control total sobre sus datos, asegurando que su información personal, confidencial y sensible permanezca segura y privada. Estén atentos para futuros artículos que exploren características y mejoras más avanzadas. ¡Feliz programación!

No dude en consultar el código completo en mi repositorio de [GitHub](https://github.com/cgrodrigues/rag-intro).

## Referencias

[1] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://research.facebook.com/file/4283170945104179/Retrieval-Augmented-Generation-for-Knowledge-Intensive-NLP-Tasks.pdf). Facebook AI Research; University College London; New York University.

[Llama 3](https://medium.com/tag/llama-3)
[Ollama](https://medium.com/tag/ollama)
[Retrieval Augmented Gen](https://medium.com/tag/retrieval-augmented-gen)
[Rag](https://medium.com/tag/rag)

-------------------

## [4 Strategies to Optimize Retrieval-Augmented Generation](https://ai.gopubby.com/4-strategies-to-optimize-retrieval-augmented-generation-0ad902b5c3e2)

### [Using Private Data and Private Infrastructure for Enhanced AI Solutions](https://ai.gopubby.com/4-strategies-to-optimize-retrieval-augmented-generation-0ad902b5c3e2)

## [New Pandas rival, FireDucks, brings the smoke!](https://ai.gopubby.com/new-pandas-rival-fireducks-brings-the-smoke-3e8553ae466a)

### [Pandas goes head-to-head with a new competitor](https://ai.gopubby.com/new-pandas-rival-fireducks-brings-the-smoke-3e8553ae466a)

## [Run the strongest open-source LLM model: Llama3 70B with just a single 4GB GPU!](https://ai.gopubby.com/run-the-strongest-open-source-llm-model-llama3-70b-with-just-a-single-4gb-gpu-7e0ea2ad8ba2)

### [The strongest open source LLM model Llama3 has been released, Here is how you can run Llama3 70B locally with just 4GB GPU, even on Macbook](https://ai.gopubby.com/run-the-strongest-open-source-llm-model-llama3-70b-with-just-a-single-4gb-gpu-7e0ea2ad8ba2)

## [LLM Knowledge Graph Builder: From Zero to GraphRAG in Five Minutes](https://medium.com/neo4j/from-zero-to-graphrag-in-5-minutes-4ffcfcb4ebc2)

### [Extract and Use Knowledge Graphs in Your GenAI App](https://medium.com/neo4j/from-zero-to-graphrag-in-5-minutes-4ffcfcb4ebc2)
