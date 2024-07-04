# Buscador de Experimentos Secretos ColdF

Este proyecto implementa un sistema de Generación Aumentada por Recuperación (RAG) para buscar y consultar experimentos secretos de la empresa ficticia ColdF, que se enfoca en experimentos de reacción nuclear de fusión fría.

## Descripción

El sistema utiliza una base de datos vectorial (ChromaDB) para almacenar y recuperar eficientemente información sobre los experimentos. Implementa un modelo de lenguaje pre-entrenado para codificar preguntas y buscar los documentos más relevantes.

## Requisitos

- Python 3.11.9
- ollama==0.2.1
- chromadb==0.5.0
- transformers==4.41.2
- torch==2.3.1

## Instalación

1. Crea un entorno virtual:
```bash
python3.11 -m venv /ruta/a/tu/entorno/rag
source /ruta/a/tu/entorno/rag/bin/activate
``` 
2. Instala las dependencias:
```bash
python3 -m pip install requests chromadb==0.5.0 transformers==4.41.2 torch==2.3.1 ollama==0.2.1
``` 
3. Verifica la instalación:
```bash
python3 -m pip list
```
## Uso

1. Primero, ejecuta el script para procesar y almacenar los experimentos en la base de datos:
```bash
python3 embeddings.py
```
2. Luego, ejecuta la aplicación de búsqueda:
```bash
python3 chat_app.py
```

3. Sigue las instrucciones en pantalla para realizar consultas sobre los experimentos.

## Estructura del Proyecto

- `embeddings.py`: Script para procesar y almacenar los experimentos en ChromaDB.
- `chat_app.py`: Aplicación principal para realizar búsquedas y consultas.
- `cromadb/`: Directorio donde se almacena la base de datos vectorial.

## Funcionamiento

El sistema utiliza un modelo de Procesamiento de Lenguaje Natural para codificar tanto los documentos como las consultas en vectores. Estos vectores se utilizan para buscar similitudes semánticas entre la consulta del usuario y los documentos almacenados, permitiendo recuperar la información más relevante.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue para discutir cambios mayores antes de hacer un pull request.

## Licencia

[Incluye aquí la licencia de tu proyecto]

## Contacto

[Tu nombre o el de tu organización]
[Información de contacto]# rag
