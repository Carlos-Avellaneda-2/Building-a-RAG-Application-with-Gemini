# RAG sobre priorización de acceso a consulta especializada en Colombia (MGTE)

Aplicación RAG (Retrieval-Augmented Generation) construida con **Gemini + LangChain + Chroma**, desarrollada como homework del workshop de RAG, extendiendo los patrones vistos allí a un caso de uso real: el proyecto de semestre *"Intelligent Prioritization of Access to Specialist Consultation — an Enterprise Architecture Problem in the Colombian Health System"* (Carlos Andres Avellaneda Franco).

## 1. Objetivo y caso de uso seleccionado

El proyecto de semestre investiga cómo priorizar, de forma transparente y clínicamente defendible, el acceso a consulta médica especializada en Colombia, en un contexto de creciente escasez de oferta frente a la demanda, y evalúa si —y bajo qué condiciones— la inteligencia artificial aporta valor frente a la priorización basada en reglas explícitas.

Esta aplicación RAG permite consultar en lenguaje natural la evidencia recopilada para ese proyecto: el nuevo marco regulatorio colombiano (Modelo de Gestión de Tiempos de Espera, MGTE), la evidencia cuantitativa del problema de acceso, y la literatura internacional sobre IA aplicada a la priorización/triage de referencias médicas.

Preguntas objetivo de la aplicación:
- ¿Qué es el MGTE, quién lo expidió y qué fases de implementación tiene?
- ¿Qué especialidades y tiempos máximos de espera prioriza la Circular 038 de 2025?
- ¿Qué evidencia cuantitativa existe sobre el problema de acceso a consulta especializada en Colombia?
- ¿Qué tan bien funcionan los modelos de machine learning para priorizar referencias médicas, según la literatura internacional?
- Preguntas fuera del alcance de las fuentes (para verificar que el sistema no alucina).

## 2. Descripción y origen de la colección de documentos

Se usaron **5 documentos públicos**, tomados directamente de la lista de referencias del artículo del proyecto de semestre (`Articulo_Proyecto_Triage.docx`), descargados y guardados como texto plano en `data/` junto con su metadata (título, URL, identificador de fuente):

| Archivo | Título | URL | Tipo |
|---|---|---|---|
| `doc1_circular_038_cups_tiempos.txt` | Circular Externa 038 de 2025 | consultorsalud.com/tiempos-de-espera-para-citas-con-especialistas | Regulatorio / operativo |
| `doc2_resolucion_2117_mgte.txt` | Resolución 2117 de 2025 (MGTE) | consultorsalud.com/modelo-tiempos-espera-salud-colombia-2025 | Regulatorio |
| `doc3_ai_referral_triage_cpc_queensland.txt` | Abdel-Hafez et al. (2023), *Frontiers in Digital Health* (CC-BY) | pmc.ncbi.nlm.nih.gov/articles/PMC10642163 | Académico / internacional |
| `doc4_radiografia_acceso_salud_colombia.txt` | Radiografía del acceso a la salud en Colombia | consultorsalud.com/radiografia-del-acceso-a-la-salud-en-colombia | Evidencia del problema (informe 2024) |
| `doc5_mgte_fases_ambitojuridico.txt` | Minsalud fija tiempos máximos de espera... | ambitojuridico.com/.../minsalud-fija-tiempos-maximos-de-espera... | Regulatorio (fuente jurídica independiente) |

**Decisión de diseño — fuentes cacheadas localmente en vez de scraping en vivo:** en lugar de que el notebook descargue las páginas web cada vez que se ejecuta (`WebBaseLoader` en vivo), el contenido se descargó una vez y se guardó como `.txt` con un encabezado de metadata. Esto se hizo porque varios de los sitios (por ejemplo, Ámbito Jurídico) tienen paywalls parciales y bloqueos a scraping automatizado, y porque garantiza que el notebook sea **reproducible** sin depender de la disponibilidad futura de esas páginas. El precio de esta decisión es que el notebook no vuelve a consultar la fuente en vivo; se documenta explícitamente como limitación en la sección 9.

Todas las fuentes son públicas y de acceso abierto. No se usó información confidencial, personal ni empresarial restringida, en línea con las notas de seguridad del enunciado.

## 3. Arquitectura

```mermaid
flowchart LR
    A[Fuentes públicas\n(5 documentos .txt con metadata)] --> B[Carga como\nLangChain Document]
    B --> C[Chunking\nRecursiveCharacterTextSplitter\nchunk_size=1000, overlap=150]
    C --> D[Embeddings\nGemini gemini-embedding-001]
    D --> E[(Chroma\nvector store local)]
    E --> F{Retrieval\ntop_k=4}
    F --> G[Prompt con contexto\n+ pregunta del usuario]
    G --> H[Gemini Flash\n(chat / generación)]
    H --> I[Respuesta + fuentes citadas]
```

Dos arquitecturas de consumo sobre el mismo índice Chroma:
- **RAG chain de dos pasos**: siempre recupera `top_k=4` chunks y luego genera la respuesta.
- **RAG agent**: el mismo retriever expuesto como herramienta (`create_retriever_tool`); un agente ReAct (`langgraph.prebuilt.create_react_agent`) decide si necesita invocarla antes de responder.

## 4. Instalación y ejecución

```bash
git clone <url-de-tu-repositorio>
cd <repositorio>

python -m venv .venv
source .venv/bin/activate   # En Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edita .env y pega tu API key gratuita de https://aistudio.google.com/apikey

jupyter notebook notebooks/rag_application.ipynb
```

Ejecuta las celdas del notebook en orden. La primera vez que corras la sección de embeddings/Chroma se creará una carpeta `chroma_db/` (excluida del repositorio vía `.gitignore`, ya que se regenera automáticamente).

## 5. Variables de entorno requeridas

| Variable | Descripción |
|---|---|
| `GOOGLE_API_KEY` | API key gratuita de Google AI Studio (Gemini Developer API, free tier) |

## 6. Modelos Gemini utilizados

- **Chat / generación:** `gemini-2.5-flash` (Gemini Developer API, free tier).
  > ⚠️ Google retira y renueva modelos Flash del free tier con frecuencia. Antes de correr el notebook, verifica en [Google AI Studio](https://aistudio.google.com/) cuál es el modelo Flash gratuito vigente y actualiza la variable `GEMINI_CHAT_MODEL` en la sección 1 del notebook si es necesario (por ejemplo, a una versión más reciente como `gemini-3-flash` si ya está disponible en tu cuenta).
- **Embeddings:** `models/gemini-embedding-001` (fijado por el enunciado de la tarea).

## 7. Decisiones de diseño principales

- **Chunking (`chunk_size=1000`, `chunk_overlap=150`):** los documentos son artículos regulatorios/periodísticos y un resumen académico con párrafos de ~150-400 caracteres; 1000 caracteres agrupa varios párrafos relacionados sin mezclar secciones distintas, y el overlap evita cortar listas o ideas en el límite entre chunks.
- **`top_k=4`:** con solo 5 documentos fuente relativamente cortos, 4 chunks cubren la mayoría de las preguntas sin diluir el contexto con chunks poco relevantes.
- **Prompt "solo evidencia, declarar huecos":** el prompt del sistema (tanto en la cadena como en el agente) instruye explícitamente a no responder más allá del contexto recuperado y a declarar qué información falta cuando el contexto es insuficiente — un requisito crítico en un dominio de salud.
- **Cacheo local de fuentes vs. scraping en vivo:** ver sección 2.
- **Citación de fuentes:** cada chunk conserva `source_id` y `url` en su metadata, y el prompt exige listar las fuentes usadas al final de cada respuesta, para que la aplicación sea auditable (coherente con uno de los atributos de calidad centrales del proyecto de semestre: explicabilidad).

## 8. Resultados de evaluación

Se probaron 3 preguntas (ver notebook, sección 9, para el detalle completo):

| Question | Retrieved source | Grounded? | Observation |
|---|---|---|---|
| ¿Cuáles son las tres fases del MGTE y cuánto dura cada una? | doc1 / doc2 / doc5 | Sí | Dato reforzado por 3 fuentes independientes → recuperación muy robusta |
| ¿Qué tan preciso es un modelo de ML para priorizar casos de alta severidad? | doc3 | Parcial | Solo hay evidencia de un estudio (Queensland/ENT, 53.8% de acuerdo); el modelo debe aclarar que no es generalizable a Colombia |
| ¿Tiempo máximo de espera para cirugía de cadera en Colombia? | Ninguna fuente relevante | Sí (rechazo correcto) | El sistema declara explícitamente que no tiene evidencia, en vez de inventar una cifra |

*(La tabla completa con las respuestas íntegras del modelo se genera al ejecutar la sección 9 del notebook, ya que las respuestas exactas dependen de la ejecución en vivo contra la API de Gemini.)*

**Caso en que la recuperación funcionó bien:** la pregunta sobre las fases del MGTE, porque el mismo dato aparece en 3 documentos distintos con redacciones diferentes.

**Falla / limitación observada:** con chunks de 1000 caracteres, cifras que aparecen en tablas extensas (como los distintos niveles de "agreement" por método en el paper de Queensland) pueden quedar repartidas en más de un chunk y no recuperarse siempre juntas con `top_k=4`.

**Mejora posible:** chunking diferenciado para contenido tabular, y un paso de re-ranking antes de pasar los chunks al LLM.

## 9. Comparación: RAG chain vs. RAG agent

Ambas arquitecturas se ejecutaron sobre la misma pregunta (notebook, sección 8). Para este caso de uso —una base de conocimiento pequeña y casi siempre relevante para las preguntas del dominio— la **cadena de dos pasos** es la opción más simple y predecible: una sola llamada de recuperación + una de generación, sin el costo ni la latencia adicional de un ciclo de razonamiento del agente. El **agente** aporta valor sobre todo si el sistema debe atender también preguntas fuera del dominio de la base de conocimiento (donde puede decidir no recuperar), un escenario que no es el foco de esta aplicación.

## 10. Limitaciones y mejoras posibles

- La colección de solo 5 documentos es representativa pero pequeña; un sistema de producción para este proyecto necesitaría integrar directamente los anexos técnicos completos del MGTE y, eventualmente, datos reales o sintéticos de referencias (ver sección "Next Steps" del artículo del proyecto).
- Las fuentes se cachearon localmente (ver sección 2); un pipeline productivo debería incluir un mecanismo de actualización periódica y control de versiones del contenido fuente.
- No se implementó re-ranking ni evaluación cuantitativa automatizada (p. ej. RAGAS); la evaluación de grounding en este homework es manual/cualitativa, siguiendo el alcance pedido.
- El modelo de chat gratuito de Gemini cambia de nombre con frecuencia (ver sección 6); el notebook aísla esa dependencia en una sola variable para facilitar la actualización.

## Estructura del repositorio

```
repository/
├── notebooks/
│   └── rag_application.ipynb
├── data/
│   ├── doc1_circular_038_cups_tiempos.txt
│   ├── doc2_resolucion_2117_mgte.txt
│   ├── doc3_ai_referral_triage_cpc_queensland.txt
│   ├── doc4_radiografia_acceso_salud_colombia.txt
│   └── doc5_mgte_fases_ambitojuridico.txt
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
```
