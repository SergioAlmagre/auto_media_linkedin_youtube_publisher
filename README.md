# Auto Media Publisher — LinkedIn & YouTube (GUI)

> App de escritorio en **Python/Tkinter** que te lleva de una **URL** a un **post** y **vídeo** publicados con todo el flujo integrado: (1) **extrae/scrapea** contenido de una web, (2) **genera** un post de LinkedIn con **prompt** personalizable (Groq – plan gratuito), (3) **crea** una **miniatura/imagen** con prompt propio (Gemini), (4) **redacta** el texto de **narración** con Groq, (5) **compone** un **vídeo** acorde al guion con clips gratuitos de **Pexels**, subtítulos **karaoke** y assets, con opción de **TTS** natural vía **ElevenLabs**, y (6) **publica** con configuración avanzada en **LinkedIn** y **YouTube**. Puedes ejecutar **paso a paso** o **lanzar todo** con un solo botón.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Tkinter](https://img.shields.io/badge/GUI-Tkinter-informational) ![ffmpeg](https://img.shields.io/badge/dep-ffmpeg-lightgrey) ![Whisper](https://img.shields.io/badge/Whisper-faster--whisper-blueviolet) ![Groq](https://img.shields.io/badge/LLM-Groq-orange) ![Gemini](https://img.shields.io/badge/API-Gemini-black) ![Pexels](https://img.shields.io/badge/API-Pexels-brightgreen) ![ElevenLabs](https://img.shields.io/badge/API-ElevenLabs-yellow) ![DeepL](https://img.shields.io/badge/API-DeepL-0A66C2) ![YouTube Data API](https://img.shields.io/badge/API-YouTube_Data_v3-red) ![LinkedIn](https://img.shields.io/badge/API-LinkedIn_Marketing-lightgrey) ![License](https://img.shields.io/badge/License-MIT-green)


<img width="981" height="2048" alt="image" src="https://github.com/user-attachments/assets/4e1c4d30-ba4b-472a-b933-67c33875b8c8" />


---

## 🧭 Tabla de contenidos

* [Descripción](#-descripción)
* [Características](#-características)
* [Requisitos](#-requisitos)
* [Instalación](#-instalación)
* [Configuración](#-configuración)
* [Ejecución (GUI)](#-ejecución-gui)
* [Estructura de carpetas](#-estructura-de-carpetas)
* [Publicación en YouTube](#-publicación-en-youtube)
* [Publicación en LinkedIn](#-publicación-en-linkedin)
* [Generación de vídeo y subtítulos](#-generación-de-vídeo-y-subtítulos)
* [Roadmap](#-roadmap)
* [Contribuir](#-contribuir)
* [Licencia](#-licencia)

---

## 📝 Descripción

La app orquesta un flujo de trabajo para creadores:

1. **Raspa** una URL (título/descripcion/texto) para inspirar el contenido.
2. **Genera** el post de LinkedIn y un **guion** de vídeo usando modelos LLM (Groq) y crea **miniatura** con Gemini.
3. **Compone** el vídeo final (clips, música, subtítulos karaoke) con MoviePy/Whisper.
4. **Sube** automáticamente a YouTube y **publica** en LinkedIn (imagen o PDF, y comentario opcional con el enlace al vídeo).

## ✨ Características

* GUI completa en Tkinter: editor de texto extraído, post, comentario, calendario por días, vista previa de vídeo.
* **Carpetas por día** (`post/YYYY-MM-DD`) con todo el material versionado.
* **YouTube upload** con título/descripcion/hashtags y miniatura.
* **LinkedIn post** con imagen o PDF + comentario secundario.
* **Whisper (faster‑whisper)** para transcripción y subtítulos **karaoke** sincronizados.
* **Pexels** para clips de stock; **ElevenLabs** para TTS opcional.
* **Gemini** para miniaturas 16:9 con modo blur o cover.

## ✅ Requisitos

* **Python 3.10+**
* **ffmpeg** instalado en el sistema (necesario para audio/vídeo)
* Credenciales:

  * **YouTube Data API v3** (OAuth — `config/client_secret.json` → genera `config/token.json` en primer login).
  * **LinkedIn**: token Bearer y tu **person URN** u **organization URN**.
* Librerías: ver `requirements.txt`.

## ⚙️ Instalación

```bash
# 1) Clonar
git clone https://github.com/SergioAlmagre/auto_media_linkedin_youtube_publisher.git
cd auto_media_linkedin_youtube_publisher

# 2) Entorno
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3) Dependencias
pip install -r requirements.txt

# 4) Sistema
ffmpeg -version   # comprueba que esté instalado
```

## 🔧 Configuración

Crea `config/global_config.json` con tus claves y ajustes (la UI también lo edita). Ejemplo real reducido y campos clave:

**Campos principales**

* `selected_day`: día activo en la UI (YYYY-MM-DD).
* `linkedin`: visibilidad, token Bearer, texto fijo que se añade al post, comentario opcional y bandera para añadir el link de YouTube en el comentario.
* `youtube`: visibilidad por defecto, playlist, texto fijo que se concatena en la descripción, y `url` final del vídeo (rellenada tras publicar).
* `groq`: modelo y API key para generar post y guion de vídeo.
* `pexels`, `deepl`, `elevenlabs`, `gemini`: API keys y opciones de cada servicio.
* `schedule`: programar ejecución automática en una fecha/hora.
* `automated`: activar pasos de la **secuencia automática** (extraer → generar post → imagen → publicar, etc.).

**Ejemplo**

```json
{
  "selected_day": "2025-10-04",
  "linkedin": {
    "visibility": "PUBLIC",
    "api_key": "<LINKEDIN_BEARER_TOKEN>",
    "fixed_description": "…",
    "comment": "Aquí puedes ver el resumen en video generado automáticamente!",
    "youtube_in_comment": "0",
    "post_urn": "urn:li:share:...",
    "prompt": "(prompt para redactar post)"
  },
  "youtube": {
    "visibility": "UNLISTED",
    "playlist": "DevsNews",
    "fixed_description": "…",
    "url": "",
    "narration": "(prompt para guion del vídeo)"
  },
  "groq": { "api_key": "…", "model_post": "…", "model_video": "…" },
  "deepl": { "api_key": "" },
  "pexels": { "api_key": "" },
  "elevenlabs": {
    "api_key_1": "",
    "voice_id": "iXa2i9eYvgmNMRQRCwqO",
    "model_id": "eleven_multilingual_v2"
  },
  "gemini": {
    "api_key": "",
    "url_model": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image-preview:generateContent",
    "imagen_cut": "0",
    "prompt": "(brief creativo para miniatura)"
  },
  "schedule": { "is_scheduled": "1", "date": "2025-09-27 11:00" },
  "automated": {
    "extraer_contenido": "0",
    "generar_post_linkedin": "0",
    "generar_imagen": "0",
    "publicar_post_linkedin": "0",
    "publicar_comentario": "0",
    "generar_narracion": "0",
    "generar_video": "0",
    "publicar_video_youtube": "0"
  }
}
```

> **Credenciales de Google**: coloca `config/client_secret.json` (OAuth). Tras el primer login se generará `config/token.json`. **No subas** estos archivos al repo.

## ▶️ Ejecución

```bash
python main_app.py
```

Interfaz principal (resumen):

* **URL** → botón **Extraer**: trae texto base del enlace.
* **Miniatura**: generación con Gemini (cover o blur) y vista previa.
* **Post LinkedIn** y **Comentario**: edición en vivo y guardado por día.
* **Narración y Vídeo**: prompts y generación con subtítulos karaoke.
* **Panel derecho**: visibilidad, playlist, checkboxes para publicar y **secuencia automática**.
* **Calendario**: crea/abre carpeta del día en `post/`.

> **Tip**: marca pasos en *Automatismos* y usa **RUN!** para lanzar toda la secuencia.

## 🗂️ Estructura de carpetas

```
assets/
  audio_news.mp3       # música o locución fija (opcional)
  final_logo.png       # marca de agua / logo final (opcional)
config/
  client_secret.json   # OAuth Google (local)
  token.json           # se crea tras el login OAuth
  global_config.json   # configuración global editada por la GUI
post/
  YYYY-MM-DD/
    video_text.txt
    linkedin_post.txt
    miniatura.jpg
    video_post.mp4
    config.json
scripts/
  main_app.py
  video_generator.py
  youtube_publisher.py
  linkedin_publisher.py
  linkedin_comment_publisher.py
  state_store.py
  gemini_image.py
requirements.txt
```

**Capturas** (recomendado)
Guarda tus screenshots en `docs/` y enlázalos:

```md
![UI principal](docs/ui_main.png)
![Estructura de carpetas](docs/tree.png)
```

## 📺 Publicación en YouTube

* Título = **primera línea** de `linkedin_post.txt`.
* Descripción = resto del archivo + `youtube.fixed_description`.
* **Hashtags**: extraídos del propio texto del post.
* **Miniatura**: `post/<día>/miniatura.jpg`.
* **Visibilidad**: `PUBLIC` | `UNLISTED` | `PRIVATE`.
* **Playlist**: define el nombre en `youtube.playlist`.

> Tras publicar, la app guarda la **URL del vídeo** en `youtube.url`.

## 💼 Publicación en LinkedIn

* Post con imagen **o** PDF desde la carpeta del día.
* Añade `linkedin.fixed_description` al final del texto.
* Comentario adicional opcional (`linkedin.comment`).
* Si `linkedin.youtube_in_comment` = `1`, el comentario incluirá también `youtube.url`.
* Publica como **persona** por defecto; para organizaciones, usa tu **Organization URN**.

## 🎬 Generación de vídeo y subtítulos

* **faster‑whisper** para transcripción con timestamps por palabra.
* **Subtítulos karaoke** renderizados en el vídeo (resaltado progresivo).
* **MoviePy** para composición de clips, música y texto.
* **Pexels** para clips de stock (si configuras API key).
* **ElevenLabs** (opcional) para narración TTS.


## 🧰 Solución de problemas

* **ffmpeg no encontrado**: instala ffmpeg y asegúrate de que está en el `PATH`.
* **OAuth de YouTube falla**: borra `config/token.json` y vuelve a iniciar sesión.
* **URN de LinkedIn inválido**: revisa que el URN sea tuyo (persona u organización) y que el token tenga permisos de `ugcPosts`.
* **Fuentes/acentos** en miniaturas: el prompt de `gemini` exige tildes correctas; si ves errores, regenera.

## 🗺️ Roadmap

* [ ] De momento se dejo del desarrollo aquí salvo que haya real interés en continuar con el desarrollo, si es así hazlo saber en linkedin!


## 🤝 Contribuir

1. Haz un fork
2. Crea rama `feat/mi-cambio`
3. Abre PR con descripción

## 📄 Licencia

MIT
