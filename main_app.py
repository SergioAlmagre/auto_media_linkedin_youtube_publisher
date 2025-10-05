import os
import sys
import subprocess
import threading
import requests
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import date, datetime
from PIL import Image, ImageTk, ImageOps, ImageDraw
from tkcalendar import Calendar
from scripts.state_store import get_nested, set_nested, GLOBAL_CONFIG, set_current_day
import shutil
from ffpyplayer.player import MediaPlayer
from tkinter import filedialog
sys.stdout.reconfigure(encoding='utf-8')


# --- Configuración ---
GROQ_API_KEY = get_nested("groq.api_key")
TODAY = date.today().strftime("%Y-%m-%d")
set_nested("selected_day",TODAY)

BASE_DIR = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "config"
POST_DIR = BASE_DIR / "post" / TODAY
POST_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)
SCRIPTS_DIR = BASE_DIR / "scripts"

# --- Scraper básico ---
def scrape_page(url: str):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    title = soup.title.string.strip() if soup.title else ""
    desc_tag = soup.find("meta", {"name": "description"})
    meta_desc = desc_tag["content"].strip() if desc_tag and desc_tag.get("content") else ""
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    body_text = "\n".join(paragraphs)

    return {"title": title, "description": meta_desc, "text": body_text}


# --- Groq helper ---
def call_groq(prompt: str, model: str, max_words: int = 500) -> str:
    if not GROQ_API_KEY:
        return "⚠ No hay GROQ_API_KEY configurada en .env"

    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Eres un redactor experto en tecnología y desarrollo de software. "
                    "Escribes en un estilo natural, conversacional y crítico, como si compartieras "
                    "tu opinión con colegas en LinkedIn. Tu tono es profesional pero cercano. "
                    "La extensión debe estar entre 250 y 500 palabras. "
                    "Evita sonar a marketing o nota de prensa."
                )
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.8,   # más creatividad y naturalidad
        "top_p": 1.0,         # no recorta la variedad
        "frequency_penalty": 0.3,  # evita repetir
        "presence_penalty": 0.2,   # fomenta nuevas ideas
        "max_tokens": int(max_words * 4),
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    r = requests.post(url, headers=headers, json=payload, timeout=45)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def create_folder(app):
    """Crea carpeta según el día seleccionado en el calendario"""
    s = app.sidebar_cal.get_date()  # 'yyyy-mm-dd'
    app.selected_day = s            # <-- día activo en memoria
    app.set_post_dir(s)
    set_nested("selected_day", s, path=GLOBAL_CONFIG)  # 🔑 guardar en global_config
    app.append_console(f"📁 Día Seleccionado: {s}\n")    
    

def delete_selected_day(app):
    """Elimina la carpeta del día seleccionado y todo su contenido."""
    day = getattr(app, "selected_day", None)
    if not day:
        messagebox.showwarning("Sin selección", "No hay ningún día seleccionado.")
        return

    folder = BASE_DIR / "post" / day

    if not folder.exists():
        messagebox.showinfo("No existe", f"La carpeta {folder} no existe.")
        return

    # Confirmación
    if not messagebox.askyesno("Confirmar", f"¿Eliminar la carpeta {folder} y todo su contenido?"):
        return

    try:
        shutil.rmtree(folder)  # 🔥 borra carpeta y contenido
        app.append_console(f"🗑 Carpeta eliminada: {folder}\n")
        messagebox.showinfo("Eliminada", f"Se eliminó la carpeta {folder}")
        app.mark_days_with_posts()  # refrescar calendario
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo eliminar {folder}:\n{e}")
        app.append_console(f"❌ Error al eliminar {folder}: {e}\n")
    
    
def copy_state_to_folder(app):
    create_folder(app)
    """Hace una copia completa de global_config.json dentro de la carpeta seleccionada,
    pero solo si no existe aún config.json en la carpeta del día."""
    
    src = CONFIG_DIR / "global_config.json"
    day = get_nested("selected_day", path=GLOBAL_CONFIG)

    if not day:
        raise ValueError("❌ No hay ningún 'selected_day' definido en la configuración global")

    dst = BASE_DIR / "post" / day / "config.json"

    if not src.exists():
        raise FileNotFoundError(f"No se encontró {src}")

    # 👇 Solo copiar si aún no existe
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"✅ Config copiado en {dst}")
    else:
        print(f"ℹ️ Config ya existe, no se sobrescribe: {dst}")



# --- GUI ---
class ScraperApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AutoMedia Post")
        self.geometry("1200x1050")

        self.url_var = tk.StringVar()
        self.running = False
        self.current_proc: subprocess.Popen | None = None

        # Estado/labels prontos para evitar errores al usarlos pronto
        self.status_var = tk.StringVar(self, value="Listo.")
        self.day_var = tk.StringVar(self, value=TODAY)
        
        self.withdraw()
        self._build_ui()
        self.set_post_dir(TODAY)
        self.deiconify()
    
    def build_config_section(self, parent, title: str, section_key: str):
        sec = CollapsibleSection(parent, title=title, start_open=False, expand_when_open=False)
        sec.pack(fill="x", padx=6, pady=6)

        body = sec.body
        state = get_nested(section_key) or {}

        self.config_vars = getattr(self, "config_vars", {})

        for key, val in state.items():
            row = ttk.Frame(body)
            row.pack(fill="x", padx=4, pady=2)

            ttk.Label(row, text=key, width=18, anchor="w").pack(side="left")

            var = tk.StringVar(value=val)
            ent = ttk.Entry(row, textvariable=var, show="*" if "key" in key.lower() else "")
            ent.pack(side="left", fill="x", expand=True, padx=4)

            # guardar variable vinculada para este campo
            self.config_vars[f"{section_key}.{key}"] = var

        # Botón guardar cambios de esta sección
        def save_section():
            for full_key, var in self.config_vars.items():
                if full_key.startswith(section_key):
                    set_nested(full_key, var.get())
            messagebox.showinfo("Guardado", f"Valores de '{title}' actualizados en global_config.json")

        ttk.Button(body, text="💾 Guardar", command=save_section).pack(pady=4, anchor="e")

        return sec
    
    
    def on_video_key(self, event):
        video_text = self.text_video.get("1.0", tk.END).strip()
        (POST_DIR / "video_text.txt").write_text(video_text, encoding="utf-8")


    def on_linkedin_key(self, event):
        text_video = self.text_post.get("1.0", tk.END).strip()
        (POST_DIR / "linkedin_post.txt").write_text(text_video, encoding="utf-8")


    def _build_ui(self):
        def add_audio(self):
            """Selecciona un archivo MP3 y lo guarda como audio_news.mp3 en assets/"""
            path = filedialog.askopenfilename(
                title="Seleccionar Banda Sonora",
                filetypes=[("Archivos MP3", "*.mp3")]
            )
            if not path:
                return
            try:
                dest = ASSETS_DIR / "audio_news.mp3"
                shutil.copy(path, dest)
                self.append_console(f"🎵 Banda sonora guardada en {dest}\n")
                messagebox.showinfo("Éxito", f"Banda sonora añadida:\n{dest}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo copiar el archivo:\n{e}")

        def add_logo(self):
            """Selecciona un JPG/PNG y lo guarda como final_logo.png en assets/"""
            path = filedialog.askopenfilename(
                title="Seleccionar Logo Final",
                filetypes=[("Imágenes", "*.jpg;*.jpeg;*.png")]
            )
            if not path:
                return
            try:
                ext = Path(path).suffix.lower()
                dest = ASSETS_DIR / "final_logo.png"

                if ext == ".png":
                    shutil.copy(path, dest)
                else:
                    # si es JPG, lo convertimos a PNG
                    from PIL import Image
                    img = Image.open(path)
                    img.save(dest, format="PNG")

                self.append_console(f"🖼 Logo final guardado en {dest}\n")
                messagebox.showinfo("Éxito", f"Logo añadido:\n{dest}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo copiar el archivo:\n{e}")
        # ---------- Contenedor raíz ----------
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)

        # ---------- Columna izquierda con scroll ----------
        left_wrapper = ttk.Frame(root)
        left_wrapper.pack(side="left", fill="y", padx=12, pady=8)

        LEFT_MAX_W = 800

        # Canvas + scrollbar
        canvas = tk.Canvas(left_wrapper, width=LEFT_MAX_W, highlightthickness=0)
        canvas.pack(side="left", fill="y", expand=True)

        scrollbar = ttk.Scrollbar(left_wrapper, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")

        canvas.configure(yscrollcommand=scrollbar.set)

        # Frame interno
        left_inner = ttk.Frame(canvas, width=LEFT_MAX_W)
        canvas.create_window((0, 0), window=left_inner, anchor="nw")

        # Actualizar scrollregion cuando cambie el contenido
        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        left_inner.bind("<Configure>", on_frame_configure)

        # Scroll con la rueda del ratón
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)


        def _on_mousewheel(event):
            # Windows y MacOS
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # Linux (X11)
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        # Solo activar cuando el ratón está sobre el canvas
        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _bind_mousewheel)
        canvas.bind("<Leave>", _unbind_mousewheel)

        # ---------- Sidebar con scroll ----------
        sidebar_wrapper = ttk.Frame(root, width=280)
        sidebar_wrapper.pack(side="left", fill="y", padx=10, pady=10)

        canvas_sidebar = tk.Canvas(sidebar_wrapper, width=280, height=400, highlightthickness=0)
        canvas_sidebar.pack(side="left", fill="both", expand=True)

        scrollbar_sidebar = ttk.Scrollbar(sidebar_wrapper, orient="vertical", command=canvas_sidebar.yview)
        scrollbar_sidebar.pack(side="right", fill="y")

        canvas_sidebar.configure(yscrollcommand=scrollbar_sidebar.set)

        # Frame interno del sidebar
        sidebar = ttk.Frame(canvas_sidebar, width=300)
        canvas_sidebar.create_window((0, 0), window=sidebar, anchor="nw")

        # Ajustar scrollregion cuando cambie el contenido
        def on_sidebar_configure(event):
            canvas_sidebar.configure(scrollregion=canvas_sidebar.bbox("all"))

        sidebar.bind("<Configure>", on_sidebar_configure)

        # Scroll con la rueda del ratón (solo cuando el puntero esté sobre el sidebar)
        def _on_mousewheel_sidebar(event):
            canvas_sidebar.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas_sidebar.bind("<Enter>", lambda e: canvas_sidebar.bind_all("<MouseWheel>", _on_mousewheel_sidebar))
        canvas_sidebar.bind("<Leave>", lambda e: canvas_sidebar.unbind_all("<MouseWheel>"))


        # ---------- Fila URL ----------
        url_row = ttk.Frame(left_inner)
        url_row.pack(fill="x", padx=8, pady=6)
        ttk.Label(url_row, text="URL:").pack(side="left")
        self.entry_url = ttk.Entry(url_row, textvariable=self.url_var)
        self.entry_url.pack(side="left", fill="x", expand=True, padx=4)
        self.btn_extract = ttk.Button(url_row, text="Extraer", command=self.on_extract)
        self.btn_extract.pack(side="left")

        # ---------- Secciones (izquierda) ----------
        # ====== Texto extraído ======
        self.sec_extracted = CollapsibleSection(
            left_inner, title="Texto extraído", start_open=True, expand_when_open=True
        )
        self.sec_extracted.pack(fill="x", padx=6, pady=6)
        self.text_extracted = tk.Text(self.sec_extracted.body, wrap="word", height=12)
        scroll_extracted = ttk.Scrollbar(self.sec_extracted.body, orient="vertical",
                                        command=self.text_extracted.yview)
        self.text_extracted.configure(yscrollcommand=scroll_extracted.set)
        scroll_extracted.pack(side="right", fill="y")
        self.text_extracted.pack(side="left", fill="both", expand=True, padx=6, pady=4)


        # Miniatura (calcula tamaño según ancho disponible)
        self.thumb_w = 750   # ancho fijo
        self.thumb_h = int(self.thumb_w * 9 / 16)

        self.thumb_section = CollapsibleSection(
            left_inner, title="Miniatura", start_open=True, expand_when_open=False
        )
        self.thumb_section.pack(fill="x", padx=6, pady=6)
        self.thumb_label = ttk.Label(self.thumb_section.body)
        self.thumb_label.pack(padx=6, pady=6)
        self._thumb_imgtk = None
        self._update_thumbnail_preview(None)

        # sección
        self.sec_post = CollapsibleSection(
            left_inner, title="Post LinkedIn", start_open=True, expand_when_open=True
        )
        self.sec_post.pack(fill="x", padx=6, pady=6)
        self.text_post = tk.Text(self.sec_post.body, wrap="word", height=10, bg="#f7f7f7")
        scroll_y = ttk.Scrollbar(self.sec_post.body, orient="vertical", command=self.text_post.yview)
        self.text_post.configure(yscrollcommand=scroll_y.set)
        scroll_y.pack(side="right", fill="y")
        self.text_post.pack(side="left", fill="both", expand=True, padx=6, pady=4)
        
        
        # Comentario LinkedIn
        self.sec_comment = CollapsibleSection(
            left_inner, title="Comentario Linkedin", start_open=True, expand_when_open=True
        )
        self.sec_comment.pack(fill="x", padx=6, pady=6)

        self.text_comment = tk.Text(self.sec_comment.body, wrap="word", height=5, bg="#f7f7f7")
        scroll_y = ttk.Scrollbar(self.sec_comment.body, orient="vertical", command=self.text_comment.yview)
        self.text_comment.configure(yscrollcommand=scroll_y.set)
        scroll_y.pack(side="right", fill="y")
        self.text_comment.pack(side="left", fill="both", expand=True, padx=6, pady=4)
        
        # Carga inicial
        init_comment = get_nested("linkedin.comment")
        self.text_comment.insert("1.0", str(init_comment or ""))
        self.text_comment.edit_modified(False)  # muy importante: resetea el flag

        # Debounce
        self._comment_save_job = None

        def _save_comment_now():
            txt = self.text_comment.get("1.0", "end-1c").strip()
            set_nested("linkedin.comment", txt)

        def _schedule_comment_save(_evt=None):
            # limpia el flag para que <<Modified>> vuelva a dispararse en el siguiente cambio
            try:
                if self.text_comment.edit_modified():
                    self.text_comment.edit_modified(False)
            except Exception:
                pass
            if self._comment_save_job:
                self.after_cancel(self._comment_save_job)
            self._comment_save_job = self.after(400, _save_comment_now)

        # Dispara al escribir y al salir del widget
        self.text_comment.bind("<<Modified>>", _schedule_comment_save)
        self.text_comment.bind("<KeyRelease>", _schedule_comment_save)  # respaldo por si <<Modified>> no se emite en tu build
        self.text_comment.bind("<FocusOut>", lambda e: _save_comment_now())
                


        # ==========================
        #  PREVISUALIZACIÓN CON AUDIO
        # ==========================
        
        # ---- estado global de la vista de vídeo ----
        self._ff_player = None
        self._ff_imgtk = None
        self._ff_running = False
        self._ff_last_ts = 0.0       # último timestamp de frame (segundos)
        self._ff_duration = 0.0      # duración en segundos (si está disponible)
        self._ff_poll_job = None     # after() id del poll de progreso
        
        
        self.video_section = CollapsibleSection(
            left_inner, title="Previsualización Video", start_open=True, expand_when_open=False
        )
        self.video_section.pack(fill="x", padx=6, pady=6)

        # Cajón fijo (mismo tamaño que miniatura)
        self.video_frame = ttk.Frame(self.video_section.body, width=self.thumb_w, height=self.thumb_h)
        self.video_frame.pack(padx=6, pady=6)
        self.video_frame.pack_propagate(False)

        # Label donde pintamos los frames
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill="both", expand=True)
        
            
        # Estado ffpyplayer
        self._ff_player = None
        self._ff_imgtk = None
        self._ff_running = False
        
        
                
        def delete_empty_post_dirs():
            """Elimina la carpeta del día seleccionado (y todo su contenido)."""
            day = self.sidebar_cal.get_date()
            if not day:
                messagebox.showwarning("Sin selección", "No hay ningún día seleccionado.")
                return

            folder = BASE_DIR / "post" / day
            if not folder.exists():
                messagebox.showinfo("No existe", f"La carpeta {folder} no existe.")
                return

            # Confirmación
            if not messagebox.askyesno("Confirmar", f"¿Eliminar la carpeta {folder} y todo su contenido?"):
                return

            try:
                shutil.rmtree(folder)  # 🔥 borra carpeta y todo dentro
                self.append_console(f"🗑 Carpeta eliminada: {folder}\n")
                messagebox.showinfo("Eliminada", f"Se eliminó la carpeta {folder}")
                self.mark_days_with_posts()  # refrescar calendario
            except Exception as e:
                self.append_console(f"❌ Error al eliminar {folder}: {e}\n")
                messagebox.showerror("Error", f"No se pudo eliminar {folder}:\n{e}")
                
                

        def _ff_stop():
            """Detiene y libera el reproductor."""
            self._ff_running = False
            if self._ff_player:
                try:
                    self._ff_player.close_player()
                except Exception:
                    pass
                self._ff_player = None

        def _ff_loop():
            if not self._ff_running or not self._ff_player:
                return

            frame, val = self._ff_player.get_frame()
            if val == 'eof':
                _ff_stop()
                return

            if frame is not None:
                img, _ts = frame  # img es el frame de vídeo de ffpyplayer

                try:
                    # ✅ conversión directa a PIL.Image (RGB)
                    pil = img.to_image()
                except Exception:
                    # Fallback por si alguna versión no trae to_image()
                    w, h = img.get_size()
                    buf = img.to_bytearray()[0]  # bytes/bytearray
                    pil = Image.frombytes('RGB', (w, h), bytes(buf))

                # Redimensiona al tamaño de tu miniatura
                pil = pil.resize((self.thumb_w, self.thumb_h), Image.BILINEAR)

                # Mantén una referencia para evitar GC
                self._ff_imgtk = ImageTk.PhotoImage(pil)
                self.video_label.configure(image=self._ff_imgtk, text="")

            # Programa el siguiente ciclo (~30fps). Si quieres más fino, baja a 16ms.
            self.video_label.after(33, _ff_loop)

        def _ff_play(folder: str):
            """Carga y reproduce video_post.mp4 desde la carpeta (con audio)."""
            path = Path(folder) / "video_post.mp4"
            if not path.exists():
                _ff_stop()
                self.video_label.configure(text="video_post.mp4 no encontrado")
                return

            _ff_stop()
            self.video_label.configure(text="")

            # out_fmt=rgb24 para recibir frames en RGB; sync=audio para sincronizar al audio
            self._ff_player = MediaPlayer(
                str(path),
                ff_opts={
                    "out_fmt": "rgb24",
                    "paused": False,
                    "sync": "audio"
                }
            )
            self._ff_running = True
            _ff_loop()

        def _ff_pause():
            if self._ff_player:
                self._ff_player.set_pause(True)

        def _ff_resume():
            if self._ff_player:
                self._ff_player.set_pause(False)
                # reanudar el loop de frames si se paró
                if not self._ff_running:
                    self._ff_running = True
                    _ff_loop()
                # asegurar polling de progreso activo
                _ensure_progress_poll()
                
                
        def _ensure_progress_poll():
            def _poll():
                if not self._ff_player:
                    return
                # obtener posición actual
                pos = None
                try:
                    # algunos builds exponen get_pts() con la posición en segundos
                    pos = float(self._ff_player.get_pts() or 0.0)
                except Exception:
                    pass
                if pos is None or pos <= 0:
                    # fallback: usa timestamp del último frame mostrado
                    pos = float(self._ff_last_ts)

                # actualizar slider sin disparar seek
                try:
                    self.video_progress.configure(to=self._ff_duration or max(self.video_progress.cget("to"), pos))
                    self.video_progress.set(pos)
                except Exception:
                    pass

                # reprograma
                self._ff_poll_job = self.video_label.after(200, _poll)

            # iniciar si no está ya
            if not self._ff_poll_job:
                _poll()
                


        # Controles
        controls = ttk.Frame(self.video_section.body)
        controls.pack(fill="x", padx=6, pady=(0,6))    
        # progreso (0 → duración). Se completa cuando conocemos la duración
        self.video_progress = ttk.Scale(controls, from_=0, to=0, orient="horizontal", length=260)
        self.video_progress.pack(side="right")
        
        ttk.Button(controls, text="▶ Reproducir", command=lambda: _ff_play(POST_DIR)).pack(side="left")
        ttk.Button(controls, text="⏸ Pausa",      command=_ff_pause).pack(side="left", padx=(6,0))
        ttk.Button(controls, text="⏯ Reanudar",   command=_ff_resume).pack(side="left", padx=(6,0))
        ttk.Button(controls, text="⏹ Detener",    command=_ff_stop).pack(side="left", padx=(6,0))   
        
        def _on_seek(_evt=None):
            """Ir a la posición elegida en el slider."""
            if not self._ff_player:
                return
            try:
                self._ff_player.seek(float(self.video_progress.get()))
            except Exception:
                pass
            
        self.video_progress.bind("<ButtonRelease-1>", _on_seek)              
                        
        # ====== Narración Video ======
        self.sec_video = CollapsibleSection(
            left_inner, title="Narración Video", start_open=True, expand_when_open=True
        )
        self.sec_video.pack(fill="x", padx=6, pady=6)
        self.text_video = tk.Text(self.sec_video.body, wrap="word", height=8, bg="#f7f7f7")
        scroll_video = ttk.Scrollbar(self.sec_video.body, orient="vertical",
                                    command=self.text_video.yview)
        self.text_video.configure(yscrollcommand=scroll_video.set)
        
        scroll_video.pack(side="right", fill="y")
        self.text_video.pack(side="left", fill="both", expand=True, padx=6, pady=4)
        self.text_video.bind("<KeyRelease>", self.on_video_key)
        self.text_post.bind("<KeyRelease>", self.on_linkedin_key)
        
        self.sec_console = CollapsibleSection(
            left_inner, title="Consola", start_open=False, expand_when_open=True
        )
        self.sec_console.pack(fill="x", padx=6, pady=6)
        self.console = ScrolledText(
            self.sec_console.body, wrap="word", height=16, state="normal",
            bg="#0d1117", fg="#e6edf3"
        )
        self.console.pack(fill="both", expand=True, padx=6, pady=(2, 6))
        self.console.pack_propagate(True)
        self.console.insert("end", "Listo.\n")

        # ---------- Sidebar: botones ----------
        style = ttk.Style(self)
        style.configure("GreenBold.TLabel", foreground="#ff569d", font=("Segoe UI", 10, "bold"))
        style.configure("Side.TButton", anchor="w")

        def sidebtn(text, cmd):
            b = ttk.Button(sidebar, text=text, command=cmd, style="Side.TButton")
            b.pack(fill="x", pady=4)
            return b

        # Panel de configuración
        self.sec_config = CollapsibleSection(left_inner, title="Configuración de APIs", start_open=False, expand_when_open=False)
        self.sec_config.pack(fill="x", padx=6, pady=6)

        # Añadir subsecciones por servicio
        self.build_config_section(self.sec_config.body, "🔑 LinkedIn", "linkedin")
        self.build_config_section(self.sec_config.body, "🤖 Groq", "groq")
        self.build_config_section(self.sec_config.body, "🌍 DeepL", "deepl")
        self.build_config_section(self.sec_config.body, "📸 Pexels", "pexels")
        self.build_config_section(self.sec_config.body, "🗣 ElevenLabs", "elevenlabs")
        self.build_config_section(self.sec_config.body, "✨ Gemini", "gemini")
        
        self.btn_clean_empty = sidebtn("🗑️ Borrar carpeta del día", delete_empty_post_dirs)

        # --- Calendario fijo en el sidebar ---
        ttk.Label(sidebar, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=6, pady=(12, 2))

        self.sidebar_cal = Calendar(
            sidebar,
            selectmode="day",
            date_pattern="yyyy-mm-dd"
        )
        self.sidebar_cal.pack(fill="x", padx=6, pady=6)
        self.mark_days_with_posts()
            

        def on_day_selected(event):
            s = self.sidebar_cal.get_date()  # 'yyyy-mm-dd'
            set_current_day(s)
            set_nested("selected_day", s, path=GLOBAL_CONFIG)  # lo sigues guardando globalmente
            self.set_post_dir(s)

            visibility_var.set(get_nested("linkedin.visibility"))
            is_url_in_comment.set(get_nested("linkedin.youtube_in_comment"))
            imagen_cut_var.set(get_nested("gemini.imagen_cut"))

            yt_vis_var.set(get_nested("youtube.visibility"))
            #yt_playlist_var.set(get_nested("youtube.playlist"))
            self.mark_days_with_posts()


        def open_selected_folder():
            """Abre en el explorador la carpeta del día seleccionado (POST_DIR)."""
            try:
                folder = POST_DIR
                if not folder.exists():
                    messagebox.showwarning("Carpeta no encontrada", f"No existe:\n{folder}")
                    return

                if sys.platform == "win32":
                    os.startfile(folder)  # Windows
                elif sys.platform == "darwin":
                    subprocess.run(["open", folder])  # macOS

                self.append_console(f"📂 Explorador abierto en {folder}\n")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo abrir la carpeta:\n{e}")
                self.append_console(f"❌ Error al abrir {folder}: {e}\n")

            # Estilo del tag para el día seleccionado
            self.sidebar_cal.tag_config("selected_day", background="lightblue", foreground="black")

        def highlight_selected_day(event):
            # Borrar marcas previas
            self.sidebar_cal.calevent_remove(tag="selected_day")

            # Obtener día actual
            s = self.sidebar_cal.get_date()  # 'yyyy-mm-dd'
            try:
                dt = datetime.strptime(s, "%Y-%m-%d").date()
            except ValueError:
                return

            # Crear evento con el tag azul
            self.sidebar_cal.calevent_create(dt, "día seleccionado", "selected_day")

        # Enlazar al evento de selección
        self.sidebar_cal.bind("<<CalendarSelected>>", highlight_selected_day)


        # 🔗 enlazamos el evento
        self.sidebar_cal.bind("<<CalendarSelected>>", on_day_selected)
        self.sidebar_cal.bind("<<CalendarSelected>>", highlight_selected_day, add="+")

        
        self.btn_open_folder = sidebtn("📂 Abrir carpeta día seleccionado", open_selected_folder)        
        self.btn_save_texts  = sidebtn("💾 Guardar", self.save_texts)
        
        ttk.Label(sidebar, text="Linkedin", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=6, pady=(12, 2))
        self.btn_prompt_li  = sidebtn("⚙️ Prompt Linkedin", self.edit_linkedin_prompt)
        self.btn_prompt_li  = sidebtn("✍️ Descripción fijada final", self.edit_fixed_description_linkedin)
        self.btn_create_post = sidebtn("✅ Generar Post", self.on_generate_linkedin)

        # Checkbox de visibilidad LinkedIn
        visibility_var = tk.StringVar(value=get_nested("linkedin.visibility"))
        is_url_in_comment = tk.StringVar(value=get_nested("linkedin.youtube_in_comment"))
        
        
        def on_change_visibility():
            copy_state_to_folder(app)
            val = visibility_var.get()
            set_nested("linkedin.visibility", val)
            self.append_console(f"🌍 Visibilidad LinkedIn cambiada a {val}\n")
            
        def on_change_url_youtube_in_comment():
            copy_state_to_folder(app)
            val = is_url_in_comment.get()
            set_nested("linkedin.youtube_in_comment", val)
            self.append_console(f"URL de youtube añadida al comentario {val}\n")

        def on_change_imagen_cut():
            copy_state_to_folder(app)
            val = imagen_cut_var.get()
            set_nested("gemini.imagen_cut", val)
            self.append_console(f"🖼️ ¿Imagen cortada? cambiada a {val}\n")
            

                
   
            

        # Checkbox para la adaptación de la imagen
        imagen_cut_var = tk.StringVar(value=get_nested("gemini.imagen_cut"))
        img_cut = ttk.Frame(sidebar)
        img_cut.pack(fill="x", pady=(4, 10))
        ttk.Label(img_cut, text="Adaptación imagen:").pack(anchor="w")
        ttk.Radiobutton(
            img_cut, text="Recortada", variable=imagen_cut_var, value="1",
            command=on_change_imagen_cut
        ).pack(anchor="w")
        ttk.Radiobutton(
            img_cut, text="Blur backgroud", variable=imagen_cut_var, value="0",
            command=on_change_imagen_cut
        ).pack(anchor="w")
        
        
        self.btn_prompt_img = sidebtn("⚙️ Prompt Image", self.edit_image_prompt)
        self.btn_create_img  = sidebtn("✅ Generar Imagen", self.on_run_gemini_image)
        
        
        
        frm_vis = ttk.Frame(sidebar)
        frm_vis.pack(fill="x", pady=(4, 10))
        ttk.Label(frm_vis, text="Visibilidad Linkedin:").pack(anchor="w")
        ttk.Radiobutton(
            frm_vis, text="Público", variable=visibility_var, value="PUBLIC",
            command=on_change_visibility
        ).pack(anchor="w")

        ttk.Radiobutton(
            frm_vis, text="Solo Conexiones", variable=visibility_var, value="CONNECTIONS",
            command=on_change_visibility
        ).pack(anchor="w")
        
        
        self.btn_pub_li = sidebtn("🚀 Publicar Post Linkedin", self.on_publish_linkedin)
        
        
        ivc_vis = ttk.Frame(sidebar)
        ivc_vis.pack(fill="x", pady=(4, 10))
        ttk.Label(ivc_vis, text="¿Incluir video youtube en comentario Linkedin?:").pack(anchor="w")
        ttk.Radiobutton(
            ivc_vis, text="Si", variable=is_url_in_comment, value="1",
            command=on_change_url_youtube_in_comment
        ).pack(anchor="w")

        ttk.Radiobutton(
            ivc_vis, text="No", variable=is_url_in_comment, value="0",
            command=on_change_url_youtube_in_comment
        ).pack(anchor="w")
        
        self.btn_pub_li      = sidebtn("🚀 Publicar Comentario", self.on_publish_linkedin_comment)
        
        ttk.Label(sidebar, text="Youtube", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=6, pady=(12, 2))
        self.btn_prompt_vid = sidebtn("⚙️ Prompt Narración", self.edit_video_prompt)
        self.btn_create_narr = sidebtn("✅ Generar Narración", self.on_generate_video)
        self.btn_add_logo = sidebtn("🖼 Añadir Logo final", lambda: add_logo(self))
        self.btn_prompt_li  = sidebtn("✍️ Descripción fijada final", self.edit_fixed_description_youtube)
        self.btn_add_audio = sidebtn("🎵 Añadir Banda sonora", lambda: add_audio(self))
        self.btn_make_video  = sidebtn("✅ Generar Video", self.on_make_video)
        
        # YouTube config
        yt_vis_var = tk.StringVar(value=get_nested("youtube.visibility", "PRIVATE"))
        yt_playlist_var = tk.StringVar(value=get_nested("youtube.playlist", ""))

        def on_change_yt_visibility():
            val = yt_vis_var.get()
            set_nested("youtube.visibility", val)
            self.append_console(f"📺 Visibilidad YouTube cambiada a {val}\n")

        def on_change_yt_playlist(*args):
            val = yt_playlist_var.get().strip()
            set_nested("youtube.playlist", val)
            self.append_console(f"📂 Playlist YouTube cambiada a: {val}\n")


        frm_yt = ttk.Frame(sidebar)
        frm_yt.pack(fill="x", pady=(4, 10))
        ttk.Label(frm_yt, text="Visibilidad YouTube").pack(anchor="w")

        # Radios de visibilidad
        ttk.Radiobutton(
            frm_yt, text="Público", variable=yt_vis_var, value="PUBLIC",
            command=on_change_yt_visibility
        ).pack(anchor="w")

        ttk.Radiobutton(
            frm_yt, text="Oculto", variable=yt_vis_var, value="UNLISTED",
            command=on_change_yt_visibility
        ).pack(anchor="w")

        ttk.Radiobutton(
            frm_yt, text="Privado", variable=yt_vis_var, value="PRIVATE",
            command=on_change_yt_visibility
        ).pack(anchor="w")

        # Campo de texto para Playlist
        ttk.Label(frm_yt, text="Youtube Playlist:").pack(anchor="w", pady=(6, 0))
        yt_playlist_entry = ttk.Entry(frm_yt, textvariable=yt_playlist_var)
        yt_playlist_entry.pack(fill="x", pady=(0, 4))
        yt_playlist_var.trace_add("write", on_change_yt_playlist)
        
        self.btn_pub_yt = sidebtn("🚀 Publicar Youtube", self.on_publish_youtube)
        
        
        ttk.Label(sidebar, text="Automatismos", font=("Segoe UI", 10, "bold")).pack(anchor="w",padx=6, pady=(12, 2))
        
        self.steps_frame = ttk.LabelFrame(sidebar, text="Secuencia automática")
        self.steps_frame.pack(fill="x", pady=10, padx=6)

        # Definimos los scripts con su nombre y ruta
        self.scripts = [
            ("Extraer contenido url", {"type": "func",   "target": self.on_extract}),
            ("Generar Post Linkedin", {"type": "func",   "target": self.on_generate_linkedin}),
            ("Generar imagen",        {"type": "func",   "target": self.on_run_gemini_image}),
            ("Publicar Post Linkedin",{"type": "script", "target": "scripts/linkedin_publisher.py"}),
            ("Publicar comentario",   {"type": "script", "target": "scripts/linkedin_comment_publisher.py"}),
            ("Generar narración",     {"type": "func",   "target": self.on_generate_video}),
            ("Generar video",         {"type": "script", "target": "scripts/video_generator.py"}),
            ("Publicar en YouTube",   {"type": "script", "target": "scripts/youtube_publisher.py"}),
        ]


        STEP_KEYS = {
            "Extraer contenido url": "extraer_contenido",
            "Generar Post Linkedin": "generar_post_linkedin",
            "Generar imagen": "generar_imagen",
            "Publicar Post Linkedin": "publicar_post_linkedin",
            "Publicar comentario": "publicar_comentario",
            "Generar narración": "generar_narracion",
            "Generar video": "generar_video",
            "Publicar en YouTube": "publicar_video_youtube",
        }

        self.step_vars = {}

        for text, cfg in self.scripts:
            key = f"automated.{STEP_KEYS[text]}"   # ej: automated.generar_video
            saved = get_nested(key, path=GLOBAL_CONFIG)

            # inicializa con lo que haya en config.json
            var = tk.BooleanVar(value=(saved == "1"))

            def on_toggle(v=var, k=key):
                val = "1" if v.get() else "0"
                set_nested(k, val, path=GLOBAL_CONFIG)
                self.append_console(f"⚙️ Guardado {k} = {val}\n")
                copy_state_to_folder(app)

            ttk.Checkbutton(
                self.steps_frame,
                text=text,
                variable=var,
                command=on_toggle
            ).pack(anchor="w", pady=2)

            self.step_vars[cfg["target"]] = var
            
        
        # --- selector horario ---
        time_frame = ttk.Frame(sidebar)
        time_frame.pack(anchor="w", padx=6, pady=4)

        ttk.Label(time_frame, text="Hora:").pack(side="left")
        self.hour_var = tk.StringVar(value="12")
        self.min_var = tk.StringVar(value="00")


        def on_scheduled():
            day = self.sidebar_cal.get_date()   # p.ej. "2025-09-25"
            time = f"{self.hour_var.get()}:{self.min_var.get()}"  # "14:30"
            val = f"{day} {time}"               # "2025-09-25 14:30"

            if self.scheduled_var.get():
                set_nested("schedule.is_scheduled", "1")
                set_nested("schedule.date", val)
                copy_state_to_folder(app)
            else:
                set_nested("schedule.is_scheduled", "0")
                set_nested("schedule.date", "")   
                copy_state_to_folder(app)    
                
        ttk.Spinbox(time_frame, from_=0, to=23, wrap=True, width=3, textvariable=self.hour_var, format="%02.0f",
                        command=on_scheduled).pack(side="left")
        ttk.Label(time_frame, text=":").pack(side="left")
        ttk.Spinbox(time_frame, from_=0, to=59, wrap=True, width=3, textvariable=self.min_var, format="%02.0f",
                        command=on_scheduled).pack(side="left")
                
        # --- checkbox programado ---
        self.scheduled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(sidebar, text="Programado", variable=self.scheduled_var,
                        command=on_scheduled).pack(anchor="w", padx=6, pady=4)


        self.btn_pub_yt = sidebtn("🚀 RUN! 🤖", self.on_auto_publish)
        self.btn_pub_yt.pack(fill="x", pady=10)
         

    # ===== Bloquear el ancho de la ventana al ancho del contenido =====
        # Calculamos el ancho que pide cada columna y usamos eso como ancho de ventana.
        self.update_idletasks()  # asegura medidas correctas

        content_w = left_inner.winfo_reqwidth() + sidebar.winfo_reqwidth() + 45 + 60  # paddings aprox.
        # Altura opcional (usa la actual si ya te vale)
        target_h = max(self.winfo_height(), 1000)# -> Alto de la ventana principal

        # Fija el tamaño y evita redimensionado horizontal (solo vertical)
        self.geometry(f"{content_w}x{target_h}")
        self.minsize(content_w, 1000) # -> Alto de la ventana principal
        self.resizable(True, True)  # ancho fijo, alto flexible
        
        

                
            
    def mark_days_with_posts(self):
        base_post = BASE_DIR / "post"
        logs = [f"[DEBUG] Revisando carpeta base: {base_post}"]

        # limpiar eventos previos
        self.sidebar_cal.calevent_remove('all')

        for folder in base_post.iterdir():
            if folder.is_dir():
                logs.append(f"[DEBUG] Encontrada carpeta: {folder.name}")
                try:
                    dt = datetime.strptime(folder.name, "%Y-%m-%d").date()
                    logs.append(f"[DEBUG] Carpeta válida como fecha: {dt}")
                except ValueError:
                    logs.append(f"[DEBUG] Carpeta ignorada (no es fecha): {folder.name}")
                    continue

                if next(folder.iterdir(), None):  # comprobación rápida
                    logs.append(f"[DEBUG] {folder.name} tiene contenido, marcando en calendario")
                    self.sidebar_cal.calevent_create(dt, "con_contenido", "post_day")
                else:
                    logs.append(f"[DEBUG] {folder.name} está vacía, no se marca")

        self.sidebar_cal.tag_config("post_day", background="lightgreen", foreground="black")
        logs.append("[DEBUG] Estilo para 'post_day' aplicado")

        # Un solo volcado de logs
        self.append_console("\n".join(logs))

    
    
    # ---------- Helpers GUI ----------
    def set_running(self, running: bool):
        self.running = running
        state = ("disabled" if running else "normal")
        for b in (
            self.btn_extract,
            self.btn_prompt_li,
            self.btn_prompt_vid,
            self.btn_prompt_img,
            self.btn_create_post,
            self.btn_create_narr,
            self.btn_create_img,
            self.btn_make_video,
            self.btn_save_texts,
            self.btn_pub_li,
            self.btn_pub_yt,
        ):
            b.config(state=state)
            
    def append_console(self, text: str):
        def _append():
            self.console.insert("end", text)
            self.console.see("end")
        self.console.after(0, _append)


    # ---------- Prompts personalizados ----------
    def edit_linkedin_prompt(self):
        self._edit_prompt_window("Prompt Linkedin", get_nested("linkedin.prompt"), setter=lambda v: setattr(self, "linkedin_prompt", v))

    def edit_fixed_description_linkedin(self):
        self._edit_prompt_window("linkedin fixed description", get_nested("linkedin.fixed_description"), setter=lambda v: setattr(self, "fixed_linkedin_description", v))

    def edit_fixed_description_youtube(self):
        self._edit_prompt_window("youtube fixed description", get_nested("youtube.fixed_description"), setter=lambda v: setattr(self, "fixed_youtube_description", v))
    
    def edit_video_prompt(self):
        self._edit_prompt_window("Prompt Narración", get_nested("youtube.narration"), setter=lambda v: setattr(self, "video_prompt", v))


    def edit_image_prompt(self):
        """Editor del prompt de imagen que PERSISTE en global_config.json al pulsar Guardar."""
        current = get_nested("gemini.prompt")

        top = tk.Toplevel(self)
        top.title("Prompt Imagen (miniatura)")
        txt = tk.Text(top, wrap="word", height=14, width=90)
        txt.pack(fill="both", expand=True, padx=6, pady=6)
        txt.insert("1.0", current)

        def save():
            new_val = txt.get("1.0", tk.END).strip()
            try:
                set_nested("gemini.prompt",new_val)  # persiste en state.json
                self.append_console("💾 Prompt de imagen guardado en global_config.json\n")
                copy_state_to_folder(app)
            except Exception as e:
                messagebox.showerror("Guardar prompt", f"No se pudo guardar el prompt: {e}")
            top.destroy()

        ttk.Button(top, text="Guardar", command=save).pack(pady=6)
        
    def _edit_prompt_window(self, title, current, setter):
        top = tk.Toplevel(self)
        top.title(title)
        txt = tk.Text(top, wrap="word", height=12, width=90)
        txt.pack(fill="both", expand=True, padx=6, pady=6)
        txt.insert("1.0", current)

        def save():
            new_val = txt.get("1.0", tk.END).strip()
            setter(new_val)
            if title == "Prompt Linkedin":
                set_nested("linkedin.prompt", new_val)
            if title == "Prompt Narración":
                set_nested("youtube.narration", new_val)
            if title == "linkedin fixed description":
                set_nested("linkedin.fixed_description", new_val)
            if title == "youtube fixed description":
                set_nested("youtube.fixed_description", new_val)
            top.destroy()
        ttk.Button(top, text="Guardar", command=save).pack(pady=6)

    # ---------- Funciones de negocio ----------
    data = ""
    def on_extract(self):
        user_url = self.url_var.get().strip()
        if not user_url:
            messagebox.showwarning("Falta URL", "Introduce una URL válida")
            return
        try:
            global data
            data = scrape_page(user_url)
            content = f"{data['title']}\n\n{data['description']}\n\n{data['text']}"

            # Mostrar en el textbox
            self.text_extracted.delete("1.0", tk.END)
            self.text_extracted.insert(tk.END, content)
            self.append_console("✅ Contenido extraído")

            # Guardar (esto ya llama a copy_state_to_folder al final)
            self.save_texts()
            

        except Exception as e:
            self.append_console(f"❌ Error: {e}")
            messagebox.showerror("Scraping", str(e))


    def on_generate_linkedin(self):
        text = self.text_extracted.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Sin texto", "Primero extrae o pega contenido")
            return
        try:
            prompt_post = get_nested("linkedin.prompt") + "\n\nTEXTO BASE:\n" + text
            post = call_groq(prompt_post, model=get_nested("groq.model_post"), max_words=500)
            self.text_post.delete("1.0", tk.END)
            self.text_post.insert(tk.END, post)
            self.append_console("✅ Post LinkedIn generado")
            self.save_texts()
        except Exception as e:
            app.append_console(f"❌ Error Groq: {e}")
            messagebox.showerror("Groq", str(e))

    def on_generate_video(self):
        post_text = self.text_post.get("1.0", tk.END).strip()
        if not post_text:
            messagebox.showwarning("Sin post", "Primero genera el post de LinkedIn")
            return
        try:
            prompt_video = get_nested("youtube.narration") + "\n\nPOST LINKEDIN:\n" + post_text
            video = call_groq(prompt_video, model=get_nested("groq.model_video"), max_words=140)
            self.text_video.delete("1.0", tk.END)
            self.text_video.insert(tk.END, video)
            app.append_console("✅ Texto video generado")
            self.save_texts()
        except Exception as e:
            app.append_console(f"❌ Error Groq: {e}")
            messagebox.showerror("Groq", str(e))

    def save_texts(self):
        post_text = self.text_post.get("1.0", tk.END).strip()
        video_text = self.text_video.get("1.0", tk.END).strip()
        url_text = self.url_var.get().strip()
        scraped_text = self.text_extracted.get("1.0", tk.END).strip()

        try:
            # 👇 Crear la carpeta del día si no existe
            POST_DIR.mkdir(parents=True, exist_ok=True)

            if post_text:
                (POST_DIR / "linkedin_post.txt").write_text(post_text, encoding="utf-8")
            if video_text:
                (POST_DIR / "video_text.txt").write_text(video_text, encoding="utf-8")
            if url_text:
                (POST_DIR / "url_text.txt").write_text(url_text, encoding="utf-8")
            if scraped_text:
                (POST_DIR / "scraped_text.txt").write_text(scraped_text, encoding="utf-8")

            copy_state_to_folder(app)
            set_nested("gemini.prompt", get_nested("gemini.prompt"))
            app.append_console("✅ Archivos guardados en carpeta post/")

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar: {e}")


            
    def _pick_latest_thumbnail(self):
        """Devuelve la ruta de la miniatura más reciente en POST_DIR."""
        p = POST_DIR / "miniatura.jpg"
        if p.exists():
            return p
        cands = sorted(
            list(POST_DIR.glob("miniatura*.jpg")) + list(POST_DIR.glob("thumbnail*.jpg")),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        return cands[0] if cands else None

    def _refresh_thumbnail_after_script(self):
        """Reintenta unos ms por si el archivo tarda en aparecer en disco."""
        def _try(retries=5):
            path = self._pick_latest_thumbnail()
            if path and path.exists():
                self._update_thumbnail_preview(path)
                self.append_console(f"🖼️ Miniatura actualizada: {path}\n")
            elif retries > 0:
                self.after(500, lambda: _try(retries-1))
            else:
                self.append_console("⚠️ No se encontró miniatura tras generar la imagen.\n")
        _try()

    # ---------- Lanzador de scripts ----------
    def on_run_gemini_image(self):
        """Ejecuta gemini_image.py y, al terminar, refresca la miniatura en la UI."""
        self.run_scripts_sequence(["gemini_image.py"], on_done=self._refresh_thumbnail_after_script)

    def on_make_video(self):
        """Genera el vídeo (sin publicar)."""
        self.run_scripts_sequence(["video_generator.py"])

    def on_publish_linkedin(self):
        self.run_scripts_sequence(["linkedin_publisher.py"])
        
    def on_publish_linkedin_comment(self):
        self.run_scripts_sequence(["linkedin_comment_publisher.py"])

    def on_publish_youtube(self):
        """Publica en YouTube (asume que el vídeo ya existe)."""
        self.run_scripts_sequence(["youtube_publisher.py"])
        
        
    def on_auto_publish(self):
        """Ejecuta en secuencia lo seleccionado"""
        selected = [cfg for _, cfg in self.scripts if self.step_vars[cfg["target"]].get()]

        for step in selected:
            if step["type"] == "script":
                self.run_scripts_sequence([step["target"]])
            elif step["type"] == "func":
                step["target"]() 
        

    def run_scripts_sequence(self, scripts: list[str], on_done=None):
        if self.running:
            messagebox.showinfo("En ejecución", "Ya hay un proceso en marcha")
            return

        def sequence():
            self.set_running(True)
            rc_total = 0
            try:
                for script in scripts:
                    rc = self._run_and_stream(script)
                    if rc != 0:
                        rc_total = rc
                        break
            finally:
                self.set_running(False)
                if rc_total == 0:
                    self.append_console("\n✔ Secuencia completada correctamente.\n")
                    if on_done:
                        # Ejecuta el callback en el hilo de la UI
                        self.after(0, on_done)
                else:
                    self.append_console(f"\n✖ Secuencia abortada. Código {rc_total}.\n")

        threading.Thread(target=sequence, daemon=True).start()
    

    def _run_and_stream(self, script_name: str) -> int:
        script_path = SCRIPTS_DIR / script_name
        if not script_path.exists():
            self.append_console(f"[ERR] No existe: {script_path}\n")
            messagebox.showerror("Script", f"No se encontró {script_path}")
            return -1

        # Fuerza modo UTF-8 en el proceso hijo (-X utf8) y salida sin buffer (-u)
        cmd = [sys.executable, "-X", "utf8", "-u", str(script_path)]
        self.append_console(f"\n▶ Ejecutando {script_name}...\n")

        # Asegura UTF-8 al leer y evita que reviente si el hijo no cumple
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        try:
            with subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,                 # modo texto
                encoding="utf-8",          # decodifica como UTF-8
                errors="replace",          # nunca lances UnicodeDecodeError
                env=env,
            ) as p:
                self.current_proc = p
                for line in p.stdout:
                    self.append_console(line)
                rc = p.wait()
        except Exception as e:
            self.append_console(f"\n[ERR] {type(e).__name__}: {e}\n")
            return -1
        finally:
            self.current_proc = None

        if rc == 0:
            self.append_console(f"✔ {script_name} finalizado OK.\n")
        else:
            self.append_console(f"✖ {script_name} terminó con código {rc}.\n")
        return rc




    # ---------- Día/carpeta + miniatura + carga de textos ----------
    def set_post_dir(self, date_str: str):
        """Actualiza la carpeta activa (post/YYYY-MM-DD), refresca labels y carga textos si existen."""
        global POST_DIR, TODAY
        TODAY = date_str
        POST_DIR = BASE_DIR / "post" / TODAY
        #POST_DIR.mkdir(parents=True, exist_ok=True)

        if hasattr(self, "day_var"):
            self.day_var.set(date_str)
        if hasattr(self, "status_var"):
            self.status_var.set(f"📂 Carpeta activa: {POST_DIR}")

        # Miniatura
        thumb_path = POST_DIR / "miniatura.jpg"
        self._update_thumbnail_preview(thumb_path if thumb_path.exists() else None)
        
        if hasattr(self, "video_label"):
            self._show_video_poster(20)
    
        # Textos
        lp = POST_DIR / "linkedin_post.txt"
        vt = POST_DIR / "video_text.txt"
        ur = POST_DIR / "url_text.txt"
        sc = POST_DIR / "scraped_text.txt"
        
        try:
            if hasattr(self, "text_post"):
                self.text_post.delete("1.0", tk.END)
                if lp.exists():
                    self.text_post.insert(tk.END, lp.read_text(encoding="utf-8"))
                    self.append_console(f"📄 Cargado: {lp}\n")
                else:
                    self.append_console("∅ No hay linkedin_post.txt en la carpeta.\n")

            if hasattr(self, "text_video"):
                self.text_video.delete("1.0", tk.END)
                if vt.exists():
                    self.text_video.insert(tk.END, vt.read_text(encoding="utf-8"))
                    self.append_console(f"📄 Cargado: {vt}\n")
                else:
                    self.append_console("∅ No hay video_text.txt en la carpeta.\n")
                    
            if hasattr(self, "entry_url"):
                self.entry_url.delete(0, tk.END)   # 👈 índices para Entry
                if ur.exists():                    # usabas lp por error
                    self.entry_url.insert(0, ur.read_text(encoding="utf-8"))
                    self.append_console(f"📄 Cargado: {ur}\n")
                else:
                    self.append_console("∅ No hay url_text.txt en la carpeta.\n")
                    
            if hasattr(self, "text_extracted"):
                self.text_extracted.delete("1.0", tk.END)
                if sc.exists():
                    self.text_extracted.insert(tk.END, sc.read_text(encoding="utf-8"))
                    self.append_console(f"📄 Cargado: {sc}\n")
                else:
                    self.append_console("∅ No hay scraped_text.txt en la carpeta.\n")
                    
        except Exception as e:
            messagebox.showerror("Cargar textos", f"No se pudieron leer archivos: {e}")

            
    
    def _ffmpeg_frame_to_photo(self, path, target):
        """Devuelve un ImageTk.PhotoImage con el frame N usando ffmpeg (rápido), o None si falla."""
        import shutil, subprocess, io
        from PIL import Image, ImageOps
        if not shutil.which("ffmpeg"):
            return None
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(path),
            "-vf", f"select=eq(n\\,{target}),scale={self.thumb_w}:-2",
            "-vframes", "1",
            "-f", "image2pipe", "-vcodec", "mjpeg",
            "-"  # stdout
        ]
        try:
            proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
            if not proc.stdout:
                return None
            im = Image.open(io.BytesIO(proc.stdout)).convert("RGB")
            im = ImageOps.fit(im, (self.thumb_w, self.thumb_h), method=Image.LANCZOS, centering=(0.5, 0.5))
            return ImageTk.PhotoImage(im)
        except Exception:
            return None    
        
    # --- Añade esto a la clase ScraperApp (p.ej. junto a _prime_first_frame) ---
    def _show_video_poster(self, frame_index=20):
        """Pinta en self.video_label el frame N del video_post.mp4 (0-based)."""
        from ffpyplayer.player import MediaPlayer
        path = POST_DIR / "video_post.mp4"

        # Siempre limpiamos primero
        self._clear_video_preview("Cargando previsualización...")

        if not path.exists() or path.stat().st_size == 0:
            self._clear_video_preview("Sin vídeo (video_post.mp4)")
            return

        # 1) Intento con ffpyplayer: avanzar hasta el frame N
        try:
            player = MediaPlayer(str(path), ff_opts={"out_fmt": "rgb24", "paused": True, "sync": "audio"})
            self._ff_player = player
        except Exception:
            self._ff_player = None

        target = int(max(0, frame_index))
        count = -1
        tries = 0

        def paint_photo(photo):
            if photo is not None:
                self._ff_imgtk = photo
                self.video_label.configure(image=self._ff_imgtk, text="")
            else:
                self._clear_video_preview(f"No se pudo extraer el frame {target}")

        def try_fallback():
            photo = self._ffmpeg_frame_to_photo(path, target)
            paint_photo(photo)

        def _grab():
            nonlocal count, tries
            if not self._ff_player:
                return try_fallback()

            frame, val = self._ff_player.get_frame()
            if frame is not None:
                count += 1
                if count == target:
                    img, _ts = frame
                    try:
                        pil = img.to_image()
                    except Exception:
                        w, h = img.get_size()
                        buf = img.to_bytearray()[0]
                        pil = Image.frombytes('RGB', (w, h), bytes(buf))
                    pil = pil.resize((self.thumb_w, self.thumb_h), Image.BILINEAR)
                    self._ff_imgtk = ImageTk.PhotoImage(pil)
                    self.video_label.configure(image=self._ff_imgtk, text="")
                    try:
                        self._ff_player.close_player()
                    except Exception:
                        pass
                    self._ff_player = None
                    return
                else:
                    return self.video_label.after(1, _grab)

            if val == "eof" or tries > 2000:
                try:
                    self._ff_player.close_player()
                except Exception:
                    pass
                self._ff_player = None
                return try_fallback()

            tries += 1
            self.video_label.after(1, _grab)

        _grab()


        
    def _clear_video_preview(self, msg="Sin vídeo"):
        """Detiene todo y pinta un placeholder vacío."""
        # parar player y polling
        try:
            if self._ff_player:
                self._ff_player.close_player()
        except Exception:
            pass
        self._ff_player = None
        self._ff_running = False

        # si usas un after() para progreso, cancelarlo
        try:
            if getattr(self, "_ff_poll_job", None):
                self.after_cancel(self._ff_poll_job)
        except Exception:
            pass
        self._ff_poll_job = None

        # limpiar slider si lo tienes
        if hasattr(self, "video_progress"):
            try:
                self.video_progress.configure(from_=0, to=0)
                self.video_progress.set(0)
            except Exception:
                pass

        # borrar referencia a la última imagen
        self._ff_imgtk = None

        # pintar un placeholder 16:9 (oscuro)

        w, h = getattr(self, "thumb_w", 720), getattr(self, "thumb_h", 405)
        ph = Image.new("RGB", (w, h), (13, 17, 23))
        draw = ImageDraw.Draw(ph)
        # mensaje centrado (sin depender de fuentes del sistema)
        try:
            txt_w, txt_h = draw.textsize(msg)
            draw.text(((w - txt_w)//2, (h - txt_h)//2), msg, fill=(200, 200, 200))
        except Exception:
            pass
        self._ff_placeholder = ImageTk.PhotoImage(ph)  # mantener referencia
        self.video_label.configure(image=self._ff_placeholder, text="")
    
            

    # ---------- Miniatura (visor 16:9) ----------
    def _update_thumbnail_preview(self, path):
        """Actualiza el visor de miniatura (16:9). Si path es None o no existe, muestra placeholder."""
        w, h = getattr(self, "thumb_w", 640), getattr(self, "thumb_h", 360)
        base = Image.new("RGB", (w, h), (13, 17, 23))  # fondo

        if path and Path(path).exists():
            try:
                img = Image.open(path).convert("RGB")
                img = ImageOps.fit(img, (w, h), method=Image.LANCZOS, centering=(0.5, 0.5))  # cover
                base.paste(img, (0, 0))
            except Exception:
                pass

        self._thumb_imgtk = ImageTk.PhotoImage(base)
        self.thumb_label.configure(image=self._thumb_imgtk)



class CollapsibleSection(ttk.Frame):
    def __init__(self, parent, title="Sección", start_open=True, expand_when_open=True):
        """
        expand_when_open: si True, la sección ocupa el espacio sobrante cuando está abierta.
                          cuando está cerrada, siempre ocupa lo mínimo.
        """
        super().__init__(parent)
        self.is_open = bool(start_open)
        self.expand_when_open = bool(expand_when_open)

        # Cabecera
        hdr = ttk.Frame(self)
        hdr.pack(fill="x")
        self._toggle_btn = ttk.Button(
            hdr, width=2, text=("▾" if self.is_open else "▸"), command=self.toggle
        )
        self._toggle_btn.pack(side="left")
        ttk.Label(hdr, text=title, font=("Segoe UI", 12, "bold")).pack(side="left")

        # Cuerpo
        self.body = ttk.Frame(self)
        if self.is_open:
            self.body.pack(fill="both", expand=True, padx=8, pady=(4, 0))

        # Ajuste del pack del propio frame según estado
        self.after(0, self._apply_pack_mode)

    def open(self):
        if not self.is_open:
            self.body.pack(fill="both", expand=True, padx=8, pady=(4, 0))
            self._toggle_btn.configure(text="▾")
            self.is_open = True
            self._apply_pack_mode()

    def close(self):
        if self.is_open:
            self.body.pack_forget()
            self._toggle_btn.configure(text="▸")
            self.is_open = False
            self._apply_pack_mode()

    def toggle(self):
        if self.is_open:
            self.close()
        else:
            self.open()

    def _apply_pack_mode(self):
        """Cerrada: fill='x', expand=False.  Abierta: fill='both', expand=True si expand_when_open=True."""
        try:
            if self.is_open and self.expand_when_open:
                self.pack_configure(fill="both", expand=True)
            else:
                self.pack_configure(fill="x", expand=False)
        except tk.TclError:
            pass



if __name__ == "__main__":
    app = ScraperApp()
    app.mainloop()
