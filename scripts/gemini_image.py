import google.generativeai as genai
import requests, base64, sys, io
from pathlib import Path
from datetime import date
from PIL import Image,  ImageFilter
from state_store import get_nested
from PIL import Image, ImageFilter, ImageOps  
sys.stdout.reconfigure(encoding='utf-8')

today = get_nested("selected_day") or date.today().strftime("%Y-%m-%d")

BASE_DIR = Path(__file__).parent.parent
post_dir = BASE_DIR / "post" / today
post_file = post_dir / "linkedin_post.txt"

API_KEY = get_nested("gemini.api_key")
URL_IMAGE_MODEL = get_nested("gemini.url_model")

if not API_KEY:
    raise ValueError("No se encontró GEMINI_API_KEY. Configura GEMINI_API_KEY primero.")

genai.configure(api_key=API_KEY)


def _save_jpeg_smart(img: Image.Image, out_path, max_kb=400, q_hi=90, q_lo=60):
    """
    Guarda como JPEG progresivo y optimizado buscando quedar <= max_kb.
    Hace una pequeña búsqueda binaria de calidad.
    """
    img = img.convert("RGB")
    lo, hi = q_lo, q_hi
    best_bytes = None
    best_q = lo
    while lo <= hi:
        q = (lo + hi) // 2
        buf = io.BytesIO()
        img.save(buf, format="JPEG",
                 quality=q, optimize=True, progressive=True, subsampling=2)
        size_kb = len(buf.getvalue()) / 1024
        if size_kb <= max_kb:
            best_bytes, best_q = buf.getvalue(), q
            lo = q + 1  # intenta mejor calidad
        else:
            hi = q - 1  # baja calidad
    if best_bytes is None:
        # Si no se logró <= max_kb, guarda al q_lo
        buf = io.BytesIO()
        img.save(buf, format="JPEG",
                 quality=q_lo, optimize=True, progressive=True, subsampling=2)
        best_bytes, best_q = buf.getvalue(), q_lo
    with open(out_path, "wb") as f:
        f.write(best_bytes)
    return best_q, len(best_bytes)/1024

def generar_imagen(prompt: str, filename: str = "miniatura.jpg"):
    url = f"{URL_IMAGE_MODEL}?key={API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    retries = 3

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            status = response.status_code
            txt = response.text

            # extraer base64 (sin guardar PNG)
            parts = result["candidates"][0]["content"]["parts"]
            image_base64 = next((p["inlineData"]["data"]
                                 for p in parts if "inlineData" in p), None)
            if not image_base64:
                print(f"⚠️ Intento {attempt}/{retries}: respuesta no-JSON (status {status}). Body:\n{txt[:1000]}")
                continue

            image_bytes = base64.b64decode(image_base64)
            src = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Lienzo final 16:9 (YouTube thumb: 1280x720 recomendado;
            # si quieres mantener 1920x1080, cambia target_size)
            target_size = (1280, 720)

            cut_mode = str(get_nested("gemini.imagen_cut", "0")).strip().lower() in {"1","true","yes","on"}

            if cut_mode:
                print("Dentro de cut_mode")
                # MODO CUT: recorta (cover) para llenar 16:9 sin bordes ni blur
                # Amplía y recorta centrado; si quieres “subir” un poco el encuadre usa centering=(0.5, 0.45)
                composed = ImageOps.fit(src, target_size, method=Image.LANCZOS, centering=(0.5, 0.5))
            else:
                print("Dentro de blur backgroud en el else")
                # MODO BLUR+CONTAIN (como ahora): fondo difuminado + imagen centrada sin deformar
                background = src.copy().resize(target_size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(30))
                fg = src.copy()
                fg.thumbnail(target_size, Image.LANCZOS)
                bg_w, bg_h = background.size
                fg_w, fg_h = fg.size
                offset = ((bg_w - fg_w) // 2, (bg_h - fg_h) // 2)
                background.paste(fg, offset)
                composed = background

            # guardar como JPG optimizado
            post_dir.mkdir(parents=True, exist_ok=True)
            out_path = post_dir / "miniatura.jpg"
            q_used, size_kb = _save_jpeg_smart(composed, out_path, max_kb=400, q_hi=88, q_lo=65)
            print(f"✅ Imagen guardada como {out_path} ({size_kb:.0f} KB, quality={q_used})")
            return str(out_path)

        except Exception as e:
            print(f"❌ Error generando la imagen: {e}")
            if 'response' in locals():
                print("Respuesta completa (truncada):", str(response.text)[:800])

    raise ValueError("❌ No se pudo generar la imagen tras varios intentos")
            
            
if __name__ == "__main__":

    # Leer el contenido del archivo
    print("today into gemimini",today)
    if not post_file.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {post_file}")

    with open(post_file, "r", encoding="utf-8") as f:
        post_text = f.read().strip()

    rules = get_nested("prompts.image")
    # Construir el prompt
    prompt = f"{rules} El contexto (no mostrar literalmente en la imagen, solo usar como inspiración): {post_text}"

    # Generar la imagen
    generar_imagen(prompt, f"mi_imagen_{today}.png")
