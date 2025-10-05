import os, re, json, argparse, requests, math, pysrt, unicodedata, hashlib, time, stat,gc, spacy
import sys, shutil, math , subprocess
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from faster_whisper import WhisperModel
from PIL import Image, ImageDraw, ImageFont
from PIL import Image, ImageDraw, ImageFont
from moviepy import (VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip, ColorClip, CompositeAudioClip)
from moviepy import vfx
from datetime import date
from state_store import get_nested
sys.stdout.reconfigure(encoding='utf-8')

today = get_nested("selected_day") or date.today().strftime("%Y-%m-%d")

BASE_DIR = Path(__file__).resolve().parent.parent
POST_DIR = BASE_DIR / "post" / today
DEFAULT_INFILE = POST_DIR / "video_text.txt"
OUT_DIR = POST_DIR
CACHE_DIR = BASE_DIR / "cache"
ASSETS_DIR = BASE_DIR / "assets"
    
GROQ_API_KEY = get_nested("groq.api_key")
PEXELS_API_KEY = get_nested("pexels.api_key")
ELEVENLABS_API_KEYS = [
    get_nested("elevenlabs.api_key_1"),
    get_nested("elevenlabs.api_key_2"),
    get_nested("elevenlabs.api_key_3"),
    get_nested("elevenlabs.api_key_4"),
    get_nested("elevenlabs.api_key_5"),
    get_nested("elevenlabs.api_key_6"),
    get_nested("elevenlabs.api_key_7"),
]

ELEVENLABS_VOICE_ID = get_nested("elevenlabs.voice_id")
ELEVENLABS_MODEL_ID = get_nested("elevenlabs.model_id") or "eleven_multilingual_v2"
ELEVENLABS_STABILITY = float(get_nested("elevenlabs.stability") or "0.35")
ELEVENLABS_SIMILARITY = float(get_nested("elevenlabs.similarity") or "0.8")
ELEVENLABS_STYLE = float(get_nested("elevenlabs.style") or "0.0")
ELEVENLABS_SPEAKER_BOOST = str(get_nested("elevenlabs.speaker_boost") or "1") in ("1","true","True")


# ---------- utils de texto ----------
STOPWORDS = set("""
a ante bajo cabe con contra de desde durante en entre hacia hasta mediante para por según sin so sobre tras
y e ni o u pero mas sino que como cual cuales cuando donde quien quienes cuyo cuya cuyos cuyas
al del la el los las un una unos unas este esta estos estas ese esa esos esas aquel aquella aquellos aquellas
yo tú tu usted ustedes él ella ello ellos ellas me mi mis mí te ti tus se sí le les lo la los las nos nosotros nosotras
su sus ya muy más menos tan tanto cada cual ser es son fue fueron era eran estoy está están estaba estaban
hay había han he ha hemos han haber puede pueden podría podrían debe deben deben debería deberían
""".split())

EN_STOPWORDS = {
    "the","a","an","and","or","but","of","in","on","at","to","for","from","by","with",
    "is","are","was","were","be","being","been","as","that","this","these","those",
    "it","its","into","over","under","up","down","out","about","than","then","so",
    "very","just","only","also","not","no","without","within"
}

def tokenize_en(s: str):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s-]+", " ", s)
    toks = [t for t in re.split(r"[\s\-]+", s) if t and t not in EN_STOPWORDS and len(t) > 2]
    return toks


def normalize(s):
    s = re.sub(r"[^\wáéíóúñü]+", " ", s.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()

def split_paragraphs(text):
    parts = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    merged = []
    buf = ""
    for p in parts:
        if len((buf+" "+p).strip()) < 120:
            buf = (buf+" "+p).strip()
        else:
            if buf: merged.append(buf); buf = ""
            merged.append(p)
    if buf: merged.append(buf)
    return merged

def keywords_for(paragraph, top_k=3):
    toks = normalize(paragraph).split()
    freq = {}
    for t in toks:
        if t in STOPWORDS or len(t) <= 2:
            continue
        freq[t] = freq.get(t, 0) + 1
    keys = [w for w,_ in sorted(freq.items(), key=lambda x: (-x[1], -len(x[0])))]
    if not keys:
        keys = toks[:2]
    return keys[:top_k]

def slugify(s: str, max_len: int = 40) -> str:
    s_ascii = unicodedata.normalize('NFKD', s).encode('ascii','ignore').decode('ascii')
    s_ascii = re.sub(r'[^a-zA-Z0-9]+', '_', s_ascii).strip('_').lower()
    if not s_ascii:
        s_ascii = "q"
    if len(s_ascii) > max_len:
        h = hashlib.sha1(s_ascii.encode()).hexdigest()[:8]
        s_ascii = f"{s_ascii[:max_len]}_{h}"
    return s_ascii


DEBUG = True  # se activará desde main con --debug 1

def dbg(msg):
    if DEBUG:
        print(msg)

def _mask(s, show=4):
    if not s: return "None"
    s = str(s)
    return s[:show] + "…" + s[-2:]


_EMBEDDER = None

def get_embedder():
    """
    Carga perezosa de un modelo multilingüe.
    Si no está instalado sentence-transformers, devolvemos None y haremos fallback.
    """
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    try:
        from sentence_transformers import SentenceTransformer
        # Modelo pequeño y multilingüe:
        _EMBEDDER = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        dbg("[EMB] SentenceTransformer cargado")
    except Exception as e:
        dbg(f"[EMB] no disponible: {e}")
        _EMBEDDER = None
    return _EMBEDDER

def embed_texts(texts):
    """
    Devuelve embeddings normalizados o None si no hay modelo.
    """
    model = get_embedder()
    if model is None:
        return None

    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
# ===== fin embeddings =====



def whisper_words_faster(audio_path, language="es", model_size="small"):
    # Fuerza HF Hub a NO usar symlinks en Windows
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    # Usa una carpeta de caché dentro de tu proyecto (evita problemas en OneDrive/Perfil)
    whisper_cache = CACHE_DIR / "whisper_models"
    whisper_cache.mkdir(parents=True, exist_ok=True)

    model = WhisperModel(
        model_size,
        device="cpu",                # o "cuda" si tienes GPU
        compute_type="int8",         # rápido en CPU
        download_root=str(whisper_cache),  # <- clave en Windows
    )

    segments, info = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        vad_filter=True,
    )
    words = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                if not w.word: continue
                words.append({"text": w.word.strip(), "start": w.start, "end": w.end})
        else:
            # fallback por si algún segmento viene sin palabras
            words.append({"text": seg.text.strip(), "start": seg.start, "end": seg.end})
    return words


def whisper_words_from_json(json_path):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    out=[]
    for seg in data.get("segments", []):
        # words o tokens -> depende de versión
        words = seg.get("words") or []
        if not words:  # intenta trocear por espacios si no hay nivel palabra
            txt = seg.get("text","").strip()
            if not txt: continue
            # reparte duración por longitud (fallback)
            t0, t1 = seg.get("start",0.0), seg.get("end",0.0)
            total = sum(len(x) for x in re.findall(r"\S+", txt)) or 1
            acc = t0
            for tok in re.findall(r"\S+", txt):
                dur = (t1 - t0) * (len(tok)/total)
                out.append({"text": tok, "start": acc, "end": acc+dur})
                acc += dur
            continue
        for w in words:
            tok = (w.get("word") or w.get("text") or "").strip()
            if not tok: continue
            out.append({"text": tok, "start": float(w["start"]), "end": float(w["end"])})
    return out




def group_words_into_lines(words, min_words=3, max_words=5, max_gap=0.65, max_line_dur=4.8):
    """
    Agrupa por CONTEO de palabras (3–5) y también corta por pausa grande,
    duración máxima de línea o fin de frase.
    """


    lines, cur = [], []
    t0, last_end = None, None

    def flush():
        nonlocal cur, t0, last_end
        if cur:
            lines.append({
                "words": cur[:],
                "start": t0 if t0 is not None else cur[0]["start"],
                "end": cur[-1]["end"],
                "text": " ".join(w["text"] for w in cur)
            })
        cur, t0, last_end = [], None, None

    for w in words:
        if not (w.get("text") or "").strip():
            continue
        if t0 is None:
            t0 = w["start"]
        gap = (w["start"] - last_end) if last_end is not None else 0.0
        cur.append(w)
        last_end = w["end"]
        dur_line = last_end - t0

        reached_max = len(cur) >= max_words
        big_gap     = gap > max_gap
        too_long    = dur_line > max_line_dur and len(cur) >= min_words
        eos         = re.search(r"[.!?…]$", w["text"]) is not None

        if reached_max or eos or big_gap or too_long:
            flush()

    if cur:
        flush()

    # fusiona líneas demasiado cortas (< min_words) con la siguiente
    i = 0
    while i < len(lines) - 1:
        if len(lines[i]["words"]) < min_words:
            lines[i]["words"].extend(lines[i+1]["words"])
            lines[i]["end"]   = lines[i+1]["end"]
            lines[i]["text"]  = " ".join(w["text"] for w in lines[i]["words"])
            del lines[i+1]
        else:
            i += 1

    return lines





def _pick_bold_font(fs):
    for fp in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/seguisb.ttf",
    ]:
        try:
            return ImageFont.truetype(fp, fs)
        except Exception:
            pass
    return ImageFont.load_default()


def _layout_line(tokens, font, fs, W, H):
    """Devuelve: base_img (RGBA), sprites=[(idx, sprite_img, x, y)], tamaño."""
    stroke_w = max(2, fs // 20)
    word_gap = int(fs * 0.22)   # << menos separación entre palabras
    pad_x    = int(fs * 0.34)   # << padding horizontal más ajustado
    pad_y    = int(fs * 0.22)
    radius   = int(fs * 0.32)
    blue =      (0, 0, 0, 200)

    dummy = Image.new("RGBA", (10,10), (0,0,0,0))
    draw  = ImageDraw.Draw(dummy)

    measures=[]
    h_line=0; total_w=0; max_cw=0
    for t in tokens:
        x0,y0,x1,y1 = draw.textbbox((0,0), t, font=font, stroke_width=stroke_w)
        tw,th=x1-x0,y1-y0
        cw,ch=tw+pad_x*2, th+pad_y*2
        measures.append((tw,th,cw,ch))
        h_line=max(h_line,ch)
        total_w += cw + (word_gap if measures[:-1] else 0)
        max_cw = max(max_cw, cw)

    # el bloque debe poder albergar el chip más ancho
    max_block = int(W * 0.85)
    w_img = min(max(max(total_w, max_cw), int(W*0.5)), max_block)
    h_img = h_line
    base  = Image.new("RGBA", (w_img, h_img), (0,0,0,0))
    bdraw = ImageDraw.Draw(base)

    sprites=[]
    cur_x = (w_img - total_w)//2
    for idx, t in enumerate(tokens):
        tw,th,cw,ch = measures[idx]
        tx = cur_x + pad_x
        ty = (h_line - th)//2
        bdraw.text((tx,ty), t, font=font, fill=(255,255,255,255),
                   stroke_width=stroke_w, stroke_fill=(0,0,0,180))

        spr = Image.new("RGBA", (cw, ch), (0,0,0,0))
        sdraw = ImageDraw.Draw(spr)
        sdraw.rounded_rectangle((0,0,cw,ch), radius=radius, fill=blue)
        sdraw.text((pad_x, (ch - th)//2), t, font=font, fill=(255,255,255,255),
                   stroke_width=stroke_w, stroke_fill=(0,0,0,180))
        sprites.append((idx, spr, cur_x, (h_line - ch)//2))
        cur_x += cw + word_gap

    return base, sprites, w_img, h_img



def add_karaoke_subs_from_words(
    base_clip,
    words,
    video_size,
    base_ratio=0.085,       # tamaño de letra global (≈8.5% de H)
    bottom_safe_ratio=0.10, # margen inferior
    uppercase=True,
    window_words=(2, 4),    # 2–4 palabras por sublínea
    shift=0.06,             # desplaza el texto unos ms para que no se adelante al audio
):

    W, H = video_size
    fs   = int(max(28, min(round(H * base_ratio), 96)))
    font = _pick_bold_font(fs)
    bottom_safe = int(H * bottom_safe_ratio)

    # --- parámetros de layout: deben coincidir con _layout_line ---
    stroke_w = max(2, fs // 18)
    word_gap = int(fs * 0.35)
    pad_x    = int(fs * 0.45)
    pad_y    = int(fs * 0.25)
    max_block_w = 0.82   # sublínea no supera ~82% del ancho del vídeo

    # ---- limpieza: fuera tokens que sean solo puntuación / apóstrofes sueltos ----
    cleaned = []
    for w in words:
        txt = (w.get("text") or "").strip()
        if not txt: 
            continue
        if re.fullmatch(r"[.,;:!?¿¡\"'’…\-]+", txt):
            continue
        txt = re.sub(r"^[\"'’\-]+", "", txt)
        txt = re.sub(r"[\"'’\-]+$", "", txt)
        if not txt:
            continue
        cleaned.append({"text": txt, "start": float(w["start"]), "end": float(w["end"])})

    # ---- agrupa temporalmente por ventana 2–4 palabras (y por pausas/duración) ----
    min_w, max_w = window_words
    try:
        lines = group_words_into_lines(
            cleaned, min_words=min_w, max_words=max_w, max_gap=0.65, max_line_dur=4.8
        )
    except TypeError:
        # compat con versión antigua basada en caracteres
        lines = group_words_into_lines(
            cleaned, max_chars=26, max_gap=0.55, max_line_dur=5.0
        )

    # ---- helpers de medida y split por ancho real (chips) ----
    from PIL import Image, ImageDraw
    _dummy = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    _draw  = ImageDraw.Draw(_dummy)

    def chip_size(token: str):
        x0, y0, x1, y1 = _draw.textbbox((0, 0), token, font=font, stroke_width=stroke_w)
        tw, th = (x1 - x0), (y1 - y0)
        return (tw + pad_x * 2, th + pad_y * 2)  # (chip_w, chip_h)

    def split_by_width(line_words):
        """Divide una 'línea' en sublíneas para no superar max_block_w * W y 2–4 palabras."""
        toks = [w["text"].strip() for w in line_words if w["text"].strip()]
        if not toks:
            return []

        chips = [chip_size(t) for t in toks]  # [(cw, ch), ...]
        max_w_px = int(W * max_block_w)

        rows, cur, cur_w_px = [], [], 0
        for idx, (tok, (cw, _)) in enumerate(zip(toks, chips)):
            add = cw if not cur else cw + word_gap
            # si ya no cabe o llegamos al máximo de palabras, cerramos fila
            if cur and (len(cur) >= max_w or cur_w_px + add > max_w_px):
                rows.append(cur)
                cur, cur_w_px = [], 0
                add = cw
            # caso extremo: el chip de una sola palabra ya excede el ancho -> va sola
            if not cur and cw > max_w_px:
                rows.append([idx])
                continue
            cur.append(idx)
            cur_w_px += add
        if cur:
            rows.append(cur)

        # intenta prestar una palabra de la siguiente si la fila quedó corta (< min_w)
        j = 0
        while j < len(rows) - 1:
            if len(rows[j]) < min_w and rows[j+1]:
                cand = rows[j] + [rows[j+1][0]]
                # vuelve a medir
                width_cand = 0
                for k, ii in enumerate(cand):
                    cw, _ = chips[ii]
                    width_cand += cw if k == 0 else cw + word_gap
                if len(cand) <= max_w and width_cand <= max_w_px:
                    rows[j]     = cand
                    rows[j+1]   = rows[j+1][1:]
                    if not rows[j+1]:
                        rows.pop(j+1)
                        continue
            j += 1

        # mapea índices a palabras originales
        out = [[line_words[i] for i in ids] for ids in rows]
        return out

    overlays = []

    # ---- pinta cada línea -> en una o varias sublíneas según ancho ----
    for line in lines:
        sublines = split_by_width(line["words"])
        if not sublines:
            continue

        for sub in sublines:
            toks = [(w["text"].strip().upper() if uppercase else w["text"].strip())
                    for w in sub if w["text"].strip()]
            if not toks:
                continue

            # imágenes preparadas con el mismo layout que la medida
            base_img, sprites, w_img, h_img = _layout_line(toks, font, fs, W, H)

            rgba = np.array(base_img, dtype=np.uint8)
            rgb  = rgba[:, :, :3]
            a_u8 = rgba[:, :, 3]
            a_rgb = np.repeat(a_u8[..., None], 3, axis=2)

            # tiempos de la sublínea = de su 1ª a su última palabra
            sub_start = max(0.0, float(sub[0]["start"]) + shift)
            sub_end   = float(sub[-1]["end"]) + shift
            if sub_end <= sub_start:
                continue

            # posición segura (centrado) y clamped
            bx = max(0, min(W - w_img, (W - w_img)//2))
            by = max(0, min(H - h_img, H - h_img - bottom_safe))

            base = (ImageClip(rgb)
                    .with_mask(ImageClip(a_rgb).to_mask())
                    .with_start(sub_start)
                    .with_duration(sub_end - sub_start)
                    .with_position((bx, by)))
            overlays.append(base)

            # --- sprites (chip azul) palabra a palabra ---
            head_pad = 0.03   # adelanta levemente el chip
            tail_pad = 0.06   # retardo para que no “corte” seco
            min_dur  = 0.08

            for idx, spr_img, sx, sy in sprites:
                w = sub[idx]  # misma indexación que 'toks'
                if w["end"] <= w["start"]:
                    continue

                s_rgba = np.array(spr_img, dtype=np.uint8)
                s_rgb  = s_rgba[:, :, :3]
                s_a    = np.repeat(s_rgba[:, :, 3][..., None], 3, axis=2)

                w_start = max(0.0, float(w["start"]) + shift - head_pad)
                w_end   = float(w["end"]) + shift + tail_pad
                dur     = max(min_dur, w_end - w_start)

                spr = (ImageClip(s_rgb)
                       .with_mask(ImageClip(s_a).to_mask())
                       .with_start(w_start)
                       .with_duration(dur))

                # posición del sprite = misma base + offset del layout
                px = int(round(bx + sx))
                py = int(round(by + sy))

                # ---- clamp + crop para NUNCA salirte del frame (evita ValueError en máscaras) ----
                sw, sh = spr.size
                vis_w = max(0, min(sw, W - max(px, 0)))
                vis_h = max(0, min(sh, H - max(py, 0)))
                if vis_w <= 0 or vis_h <= 0:
                    continue  # completamente fuera

                if vis_w < sw or vis_h < sh:
                    try:
                        spr = spr.cropped(x1=0, y1=0, x2=vis_w, y2=vis_h)
                    except AttributeError:
                        from moviepy.video.fx.crop import crop
                        spr = crop(spr, x1=0, y1=0, x2=vis_w, y2=vis_h)
                    sw, sh = spr.size

                px = max(0, min(W - sw, px))
                py = max(0, min(H - sh, py))

                overlays.append(spr.with_position((px, py)))

    return CompositeVideoClip([base_clip, *overlays], size=base_clip.size)





def fit_cover(clip, target_w, target_h):
    """
    Ajusta un clip de vídeo al tamaño destino con efecto 'cover':
    rellena todo el marco, manteniendo proporción y recortando el exceso.
    """
    cw, ch = clip.w, clip.h
    if not cw or not ch:
        return clip  # fallback de seguridad
    
    try:

        # factor de escala para cubrir el área
        scale = max(target_w / cw, target_h / ch)

        # dimensiones escaladas
        new_w, new_h = int(math.ceil(cw * scale)), int(math.ceil(ch * scale))
        scaled = clip.resized(width=new_w)  # 👈 en 2.x es resized()

        # calcular recorte centrado
        x1 = max(0, (new_w - target_w) // 2)
        y1 = max(0, (new_h - target_h) // 2)

        return (
            scaled.cropped(x1=x1, y1=y1, x2=x1 + target_w, y2=y1 + target_h)
                .with_duration(clip.duration)
        )
        
    except Exception:
        ar_clip = cw / ch
        ar_target = target_w / target_h
        if abs(ar_clip - ar_target) < 0.01 and cw >= target_w and ch >= target_h:
            x1 = (cw - target_w) // 2
            y1 = (ch - target_h) // 2
            return clip.cropped(x1=x1, y1=y1, x2=x1+target_w, y2=y1+target_h).with_duration(clip.duration)
        raise


def pexels_candidates(query, api_key, per_page=80,
                      min_width=1280, min_dur=3, max_dur=30,
                      aspect_min=1.3, debug=False,
                      relevance_tokens=None, ctx_text=None):
    """
    Devuelve lista de dicts:
      {'w','h','dur','link','slug','desc','tok_score','sim','score2'}
    - tok_score = solape de tokens del slug con tokens relevantes de la frase
    - desc      = texto que usamos para embeddings (slug + query)
    """
    if not api_key:
        raise RuntimeError("Falta PEXELS_API_KEY")

    def slug_tokens(v):
        url = (v.get("url") or "").strip()
        if not url:
            return []
        seg = url.strip("/").split("/")[-1]
        seg = re.sub(r"-\d+$", "", seg)
        print(f"[PEXELS QUERY] {query}")
        print(f"[PEXELS URL] {url}")
        return tokenize_en(seg)  # usa tu tokenize_en


    def _pass(per_page, min_w, a_min, use_dur=True):
        url = "https://api.pexels.com/videos/search"
        params = {"query": query, "per_page": int(per_page)}
        if use_dur:
            params["min_duration"] = int(min_dur)
            params["max_duration"] = int(max_dur)
        r = requests.get(url, headers={"Authorization": api_key}, params=params, timeout=30)
        r.raise_for_status()
        videos = r.json().get("videos", []) or []
        cand = []
        for v in videos:
            dur = int(v.get("duration") or 0)
            files = sorted(v.get("video_files", []), key=lambda f: (f.get("width") or 0), reverse=True)
            sl_toks = slug_tokens(v)
            tok_score = len(set(sl_toks) & set(relevance_tokens or []))
            for f in files:
                w = int(f.get("width") or 0)
                h = int(f.get("height") or 0)
                link = f.get("link")
                if not link or not link.endswith(".mp4") or h <= 0:
                    continue
                if use_dur and (dur < min_dur or dur > max_dur):
                    continue
                if w < min_w:
                    continue
                if (w / h) < float(a_min):
                    continue
                desc = f"{' '.join(sl_toks)} | {query} | {ctx_text or ''}"
                cand.append({"w": w, "h": h, "dur": dur, "link": link,
                             "slug": "-".join(sl_toks), "desc": desc,
                             "tok_score": tok_score, "sim": 0.0, "score2": 0.0})
                break
        return cand

    cands = _pass(per_page, max(1280, min_width), aspect_min, True)
    if debug: print(f"[PEXELS] '{query}' P1={len(cands)}")
    if not cands:
        cands = _pass(per_page, 1024, 1.3, True)
        if debug: print(f"[PEXELS] '{query}' P2={len(cands)}")
    if not cands:
        cands = _pass(per_page, 800, 1.25, False)
        if debug: print(f"[PEXELS] '{query}' P3={len(cands)}")
    return cands



def translate_text_deepl_full(text: str, target_lang="EN"):
    """
    Traduce un bloque completo (mejor contexto para tokens).
    Usa los mismos endpoints que translate_phrases_deepl.
    """
    api_key = get_nested("deepl.api_key")
    if not api_key:
        dbg("[DeepL] Sin clave, contexto EN = texto ES.")
        return text

    endpoints = ["https://api.deepl.com/v2/translate", "https://api-free.deepl.com/v2/translate"]
    last_err = None
    for ep in endpoints:
        try:
            r = requests.post(ep, data={"auth_key": api_key, "text": text, "target_lang": target_lang}, timeout=15)
            if r.status_code == 200:
                tr = r.json().get("translations", [{}])[0].get("text", "").strip()
                dbg(f"[DeepL][full] OK via {ep.split('//')[1]}")
                return tr or text
            else:
                last_err = f"HTTP {r.status_code} {r.text[:120]}"
                dbg(f"[DeepL][full] FAIL {ep.split('//')[1]} :: {last_err}")
        except Exception as e:
            last_err = str(e)
            dbg(f"[DeepL][full] EXC  {ep.split('//')[1]} :: {last_err}")
    dbg(f"[DeepL][full] FALLBACK ES por error: {last_err}")
    return text


def translate_phrases_deepl(phrases, target_lang="EN", debug=None):
    """
    Traduce con DeepL si hay clave. Prueba Pro y Free. Loguea estado.
    """
    if debug is None:
        debug = DEBUG

    api_key = get_nested("deepl.api_key")
    if not api_key:
        if debug: print("[DeepL] Sin DEEPL_API_KEY → no se traduce.")
        return phrases

    # probamos ambos endpoints
    endpoints = ["https://api.deepl.com/v2/translate", "https://api-free.deepl.com/v2/translate"]

    out = []
    for ph in phrases:
        src = ph.strip()
        translated = None
        last = None
        for ep in endpoints:
            try:
                r = requests.post(
                    ep,
                    data={"auth_key": api_key, "text": src, "target_lang": target_lang},
                    timeout=15
                )
                if r.status_code == 200:
                    tr = r.json().get("translations", [{}])[0].get("text", "").strip()
                    translated = tr or src
                    if debug: print(f"[DeepL] OK via {ep.split('//')[1]} :: '{src}' → '{translated}'")
                    break
                else:
                    last = f"HTTP {r.status_code} {r.text[:120]}"
                    if debug: print(f"[DeepL] FAIL {ep.split('//')[1]} :: {last}")
            except Exception as e:
                last = f"EXC {type(e).__name__}: {e}"
                if debug: print(f"[DeepL] EXC  {ep.split('//')[1]} :: {last}")
        if translated is None and debug:
            print(f"[DeepL] FALLBACK (sin traducir): '{src}'  Motivo: {last}")
        out.append(translated or src)

    if debug:
        print("[DeepL] ES →", phrases)
        print("[DeepL] EN →", out)
    return out



def extract_keyphrases_rake(text, stopwords=STOPWORDS, max_words=3, top_k=10):
    tokens = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", text, flags=re.UNICODE)
    lowers = [t.lower() for t in tokens]
    seq = []
    for t in lowers:
        if (t in stopwords) or len(t) <= 2:
            seq.append("|")
        else:
            seq.append(t)
    raw_phrases = " ".join(seq).split("|")
    cand_phrases = []
    for ph in raw_phrases:
        words = [w for w in ph.split() if w]
        if 1 <= len(words) <= max_words:
            cand_phrases.append(words)
    if not cand_phrases:
        return []
    freqs = Counter(); degree = Counter()
    for words in cand_phrases:
        uniq = set(words)
        for w in words:
            freqs[w] += 1
            degree[w] += len(uniq)
    word_score = {w: degree[w] / freqs[w] for w in freqs}
    phrase_scores = []
    for words in cand_phrases:
        phrase_scores.append((" ".join(words), sum(word_score.get(w, 0.0) for w in words)))
    ranked = sorted(phrase_scores, key=lambda kv: (-kv[1], -len(kv[0])))
    out, seen = [], set()
    for ph, _ in ranked:
        if ph not in seen:
            out.append(ph); seen.add(ph)
        if len(out) >= top_k:
            break
    return out


def rank_with_embeddings(cands, ctx_text, debug=False):
    """
    score2 = 0.65*sim(emb) + 0.25*tok_score_norm + 0.10*resol_norm
    Si no hay embeddings, usamos (tok_score, w, dur) como fallback.
    """
    if not cands:
        return cands

    vecs = embed_texts([ctx_text] + [c["desc"] for c in cands])
    if vecs is None:
        # fallback sin embeddings
        cands.sort(key=lambda c: (c["tok_score"], c["w"], c["dur"]), reverse=True)
        return cands

    ctx_vec = vecs[0]
    cand_vecs = vecs[1:]
    max_tok = max([c["tok_score"] for c in cands] + [1])
    max_w   = max([c["w"] for c in cands] + [1])

    for c, v in zip(cands, cand_vecs):
        sim = float(np.dot(ctx_vec, v))  # ya normalizados
        c["sim"] = sim
        c["score2"] = 0.65*sim + 0.25*(c["tok_score"]/max_tok) + 0.10*(c["w"]/max_w)

    cands.sort(key=lambda c: (c["score2"], c["sim"], c["w"], c["dur"]), reverse=True)
    if debug:
        tops = [(c["slug"], round(c["sim"],3), c["tok_score"]) for c in cands[:5]]
        print(f"[RANK] top: {tops}")
    return cands





def call_groq_visual_queries(text: str, ctx_text: str = "", n: int = 3, model: str = "llama-3.1-8b-instant") -> list[str]:
    """
    Genera queries visuales concretas (en inglés) para buscadores de vídeo.
    Usa Groq con más contexto (texto original + contexto narrativo).
    """

    if not GROQ_API_KEY:
        raise ValueError("⚠ No hay GROQ_API_KEY configurada en .env")

    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an assistant that converts abstract text into short, concrete, "
                    "visual search queries in English for stock video platforms like Pexels. "
                    "Avoid abstract or vague words. Always describe a clear visual scene "
                    "with people, objects, environments, or actions."
                )
            },
            {
                "role": "user",
                "content": f"""
                Original text: "{text}"
                Narrative context: "{ctx_text or text}"

                Generate {n} short, highly visual search queries (2–6 words).
                Example output: ["developer coding on laptop", "AI hologram interface", "team working in office"]

                Return ONLY a valid JSON list.
                """
            },
        ],
        "temperature": 0.7,
        "max_tokens": 200,
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    r = requests.post(url, headers=headers, json=payload, timeout=45)
    r.raise_for_status()
    data = r.json()
    response = data["choices"][0]["message"]["content"].strip()

    try:
        return json.loads(response)
    except Exception:
        return [line.strip("-• ") for line in response.splitlines() if line.strip()]





def gather_candidates_round_robin(
    queries, api_key, *,
    min_width=1280, aspect_min=1.3,
    min_dur=3, max_dur=30,
    pool_target=12, per_query_cap=3, per_page=80,
    debug=True, relevance_tokens=None, ctx_text=None
):
    """
    Recorre queries en varias pasadas; por cada query añade el mejor clip no repetido,
    ya re-ordenado por rank_with_embeddings(ctx_text). Devuelve (pool, dist_por_query)
    """

    pool = []
    taken_by_q = defaultdict(int)
    cache_cands = {}

    # 🔹 Paso 1: Expandir queries con Groq → queries visuales
    expanded_queries = []
    for q in queries:
        try:
            visual_qs = call_groq_visual_queries(q, ctx_text=ctx_text, n=2)
            if debug:
                print(f"[GROQ] '{q}' -> {visual_qs}")
            expanded_queries.extend(visual_qs)
        except Exception as e:
            if debug:
                print(f"[GROQ ERROR] {e} - usando query original '{q}'")
            expanded_queries.append(q)

    queries = expanded_queries or queries

    # 🔹 Paso 2: Usar las queries expandidas en el bucle
    for pass_idx in range(per_query_cap):
        any_added = False
        for q_idx, q in enumerate(expanded_queries):
            if taken_by_q[q] >= per_query_cap:
                continue

            if q not in cache_cands:
                print(f"[PEXELS QUERY] {q}")
                cc = pexels_candidates(
                    q, api_key, per_page=per_page,
                    min_width=min_width, min_dur=min_dur, max_dur=max_dur,
                    aspect_min=aspect_min, debug=debug,
                    relevance_tokens=relevance_tokens,
                    ctx_text=ctx_text
                )
                cc = rank_with_embeddings(cc, ctx_text or "", debug=debug)
                cache_cands[q] = cc
                if debug:
                    topscore = round(cc[0]["score2"], 3) if cc else 0
                    print(f"[POOL] q[{q_idx}] '{q}' -> {len(cc)} cand (topscore2={topscore})")

            # coge el mejor de esa query que aún no esté en pool
            added_here = 0
            for c in cache_cands[q]:
                if any(c["link"] == x["link"] for x in pool):
                    continue
                pool.append({**c, "q": q})
                taken_by_q[q] += 1
                added_here += 1
                if debug:
                    print(f"[POOL] + {c.get('slug') or 'no-slug'}  from '{q}'  (pass {pass_idx})  score2={round(c.get('score2',0),3)}")
                break

            if added_here:
                any_added = True
            if len(pool) >= pool_target:
                break

        if len(pool) >= pool_target or not any_added:
            break

    if debug:
        print(f"[POOL] total={len(pool)} · dist={dict(taken_by_q)}")

    return pool, dict(taken_by_q)



def build_queries_generic(text: str, top_k=8):
    """Queries solo a partir del chunk (spaCy si existe, si no RAKE) + variantes + DeepL opcional."""
    phrases = []
    # spaCy (si está)
    try:

        try:
            nlp = spacy.load("es_core_news_sm")
        except Exception:
            nlp = spacy.blank("es")
        doc = nlp(text)
        noun_chunks = []
        if hasattr(doc, "noun_chunks"):
            noun_chunks += [nc.text.strip() for nc in doc.noun_chunks]
        ents = [e.text.strip() for e in getattr(doc, "ents", []) if getattr(e, "label_", "") in ("LOC","GPE","FAC","ORG","PERSON","NORP")]
        pos_chunks = []
        buf=[]
        for tok in doc:
            if getattr(tok, "pos_", "") in ("NOUN","PROPN","ADJ") and len(tok.text) > 2:
                buf.append(tok.text)
            else:
                if buf:
                    pos_chunks.append(" ".join(buf)); buf=[]
        if buf: pos_chunks.append(" ".join(buf))
        phrases = [*noun_chunks, *ents, *pos_chunks]
    except Exception as e:
        dbg(f"[spaCy] no disponible: {e}")

    # RAKE de respaldo
    phrases += extract_keyphrases_rake(text, top_k=top_k, max_words=3)

    # limpieza/dedup
    clean, seen = [], set()
    for ph in phrases:
        ph = re.sub(r"\s+", " ", ph).strip()
        if 1 <= len(ph) <= 60 and ph.lower() not in seen:
            clean.append(ph); seen.add(ph.lower())
    if not clean:
        clean = [" ".join(text.split()[:4])]

    dbg(f"[Queries] base ES: {clean}")

    # Traducción opcional
    en = translate_phrases_deepl(clean, target_lang="EN", debug=DEBUG)

    # Variantes
    queries, seenq = [], set()
    for es, enq in zip(clean, en):
        for q in (es, enq, f"{es} b-roll", f"{enq} b-roll", f"{es} cinematic", f"{enq} cinematic"):
            q = (q or "").strip()
            if q and q.lower() not in seenq:
                queries.append(q); seenq.add(q.lower())
        if len(queries) >= 24:
            break

    dbg(f"[Queries] finales: {queries}")
    return queries


def _plural_variants(word: str):
    w = word.lower()
    variants = {w}
    # plural -> singular
    if w.endswith("ies") and len(w) > 3: variants.add(w[:-3] + "y")
    elif re.search(r"(s|x|z|ch|sh)es$", w): variants.add(re.sub(r"es$", "", w))
    elif w.endswith("s") and not w.endswith("ss"): variants.add(w[:-1])
    # singular -> plural
    if re.search(r"(s|x|z|ch|sh)$", w): variants.add(w + "es")
    elif w.endswith("y") and len(w) > 1 and w[-2] not in "aeiou": variants.add(w[:-1] + "ies")
    else: variants.add(w + "s")
    return list(variants)

def _ngrams(seq, n):
    return [" ".join(seq[i:i+n]) for i in range(len(seq)-n+1)]

def build_queries_for_sentence(sentence_es: str, top_k=8, debug=False):
    """
    Sin sinónimos manuales:
    - Traduce la frase completa con DeepL
    - RAKE + n-grams + (opcional) noun_chunks de spaCy EN
    - Variantes (sing/plur + b-roll/cinematic)
    """
    ctx_en = translate_text_deepl_full(sentence_es, target_lang="EN")
    ctx_tokens = tokenize_en(ctx_en)

    # candidatos base
    phrases = []
    try:
        try:
            nlp_en = spacy.load("en_core_web_sm")
        except Exception:
            nlp_en = spacy.blank("en")
        doc = nlp_en(ctx_en)
        if doc.has_annotation("DEP"):
            phrases += [nc.text.strip().lower() for nc in getattr(doc, "noun_chunks", [])]
        if doc.has_annotation("TAG"):
            buf=[]
            for tok in doc:
                if tok.is_alpha and len(tok.text) > 2 and tok.pos_ in ("NOUN","PROPN","ADJ"):
                    buf.append(tok.lemma_.lower())
                else:
                    if buf: phrases.append(" ".join(buf)); buf=[]
            if buf: phrases.append(" ".join(buf))
    except Exception as e:
        if debug: print(f"[spaCy EN] no disponible: {e}")

    phrases += extract_keyphrases_rake(ctx_en, top_k=top_k*2, max_words=3)
    for n in (3,2):
        phrases += _ngrams(ctx_tokens, n)

    # filtro simple para quitar frases con verbos/funcionales típicos
    banned = {
        "is","are","was","were","be","being","been","have","has","had",
        "do","does","did","make","makes","made","live","lives","lived",
        "mark","marks","marked","resist","resists","resisted","disappear",
        "disappears","disappeared","still","then","they","very","just"
    }
    counts={}
    for ph in phrases:
        ph = re.sub(r"\s+"," ",ph).strip().lower()
        if not ph or len(ph)<3: continue
        toks = ph.split()
        if any(t in banned for t in toks):  # evita "lives federico", etc.
            continue
        counts[ph] = counts.get(ph,0)+1

    ranked = sorted(counts.items(), key=lambda kv:(len(kv[0].split()), kv[1], len(kv[0])), reverse=True)
    base = [p for p,_ in ranked][:max(10, top_k*2)]

    # no pluralizar nombres propios (heurística: palabras capitalizadas en ctx_en)
    proper = {m.group(0).lower() for m in re.finditer(r"\b[A-Z][a-z]+\b", ctx_en)}

    variants, seen = [], set()
    for q in base:
        words = q.split()
        last = words[-1]
        # si el último es nombre propio, no pluralizamos
        last_is_propn = last in proper
        forms = [last] if last_is_propn else _plural_variants(last)
        for vlast in forms:
            cand = " ".join(words[:-1] + [vlast])
            for v in (cand, f"{cand} b-roll", f"{cand} cinematic"):
                v = v.strip()
                if v and v.lower() not in seen:
                    variants.append(v); seen.add(v.lower())
        if len(variants) >= 36: break

    final, seen2 = [], set()
    for q in variants + base:
        if q and q.lower() not in seen2:
            final.append(q); seen2.add(q.lower())
        if len(final) >= 36: break

    if debug:
        print(f"[CTX EN] {ctx_en}")
        print(f"[CTX TOKENS] {ctx_tokens[:20]}{'...' if len(ctx_tokens)>20 else ''}")
        print(f"[Q_SENT] queries({len(final)}): {final[:24]}{'...' if len(final)>24 else ''}")
    return final, ctx_en, ctx_tokens




def download_mp4_checked(link, out_path, timeout=120, size_tolerance=0.95, debug=False):
    """
    Descarga un MP4 y verifica (si hay Content-Length) que no esté truncado.
    """
    if debug:
        dbg(f"[DL] GET {link}")
    with requests.get(link, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        expected = int(r.headers.get("Content-Length", "0") or 0)
        tmp = f"{out_path}.part"
        with open(tmp, "wb") as f:
            for ch in r.iter_content(1024 * 256):
                if ch:
                    f.write(ch)
    # verifica tamaño esperado si el servidor lo envió
    actual = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if expected and actual < expected * size_tolerance:
        try: os.remove(tmp)
        except Exception: pass
        raise RuntimeError(f"Descarga incompleta ({actual}/{expected} bytes)")

    # renombra a definitivo (descarga atómica)
    os.replace(tmp, out_path)
    if debug:
        dbg(f"[DL] OK -> {out_path} ({actual or os.path.getsize(out_path)} bytes)")
    return out_path


# ---------- Pexels ----------
def pexels_download(query, api_key, out_path, pick_idx=0, per_page=24,
                    min_width=1280, min_dur=3, max_dur=30, ban_links=None):
    if not api_key:
        raise RuntimeError("Falta PEXELS_API_KEY")
    ban_links = ban_links or set()

    url = "https://api.pexels.com/videos/search"
    params = {
        "query": query,
        "per_page": per_page,
        "min_duration": int(min_dur),
        "max_duration": int(max_dur),
    }
    r = requests.get(url, headers={"Authorization": api_key}, params=params, timeout=30)
    r.raise_for_status()
    videos = r.json().get("videos", []) or []
    if not videos:
        raise RuntimeError(f"No hay videos para: {query}")

    # candidatos: mejor archivo de cada vídeo
    candidates = []
    for v in videos:
        dur = int(v.get("duration") or 0)
        files = sorted(v.get("video_files", []), key=lambda f: (f.get("width") or 0), reverse=True)
        for f in files:
            link = f.get("link")
            w = int(f.get("width") or 0)
            if not link or not link.endswith(".mp4"):
                continue
            if dur < min_dur or dur > max_dur or w < min_width:
                continue
            if link in ban_links:
                continue
            candidates.append((w, dur, link))
            break

    # si no hay suficientes, relaja resolución y ban_links
    if not candidates:
        for v in videos:
            files = sorted(v.get("video_files", []), key=lambda f: (f.get("width") or 0), reverse=True)
            for f in files:
                link = f.get("link")
                if link and link.endswith(".mp4") and link not in ban_links:
                    candidates.append((int(f.get("width") or 0), int(v.get("duration") or 0), link))
                    break

    if not candidates:
        raise RuntimeError(f"No hay MP4 útiles para: {query}")

    pick = candidates[pick_idx % len(candidates)][2]

    with requests.get(pick, stream=True, timeout=120) as mp4:
        mp4.raise_for_status()
        with open(out_path, "wb") as f:
            for ch in mp4.iter_content(1024 * 256):
                if ch:
                    f.write(ch)
    return str(out_path), pick


# ---------- TTS ----------
def tts_elevenlabs(text, out_audio_mp3, api_key, voice_id, model_id="eleven_multilingual_v2",
                   stability=0.35, similarity=0.8, style=0.0, speaker_boost=True):
    """
    Genera narración con ElevenLabs y guarda MP3.
    Requiere: api_key, voice_id válidos.
    """
    if not api_key:
        raise RuntimeError("Falta ELEVENLABS_API_KEY")
    if not voice_id:
        raise RuntimeError("Falta ELEVENLABS_VOICE_ID")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "accept": "audio/mpeg",
        "content-type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": float(stability),
            "similarity_boost": float(similarity),
            "style": float(style),
            "use_speaker_boost": bool(speaker_boost),
            "speaking_rate": 1.5
        },
    }

    with requests.post(url, headers=headers, json=payload, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out_audio_mp3, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)
    return out_audio_mp3


# ---------- SRT ----------
def to_tc(x):
    h=int(x//3600); m=int((x%3600)//60); s=int(x%60); ms=int((x-int(x))*1000)
    return pysrt.SubRipTime(hours=h, minutes=m, seconds=s, milliseconds=ms)

def write_srt(segments, path):
    subs = pysrt.SubRipFile()
    for i,(txt,a,b) in enumerate(segments,1):
        subs.append(pysrt.SubRipItem(index=i,start=to_tc(a),end=to_tc(b),text=txt))
    subs.save(path, encoding="utf-8")

def sentences(text):
    parts = re.split(r'(?<=[\.\!\?])\s+', text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [text.strip()]

def build_segments(paragraphs, scene_times):
    segs=[]
    for p,(a,b) in zip(paragraphs, scene_times):
        ss = sentences(p)
        total_chars = sum(len(s) for s in ss) or 1
        t=a
        for s in ss:
            d=max(1.0, (b-a)*(len(s)/total_chars))
            segs.append((s, t, min(b, t+d))); t+=d
        if segs and segs[-1][2] < b: segs[-1]=(segs[-1][0], segs[-1][1], b)
    return segs

# ---------- subtítulos quemados ----------
def textimg(
    txt, W, H,
    max_lines=2,
    base_ratio=0.10,          # ~10% de la altura del vídeo
    margin_ratio=0.03,        # margen vertical interno
    highlight="last",         # "none" | "last" | int (n últimas) | set/list de palabras a resaltar
    blue_rgba=(0, 0, 0, 200),  # color del chip negro (RGBA)
    uppercase=True,
):
    """
    Subtítulo con estilo moderno: MAYÚSCULAS + trazo + 'chip' azul tras palabras destacadas.
    Devuelve una PIL.Image RGBA.
    """
    import re
    from PIL import Image, ImageDraw, ImageFont

    # ===== 1) Fuente negrita =====
    fs = int(max(32, min(round(H * base_ratio), 96)))
    margin = int(H * margin_ratio)
    font = None
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",  # Segoe UI Semibold
        "C:/Windows/Fonts/seguisb.ttf",   # Segoe UI Semibold alt
    ]
    for fp in font_candidates:
        try:
            font = ImageFont.truetype(fp, fs)
            break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()

    # ===== 2) Texto y tokens =====
    raw = txt.strip()
    text = raw.upper() if uppercase else raw
    tokens = re.findall(r"\S+", text)

    # ===== 3) Ajuste a líneas =====
    dummy = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    draw  = ImageDraw.Draw(dummy)
    stroke_w = max(2, fs // 18)
    word_gap = int(fs * 0.25)
    pad_x    = int(fs * 0.45)
    pad_y    = int(fs * 0.25)
    max_w    = int(W * 0.88)
    radius   = int(fs * 0.35)

    def _measure_word(w):
        x0, y0, x1, y1 = draw.textbbox((0, 0), w, font=font, stroke_width=stroke_w)
        tw, th = (x1 - x0), (y1 - y0)
        return tw, th

    measured = [(_measure_word(w)) for w in tokens]
    line_sets = []
    cur = []
    cur_w = 0
    for idx, w in enumerate(tokens):
        tw, th = measured[idx]
        cw = tw + pad_x * 2
        add = (cw if not cur else (cw + word_gap))
        if cur and (cur_w + add) > max_w and len(line_sets) + 1 < max_lines:
            line_sets.append(cur)
            cur, cur_w = [], 0
            add = cw
        cur.append(idx)
        cur_w += add
    if cur:
        line_sets.append(cur)

    # ===== 4) Qué palabras van destacadas =====
    highlight_set = set()
    if highlight == "last" and tokens:
        highlight_set.add(len(tokens) - 1)
    elif isinstance(highlight, int) and highlight > 0:
        for k in range(1, min(highlight, len(tokens)) + 1):
            highlight_set.add(len(tokens) - k)
    elif isinstance(highlight, (list, set, tuple)):
        want = {str(w).upper() for w in highlight}
        for i, w in enumerate(tokens):
            if w.upper().strip() in want:
                highlight_set.add(i)

    # ===== 5) Dimensiones del lienzo =====
    line_heights, line_widths = [], []
    for ids in line_sets:
        tot = 0
        h_line = 0
        for j, idx in enumerate(ids):
            tw, th = measured[idx]
            cw = tw + pad_x * 2
            tot += (cw if j == 0 else cw + word_gap)
            h_line = max(h_line, th + pad_y * 2)
        line_widths.append(tot)
        line_heights.append(h_line)

    w_img = min(max(line_widths and max(line_widths) or 0, int(W * 0.5)), max_w)
    h_img = sum(line_heights) + margin * 2 + (len(line_heights) - 1) * int(fs * 0.2)
    img   = Image.new("RGBA", (w_img, h_img), (0, 0, 0, 0))
    draw  = ImageDraw.Draw(img)

    # ===== 6) Pintado =====
    cur_y = margin
    for row, ids in enumerate(line_sets):
        row_h  = line_heights[row]
        row_w  = line_widths[row]
        cur_x  = (w_img - row_w) // 2  # centrado
        for j, idx in enumerate(ids):
            word = tokens[idx]
            tw, th = measured[idx]
            cw = tw + pad_x * 2
            # coordenadas base del texto
            txt_x = cur_x + pad_x
            txt_y = cur_y + (row_h - th) // 2

            if idx in highlight_set:
                # rect azul detrás del texto
                rect_left   = cur_x
                rect_top    = txt_y - pad_y
                rect_right  = cur_x + cw
                rect_bottom = txt_y + th + pad_y
                draw.rounded_rectangle(
                    (rect_left, rect_top, rect_right, rect_bottom),
                    radius=radius,
                    fill=blue_rgba
                )

            # texto en blanco con borde negro
            draw.text(
                (txt_x, txt_y),
                word,
                font=font,
                fill=(255, 255, 255, 255),
                stroke_width=stroke_w,
                stroke_fill=(0, 0, 0, 180),
            )
            cur_x += cw + word_gap
        cur_y += row_h + int(fs * 0.2)

    return img




def add_burned_subs(base_clip, segments, video_size):
    W, H = video_size
    overlays = []
    bottom_safe = int(H * 0.10)  # margen inferior de seguridad

    for txt, a, b in segments:
        # 1) Imagen RGBA del subtítulo (usa tu textimg con fondo claro translúcido)
        img = textimg(txt, W, H, max_lines=2, base_ratio=0.10)  # tamaño grande (~10% de H)

        # 2) Separa RGB y ALPHA
        rgba = np.array(img, dtype=np.uint8)              # (h, w, 4)
        rgb = rgba[:, :, :3]                              # (h, w, 3) uint8 0..255
        alpha_u8 = rgba[:, :, 3]                          # (h, w) uint8 0..255

        # 3) Convierte el alfa 2D a 3 canales para que .to_mask() no falle
        alpha_rgb = np.repeat(alpha_u8[..., None], 3, axis=2)  # (h, w, 3) uint8

        # 4) Crea el clip de texto y su máscara
        txt_rgb = (
            ImageClip(rgb)
            .with_duration(b - a)
            .with_start(a)
            .with_position(("center", H - img.height - bottom_safe))
        )

        # .to_mask() convertirá el 0..255 en 0..1 internamente
        alpha_clip = (
            ImageClip(alpha_rgb)
            .to_mask()                      # -> MaskClip (is_mask=True)
            .with_duration(b - a)
            .with_start(a)
        )

        # 5) Aplica la máscara al texto y acumula
        txt_rgb = txt_rgb.with_mask(alpha_clip)
        overlays.append(txt_rgb)

    # 6) Devuelve el compuesto (base + subtítulos)
    return CompositeVideoClip([base_clip, *overlays], size=base_clip.size)




# ---------- video helpers ----------
def fit_clip_duration(clip, target):
    """Recorta si sobra. Si falta, NO congela: eso se hace al final de la escena."""
    dur = clip.duration or 0.0
    if dur > target + 1e-3:
        return clip.subclipped(0, target)
    return clip  # no rellenes aquí


def safe_videoclip(path, max_w=1920, max_h=1080, fps=None, force_transcode=True):
    path = str(path)
    # Sonda inicial
    probe = VideoFileClip(path, audio=False)
    w, h = probe.w, probe.h
    probe.close()

    # Decide si transcodificar SIEMPRE (recomendado en Windows) o por tamaño
    need_transcode = force_transcode or (w > max_w or h > max_h)

    if not need_transcode:
        clip = VideoFileClip(path, audio=False)
        clip.is_mask = False
        if fps: clip = clip.with_fps(fps)
        return clip

    # Re-encode → libx264 + yuv420p (muy compatible)
    ar_src = w / h
    ar_dst = max_w / max_h
    if ar_src >= ar_dst:
        vf = f"scale={max_w}:-2,format=yuv420p"
    else:
        vf = f"scale=-2:{max_h},format=yuv420p"

    reduced = str(Path(path).with_name(Path(path).stem + f"_{max_h}_yuv420p.mp4"))
    if not Path(reduced).exists():
        cmd = [
            "ffmpeg","-y","-loglevel","error",
            "-i", path,
            "-vf", vf,
            "-c:v","libx264","-preset","veryfast","-crf","20",
            "-an", reduced
        ]
        subprocess.run(cmd, check=True)

    clip = VideoFileClip(reduced, audio=False)
    clip.is_mask = False
    if fps: clip = clip.with_fps(fps)
    print(f"[SAFE] {Path(path).name} → {clip.w}x{clip.h} yuv420p (forzado)")
    return clip

    
def smart_freeze_from_clip(clip, duration, W, H):
    """
    Devuelve un freeze-frame con un leve push-in (zoom 2%/s) para evitar 'congelados' feos.
    """
    t_last = max((clip.duration or 0) - 1e-3, 0.0)
    fr = clip.to_ImageClip(t=t_last).with_duration(duration).without_audio()
    # sutil zoom-in y ligero pan vertical (funciones dependientes de t)
    try:
        fr = fr.resized(lambda t: 1.0 + 0.02 * t) \
               .with_position(lambda t: ("center", int(-8 * t)))  # sube ~8px/s
    except Exception:
        # si la versión no soporta funciones en resized/position, deja el freeze simple
        pass
    # garantiza que el freeze cubre el lienzo (por si el frame no es exacto al W,H)
    if fr.w != W or fr.h != H:
        fr = fit_cover(fr, W, H)
    return fr

def clear_cache():
    for item in CACHE_DIR.iterdir():
        if item.is_file():
            item.unlink()  # elimina archivo
        elif item.is_dir() and item.name != "whisper_models":
            shutil.rmtree(item)  # elimina carpeta entera

    print("🧹 Caché limpiada (excepto whisper_models)")

# ---------- main ----------
def main():
    clear_cache()
    
    FADES = False         # ← nuevo flag
    transition = 0.15      # ← fuerza cortes duros
    opening_fade = 0.15 
    
    # ------ config / args ------
    global DEBUG
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default=str(DEFAULT_INFILE),help="Ruta del texto (UTF-8). Por defecto usa post/<hoy>/video_text.txt")
    ap.add_argument("--out", default=str(OUT_DIR / "video_post.mp4"))
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--transition", type=float, default=0.5)
    ap.add_argument("--burn-subs", type=int, default=1)
    ap.add_argument("--debug", type=int, choices=[0, 1], default=0)
    args = ap.parse_args()
    
      # Aquí ya puedes usar args.infile aunque no lo pases manualmente
    infile_path = Path(args.infile)
    if not infile_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {infile_path}")

    print(f"📄 Usando texto: {infile_path}")
    print(f"🎬 Generando video en: {args.out}")

    DEBUG = bool(args.debug)
    print(f"[INFO] DEBUG = {DEBUG}")

    if not PEXELS_API_KEY:
        raise SystemExit("Falta PEXELS_API_KEY en entorno o .env")

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    CACHE_DIR.mkdir(exist_ok=True, parents=True)

    text_in = Path(args.infile).read_text(encoding="utf-8").strip()
    paragraphs = split_paragraphs(text_in)
    if not paragraphs:
        raise SystemExit("El archivo de entrada está vacío.")

    W, H = args.width, args.height
    print(f"[INFO] Resolución destino: {W}x{H}")
    print(f"[INFO] Párrafos: {len(paragraphs)}")
    audio_mp3 = str(OUT_DIR / "voice.mp3")

    
    audio_ok = False
    for idx, api_key in enumerate(ELEVENLABS_API_KEYS, start=1):
        if not api_key:
            continue

        try:
            print(f"🔄 Intentando con API Key #{idx}...")
            tts_elevenlabs(
                text_in,
                audio_mp3,
                api_key=api_key,
                voice_id=ELEVENLABS_VOICE_ID,
                model_id=ELEVENLABS_MODEL_ID,
                stability=ELEVENLABS_STABILITY,
                similarity=ELEVENLABS_SIMILARITY,
                style=ELEVENLABS_STYLE,
                speaker_boost=ELEVENLABS_SPEAKER_BOOST,
            )

            if os.path.exists(audio_mp3) and os.path.getsize(audio_mp3) > 0:
                print(f"✅ Generado con la API Key #{idx}")
                audio_ok = True
                break   # 👈 salir del bucle, no del main()

        except Exception as e:
            print(f"⚠️ Error con API Key #{idx}: {e}")
            continue

    if not audio_ok:
        raise RuntimeError("❌ Ninguna API key funcionó para generar el audio")
    
            
    words = whisper_words_faster(audio_mp3, language="es", model_size="small")
    audio_clip = AudioFileClip(audio_mp3)

    # ------ tiempos por párrafo (una sola vez) ------
    total_chars = sum(len(p) for p in paragraphs) or 1
    t = 0.0
    scene_times = []
    for p in paragraphs:
        d = max(2.5, audio_clip.duration * (len(p) / total_chars))
        end = min(audio_clip.duration, t + d)
        scene_times.append((t, end))
        t = end
    # asegura que el último termina justo en la duración del audio
    if scene_times:
        a_last, _ = scene_times[-1]
        scene_times[-1] = (a_last, audio_clip.duration)

    # frases (segmentos) con tiempos absolutos (a,b)
    segments = build_segments(paragraphs, scene_times)
    print(f"[INFO] Segmentos: {len(segments)} · Audio: {audio_clip.duration:.2f}s")

    # ------ montaje por escena ------
    clips = []
    used_links = set()
    NEG_TOKENS = {"car","cars","traffic","highway","road","driving","street","city","urban","desert","timelapse","audi"}

    for i, (sent_text, a, b) in enumerate(segments, start=1):
        dur = max(0.1, b - a)
        # nº razonable de cortes y duración objetivo por corte
        n_parts = max(1, min(3, round(dur / 3.0)))
        micro_target = max(1.8, min(4.0, dur / n_parts))

        print(f"\n[SCENE {i}] dur={dur:.2f}s · micro_target≈{micro_target:.2f}s")
        if DEBUG:
            print(f"[SCENE {i}] Texto: {sent_text}")

        # --- queries por frase ---
        queries, ctx_en, ctx_tokens = build_queries_for_sentence(sent_text, top_k=8, debug=DEBUG)

        # --- ventana de duración aceptable para Pexels ---
        min_d = max(3, int(micro_target * 0.7))
        max_d = min(60, int(micro_target * 1.8))
        pool_target = max(n_parts * 2, 6)

        pool, dist = gather_candidates_round_robin(
            queries, PEXELS_API_KEY,
            min_width=1280, 
            aspect_min=1.4,
            min_dur=min_d, max_dur=max_d,
            pool_target=pool_target, per_query_cap=2,
            per_page=80, debug=DEBUG,
            relevance_tokens=ctx_tokens, ctx_text=ctx_en
        )

        # --- filtra repetidos y poco relacionados (roads/cars/etc.) ---
        pool = [
            c for c in pool
            if c["link"] not in used_links and not any(tok in (c.get("slug", "") or "") for tok in NEG_TOKENS)
        ]
        # si hay embeddings decentes, sube el umbral un poco
        if any(c.get("sim", 0.0) >= 0.45 for c in pool):
            pool = [c for c in pool if c.get("sim", 0.0) >= 0.40]

        pool.sort(
            key=lambda x: (x.get("score2", 0.0), x.get("sim", 0.0), x.get("w", 0), x.get("dur", 0)),
            reverse=True
        )

        if DEBUG:
            tops = [round(x.get('score2', 0.0), 3) for x in pool[:5]]
            print(f"[SCENE {i}] pool usable={len(pool)} · top score2={tops}")

        # --- compón la escena (t=0 relativo dentro de la escena) ---
        scene_parts, filled, idx_local = [], 0.0, 0
        # usa un fade de apertura pequeño para evitar negro (o pon 0.0 si no quieres fade)

        while filled < dur - 0.05 and pool:
            c = pool.pop(0)
            link, q_src = c["link"], c["q"]
            cache_mp4 = str(CACHE_DIR / f"px_{i:02d}_{idx_local}_{hashlib.sha1(link.encode()).hexdigest()[:8]}.mp4")
            idx_local += 1

            if not Path(cache_mp4).exists():
                try:
                    download_mp4_checked(link, cache_mp4, timeout=120, debug=DEBUG)
                    if DEBUG:
                        print(f"[DL] {os.path.basename(cache_mp4)} ← {link.split('/')[-1]}")
                except Exception as e:
                    print(f"[SCENE {i}][SKIP] download: {e}")
                    continue

            try:
                part_raw = safe_videoclip(cache_mp4, max_w=W, max_h=H, fps=args.fps, force_transcode=True)
            except Exception as e:
                print(f"[SCENE {i}][SKIP] corrupt: {e}")
                continue

            # encuadre tipo cover y recorte si sobra (NO rellenes aquí)
            part = fit_cover(part_raw, W, H)

            remaining = dur - filled
            need = min(micro_target, remaining)
            if (part.duration or 0.0) > need + 1e-3:
                part = part.subclipped(0, need)

            # descarta microclips demasiado cortos (evita parpadeos)
            if (part.duration or 0.0) < 0.06:
                continue
            
            part.is_mask = False

            # crossfade sólo si hay tiempo suficiente para solapar
            use_overlap = (len(scene_parts) > 0) and ((part.duration or 0.0) > transition + 0.1)

            if FADES and use_overlap:
                scene_parts[-1] = scene_parts[-1].with_effects([vfx.FadeOut(duration=transition)])
                start_time = max(filled - transition, 0.0)
                part = part.with_effects([vfx.FadeIn(duration=transition)]).with_start(start_time)
                filled += (part.duration or 0.0) - transition
            else:
                # corte duro, sin solape
                part = part.with_start(filled)
                filled += (part.duration or 0.0)


            scene_parts.append(part)
            used_links.add(link)

            if DEBUG:
                print(f"[SCENE {i}] +seg {part.duration:.2f}s from '{q_src}'  ({filled:.2f}/{dur:.2f})")

        # --- relleno final (único) con freeze-frame si falta colilla ---
        if filled < dur - 0.01 and scene_parts:
                leftover = dur - filled

                # 2º intento: búsqueda más relajada para cubrir el hueco
                more, _ = gather_candidates_round_robin(
                    queries, PEXELS_API_KEY,
                    min_width=960,              # baja un poco la exigencia
                    aspect_min=1.25,            # acepta algo menos panorámico
                    min_dur=max(2, int(leftover * 0.7)),
                    max_dur=min(20, int(leftover * 2.0)),
                    pool_target=6, per_query_cap=1, per_page=40,
                    debug=DEBUG,
                    relevance_tokens=ctx_tokens, ctx_text=ctx_en
                )
                more = [c for c in more if c["link"] not in used_links]
                more.sort(key=lambda x: (x.get("score2",0.0), x.get("sim",0.0), x.get("w",0)), reverse=True)

                added_extra = False
                if more:
                    c2 = more[0]
                    link2, q_src2 = c2["link"], c2["q"]
                    cache2 = str(CACHE_DIR / f"px_{i:02d}_extra_{hashlib.sha1(link2.encode()).hexdigest()[:8]}.mp4")
                    if not Path(cache2).exists():
                        try:
                            download_mp4_checked(link2, cache2, timeout=120, debug=DEBUG)
                            if DEBUG: print(f"[DL] {os.path.basename(cache2)} ← {link2.split('/')[-1]}")
                        except Exception as e:
                            if DEBUG: print(f"[SCENE {i}][SKIP extra] download: {e}")
                            cache2 = None
                    if cache2:
                        try:
                            extra_raw = safe_videoclip(cache2).without_audio().with_fps(args.fps)
                            extra = fit_cover(extra_raw, W, H)
                            # recorta o deja tal cual según lo que falte
                            if (extra.duration or 0.0) > leftover + 1e-3:
                                extra = extra.subclipped(0, leftover)

                            # si hay margen, solapamos con fade; si no, corte duro
                            if FADES and (scene_parts and (extra.duration or 0.0) > transition + 0.1):
                                scene_parts[-1] = scene_parts[-1].with_effects([vfx.FadeOut(duration=transition)])
                                start_time = max(filled - transition, 0.0)
                                extra = extra.with_effects([vfx.FadeIn(duration=transition)]).with_start(start_time)
                                filled += (extra.duration or 0.0) - transition
                            else:
                                extra = extra.with_start(filled)   # ← corte duro
                                filled += (extra.duration or 0.0)


                            scene_parts.append(extra)
                            used_links.add(link2)
                            added_extra = True
                            if DEBUG:
                                print(f"[SCENE {i}] +extra {extra.duration:.2f}s from '{q_src2}'  ({filled:.2f}/{dur:.2f})")
                        except Exception as e:
                            if DEBUG: print(f"[SCENE {i}][SKIP extra] corrupt: {e}")

                # si sigue faltando tiempo, usa un freeze con micro-movimiento
                if filled < dur - 0.01:
                    tail = smart_freeze_from_clip(scene_parts[-1], dur - filled, W, H).with_start(filled)
                    scene_parts.append(tail)
                    filled = dur

        # --- fallback si no hubo material ---
        if not scene_parts:
            scene = ColorClip(size=(W, H), color=(16, 16, 16)).with_duration(dur)
            print(f"[SCENE {i}] ⚠️ sin material, fallback color")
        else:
            bg = ColorClip(size=(W, H), color=(0, 0, 0)).with_duration(dur)
            scene = CompositeVideoClip(
                [bg, *scene_parts], size=(W, H), use_bgclip=True
            ).with_duration(dur)

        # fades sólo en apertura/cierre del vídeo (apertura corta para no ver negro)
        if i == 1 and opening_fade > 0:
            scene = scene.with_effects([vfx.FadeIn(duration=opening_fade)])
        if i == len(segments) and transition > 0:
            scene = scene.with_effects([vfx.FadeOut(duration=transition)])

        clips.append(scene)
        print(f"[SCENE {i}] ✅ Partes: {len(scene_parts)} · total={scene.duration:.2f}s")


    # ------ timeline final (secuencial) ------
    if not clips:
        raise SystemExit("No se generaron escenas.")

    timeline, t_acc = [], 0.0
    for sc in clips:
        if (sc.duration or 0) < 0.05:
            continue
        timeline.append(sc.with_start(t_acc))
        t_acc += sc.duration or 0.0

    if not timeline:
        raise SystemExit("Timeline vacío tras filtrar duraciones.")

    # 1) Construye el master desde timeline
    video = CompositeVideoClip(timeline, size=(W, H), use_bgclip=True).with_fps(args.fps)

    # 2) **Asegura** la narración (luego ya mezclas música)
    video = video.with_audio(audio_clip)

    # 🖼️ Imagen final (3s)
    logo_path = ASSETS_DIR / "final_logo.png"
    if logo_path.exists():
        outro_img = (
            ImageClip(str(logo_path))
            .resized(width=W)              # mantiene aspecto y ajusta por ancho
            .with_duration(3)
            .with_fps(args.fps)
        )
        outro_bg = ColorClip(size=(W, H), color=(16,16,16)).with_duration(3).with_fps(args.fps)
        outro = CompositeVideoClip(
            [outro_bg, outro_img.with_position(("center","center"))],
            size=(W, H),
            use_bgclip=True
        ).with_fps(args.fps)

        # empieza al terminar el principal
        outro = outro.with_start(video.duration)

        # 3) Encadena master + outro
        video = CompositeVideoClip([video, outro], size=(W, H), use_bgclip=True).with_fps(args.fps)

          
    # 🎵 Música de fondo
    bg_path = ASSETS_DIR / "audio_news.mp3"
    if bg_path.exists():
        bg_music = AudioFileClip(str(bg_path))
        # Ajusta volumen con el efecto oficial (no convierte a ndarray)
        bg_music = bg_music.with_volume_scaled(0.15)

        # Empareja duración al vídeo
        if bg_music.duration > (video.duration or 0):
            bg_music = bg_music.subclipped(0, video.duration)
        else:
            bg_music = bg_music.with_duration(video.duration)

        # Mezcla narración + música
        mixed = CompositeAudioClip([audio_clip, bg_music])
        video = video.with_audio(mixed)


    # ------ Subtítulos (SRT + quemado opcional) ------
    srt_path = str(OUT_DIR / "subs.srt")
    write_srt(segments, srt_path)

    if args.burn_subs == 1:
        video = add_karaoke_subs_from_words(video, words, (W, H),
                                    base_ratio=0.085, bottom_safe_ratio=0.10, uppercase=True)

    # ------ Export ------

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # helper: ¿está bloqueado?
    def _is_locked(p: Path) -> bool:
        try:
            # abrir lectura+escritura sin truncar
            with open(p, "r+b"):
                return False
        except FileNotFoundError:
            return False
        except PermissionError:
            return True
        except OSError:
            # algunos locks devuelven OSError genérico
            return True

    # quitar Read-only si lo tuviera
    if out_path.exists():
        try:
            os.chmod(out_path, stat.S_IWRITE)
        except Exception:
            pass

    # si sigue bloqueado, no escribas encima: usa nombre alternativo
    dest = out_path
    if out_path.exists() and _is_locked(out_path):
        ts = time.strftime("%Y%m%d-%H%M%S")
        dest = out_path.with_name(f"{out_path.stem}_{ts}{out_path.suffix}")
        print(f"[WARN] '{out_path}' parece bloqueado. Exportaré a '{dest.name}'.")

    temp_out = dest.with_name(dest.stem + ".temp.mp4")
    # limpia temporales previos
    for p in (temp_out,):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
        except PermissionError:
            # último recurso: renómbralo para no chocar
            ts = time.strftime("%Y%m%d-%H%M%S")
            try:
                os.replace(p, p.with_name(p.stem + f".old_{ts}" + p.suffix))
            except Exception:
                raise SystemExit(f"No puedo manipular el temporal '{p}'. Cierra procesos y reintenta.")

    ffmpeg_params = ["-movflags", "+faststart", "-pix_fmt", "yuv420p", "-y"]



    # usa un stem “seguro” (sin puntos raros) para que MoviePy no derive nombres chungos
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", dest.stem)

    temp_out = dest.with_name(f"{safe_stem}__temp.mp4")
    temp_audio = dest.with_name(f"{safe_stem}__temp_snd_{int(time.time())}.m4a")

    # limpia temporales previos si quedaron colgados
    for p in (temp_out, temp_audio):
        try:
            if p.exists():
                try: os.chmod(p, stat.S_IWRITE)
                except Exception: pass
                p.unlink()
        except PermissionError:
            # último recurso: renombra para no chocar
            try:
                os.replace(p, p.with_name(p.stem + f".old_{int(time.time())}" + p.suffix))
            except Exception:
                raise SystemExit(f"No puedo manipular el temporal '{p}'. Cierra procesos y reintenta.")


    # --- render: forza dónde va el audio temporal
    video.write_videofile(
        str(temp_out),
        codec="libx264",
        audio_codec="aac",
        fps=args.fps,
        bitrate="5000k",
        threads=2,
        ffmpeg_params=ffmpeg_params,
        temp_audiofile=str(temp_audio),  
        remove_temp=True,               
        audio=True,
        audio_fps=44100,
        audio_nbytes=2,
    )

    # liberar handles antes del rename (Windows)
    try: video.close()
    except: pass
    try: audio_clip.close()
    except: pass
    for sc in clips:
        try: sc.close()
        except: pass
    gc.collect()
    time.sleep(0.1)

    # intenta colocar el archivo final
    try:
        if dest.exists():
            try: os.chmod(dest, stat.S_IWRITE)
            except Exception: pass
            try: dest.unlink()
            except Exception: pass
        os.replace(temp_out, dest)
    except PermissionError:
        alt = dest.with_name(f"{safe_stem}_{int(time.time())}{dest.suffix}")
        print(f"[WARN] '{dest.name}' bloqueado. Guardaré como '{alt.name}'.")
        os.replace(temp_out, alt)
        dest = alt

    print(f"[OK] Vídeo exportado en: {dest}")


if __name__ == "__main__":
    main()
