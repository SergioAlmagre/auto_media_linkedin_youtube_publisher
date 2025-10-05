# linkedin_post_publisher.py

import requests, datetime
from pathlib import Path
import sys
from state_store import get_nested, set_nested
sys.stdout.reconfigure(encoding='utf-8')

# ---------- Configuración ----------
BASE_DIR = Path(__file__).parent.parent
POSTS_DIR = BASE_DIR / "post"
ACCESS_TOKEN = get_nested("linkedin.api_key")
PERSON_URN = "urn:li:person:EZxDW6SMnO"   # Tu person URN

headers = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "X-Restli-Protocol-Version": "2.0.0",
    "LinkedIn-Version": "202307"
}

# ---------- 1. Localizar carpeta de hoy ----------
today = get_nested("selected_day") or datetime.date.today().strftime("%Y-%m-%d")
today_dir = POSTS_DIR / today

if not today_dir.exists():
    raise FileNotFoundError(f"No existe la carpeta de hoy: {today_dir}")

# Leer texto
txt_path = today_dir / "linkedin_post.txt"
if not txt_path.exists():
    raise FileNotFoundError(f"No existe linkedin_post.txt en {today_dir}")

with open(txt_path, "r", encoding="utf-8") as f:
    post_text = f.read().strip()

# Buscar imágenes o PDF
image_files = []
for ext in ("*.jpg", "*.jpeg", "*.png"):
    image_files.extend(today_dir.glob(ext))

image_path, pdf_path = None, None

if image_files:
    image_path = image_files[0]
else:
    pdf_files = list(today_dir.glob("*.pdf"))
    if pdf_files:
        pdf_path = pdf_files[0]

if not image_path and not pdf_path:
    raise FileNotFoundError(f"No hay archivo de imagen (.jpg/.jpeg/.png) ni PDF en {today_dir}")

print(f"📄 Texto: {post_text[:50]}...")


if image_path:
    print(f"🖼 Imagen: {image_path}")
if pdf_path:
    print(f"📑 PDF: {pdf_path}")

# ---------- 2. Inicializar subida ----------
if image_path:
    init_body = {"initializeUploadRequest": {"owner": PERSON_URN}}
    init_resp = requests.post(
        "https://api.linkedin.com/rest/images?action=initializeUpload",
        headers={**headers, "Content-Type": "application/json"},
        json=init_body
    )
    upload_info = init_resp.json()["value"]
    upload_url, asset_urn = upload_info["uploadUrl"], upload_info["image"]

    with open(image_path, "rb") as f:
        upload_resp = requests.put(upload_url, data=f, headers={"Authorization": f"Bearer {ACCESS_TOKEN}"})
    print(f"✅ Imagen subida: {asset_urn}")

elif pdf_path:
    init_body = {"initializeUploadRequest": {"owner": PERSON_URN}}
    init_resp = requests.post(
        "https://api.linkedin.com/rest/documents?action=initializeUpload",
        headers={**headers, "Content-Type": "application/json"},
        json=init_body
    )
    upload_info = init_resp.json()["value"]
    upload_url, asset_urn = upload_info["uploadUrl"], upload_info["document"]

    with open(pdf_path, "rb") as f:
        upload_resp = requests.put(upload_url, data=f, headers={"Authorization": f"Bearer {ACCESS_TOKEN}"})
    print(f"✅ PDF subido: {asset_urn}")

if image_path:
    content_block = {
        "media": {"id": asset_urn}
    }
elif pdf_path:
    content_block = {
        "multiDocument": {
            "documents": [
                {"id": asset_urn}
            ],
            "title": "📑 Documento adjunto"
        }
    }

fixed_description = get_nested("linkedin.fixed_description")
final_post = f"{post_text}\n\n{fixed_description}"

# ---------- 3. Crear post ----------
post_data = {
    "author": PERSON_URN,
    "commentary": final_post,
    "visibility": get_nested("linkedin.visibility"),
    "distribution": {
        "feedDistribution": "MAIN_FEED",
        "targetEntities": [],
        "thirdPartyDistributionChannels": []
    },
    "content": content_block,
    "lifecycleState": "PUBLISHED",
    "isReshareDisabledByAuthor": False
}

resp = requests.post(
    "https://api.linkedin.com/rest/posts",
    headers={**headers, "Content-Type": "application/json"},
    json=post_data
)

print("📡 Publicación enviada →", resp.status_code)
post_urn = None

# Algunos 201/202 devuelven cuerpo vacío: coger de cabecera
try:
    if resp.headers.get("x-restli-id"):
        post_urn = resp.headers["x-restli-id"]
        set_nested("linkedin.post_urn",post_urn)
    else:
        # si hay JSON, úsalo
        j = resp.json()
        post_urn = j.get("id")
        set_nested("linkedin.post_urn",post_urn)
except requests.exceptions.JSONDecodeError:
    # cuerpo vacío o no-JSON; ya intentamos con cabecera
    pass

if not post_urn:
    # Para depurar: muestra un poco de la respuesta
    body_preview = (resp.text or "")[:500]
    raise RuntimeError(f"No pude obtener el URN del post. Status={resp.status_code}. Body: {body_preview}")

print("📡 Publicación enviada")
print("Status:", resp.status_code)
print(resp.text)
