# linkedin_post_publisher.py

import requests, sys
from pathlib import Path
from state_store import get_nested
sys.stdout.reconfigure(encoding='utf-8')

# ---------- Configuración ----------
BASE_DIR = Path(__file__).parent.parent
POSTS_DIR = BASE_DIR / "post"
ACCESS_TOKEN = get_nested("linkedin.api_key")
PERSON_URN = "urn:li:person:EZxDW6SMnO"

headers = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "X-Restli-Protocol-Version": "2.0.0",
    "LinkedIn-Version": "202307"
}

post_urn = get_nested("linkedin.post_urn")

# ====== Publicar un comentario en el post recién creado ======
comment_text = get_nested("linkedin.comment")
youtube_url_in_comment = get_nested("linkedin.youtube_in_commnet")
final_comment = ""

if youtube_url_in_comment == 1:
    youtube_url = get_nested("youtube.url") 
    final_comment = f"{comment_text} \n {youtube_url}"
else:
    final_comment = comment_text
    
comment_body = {
    "actor": PERSON_URN,
    "object": post_urn,
    "message": {"text": final_comment},
}

# URN debe ir URL-encoded
import urllib.parse as _url
comment_url = f"https://api.linkedin.com/rest/socialActions/{_url.quote(post_urn, safe='')}/comments"

c_resp = requests.post(
    comment_url,
    headers={**headers, "Content-Type": "application/json"},
    json=comment_body,
)
print("💬 Comentario:", c_resp.status_code)
if not c_resp.ok:
    print("↳", (c_resp.text or "")[:500])
c_resp.raise_for_status()

print("📡 Publicación enviada")
print("Status:", c_resp.status_code)
print(c_resp.text)
