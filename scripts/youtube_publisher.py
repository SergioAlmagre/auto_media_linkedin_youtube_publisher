import os, sys, re, google_auth_oauthlib.flow, googleapiclient.discovery
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from datetime import date
from pathlib import Path
from state_store import get_nested
from time import sleep
sys.stdout.reconfigure(encoding='utf-8')


SCOPES = ["https://www.googleapis.com/auth/youtube"]
BASE_DIR = Path(__file__).parent.parent
today = get_nested("selected_day") or date.today().strftime("%Y-%m-%d")
POST_DIR = BASE_DIR / "post" / today
POST_VIDEO = POST_DIR / "video_post.mp4"
POST_TEXT = POST_DIR / "linkedin_post.txt"
POST_THUMBNAIL = POST_DIR / "miniatura.jpg"


def read_post_file(file_path):
    """Usa primera línea como título, resto como descripción, y extrae hashtags"""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    title = lines[0] if lines else "Video sin título"
    description = "\n".join(lines[1:]) if len(lines) > 1 else ""

    # Extraer hashtags (ej: #seguridad, #tecnologia)
    hashtags = re.findall(r"#\w+", description)
    hashtags_str = " ".join(hashtags)

    return title, description, hashtags_str

def wait_until_processed(youtube, video_id, timeout=180, poll_secs=3):
    """
    Espera hasta que el vídeo esté procesado (máx. 'timeout' segundos).
    Lanza excepción si falla/termina o si expira el tiempo.
    """
    waited = 0
    last = None
    while waited < timeout:
        resp = youtube.videos().list(
            part="processingDetails,status",
            id=video_id
        ).execute()
        items = resp.get("items", [])
        if not items:
            raise RuntimeError("Vídeo no encontrado o no pertenece al canal autenticado.")

        pd = items[0].get("processingDetails", {}) or {}
        st = pd.get("processingStatus")              # pending/processing/succeeded/failed/terminated
        up = items[0]["status"].get("uploadStatus")  # uploaded/processed/failed/rejected

        # Opcional: tiempo estimado restante, si lo hay
        tl = pd.get("processingProgress", {}).get("timeLeftMs")
        if st != last:
            print(f"[YT] processingStatus={st}, uploadStatus={up}, timeLeftMs={tl}")
            last = st

        if st == "succeeded" or up == "processed":
            return  # listo

        if st in ("failed", "terminated") or up in ("failed", "rejected"):
            raise RuntimeError(f"Procesamiento falló: processingStatus={st}, uploadStatus={up}")

        sleep(poll_secs)
        waited += poll_secs

    raise TimeoutError("El vídeo no terminó de procesarse a tiempo.")

def main():
    creds = None
    if os.path.exists("config/token.json"):
        creds = Credentials.from_authorized_user_file("config/token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                "config/client_secret.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open("config/token.json", "w") as token:
            token.write(creds.to_json())

    youtube = googleapiclient.discovery.build("youtube", "v3", credentials=creds)

    # Leer datos desde linkedin_post.txt
    title, description, hashtags = read_post_file(POST_TEXT)
    fixed_description = get_nested("youtube.fixed_description")
    full_description = f"{description}\n\n{hashtags}\n\n{fixed_description}"

    print("Título:", title)
    print("Descripción:", full_description)

    # Ruta del video y miniatura
    video_path = POST_VIDEO
    thumbnail_path = POST_DIR / "miniatura.jpg"
    print(f"[DEBUG] len(title)={len(title)} repr={repr(title)}")
    
    # Subir video
    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": title,
                "description": full_description,
                "tags": hashtags.replace("#", "").split(),
                "categoryId": "22",  # People & Blogs
            },
            "status": {"privacyStatus": get_nested("youtube.visibility")},
        },
        media_body=MediaFileUpload(video_path),
    )
    response = request.execute()
    video_id = response["id"]
    print(f"✅ Video publicado: https://www.youtube.com/watch?v={video_id}")

    try:
        wait_until_processed(youtube, video_id, timeout=240, poll_secs=4)
    except Exception as e:
        print(f"⚠️ Aviso durante espera: {e} (intento igualmente subir miniatura)")
        
    # Subir miniatura si existe
    if thumbnail_path.exists():
        youtube.thumbnails().set(
            videoId=video_id,
            media_body=MediaFileUpload(str(thumbnail_path))
        ).execute()
        print(f"🖼️ Miniatura subida: {thumbnail_path.name}")
    else:
        print("⚠️ No se encontró miniatura, se usará la generada por YouTube")

    # Añadir a playlist
    playlist_name = get_nested("youtube.playlist")
    playlists = youtube.playlists().list(part="snippet", mine=True, maxResults=50).execute()
    playlist_id = None
    for pl in playlists["items"]:
        if pl["snippet"]["title"] == playlist_name:
            playlist_id = pl["id"]
            break

    if playlist_id:
        youtube.playlistItems().insert(
            part="snippet",
            body={
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {"kind": "youtube#video", "videoId": video_id},
                }
            },
        ).execute()
        print(f"📂 Video añadido a la lista de reproducción: {playlist_name}")
    else:
        print(f"⚠️ No se encontró la playlist '{playlist_name}' en tu canal.")
        

if __name__ == "__main__": main()