import os
import time
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google import genai


def _load_dotenv_file(dotenv_path: str = ".env"):
    if not os.path.exists(dotenv_path):
        return

    with open(dotenv_path, "r", encoding="utf-8") as dotenv_file:
        for raw_line in dotenv_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv_file()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_REQUEST_GAP_SECONDS = 5.0
SEMANTIC_SCHOLAR_429_COOLDOWN_SECONDS = 60
SEMANTIC_SCHOLAR_MAX_RETRIES = 2
_semantic_scholar_next_request_time = 0.0


def _wait_for_semantic_scholar_slot():
    global _semantic_scholar_next_request_time

    now = time.time()
    if now < _semantic_scholar_next_request_time:
        wait_seconds = _semantic_scholar_next_request_time - now
        print(f"Waiting {wait_seconds:.1f}s to comply with Semantic Scholar Rate Limits...")
        time.sleep(wait_seconds)


def _set_semantic_scholar_next_request_time(wait_seconds: float):
    global _semantic_scholar_next_request_time
    _semantic_scholar_next_request_time = max(
        _semantic_scholar_next_request_time,
        time.time() + wait_seconds,
    )

def _semantic_scholar_headers():
    headers = {
        "Accept": "application/json",
        "User-Agent": "paper-digest-bot/1.0",
    }
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
    return headers

def get_published_papers(query: str, limit: int = 5):
    """
    Fetch high-quality published papers (2021-2026) from Semantic Scholar.
    Filtering specifically for Journal & Conference publications.
    Strictly enforcing < 1 RPS rate limit. Reduced limit for payload optimization.
    """
    params = {
        'query': query,
        'limit': min(limit, 5),
        'year': '2021-2026',
        'fields': 'title,abstract,authors,year,venue,url',
        'publicationTypes': 'JournalArticle,Conference'
    }
    
    last_error = None
    data = None

    for attempt in range(SEMANTIC_SCHOLAR_MAX_RETRIES):
        try:
            _wait_for_semantic_scholar_slot()
            response = requests.get(
                SEMANTIC_SCHOLAR_URL,
                params=params,
                headers=_semantic_scholar_headers(),
                timeout=15,
            )
            
            # 處理 429 Rate Limit
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_seconds = int(retry_after) if retry_after and retry_after.isdigit() else SEMANTIC_SCHOLAR_429_COOLDOWN_SECONDS
                wait_seconds = max(wait_seconds, SEMANTIC_SCHOLAR_429_COOLDOWN_SECONDS)
                last_error = f"HTTP 429 Too Many Requests (Wait suggested: {wait_seconds}s)"
                print(f"⚠️ [429 Rate Limited] Retrying in {wait_seconds}s (Attempt {attempt + 1}/{SEMANTIC_SCHOLAR_MAX_RETRIES})...")
                _set_semantic_scholar_next_request_time(wait_seconds)
                time.sleep(wait_seconds)
                continue
                
            response.raise_for_status()
            data = response.json()
            _set_semantic_scholar_next_request_time(SEMANTIC_SCHOLAR_REQUEST_GAP_SECONDS)
            break  # 成功取得資料，跳出迴圈
            
        except requests.exceptions.RequestException as e:
            last_error = e
            wait_seconds = max(10 * (attempt + 1), SEMANTIC_SCHOLAR_429_COOLDOWN_SECONDS // 2)
            print(f"⚠️ Request failed: {e}. Retrying in {wait_seconds}s...")
            _set_semantic_scholar_next_request_time(wait_seconds)
            time.sleep(wait_seconds)
    else:
        print(f"❌ Failed to fetch Semantic Scholar data for query '{query}'. Last error: {last_error}")
        return []

    published_papers = []
    if data and 'data' in data:
        for item in data.get('data', []):
            if item.get('abstract') and item.get('venue'):
                published_papers.append({
                    "title": item.get("title", "N/A"),
                    "authors": [a['name'] for a in item.get("authors", [])[:2]],
                    "venue": item.get("venue", "N/A"),
                    "year": item.get("year", "N/A"),
                    "abstract": item.get("abstract", "")[:400],
                    "url": item.get("url") or "N/A"
                })

    return published_papers[:3]

def generate_daily_digest():
    print("Fetching Topic 1 papers...")
    depth_query = "depth estimation"  # Simplified query
    depth_papers = get_published_papers(depth_query, limit=2)

    if not depth_papers:
        print("⚠️ [Warning] Topic 1 returned 0 candidate papers. Continuing with Topic 2.")

    print("Fetching Topic 2 papers...")
    world_model_query = "world model"  # Simplified query
    world_model_papers = get_published_papers(world_model_query, limit=2)

    if not world_model_papers:
        print("⚠️ [Warning] Topic 2 returned 0 candidate papers. Continuing to digest generation.")

    if not depth_papers or not world_model_papers:
        raise RuntimeError("❌ [Error] 包含未抓取成功的論文主題 (可能觸發 429 Rate Limit)，取消呼叫 Gemini 並終止發信。")

    import json
    papers_data = json.dumps({"topic1": depth_papers, "topic2": world_model_papers})
    
    prompt = f"""Summarize these research papers:\n{papers_data}\n\nFormat each:\n**Title** (Year, Venue)\nLink: URL | Authors: Names\nSummary: 1-2 sentences on method\nImpact: 2 key contributions"""

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is missing.")

    print("Generating summary via Gemini API...")
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"⚠️ Gemini API 呼叫失敗 (嘗試 {attempt + 1}/3): {e}")
            if attempt < 2:
                time.sleep(5)
            else:
                raise e

def send_email(content: str):
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_password = os.environ.get("SENDER_PASSWORD")
    receiver_email = os.environ.get("RECEIVER_EMAIL")

    if not sender_email or not sender_password or not receiver_email:
        print("Email credentials missing. Outputting content to stdout:")
        print(content)
        return

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "📅 Daily Academic Paper Digest"

    msg.attach(MIMEText(content, 'plain', 'utf-8'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

if __name__ == "__main__":
    digest = generate_daily_digest()
    send_email(digest)