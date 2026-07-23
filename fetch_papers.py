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
SEMANTIC_SCHOLAR_REQUEST_GAP_SECONDS = 3.0
SEMANTIC_SCHOLAR_429_COOLDOWN_SECONDS = 30
SEMANTIC_SCHOLAR_MAX_RETRIES = 3


def _semantic_scholar_headers():
    headers = {
        "Accept": "application/json",
        "User-Agent": "paper-digest-bot/1.0",
    }
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY.strip()
        print(f"🔑 [DEBUG] Semantic Scholar API Key Active (Len: {len(SEMANTIC_SCHOLAR_API_KEY)})")
    else:
        print("⚠️ [DEBUG] No Semantic Scholar API Key detected. Running in Anonymous Mode.")
    return headers


def get_published_papers(query: str, limit: int = 3):
    """
    Fetch research papers (2021-2026) directly from Semantic Scholar.
    """
    params = {
        'query': query,
        'limit': min(limit, 5),
        'year': '2021-2026',
        'fieldsOfStudy': 'Computer Science',
        'fields': 'title,abstract,authors,year,venue,url'
    }

    last_error = None
    data = None

    for attempt in range(SEMANTIC_SCHOLAR_MAX_RETRIES):
        try:
            time.sleep(SEMANTIC_SCHOLAR_REQUEST_GAP_SECONDS)
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
                last_error = f"HTTP 429 Too Many Requests (Wait suggested: {wait_seconds}s)"
                print(f"⚠️ [429 Rate Limited] Retrying in {wait_seconds}s (Attempt {attempt + 1}/{SEMANTIC_SCHOLAR_MAX_RETRIES})...")
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            data = response.json()
            break  # 成功取得資料，跳出迴圈

        except requests.exceptions.RequestException as e:
            last_error = e
            wait_seconds = 10 * (attempt + 1)
            print(f"⚠️ Request failed: {e}. Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)
    else:
        print(f"❌ Failed to fetch Semantic Scholar data for query '{query}'. Last error: {last_error}")
        return []

    published_papers = []
    if data and 'data' in data:
        for item in data.get('data', []):
            # 只要有 title 即可採納，abstract 有就抓，沒有給預設文字
            title = item.get("title")
            if title:
                abstract_text = item.get("abstract") or "No abstract available."
                authors_list = [a['name'] for a in item.get("authors", [])[:2] if 'name' in a]
                published_papers.append({
                    "title": title,
                    "authors": authors_list if authors_list else ["Unknown Author"],
                    "venue": item.get("venue") or "Journal/Conference",
                    "year": item.get("year") or "N/A",
                    "abstract": abstract_text[:400],
                    "url": item.get("url") or f"https://www.semanticscholar.org/search?q={title}"
                })

    return published_papers[:limit]


def generate_daily_digest():
    time.sleep(3.0)

    print("Fetching Topic 1 papers (Depth Estimation)...")
    depth_papers = get_published_papers("depth estimation", limit=2)

    if not depth_papers:
        print("⚠️ [Warning] Topic 1 returned 0 candidate papers.")

    time.sleep(3.0)  # 兩次請求中間適度冷卻

    print("Fetching Topic 2 papers (Generative World Models)...")
    world_model_papers = get_published_papers(
        "generative world models video prediction simulation",
        limit=2
    )

    if not world_model_papers:
        print("⚠️ [Warning] Topic 2 returned 0 candidate papers.")

    # 任一主題沒抓到就終止，確保信件完整性
    if not depth_papers or not world_model_papers:
        raise RuntimeError("❌ [Error] 包含未抓取成功的論文主題 (可能觸發 Semantic Scholar 429 Limit)，取消呼叫 Gemini 並終止發信。")

    import json
    papers_data = json.dumps({"topic1": depth_papers, "topic2": world_model_papers}, ensure_ascii=False)

    prompt = f"""Summarize these research papers:\n{papers_data}\n\nFormat each:\n**Title** (Year, Venue)\nLink: URL | Authors: Names\nSummary: 1-2 sentences on method\nImpact: 2 key contributions"""

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is missing.")

    print("Generating summary via Gemini API...")
    client = genai.Client(api_key=GEMINI_API_KEY)

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-3.6-flash',
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