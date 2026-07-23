import hashlib
import os
import random
import smtplib
import time
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests
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
SEMANTIC_SCHOLAR_API_KEY = os.environ.get(
    "SEMANTIC_SCHOLAR_API_KEY"
)

SEMANTIC_SCHOLAR_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/search"
)

# Each topic fetches more candidates, then selects a daily subset.
SEMANTIC_SCHOLAR_CANDIDATE_LIMIT = 10

# Semantic Scholar API keys start at 1 request per second, but the
# service may apply additional throttling during periods of heavy use.
SEMANTIC_SCHOLAR_REQUEST_GAP_SECONDS = 5.0
SEMANTIC_SCHOLAR_TOPIC_GAP_SECONDS = 120
SEMANTIC_SCHOLAR_429_COOLDOWN_SECONDS = 120
SEMANTIC_SCHOLAR_MAX_RETRIES = 3


def _semantic_scholar_headers():
    headers = {
        "Accept": "application/json",
        "User-Agent": "paper-digest-bot/1.0",
    }

    if SEMANTIC_SCHOLAR_API_KEY:
        api_key = SEMANTIC_SCHOLAR_API_KEY.strip()
        headers["x-api-key"] = api_key
        print(
            "🔑 [DEBUG] Semantic Scholar API Key Active "
            f"(Len: {len(api_key)})"
        )
    else:
        print(
            "⚠️ [DEBUG] No Semantic Scholar API Key detected. "
            "Running in Anonymous Mode."
        )

    return headers


def _select_daily_papers(papers, limit: int, query: str):
    """Select a deterministic subset using the current UTC date."""
    if len(papers) <= limit:
        return papers

    utc_date = datetime.now(timezone.utc).date().isoformat()
    seed_text = f"{utc_date}:{query}"
    seed_value = int(
        hashlib.sha256(
            seed_text.encode("utf-8")
        ).hexdigest(),
        16,
    )

    daily_random = random.Random(seed_value)
    selected_papers = papers.copy()
    daily_random.shuffle(selected_papers)

    print(
        f"🎲 Selecting {limit} paper(s) from "
        f"{len(papers)} candidates using UTC date {utc_date}."
    )

    return selected_papers[:limit]


def get_published_papers(query: str, limit: int = 3):
    """
    Fetch research papers (2021-2026) from Semantic Scholar.
    """
    candidate_limit = max(
        limit,
        SEMANTIC_SCHOLAR_CANDIDATE_LIMIT,
    )

    params = {
        "query": query,
        "limit": candidate_limit,
        "year": "2021-2026",
        "fieldsOfStudy": "Computer Science",
        "fields": "title,abstract,authors,year,venue,url",
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
                timeout=30,
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")

                fallback_wait = (
                    SEMANTIC_SCHOLAR_429_COOLDOWN_SECONDS
                    * (2 ** attempt)
                )

                server_wait = (
                    int(retry_after)
                    if retry_after and retry_after.isdigit()
                    else 0
                )

                wait_seconds = max(
                    server_wait,
                    fallback_wait,
                )

                last_error = (
                    "HTTP 429 Too Many Requests "
                    f"(wait: {wait_seconds}s)"
                )

                print(
                    "⚠️ [429 Rate Limited] "
                    f"Attempt {attempt + 1}/"
                    f"{SEMANTIC_SCHOLAR_MAX_RETRIES}."
                )

                # Do not wait after the final failed attempt.
                if attempt < SEMANTIC_SCHOLAR_MAX_RETRIES - 1:
                    print(f"Retrying in {wait_seconds}s...")
                    time.sleep(wait_seconds)

                continue

            response.raise_for_status()
            data = response.json()
            break

        except requests.exceptions.RequestException as error:
            last_error = error

            if attempt < SEMANTIC_SCHOLAR_MAX_RETRIES - 1:
                wait_seconds = 10 * (attempt + 1)
                print(
                    f"⚠️ Request failed: {error}. "
                    f"Retrying in {wait_seconds}s..."
                )
                time.sleep(wait_seconds)

    if data is None:
        print(
            "❌ Failed to fetch Semantic Scholar data "
            f"for query {query!r}. Last error: {last_error}"
        )
        return []

    published_papers = []

    for item in data.get("data", []):
        title = item.get("title")

        if not title:
            continue

        abstract_text = (
            item.get("abstract")
            or "No abstract available."
        )

        authors_list = [
            author["name"]
            for author in item.get("authors", [])[:2]
            if "name" in author
        ]

        published_papers.append(
            {
                "title": title,
                "authors": (
                    authors_list
                    if authors_list
                    else ["Unknown Author"]
                ),
                "venue": (
                    item.get("venue")
                    or "Journal/Conference"
                ),
                "year": item.get("year") or "N/A",
                "abstract": abstract_text[:400],
                "url": (
                    item.get("url")
                    or (
                        "https://www.semanticscholar.org/"
                        f"search?q={title}"
                    )
                ),
            }
        )

    return _select_daily_papers(
        published_papers,
        limit,
        query,
    )


def generate_daily_digest():
    time.sleep(3.0)

    print("Fetching Topic 1 papers (Depth Estimation)...")
    depth_papers = get_published_papers(
        "depth estimation",
        limit=2,
    )

    if not depth_papers:
        print(
            "⚠️ [Warning] Topic 1 returned "
            "0 candidate papers."
        )

    print(
        "Cooling down before Topic 2 "
        f"for {SEMANTIC_SCHOLAR_TOPIC_GAP_SECONDS}s..."
    )
    time.sleep(SEMANTIC_SCHOLAR_TOPIC_GAP_SECONDS)

    print(
        "Fetching Topic 2 papers "
        "(Generative World Models)..."
    )
    world_model_papers = get_published_papers(
        (
            "generative world models "
            "video prediction simulation"
        ),
        limit=2,
    )

    if not world_model_papers:
        print(
            "⚠️ [Warning] Topic 2 returned "
            "0 candidate papers."
        )

    # Keep the private bot strict: both topics are required.
    if not depth_papers or not world_model_papers:
        raise RuntimeError(
            "❌ [Error] At least one paper topic could not "
            "be fetched, so Gemini and email were skipped."
        )

    import json

    papers_data = json.dumps(
        {
            "topic1": depth_papers,
            "topic2": world_model_papers,
        },
        ensure_ascii=False,
    )

    prompt = f"""Summarize these research papers:
{papers_data}

Format each:
**Title** (Year, Venue)
Link: URL | Authors: Names
Summary: 1-2 sentences on method
Impact: 2 key contributions"""

    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY environment variable is missing."
        )

    print("Generating summary via Gemini API...")
    client = genai.Client(api_key=GEMINI_API_KEY)

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
            )
            return response.text

        except Exception as error:
            print(
                "⚠️ Gemini API call failed "
                f"(Attempt {attempt + 1}/3): {error}"
            )

            if attempt < 2:
                time.sleep(5)
            else:
                raise


def send_email(content: str):
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_password = os.environ.get("SENDER_PASSWORD")
    receiver_email = os.environ.get("RECEIVER_EMAIL")

    if not sender_email or not sender_password or not receiver_email:
        print(
            "Email credentials missing. "
            "Outputting content to stdout:"
        )
        print(content)
        return

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "📅 Daily Academic Paper Digest"
    message.attach(MIMEText(content, "plain", "utf-8"))

    try:
        server = smtplib.SMTP(
            "smtp.gmail.com",
            587,
            timeout=30,
        )
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(message)
        server.quit()
        print("Email sent successfully!")

    except Exception as error:
        raise RuntimeError(
            f"Failed to send email: {error}"
        ) from error


if __name__ == "__main__":
    digest = generate_daily_digest()
    send_email(digest)