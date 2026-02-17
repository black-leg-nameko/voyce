import json
import os
import tempfile
import time
import asyncio
import logging
import uuid
import re
import hashlib
import hmac
import secrets
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple
import contextlib

import argostranslate.package
import argostranslate.translate
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from rapidfuzz import fuzz
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

try:
  import redis
except ImportError:
  redis = None

BASE_DIR = Path(__file__).resolve().parent
FAQ_PATH = BASE_DIR / "data" / "faq_ja.json"

app = FastAPI(title="Voyce Guide API")
logger = logging.getLogger("voyce.api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(message)s")

CORS_ALLOWED_ORIGINS = [
  origin.strip()
  for origin in os.getenv(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:8081,http://127.0.0.1:8081,http://localhost:19006",
  ).split(",")
  if origin.strip()
]
APP_ENV = os.getenv("APP_ENV", "dev").lower().strip()
DEFAULT_API_KEYS = "dev-local-key" if APP_ENV == "dev" else ""
API_KEYS = {key.strip() for key in os.getenv("API_KEYS", DEFAULT_API_KEYS).split(",") if key.strip()}
ALLOW_API_KEY_AUTH = os.getenv("ALLOW_API_KEY_AUTH", "true").lower() == "true"
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
AUTH_RATE_LIMIT_PER_MINUTE = int(os.getenv("AUTH_RATE_LIMIT_PER_MINUTE", "10"))
RATE_WINDOW_SECONDS = 60
TRUSTED_PROXY_IPS = {
  value.strip()
  for value in os.getenv("TRUSTED_PROXY_IPS", "127.0.0.1,::1").split(",")
  if value.strip()
}
ENFORCE_HTTPS = os.getenv("ENFORCE_HTTPS", "false").lower() == "true"
REQUIRE_CONSENT = os.getenv("REQUIRE_CONSENT", "true").lower() == "true"
AUTH_TOKEN_SECRET = os.getenv("AUTH_TOKEN_SECRET", "")
AUTH_TOKEN_TTL_SECONDS = int(os.getenv("AUTH_TOKEN_TTL_SECONDS", "28800"))
OPERATOR_PIN = os.getenv("OPERATOR_PIN", "")
OPERATOR_PIN_HASH = os.getenv("OPERATOR_PIN_HASH", "").strip()
REDIS_URL = os.getenv("REDIS_URL", "")
ARGOS_ALLOW_RUNTIME_INSTALL = os.getenv("ARGOS_ALLOW_RUNTIME_INSTALL", "false").lower() == "true"
ARGOS_LANGUAGE_PAIRS = os.getenv(
  "ARGOS_LANGUAGE_PAIRS",
  "ja-en,en-ja,ja-zh,zh-ja,ja-ko,ko-ja,ja-fr,fr-ja,ja-es,es-ja",
)
TRANSLATIONS: Dict[Tuple[str, str], object] = {}
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(8 * 1024 * 1024)))
ALLOWED_AUDIO_EXTENSIONS = {
  value.strip().lower()
  for value in os.getenv("ALLOWED_AUDIO_EXTENSIONS", ".m4a,.mp3,.wav,.webm,.ogg").split(",")
  if value.strip()
}
ALLOWED_AUDIO_MIME_TYPES = {
  value.strip().lower()
  for value in os.getenv(
    "ALLOWED_AUDIO_MIME_TYPES",
    "audio/m4a,audio/mp4,audio/mpeg,audio/wav,audio/x-wav,audio/webm,audio/ogg,application/octet-stream",
  ).split(",")
  if value.strip()
}
MAX_CONCURRENT_TRANSCRIBE = int(os.getenv("MAX_CONCURRENT_TRANSCRIBE", "2"))
QUEUE_WAIT_TIMEOUT_SECONDS = float(os.getenv("QUEUE_WAIT_TIMEOUT_SECONDS", "2.0"))
TRANSCRIBE_TIMEOUT_SECONDS = float(os.getenv("TRANSCRIBE_TIMEOUT_SECONDS", "45.0"))
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
WHISPER_BEST_OF = int(os.getenv("WHISPER_BEST_OF", "1"))
WHISPER_TEMPERATURE = float(os.getenv("WHISPER_TEMPERATURE", "0.0"))
ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"
METRICS_TOKEN = os.getenv("METRICS_TOKEN", "")
READY_TOKEN = os.getenv("READY_TOKEN", "")
REQUEST_COUNTERS: Dict[str, int] = defaultdict(int)
REQUEST_LATENCY_MS: Deque[float] = deque(maxlen=1000)
MEMORY_RATE_BUCKETS: Dict[str, Deque[float]] = defaultdict(deque)
MEMORY_SESSION_EXPIRY: Dict[str, int] = {}
WHISPER_MODEL = None
MIN_FAQ_CONFIDENCE = float(os.getenv("MIN_FAQ_CONFIDENCE", "68"))
FAQ_AMBIGUITY_MARGIN = float(os.getenv("FAQ_AMBIGUITY_MARGIN", "8"))
FAQ_HIGH_CONFIDENCE = float(os.getenv("FAQ_HIGH_CONFIDENCE", "85"))
SUPPORTED_TARGET_LANGUAGES = {
  value.strip()
  for value in os.getenv("SUPPORTED_TARGET_LANGUAGES", "ja,en,zh,ko,fr,es").split(",")
  if value.strip()
}
TRANSLATION_MIN_RATIO = float(os.getenv("TRANSLATION_MIN_RATIO", "0.25"))
TRANSLATION_MAX_RATIO = float(os.getenv("TRANSLATION_MAX_RATIO", "4.0"))
TRANSLATION_MIN_CHARS = int(os.getenv("TRANSLATION_MIN_CHARS", "2"))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "500"))
MIN_AUTH_SECRET_LENGTH = int(os.getenv("MIN_AUTH_SECRET_LENGTH", "32"))

REDIS_CLIENT = None
REDIS_INIT_FAILED = False

app.add_middleware(
  CORSMiddleware,
  allow_origins=CORS_ALLOWED_ORIGINS,
  allow_credentials=False,
  allow_methods=["POST", "GET", "OPTIONS"],
  allow_headers=["Content-Type", "authorization", "x-api-key", "x-ready-token", "x-metrics-token"],
)


@app.middleware("http")
async def enforce_https(request: Request, call_next):
  # Health check is often terminated inside a private network.
  if ENFORCE_HTTPS and request.url.path != "/health":
    proto = request.headers.get("x-forwarded-proto", request.url.scheme).lower()
    if proto != "https":
      return JSONResponse(status_code=400, content={"detail": "HTTPS is required."})

  response = await call_next(request)
  if ENFORCE_HTTPS:
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
  response.headers["X-Content-Type-Options"] = "nosniff"
  response.headers["X-Frame-Options"] = "DENY"
  return response


@app.middleware("http")
async def access_log_and_metrics(request: Request, call_next):
  request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
  start = time.perf_counter()
  method = request.method
  path = request.url.path
  client_ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "unknown")

  try:
    response = await call_next(request)
    status_code = response.status_code
  except Exception:
    status_code = 500
    client = get_redis_client()
    if client is not None:
      pipe = client.pipeline()
      pipe.incr("metrics:requests_total")
      pipe.hincrby("metrics:status_codes", str(status_code), 1)
      pipe.execute()
    else:
      REQUEST_COUNTERS["requests_total"] += 1
      REQUEST_COUNTERS[f"status_{status_code}"] += 1
    logger.exception(
      json.dumps(
        {
          "event": "request_error",
          "request_id": request_id,
          "method": method,
          "path": path,
          "status_code": status_code,
          "client_ip": client_ip,
        },
        ensure_ascii=False,
      )
    )
    raise

  latency_ms = (time.perf_counter() - start) * 1000
  client = get_redis_client()
  if client is not None:
    pipe = client.pipeline()
    pipe.incr("metrics:requests_total")
    pipe.hincrby("metrics:status_codes", str(status_code), 1)
    pipe.lpush("metrics:latency_ms", f"{latency_ms:.6f}")
    pipe.ltrim("metrics:latency_ms", 0, 999)
    pipe.expire("metrics:latency_ms", 86400)
    pipe.execute()
  else:
    REQUEST_COUNTERS["requests_total"] += 1
    REQUEST_COUNTERS[f"status_{status_code}"] += 1
    REQUEST_LATENCY_MS.append(latency_ms)

  response.headers["x-request-id"] = request_id
  logger.info(
    json.dumps(
      {
        "event": "request_completed",
        "request_id": request_id,
        "method": method,
        "path": path,
        "status_code": status_code,
        "latency_ms": round(latency_ms, 2),
        "client_ip": client_ip,
      },
      ensure_ascii=False,
    )
  )
  return response


def load_faq() -> List[Dict[str, str]]:
  with FAQ_PATH.open("r", encoding="utf-8") as file:
    return json.load(file)


FAQ_DATA = load_faq()
TRANSCRIBE_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_TRANSCRIBE)


def parse_language_pairs(value: str) -> List[Tuple[str, str]]:
  pairs: List[Tuple[str, str]] = []
  for raw in value.split(","):
    item = raw.strip()
    if "-" not in item:
      continue
    from_code, to_code = item.split("-", 1)
    if from_code and to_code:
      pairs.append((from_code.strip(), to_code.strip()))
  return pairs


def enforce_startup_guards() -> None:
  if APP_ENV in {"stg", "prod"}:
    if not AUTH_TOKEN_SECRET:
      raise RuntimeError("AUTH_TOKEN_SECRET is required in stg/prod")
    if len(AUTH_TOKEN_SECRET) < MIN_AUTH_SECRET_LENGTH:
      raise RuntimeError(f"AUTH_TOKEN_SECRET must be at least {MIN_AUTH_SECRET_LENGTH} chars in stg/prod")
    if not OPERATOR_PIN:
      if not OPERATOR_PIN_HASH:
        raise RuntimeError("OPERATOR_PIN_HASH is required in stg/prod")
    if OPERATOR_PIN:
      raise RuntimeError("OPERATOR_PIN plaintext is not allowed in stg/prod")
    if ALLOW_API_KEY_AUTH:
      raise RuntimeError("ALLOW_API_KEY_AUTH must be false in stg/prod")
    if not METRICS_TOKEN:
      raise RuntimeError("METRICS_TOKEN is required in stg/prod")
    if not READY_TOKEN:
      raise RuntimeError("READY_TOKEN is required in stg/prod")
    if not REDIS_URL:
      raise RuntimeError("REDIS_URL is required in stg/prod")
    if redis is None:
      raise RuntimeError("redis package is required in stg/prod")
    client = get_redis_client()
    if client is None:
      raise RuntimeError("Redis connection is required in stg/prod")


def get_redis_client():
  global REDIS_CLIENT, REDIS_INIT_FAILED
  if not REDIS_URL or redis is None or REDIS_INIT_FAILED:
    return None
  if REDIS_CLIENT is None:
    try:
      REDIS_CLIENT = redis.Redis.from_url(
        REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=0.5,
        socket_timeout=0.5,
      )
      REDIS_CLIENT.ping()
    except Exception:
      REDIS_INIT_FAILED = True
      logger.exception(json.dumps({"event": "redis_init_failed"}, ensure_ascii=False))
      REDIS_CLIENT = None
      return None
  return REDIS_CLIENT


def hash_token(token: str) -> str:
  return hmac.new(AUTH_TOKEN_SECRET.encode("utf-8"), token.encode("utf-8"), hashlib.sha256).hexdigest()


def store_session_token(token: str) -> None:
  token_hash = hash_token(token)
  expires_at = int(time.time()) + AUTH_TOKEN_TTL_SECONDS
  client = get_redis_client()
  if client is not None:
    client.setex(f"session:{token_hash}", AUTH_TOKEN_TTL_SECONDS, "1")
    return
  if APP_ENV in {"stg", "prod"}:
    raise HTTPException(status_code=503, detail="Session store is unavailable.")
  MEMORY_SESSION_EXPIRY[token_hash] = expires_at


def is_session_token_valid(token: str) -> bool:
  token_hash = hash_token(token)
  client = get_redis_client()
  if client is not None:
    return client.exists(f"session:{token_hash}") == 1
  if APP_ENV in {"stg", "prod"}:
    raise HTTPException(status_code=503, detail="Session store is unavailable.")

  expires_at = MEMORY_SESSION_EXPIRY.get(token_hash, 0)
  if int(time.time()) >= expires_at:
    if token_hash in MEMORY_SESSION_EXPIRY:
      del MEMORY_SESSION_EXPIRY[token_hash]
    return False
  return expires_at > 0


def revoke_session_token(token: str) -> None:
  token_hash = hash_token(token)
  client = get_redis_client()
  if client is not None:
    client.delete(f"session:{token_hash}")
    return
  if APP_ENV in {"stg", "prod"}:
    raise HTTPException(status_code=503, detail="Session store is unavailable.")
  MEMORY_SESSION_EXPIRY.pop(token_hash, None)


def issue_access_token() -> str:
  if not AUTH_TOKEN_SECRET:
    raise HTTPException(status_code=500, detail="Token auth is not configured.")
  token = secrets.token_urlsafe(32)
  store_session_token(token)
  return token


def verify_access_token(token: str) -> None:
  if not AUTH_TOKEN_SECRET:
    raise HTTPException(status_code=500, detail="Token auth is not configured.")
  if not token or not is_session_token_valid(token):
    raise HTTPException(status_code=401, detail="Invalid or expired token.")


def verify_operator_pin(operator_pin: str) -> bool:
  if OPERATOR_PIN_HASH:
    parts = OPERATOR_PIN_HASH.split("$")
    if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
      logger.error(json.dumps({"event": "invalid_operator_pin_hash_format"}, ensure_ascii=False))
      return False

    _, iterations_text, salt, expected_hex = parts
    try:
      iterations = int(iterations_text)
    except ValueError:
      logger.error(json.dumps({"event": "invalid_operator_pin_hash_iterations"}, ensure_ascii=False))
      return False

    derived = hashlib.pbkdf2_hmac("sha256", operator_pin.encode("utf-8"), salt.encode("utf-8"), iterations).hex()
    return hmac.compare_digest(derived, expected_hex)

  if OPERATOR_PIN:
    return hmac.compare_digest(operator_pin, OPERATOR_PIN)
  return False


def get_whisper_model():
  global WHISPER_MODEL
  if WHISPER_MODEL is None:
    WHISPER_MODEL = WhisperModel(
      os.getenv("WHISPER_MODEL_SIZE", "small"),
      device=os.getenv("WHISPER_DEVICE", "cpu"),
      compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
    )
  return WHISPER_MODEL


def get_translation(from_code: str, to_code: str) -> object | None:
  installed_languages = argostranslate.translate.get_installed_languages()
  from_lang = next((lang for lang in installed_languages if lang.code == from_code), None)
  to_lang = next((lang for lang in installed_languages if lang.code == to_code), None)

  if not from_lang or not to_lang:
    return None

  return from_lang.get_translation(to_lang)


def install_argos_package(from_code: str, to_code: str) -> bool:
  argostranslate.package.update_package_index()
  packages = argostranslate.package.get_available_packages()
  pkg = next((p for p in packages if p.from_code == from_code and p.to_code == to_code), None)
  if not pkg:
    return False

  download_path = pkg.download()
  argostranslate.package.install_from_path(download_path)
  return True


def warmup_translations() -> None:
  for from_code, to_code in parse_language_pairs(ARGOS_LANGUAGE_PAIRS):
    translation = get_translation(from_code, to_code)
    if translation is not None:
      TRANSLATIONS[(from_code, to_code)] = translation

  if ARGOS_ALLOW_RUNTIME_INSTALL:
    for from_code, to_code in parse_language_pairs(ARGOS_LANGUAGE_PAIRS):
      if (from_code, to_code) in TRANSLATIONS:
        continue
      if install_argos_package(from_code, to_code):
        translation = get_translation(from_code, to_code)
        if translation is not None:
          TRANSLATIONS[(from_code, to_code)] = translation


def translate_text(text: str, from_code: str, to_code: str, allow_runtime_install: bool = False) -> str:
  if not text or from_code == to_code:
    return text

  translation = TRANSLATIONS.get((from_code, to_code))
  if translation is None and allow_runtime_install and install_argos_package(from_code, to_code):
    translation = get_translation(from_code, to_code)
    if translation is not None:
      TRANSLATIONS[(from_code, to_code)] = translation
  if translation is None:
    return text

  return translation.translate(text)


def has_script_hint(text: str, language: str) -> bool:
  if language == "ko":
    return bool(re.search(r"[\uac00-\ud7af]", text))
  if language == "zh":
    return bool(re.search(r"[\u4e00-\u9fff]", text))
  if language == "ja":
    return bool(re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", text))
  return True


def assess_translation_quality(source_text: str, translated_text: str, target_language: str) -> Dict[str, object]:
  source = source_text.strip()
  translated = translated_text.strip()

  if not translated:
    return {"ok": False, "reason": "empty_translation"}
  if len(translated) < TRANSLATION_MIN_CHARS:
    return {"ok": False, "reason": "too_short"}
  if source.lower() == translated.lower() and target_language != "ja":
    return {"ok": False, "reason": "unchanged_text"}

  source_len = max(1, len(source))
  ratio = len(translated) / source_len
  if ratio < TRANSLATION_MIN_RATIO or ratio > TRANSLATION_MAX_RATIO:
    return {"ok": False, "reason": "length_ratio_out_of_bounds", "ratio": round(ratio, 3)}
  if not has_script_hint(translated, target_language):
    return {"ok": False, "reason": "script_hint_missing"}

  return {"ok": True, "reason": "ok", "ratio": round(ratio, 3)}


def normalize_text(text: str) -> str:
  lowered = text.lower().strip()
  lowered = re.sub(r"\s+", " ", lowered)
  return lowered


def score_faq_item(text: str, keywords: List[str]) -> Tuple[float, str]:
  best_keyword = ""
  best_keyword_score = 0.0
  hit_count = 0
  exact_hit = False

  for keyword in keywords:
    normalized_keyword = normalize_text(keyword)
    if not normalized_keyword:
      continue

    fuzzy_score = float(fuzz.partial_ratio(text, normalized_keyword))
    if fuzzy_score > best_keyword_score:
      best_keyword_score = fuzzy_score
      best_keyword = keyword

    if normalized_keyword in text:
      hit_count += 1
      if text == normalized_keyword:
        exact_hit = True

  hit_bonus = min(2, hit_count) * 10
  exact_bonus = 12 if exact_hit else 0
  final_score = (best_keyword_score * 0.78) + hit_bonus + exact_bonus
  return min(100.0, final_score), best_keyword


def generate_answer_ja(transcript: str) -> Dict[str, object]:
  text = normalize_text(transcript)
  scored_items: List[Dict[str, object]] = []

  for item in FAQ_DATA:
    keywords = item.get("keywords", [])
    score, matched_keyword = score_faq_item(text, keywords)
    topic = keywords[0] if keywords else "案内"
    scored_items.append(
      {
        "score": score,
        "topic": topic,
        "matched_keyword": matched_keyword,
        "answer": item.get("answer", ""),
      }
    )

  scored_items.sort(key=lambda value: float(value["score"]), reverse=True)
  best = scored_items[0] if scored_items else {"score": 0.0, "answer": ""}
  second = scored_items[1] if len(scored_items) > 1 else {"score": 0.0, "topic": "案内"}

  best_score = float(best["score"])
  second_score = float(second["score"])
  ambiguous = (best_score - second_score) < FAQ_AMBIGUITY_MARGIN and best_score < FAQ_HIGH_CONFIDENCE
  low_confidence = best_score < MIN_FAQ_CONFIDENCE

  if low_confidence or ambiguous:
    followup = f"「{best.get('topic', '案内')}」と「{second.get('topic', '案内')}」のどちらについて知りたいですか。"
    return {
      "answer": f"ご質問ありがとうございます。{followup}",
      "confidence": round(best_score, 2),
      "topic": "clarification",
      "matched_keyword": best.get("matched_keyword", ""),
    }

  return {
    "answer": str(best["answer"]),
    "confidence": round(best_score, 2),
    "topic": str(best.get("topic", "案内")),
    "matched_keyword": str(best.get("matched_keyword", "")),
  }


warmup_translations()
enforce_startup_guards()


def get_client_ip(request: Request) -> str:
  forwarded_for = request.headers.get("x-forwarded-for", "")
  remote_host = request.client.host if request.client else "unknown"
  if forwarded_for and remote_host in TRUSTED_PROXY_IPS:
    return forwarded_for.split(",")[0].strip()
  if remote_host:
    return remote_host
  return "unknown"


def enforce_rate_limit(client_ip: str, limit: int = RATE_LIMIT_PER_MINUTE, kind: str = "api") -> None:
  key = f"rate:{kind}:{client_ip}"
  client = get_redis_client()
  if client is not None:
    current = client.incr(key)
    if current == 1:
      client.expire(key, RATE_WINDOW_SECONDS)
    if current > limit:
      raise HTTPException(status_code=429, detail="Too many requests. Please retry later.")
    return
  if APP_ENV in {"stg", "prod"}:
    raise HTTPException(status_code=503, detail="Rate limiter is unavailable.")

  now = time.time()
  bucket = MEMORY_RATE_BUCKETS[key]

  while bucket and now - bucket[0] > RATE_WINDOW_SECONDS:
    bucket.popleft()

  if len(bucket) >= limit:
    raise HTTPException(status_code=429, detail="Too many requests. Please retry later.")

  bucket.append(now)


def verify_request(
  request: Request,
  x_api_key: str = Header(default="", alias="x-api-key"),
  authorization: str = Header(default="", alias="authorization"),
) -> None:
  auth_ok = False
  if authorization.lower().startswith("bearer "):
    token = authorization[7:].strip()
    verify_access_token(token)
    auth_ok = True

  if not auth_ok and ALLOW_API_KEY_AUTH and x_api_key in API_KEYS:
    auth_ok = True

  if not auth_ok:
    raise HTTPException(status_code=401, detail="Unauthorized.")

  enforce_rate_limit(get_client_ip(request))


def validate_upload(file: UploadFile, content: bytes, suffix: str) -> None:
  if not content:
    raise HTTPException(status_code=400, detail="Empty audio file.")
  if len(content) > MAX_AUDIO_BYTES:
    raise HTTPException(status_code=413, detail="Audio file too large.")

  normalized_suffix = suffix.lower()
  if normalized_suffix not in ALLOWED_AUDIO_EXTENSIONS:
    raise HTTPException(status_code=415, detail=f"Unsupported file extension: {normalized_suffix}")

  content_type = (file.content_type or "").lower().strip()
  if content_type and content_type not in ALLOWED_AUDIO_MIME_TYPES:
    raise HTTPException(status_code=415, detail=f"Unsupported content type: {content_type}")


def transcribe_sync(audio_path: str, whisper_lang: str | None):
  return get_whisper_model().transcribe(
    audio_path,
    language=whisper_lang,
    beam_size=WHISPER_BEAM_SIZE,
    best_of=WHISPER_BEST_OF,
    temperature=WHISPER_TEMPERATURE,
  )


def build_guide_response(transcript: str, detected: str, target_language: str) -> Dict[str, object]:
  if target_language not in SUPPORTED_TARGET_LANGUAGES:
    raise HTTPException(status_code=400, detail=f"Unsupported target language: {target_language}")

  input_translation_quality = {"ok": True, "reason": "not_required"}
  if detected != "ja":
    transcript_ja_candidate = translate_text(transcript, detected, "ja", allow_runtime_install=ARGOS_ALLOW_RUNTIME_INSTALL)
    input_translation_quality = assess_translation_quality(transcript, transcript_ja_candidate, "ja")
    transcript_ja = transcript_ja_candidate if bool(input_translation_quality["ok"]) else transcript
  else:
    transcript_ja = transcript

  answer_eval = generate_answer_ja(transcript_ja)
  answer_ja = str(answer_eval["answer"])

  output_translation_quality = {"ok": True, "reason": "not_required"}
  translation_fallback = False
  if target_language != "ja":
    translated_candidate = translate_text(answer_ja, "ja", target_language, allow_runtime_install=ARGOS_ALLOW_RUNTIME_INSTALL)
    output_translation_quality = assess_translation_quality(answer_ja, translated_candidate, target_language)
    if bool(output_translation_quality["ok"]):
      answer_translated = translated_candidate
    else:
      answer_translated = answer_ja
      translation_fallback = True
  else:
    answer_translated = answer_ja

  return {
    "transcript": transcript,
    "answer_source_ja": answer_ja,
    "answer_translated": answer_translated,
    "answer_confidence": answer_eval["confidence"],
    "answer_topic": answer_eval["topic"],
    "matched_keyword": answer_eval["matched_keyword"],
    "input_translation_quality": input_translation_quality,
    "output_translation_quality": output_translation_quality,
    "translation_fallback": translation_fallback,
    "source_language": detected,
    "target_language": target_language,
  }


@app.get("/health")
def health() -> Dict[str, str]:
  return {"status": "ok"}


@app.get("/ready")
def ready(x_ready_token: str = Header(default="", alias="x-ready-token")) -> Dict[str, object]:
  if READY_TOKEN and x_ready_token != READY_TOKEN:
    raise HTTPException(status_code=401, detail="Invalid ready token.")
  required_pairs = parse_language_pairs(ARGOS_LANGUAGE_PAIRS)
  loaded_pairs = list(TRANSLATIONS.keys())
  missing_pairs = [f"{src}-{dst}" for (src, dst) in required_pairs if (src, dst) not in TRANSLATIONS]

  redis_connected = get_redis_client() is not None
  checks = {
    "faq_loaded": len(FAQ_DATA) > 0,
    "api_keys_set": len(API_KEYS) > 0,
    "translations_loaded": len(TRANSLATIONS) > 0,
    "required_pairs_ready": len(missing_pairs) == 0 or ARGOS_ALLOW_RUNTIME_INSTALL,
    "shared_state_ready": redis_connected if APP_ENV in {"stg", "prod"} else True,
  }
  ok = all(checks.values())

  return {
    "status": "ok" if ok else "degraded",
    "checks": checks,
    "faq_count": len(FAQ_DATA),
    "translation_pairs_loaded": [f"{src}-{dst}" for (src, dst) in loaded_pairs],
    "translation_pairs_missing": missing_pairs,
    "runtime_install_enabled": ARGOS_ALLOW_RUNTIME_INSTALL,
  }


@app.get("/metrics")
def metrics(x_metrics_token: str = Header(default="", alias="x-metrics-token")) -> Dict[str, object]:
  if not ENABLE_METRICS:
    raise HTTPException(status_code=404, detail="Metrics endpoint is disabled.")
  if METRICS_TOKEN and x_metrics_token != METRICS_TOKEN:
    raise HTTPException(status_code=401, detail="Invalid metrics token.")

  client = get_redis_client()
  if client is not None:
    requests_total_raw = client.get("metrics:requests_total") or "0"
    requests_total = int(requests_total_raw)
    status_codes_raw = client.hgetall("metrics:status_codes")
    status_codes = {f"status_{code}": int(value) for code, value in status_codes_raw.items()}
    latencies_raw = client.lrange("metrics:latency_ms", 0, 999)
    latencies = [float(value) for value in latencies_raw]
  else:
    requests_total = REQUEST_COUNTERS.get("requests_total", 0)
    status_codes = {k: v for k, v in REQUEST_COUNTERS.items() if k.startswith("status_")}
    latencies = list(REQUEST_LATENCY_MS)

  count = len(latencies)
  avg_latency_ms = (sum(latencies) / count) if count > 0 else 0.0
  p95_latency_ms = 0.0
  if count > 0:
    sorted_latencies = sorted(latencies)
    p95_index = min(count - 1, int(count * 0.95))
    p95_latency_ms = sorted_latencies[p95_index]

  return {
    "requests_total": requests_total,
    "status_codes": status_codes,
    "latency_avg_ms": round(avg_latency_ms, 2),
    "latency_p95_ms": round(p95_latency_ms, 2),
    "window_size": count,
    "timestamp_unix": int(time.time()),
  }


@app.post("/auth/session")
def create_session(
  request: Request,
  operator_pin: str = Form(...),
) -> Dict[str, object]:
  enforce_rate_limit(get_client_ip(request), limit=AUTH_RATE_LIMIT_PER_MINUTE, kind="auth")
  if not OPERATOR_PIN and not OPERATOR_PIN_HASH:
    raise HTTPException(status_code=500, detail="Operator auth is not configured.")
  if not verify_operator_pin(operator_pin):
    raise HTTPException(status_code=401, detail="Invalid operator PIN.")
  token = issue_access_token()
  return {"access_token": token, "token_type": "bearer", "expires_in": AUTH_TOKEN_TTL_SECONDS}


@app.post("/auth/revoke")
def revoke_session(
  authorization: str = Header(default="", alias="authorization"),
) -> Dict[str, str]:
  if not authorization.lower().startswith("bearer "):
    raise HTTPException(status_code=401, detail="Missing bearer token.")
  token = authorization[7:].strip()
  verify_access_token(token)
  revoke_session_token(token)
  return {"status": "revoked"}


@app.post("/guide")
async def guide(
  file: UploadFile = File(...),
  source_language: str = Form("auto"),
  target_language: str = Form("en"),
  consent: bool = Form(False),
  _: None = Depends(verify_request),
) -> Dict[str, object]:
  if REQUIRE_CONSENT and not consent:
    raise HTTPException(status_code=400, detail="User consent is required.")

  suffix = Path(file.filename or "audio.m4a").suffix or ".m4a"
  suffix = suffix.lower()
  content = await file.read()
  validate_upload(file, content, suffix)

  with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(content)
    audio_path = tmp.name

  whisper_lang = None if source_language == "auto" else source_language

  acquired = False
  try:
    await asyncio.wait_for(TRANSCRIBE_SEMAPHORE.acquire(), timeout=QUEUE_WAIT_TIMEOUT_SECONDS)
    acquired = True
    segments, info = await asyncio.wait_for(
      run_in_threadpool(transcribe_sync, audio_path, whisper_lang),
      timeout=TRANSCRIBE_TIMEOUT_SECONDS,
    )
    transcript = " ".join(segment.text.strip() for segment in segments).strip()
  except asyncio.TimeoutError as error:
    if not acquired:
      raise HTTPException(status_code=503, detail="Server busy. Please retry.") from error
    raise HTTPException(status_code=504, detail="Transcription timed out.") from error
  except Exception as error:
    raise HTTPException(status_code=500, detail=f"transcribe failed: {error}") from error
  finally:
    if acquired:
      TRANSCRIBE_SEMAPHORE.release()
    with contextlib.suppress(FileNotFoundError):
      os.unlink(audio_path)

  if not transcript:
    raise HTTPException(status_code=400, detail="音声を認識できませんでした。")

  detected = info.language if source_language == "auto" else source_language
  return build_guide_response(transcript, detected, target_language)


@app.post("/guide/text")
async def guide_text(
  question: str = Form(...),
  source_language: str = Form("ja"),
  target_language: str = Form("en"),
  consent: bool = Form(False),
  _: None = Depends(verify_request),
) -> Dict[str, object]:
  if REQUIRE_CONSENT and not consent:
    raise HTTPException(status_code=400, detail="User consent is required.")

  transcript = question.strip()
  if not transcript:
    raise HTTPException(status_code=400, detail="Question is empty.")
  if len(transcript) > MAX_TEXT_CHARS:
    raise HTTPException(status_code=413, detail="Question is too long.")
  detected = source_language if source_language != "auto" else "ja"
  return build_guide_response(transcript, detected, target_language)
