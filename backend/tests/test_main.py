from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
import hashlib

import main


@pytest.fixture(autouse=True)
def reset_state():
  main.MEMORY_RATE_BUCKETS.clear()
  main.MEMORY_SESSION_EXPIRY.clear()
  main.REQUEST_COUNTERS.clear()
  main.REQUEST_LATENCY_MS.clear()


def test_validate_upload_empty_file():
  upload = SimpleNamespace(content_type="audio/m4a")
  with pytest.raises(HTTPException) as exc:
    main.validate_upload(upload, b"", ".m4a")
  assert exc.value.status_code == 400


def test_validate_upload_too_large(monkeypatch):
  monkeypatch.setattr(main, "MAX_AUDIO_BYTES", 4)
  upload = SimpleNamespace(content_type="audio/m4a")
  with pytest.raises(HTTPException) as exc:
    main.validate_upload(upload, b"12345", ".m4a")
  assert exc.value.status_code == 413


def test_guide_unauthorized():
  client = TestClient(main.app)
  response = client.post(
    "/guide",
    files={"file": ("q.m4a", b"abc", "audio/m4a")},
    data={"source_language": "auto", "target_language": "en", "consent": "true"},
  )
  assert response.status_code == 401


def test_guide_success(monkeypatch):
  client = TestClient(main.app)

  monkeypatch.setattr(main, "get_whisper_model", lambda: SimpleNamespace(transcribe=lambda *args, **kwargs: ([SimpleNamespace(text="こんにちは")], SimpleNamespace(language="ja"))))
  monkeypatch.setattr(
    main,
    "generate_answer_ja",
    lambda transcript: {"answer": "テスト回答", "confidence": 96.0, "topic": "営業時間", "matched_keyword": "営業時間"},
  )
  monkeypatch.setattr(main, "translate_text", lambda text, from_code, to_code, allow_runtime_install=False: f"{text}-{to_code}")

  response = client.post(
    "/guide",
    headers={"x-api-key": "dev-local-key"},
    files={"file": ("q.m4a", b"abc", "audio/m4a")},
    data={"source_language": "auto", "target_language": "en", "consent": "true"},
  )

  assert response.status_code == 200
  payload = response.json()
  assert payload["transcript"] == "こんにちは"
  assert payload["answer_source_ja"] == "テスト回答"
  assert payload["answer_translated"] == "テスト回答-en"
  assert payload["answer_confidence"] == 96.0
  assert payload["answer_topic"] == "営業時間"
  assert payload["matched_keyword"] == "営業時間"


def test_generate_answer_ja_low_confidence(monkeypatch):
  monkeypatch.setattr(main, "FAQ_DATA", [{"keywords": ["営業時間"], "answer": "営業時間は9時からです。"}])
  result = main.generate_answer_ja("今日は天気がいいですね")
  assert result["topic"] == "clarification"
  assert "どちら" in result["answer"] or "知りたいですか" in result["answer"]


def test_generate_answer_ja_high_confidence(monkeypatch):
  monkeypatch.setattr(main, "FAQ_DATA", [{"keywords": ["駐車場", "parking"], "answer": "駐車場はあります。"}])
  result = main.generate_answer_ja("駐車場はありますか")
  assert result["topic"] == "駐車場"
  assert result["answer"] == "駐車場はあります。"


def test_assess_translation_quality_unchanged_text():
  result = main.assess_translation_quality("営業時間は？", "営業時間は？", "en")
  assert result["ok"] is False
  assert result["reason"] == "unchanged_text"


def test_guide_translation_fallback(monkeypatch):
  client = TestClient(main.app)

  monkeypatch.setattr(main, "get_whisper_model", lambda: SimpleNamespace(transcribe=lambda *args, **kwargs: ([SimpleNamespace(text="こんにちは")], SimpleNamespace(language="ja"))))
  monkeypatch.setattr(
    main,
    "generate_answer_ja",
    lambda transcript: {"answer": "営業時間は9時からです。", "confidence": 93.0, "topic": "営業時間", "matched_keyword": "営業時間"},
  )
  monkeypatch.setattr(main, "translate_text", lambda text, from_code, to_code, allow_runtime_install=False: text)

  response = client.post(
    "/guide",
    headers={"x-api-key": "dev-local-key"},
    files={"file": ("q.m4a", b"abc", "audio/m4a")},
    data={"source_language": "auto", "target_language": "en", "consent": "true"},
  )
  assert response.status_code == 200
  payload = response.json()
  assert payload["translation_fallback"] is True
  assert payload["output_translation_quality"]["ok"] is False


def test_guide_unsupported_target_language():
  client = TestClient(main.app)
  response = client.post(
    "/guide",
    headers={"x-api-key": "dev-local-key"},
    files={"file": ("q.m4a", b"abc", "audio/m4a")},
    data={"source_language": "auto", "target_language": "xx", "consent": "true"},
  )
  assert response.status_code == 400


def test_guide_text_success(monkeypatch):
  client = TestClient(main.app)
  monkeypatch.setattr(
    main,
    "generate_answer_ja",
    lambda transcript: {"answer": "テキスト回答", "confidence": 91.0, "topic": "案内", "matched_keyword": "案内"},
  )
  monkeypatch.setattr(main, "translate_text", lambda text, from_code, to_code, allow_runtime_install=False: f"{text}-{to_code}")

  response = client.post(
    "/guide/text",
    headers={"x-api-key": "dev-local-key"},
    data={"question": "営業時間を教えて", "source_language": "ja", "target_language": "en", "consent": "true"},
  )
  assert response.status_code == 200
  payload = response.json()
  assert payload["answer_source_ja"] == "テキスト回答"


def test_auth_session_success(monkeypatch):
  client = TestClient(main.app)
  monkeypatch.setattr(main, "OPERATOR_PIN", "1234")
  monkeypatch.setattr(main, "OPERATOR_PIN_HASH", "")
  monkeypatch.setattr(main, "AUTH_TOKEN_SECRET", "secret")
  response = client.post("/auth/session", data={"operator_pin": "1234"})
  assert response.status_code == 200
  payload = response.json()
  assert payload["token_type"] == "bearer"
  assert "access_token" in payload


def test_auth_session_success_with_pin_hash(monkeypatch):
  client = TestClient(main.app)
  pin = "5678"
  salt = "testsalt"
  iterations = 1000
  digest = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt.encode("utf-8"), iterations).hex()
  monkeypatch.setattr(main, "OPERATOR_PIN", "")
  monkeypatch.setattr(main, "OPERATOR_PIN_HASH", f"pbkdf2_sha256${iterations}${salt}${digest}")
  monkeypatch.setattr(main, "AUTH_TOKEN_SECRET", "secret")
  response = client.post("/auth/session", data={"operator_pin": pin})
  assert response.status_code == 200


def test_get_client_ip_trust_proxy(monkeypatch):
  monkeypatch.setattr(main, "TRUSTED_PROXY_IPS", {"10.0.0.1"})
  request = SimpleNamespace(headers={"x-forwarded-for": "198.51.100.2"}, client=SimpleNamespace(host="10.0.0.1"))
  assert main.get_client_ip(request) == "198.51.100.2"


def test_get_client_ip_ignore_spoofed_header(monkeypatch):
  monkeypatch.setattr(main, "TRUSTED_PROXY_IPS", {"10.0.0.1"})
  request = SimpleNamespace(headers={"x-forwarded-for": "198.51.100.2"}, client=SimpleNamespace(host="203.0.113.10"))
  assert main.get_client_ip(request) == "203.0.113.10"


def test_guide_text_too_long(monkeypatch):
  client = TestClient(main.app)
  monkeypatch.setattr(main, "MAX_TEXT_CHARS", 5)
  response = client.post(
    "/guide/text",
    headers={"x-api-key": "dev-local-key"},
    data={"question": "123456", "source_language": "ja", "target_language": "en", "consent": "true"},
  )
  assert response.status_code == 413
