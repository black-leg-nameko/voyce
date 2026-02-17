import os
import sys
import re


def fail(message: str) -> None:
  print(f"[env-check] ERROR: {message}")
  sys.exit(1)


def validate_common() -> None:
  required = ["API_KEYS", "CORS_ALLOWED_ORIGINS"]
  for key in required:
    if not os.getenv(key, "").strip():
      fail(f"{key} is required")


def validate_secure_defaults() -> None:
  if os.getenv("ENFORCE_HTTPS", "false").lower() != "true":
    fail("ENFORCE_HTTPS must be true in stg/prod")
  if os.getenv("ARGOS_ALLOW_RUNTIME_INSTALL", "false").lower() == "true":
    fail("ARGOS_ALLOW_RUNTIME_INSTALL must be false in stg/prod")
  if os.getenv("REQUIRE_CONSENT", "true").lower() != "true":
    fail("REQUIRE_CONSENT must be true in stg/prod")
  if os.getenv("ALLOW_API_KEY_AUTH", "true").lower() == "true":
    fail("ALLOW_API_KEY_AUTH must be false in stg/prod")


def validate_redis_url() -> None:
  redis_url = os.getenv("REDIS_URL", "").strip()
  if not redis_url:
    fail("REDIS_URL must be set")
  if not redis_url.startswith(("redis://", "rediss://")):
    fail("REDIS_URL must start with redis:// or rediss://")


def validate_operator_pin_hash() -> None:
  operator_pin_plain = os.getenv("OPERATOR_PIN", "").strip()
  if operator_pin_plain:
    fail("OPERATOR_PIN plaintext is not allowed in stg/prod")

  operator_pin_hash = os.getenv("OPERATOR_PIN_HASH", "").strip()
  if not operator_pin_hash:
    fail("OPERATOR_PIN_HASH must be set in stg/prod")

  pattern = r"^pbkdf2_sha256\$\d+\$[^$]+\$[0-9a-fA-F]{64}$"
  if not re.match(pattern, operator_pin_hash):
    fail("OPERATOR_PIN_HASH format must be pbkdf2_sha256$<iterations>$<salt>$<64hex>")


def validate_prod() -> None:
  validate_common()
  validate_secure_defaults()

  api_keys = [v.strip() for v in os.getenv("API_KEYS", "").split(",") if v.strip()]
  if any(v in {"dev-local-key", "replace-with-prod-key", "replace-with-stg-key"} for v in api_keys):
    fail("API_KEYS contains default or placeholder values")

  metrics_token = os.getenv("METRICS_TOKEN", "").strip()
  if not metrics_token or metrics_token.startswith("replace-with"):
    fail("METRICS_TOKEN must be set to a strong secret in prod")
  ready_token = os.getenv("READY_TOKEN", "").strip()
  if not ready_token or ready_token.startswith("replace-with"):
    fail("READY_TOKEN must be set to a strong secret in prod")
  auth_secret = os.getenv("AUTH_TOKEN_SECRET", "").strip()
  if not auth_secret or auth_secret.startswith("replace-with"):
    fail("AUTH_TOKEN_SECRET must be set to a strong secret in prod")
  min_auth_secret_length = int(os.getenv("MIN_AUTH_SECRET_LENGTH", "32"))
  if len(auth_secret) < min_auth_secret_length:
    fail(f"AUTH_TOKEN_SECRET must be at least {min_auth_secret_length} chars in prod")
  validate_operator_pin_hash()
  validate_redis_url()

  origins = os.getenv("CORS_ALLOWED_ORIGINS", "")
  if "localhost" in origins or "127.0.0.1" in origins:
    fail("CORS_ALLOWED_ORIGINS must not include localhost in prod")


def validate_stg() -> None:
  validate_common()
  validate_secure_defaults()
  validate_redis_url()
  auth_secret = os.getenv("AUTH_TOKEN_SECRET", "").strip()
  if not auth_secret:
    fail("AUTH_TOKEN_SECRET is required in stg")
  min_auth_secret_length = int(os.getenv("MIN_AUTH_SECRET_LENGTH", "32"))
  if len(auth_secret) < min_auth_secret_length:
    fail(f"AUTH_TOKEN_SECRET must be at least {min_auth_secret_length} chars in stg")
  validate_operator_pin_hash()


def main() -> None:
  env = os.getenv("APP_ENV", "dev").lower().strip()
  if env == "prod":
    validate_prod()
  elif env == "stg":
    validate_stg()
  else:
    validate_common()

  print(f"[env-check] OK (APP_ENV={env})")


if __name__ == "__main__":
  main()
