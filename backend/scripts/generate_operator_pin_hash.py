#!/usr/bin/env python3
import argparse
import getpass
import hashlib
import secrets


def main() -> None:
  parser = argparse.ArgumentParser(description="Generate OPERATOR_PIN_HASH for Voyce backend.")
  parser.add_argument("--iterations", type=int, default=310000, help="PBKDF2 iterations (default: 310000)")
  parser.add_argument("--salt", type=str, default="", help="Optional fixed salt (for deterministic output)")
  args = parser.parse_args()

  pin = getpass.getpass("Operator PIN: ").strip()
  if not pin:
    raise SystemExit("PIN is required.")

  salt = args.salt.strip() or secrets.token_hex(16)
  digest = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt.encode("utf-8"), args.iterations).hex()
  print(f"OPERATOR_PIN_HASH=pbkdf2_sha256${args.iterations}${salt}${digest}")


if __name__ == "__main__":
  main()
