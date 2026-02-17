import os

import argostranslate.package


def parse_pairs(raw: str):
  pairs = []
  for item in raw.split(","):
    value = item.strip()
    if "-" not in value:
      continue
    from_code, to_code = value.split("-", 1)
    pairs.append((from_code.strip(), to_code.strip()))
  return pairs


def main():
  pairs_raw = os.getenv(
    "ARGOS_LANGUAGE_PAIRS",
    "ja-en,en-ja,ja-zh,zh-ja,ja-ko,ko-ja,ja-fr,fr-ja,ja-es,es-ja",
  )
  pairs = parse_pairs(pairs_raw)

  argostranslate.package.update_package_index()
  available = argostranslate.package.get_available_packages()

  installed = 0
  skipped = 0
  for from_code, to_code in pairs:
    pkg = next((p for p in available if p.from_code == from_code and p.to_code == to_code), None)
    if not pkg:
      print(f"[skip] package not found: {from_code}->{to_code}")
      skipped += 1
      continue

    print(f"[install] {from_code}->{to_code}")
    path = pkg.download()
    argostranslate.package.install_from_path(path)
    installed += 1

  print(f"done installed={installed} skipped={skipped}")


if __name__ == "__main__":
  main()
