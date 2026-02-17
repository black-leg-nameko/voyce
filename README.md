# Voyce - 観光地向け 音声ガイド + 翻訳

1. 音声質問を録音
2. `faster-whisper` (無料の学習済み音声認識) で文字起こし
3. `argostranslate` (無料の翻訳モデル) で多言語化
4. Expoの `expo-speech` で音声返答

## ディレクトリ構成

- `app/`: Expoアプリ
- `backend/`: Whisper + 翻訳API

## 1) バックエンド起動

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.dev.example .env
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

初回はWhisperモデルと翻訳パッケージのダウンロードが入るため、起動と初回リクエストに時間がかかります。
本番では `ARGOS_ALLOW_RUNTIME_INSTALL=false` を維持し、必要な翻訳パッケージは `backend/preload_argos.py` で事前導入してください。
アプリ利用前に `POST /auth/session` で運用者PINからBearerトークンを取得してください。
本番/ステージングは単一インスタンス前提ではなく、セッション・レート制御を Redis で共有します（`REDIS_URL` 必須）。

## 2) Expoアプリ起動

```bash
cd app
npm install
cp .env.development.example .env.development
npm run start
```

- 実機で検証する場合、`<PCのLAN_IP>` をPCのローカルIPにしてください。
- iOS/Androidでマイク権限を許可してください。
- 本番配布では `EXPO_PUBLIC_API_BASE_URL` は必ず `https://` を使用してください。

## API

- `GET /health`
- `GET /ready`（運用向けの準備状態チェック、`x-ready-token` で保護可能）
- `GET /metrics`（運用向け、`x-metrics-token` で保護）
- `POST /auth/session`（運用者PINでBearerトークン発行）
  - `operator_pin`: 運用者PIN
- `POST /auth/revoke`（現在のBearerトークン失効）
- `POST /guide` (`multipart/form-data`)
  - header: `Authorization: Bearer <token>`（推奨）
  - header: `x-api-key: <your-api-key>`（開発互換）
  - `file`: 音声ファイル
  - `source_language`: `auto` など
  - `target_language`: `en`, `ja`, `zh`, `ko`, `fr`, `es`
  - 音声ファイルはサイズ・拡張子・MIMEタイプの検証があります（`backend/.env` で調整可能）。
  - 混雑時は `503`（キュー満杯）や `504`（推論タイムアウト）を返します。
  - アプリ側は `401/429/503/504` を判別表示し、再送信ボタンを表示します。
  - FAQ応答は信頼度判定付きで、曖昧な質問には確認質問へフォールバックします。
  - 翻訳品質が低い場合、`translation_fallback=true` で日本語回答に安全フォールバックします。
- `POST /guide/text` (`multipart/form-data`)
  - header: `Authorization: Bearer <token>`（推奨）
  - header: `x-api-key: <your-api-key>`（開発互換）
  - `question`: テキスト質問
  - `source_language`: `ja` など
  - `target_language`: `en`, `ja`, `zh`, `ko`, `fr`, `es`
  - `consent`: `true` 必須（本番設定）

## 補足

FAQベース回答は `backend/data/faq_ja.json` を編集すると、現地施設ごとの案内に差し替えできます。
本番では `OPERATOR_PIN` の平文利用を避け、`OPERATOR_PIN_HASH` を使ってください（`backend/scripts/generate_operator_pin_hash.py`）。
個人情報保護方針のドラフトは `PRIVACY_POLICY.md` を参照してください。
アプリにはアクセシビリティ向上のため、テキスト質問導線と読み上げ停止ボタンを実装しています。

## 本番配備

ローカルPoCから本番移行する場合は `DEPLOYMENT.md` を参照してください。
`backend` のコンテナ化、`docker-compose.prod.yml`、環境別テンプレート (`backend/.env.dev.example`, `backend/.env.stg.example`, `backend/.env.prod.example`) を用意しています。

## テスト

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
```

## CI/CD

- GitHub Actions: `.github/workflows/ci.yml`
- `push` / `pull_request` 時に以下を自動実行します。
  - Backend: 依存インストール、`py_compile`、`pytest`
  - Mobile: 依存インストール、`npx tsc --noEmit`

## 設定管理

- 環境分離とシークレット運用ルールは `CONFIGURATION.md` を参照してください。

## 運用Runbook

- 日次監視、障害初動、FAQバックアップは `OPS_RUNBOOK.md` を参照してください。

## リリース判定

- リリース可否の最終確認は `RELEASE_GATE.md` を使用してください。

## ストア提出

Expo/EAS での提出チェックは `app/STORE_SUBMISSION.md` を参照してください。
設定テンプレートは `app/eas.json` と `app/store.config.example.json` です。
