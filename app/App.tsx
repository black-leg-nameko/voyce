import { useMemo, useState } from 'react';
import { ActivityIndicator, Alert, Pressable, SafeAreaView, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { Audio } from 'expo-av';
import * as Speech from 'expo-speech';
import { StatusBar } from 'expo-status-bar';

type GuideResponse = {
  transcript: string;
  answer_source_ja: string;
  answer_translated: string;
  source_language: string;
  target_language: string;
};

const API_BASE_URL = process.env.EXPO_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';
const DEV_API_KEY = process.env.EXPO_PUBLIC_API_KEY || '';
const ALLOW_INSECURE_HTTP = process.env.EXPO_PUBLIC_ALLOW_INSECURE_HTTP === 'true';
const REQUEST_TIMEOUT_MS = 45000;

type LastRequest = {
  kind: 'audio' | 'text';
  payload: string;
  targetLang: (typeof TARGET_LANGUAGES)[number]['code'];
};

const TARGET_LANGUAGES = [
  { label: 'English', code: 'en', speech: 'en-US' },
  { label: '日本語', code: 'ja', speech: 'ja-JP' },
  { label: '中文', code: 'zh', speech: 'zh-CN' },
  { label: '한국어', code: 'ko', speech: 'ko-KR' },
  { label: 'Français', code: 'fr', speech: 'fr-FR' },
  { label: 'Español', code: 'es', speech: 'es-ES' },
] as const;

export default function App() {
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<GuideResponse | null>(null);
  const [targetLang, setTargetLang] = useState<(typeof TARGET_LANGUAGES)[number]['code']>('en');
  const [consentAccepted, setConsentAccepted] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [lastRequest, setLastRequest] = useState<LastRequest | null>(null);
  const [manualQuestion, setManualQuestion] = useState('');
  const [operatorPin, setOperatorPin] = useState('');
  const [authToken, setAuthToken] = useState('');

  const selectedSpeechLang = useMemo(
    () => TARGET_LANGUAGES.find((item) => item.code === targetLang)?.speech || 'en-US',
    [targetLang],
  );

  const startRecording = async () => {
    try {
      setErrorMessage('');
      if (!consentAccepted) {
        Alert.alert('同意が必要です', '録音・文字起こし・翻訳処理に関する同意を先に有効にしてください。');
        return;
      }

      setResult(null);
      const permission = await Audio.requestPermissionsAsync();
      if (!permission.granted) {
        Alert.alert('権限エラー', 'マイク権限が必要です。');
        return;
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      const rec = new Audio.Recording();
      await rec.prepareToRecordAsync(Audio.RecordingOptionsPresets.HIGH_QUALITY);
      await rec.startAsync();
      setRecording(rec);
    } catch (error) {
      Alert.alert('録音開始エラー', String(error));
    }
  };

  const buildErrorMessage = (status: number, detail: string) => {
    if (status === 401) return '認証エラーです。APIキー設定を確認してください。';
    if (status === 429) return 'アクセスが集中しています。少し待ってから再試行してください。';
    if (status === 503) return 'サーバーが混雑しています。時間をおいて再試行してください。';
    if (status === 504) return '処理がタイムアウトしました。短めの音声で再試行してください。';
    if (status >= 500) return 'サーバーエラーが発生しました。時間をおいて再試行してください。';
    return detail || `リクエストエラー (HTTP ${status})`;
  };

  const ensureSecureApiUrl = () => {
    const isHttp = API_BASE_URL.startsWith('http://');
    const isLocalDevTarget = API_BASE_URL.includes('127.0.0.1') || API_BASE_URL.includes('localhost') || API_BASE_URL.includes('192.168.');
    if (isHttp && !(__DEV__ && ALLOW_INSECURE_HTTP && isLocalDevTarget)) {
      throw new Error('本番ではHTTPSのAPI URLが必要です。EXPO_PUBLIC_API_BASE_URLをhttps://で設定してください。');
    }
  };

  const sendAudio = async (uri: string, targetLanguage: (typeof TARGET_LANGUAGES)[number]['code']) => {
    setErrorMessage('');
    ensureSecureApiUrl();

    if (!authToken && !DEV_API_KEY) {
      throw new Error('先に運用者PINで認証してください。');
    }

    const formData = new FormData();
    formData.append('file', {
      uri,
      name: 'question.m4a',
      type: 'audio/m4a',
    } as any);
    formData.append('source_language', 'auto');
    formData.append('target_language', targetLanguage);
    formData.append('consent', consentAccepted ? 'true' : 'false');

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    try {
      const response = await fetch(`${API_BASE_URL}/guide`, {
        method: 'POST',
        headers: {
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
          ...(DEV_API_KEY ? { 'x-api-key': DEV_API_KEY } : {}),
        },
        body: formData,
        signal: controller.signal,
      });

      if (!response.ok) {
        const bodyText = await response.text();
        throw new Error(buildErrorMessage(response.status, bodyText));
      }

      const json: GuideResponse = await response.json();
      setResult(json);
      setLastRequest({ kind: 'audio', payload: uri, targetLang: targetLanguage });
      const speechLanguage = TARGET_LANGUAGES.find((item) => item.code === targetLanguage)?.speech || selectedSpeechLang;

      await Speech.speak(json.answer_translated, {
        language: speechLanguage,
        pitch: 1,
        rate: 0.95,
      });
    } catch (error: any) {
      const message = String(error?.message || error);
      if (message.includes('aborted') || message.includes('AbortError')) {
        throw new Error('通信がタイムアウトしました。通信状況を確認して再試行してください。');
      }
      throw new Error(message);
    } finally {
      clearTimeout(timeout);
    }
  };

  const sendTextQuestion = async (question: string, targetLanguage: (typeof TARGET_LANGUAGES)[number]['code']) => {
    setErrorMessage('');
    ensureSecureApiUrl();
    if (!authToken && !DEV_API_KEY) {
      throw new Error('先に運用者PINで認証してください。');
    }

    const text = question.trim();
    if (!text) {
      throw new Error('テキスト質問が空です。');
    }

    const formData = new FormData();
    formData.append('question', text);
    formData.append('source_language', 'ja');
    formData.append('target_language', targetLanguage);
    formData.append('consent', consentAccepted ? 'true' : 'false');

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    try {
      const response = await fetch(`${API_BASE_URL}/guide/text`, {
        method: 'POST',
        headers: {
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
          ...(DEV_API_KEY ? { 'x-api-key': DEV_API_KEY } : {}),
        },
        body: formData,
        signal: controller.signal,
      });
      if (!response.ok) {
        const bodyText = await response.text();
        throw new Error(buildErrorMessage(response.status, bodyText));
      }
      const json: GuideResponse = await response.json();
      setResult(json);
      setLastRequest({ kind: 'text', payload: text, targetLang: targetLanguage });
      const speechLanguage = TARGET_LANGUAGES.find((item) => item.code === targetLanguage)?.speech || selectedSpeechLang;
      await Speech.speak(json.answer_translated, {
        language: speechLanguage,
        pitch: 1,
        rate: 0.95,
      });
    } catch (error: any) {
      const message = String(error?.message || error);
      if (message.includes('aborted') || message.includes('AbortError')) {
        throw new Error('通信がタイムアウトしました。通信状況を確認して再試行してください。');
      }
      throw new Error(message);
    } finally {
      clearTimeout(timeout);
    }
  };

  const stopAndSend = async () => {
    if (!recording) {
      return;
    }

    setLoading(true);
    try {
      await recording.stopAndUnloadAsync();
      const uri = recording.getURI();
      setRecording(null);

      if (!uri) {
        throw new Error('録音ファイルが見つかりませんでした。');
      }

      await sendAudio(uri, targetLang);
    } catch (error) {
      const message = String(error);
      setErrorMessage(message);
      Alert.alert('送信エラー', message);
    } finally {
      setLoading(false);
    }
  };

  const retryLastRequest = async () => {
    if (!lastRequest) {
      return;
    }
    setLoading(true);
    try {
      if (lastRequest.kind === 'audio') {
        await sendAudio(lastRequest.payload, lastRequest.targetLang);
      } else {
        await sendTextQuestion(lastRequest.payload, lastRequest.targetLang);
      }
    } catch (error) {
      const message = String(error);
      setErrorMessage(message);
      Alert.alert('再試行エラー', message);
    } finally {
      setLoading(false);
    }
  };

  const submitManualQuestion = async () => {
    if (!consentAccepted) {
      Alert.alert('同意が必要です', '録音・文字起こし・翻訳処理に関する同意を先に有効にしてください。');
      return;
    }
    setLoading(true);
    try {
      await sendTextQuestion(manualQuestion, targetLang);
    } catch (error) {
      const message = String(error);
      setErrorMessage(message);
      Alert.alert('送信エラー', message);
    } finally {
      setLoading(false);
    }
  };

  const authenticateOperator = async () => {
    ensureSecureApiUrl();
    const pin = operatorPin.trim();
    if (!pin) {
      Alert.alert('認証エラー', '運用者PINを入力してください。');
      return;
    }
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('operator_pin', pin);
      const response = await fetch(`${API_BASE_URL}/auth/session`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || `HTTP ${response.status}`);
      }
      const json = await response.json();
      setAuthToken(String(json.access_token || ''));
      setOperatorPin('');
      Alert.alert('認証完了', 'セッショントークンを取得しました。');
    } catch (error) {
      Alert.alert('認証失敗', String(error));
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="dark" />
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>Voyce Guide</Text>
        <Text style={styles.subtitle}>観光地向け 音声ガイド + 翻訳</Text>

        <View style={styles.authCard}>
          <Text style={styles.sectionTitle}>運用者認証</Text>
          <TextInput
            style={styles.pinInput}
            value={operatorPin}
            onChangeText={setOperatorPin}
            placeholder="運用者PIN"
            secureTextEntry
            accessibilityLabel="運用者PIN入力"
          />
          <Pressable style={styles.authButton} onPress={authenticateOperator} accessibilityRole="button" accessibilityLabel="運用者PINで認証">
            <Text style={styles.authButtonText}>{authToken ? '再認証する' : '認証する'}</Text>
          </Pressable>
          <Text style={styles.authStatus}>{authToken ? '認証状態: 有効' : '認証状態: 未認証'}</Text>
        </View>

        <View style={styles.langWrap}>
          {TARGET_LANGUAGES.map((lang) => (
            <Pressable
              key={lang.code}
              onPress={() => setTargetLang(lang.code)}
              style={[styles.langButton, targetLang === lang.code && styles.langButtonActive]}
              accessibilityRole="button"
              accessibilityLabel={`回答言語を${lang.label}に設定`}
            >
              <Text style={[styles.langText, targetLang === lang.code && styles.langTextActive]}>{lang.label}</Text>
            </Pressable>
          ))}
        </View>

        <View style={styles.recordCard}>
          <Pressable style={styles.consentRow} onPress={() => setConsentAccepted((prev) => !prev)} accessibilityRole="checkbox" accessibilityState={{ checked: consentAccepted }} accessibilityLabel="音声データ処理への同意">
            <View style={[styles.checkbox, consentAccepted && styles.checkboxActive]}>
              {consentAccepted && <Text style={styles.checkboxMark}>✓</Text>}
            </View>
            <Text style={styles.consentText}>音声データの処理に同意する（保存しない一時処理）</Text>
          </Pressable>
          <Text style={styles.recordLabel}>{recording ? '録音中...' : 'マイクを押して質問してください'}</Text>
          <Pressable style={[styles.micButton, recording && styles.micButtonActive]} onPress={recording ? stopAndSend : startRecording} accessibilityRole="button" accessibilityLabel={recording ? '録音停止して送信' : '録音開始'}>
            <Text style={styles.micText}>{recording ? '停止して送信' : '録音開始'}</Text>
          </Pressable>
        </View>

        <View style={styles.manualCard}>
          <Text style={styles.sectionTitle}>テキストで質問する</Text>
          <TextInput
            style={styles.manualInput}
            value={manualQuestion}
            onChangeText={setManualQuestion}
            placeholder="例: 駐車場はありますか？"
            multiline
            accessibilityLabel="テキスト質問入力"
          />
          <Pressable style={styles.manualSendButton} onPress={submitManualQuestion} accessibilityRole="button" accessibilityLabel="テキスト質問を送信">
            <Text style={styles.manualSendText}>テキスト送信</Text>
          </Pressable>
        </View>

        {loading && (
          <View style={styles.loading}>
            <ActivityIndicator />
            <Text style={styles.loadingText}>Whisperで認識して翻訳中...</Text>
          </View>
        )}

        {result && (
          <View style={styles.resultCard}>
            <Text style={styles.sectionTitle}>認識結果</Text>
            <Text style={styles.resultText}>{result.transcript}</Text>
            <Text style={styles.sectionTitle}>回答（翻訳後）</Text>
            <Text style={styles.resultText}>{result.answer_translated}</Text>
            <Pressable style={styles.stopSpeechButton} onPress={() => Speech.stop()} accessibilityRole="button" accessibilityLabel="読み上げを停止">
              <Text style={styles.stopSpeechText}>読み上げ停止</Text>
            </Pressable>
          </View>
        )}

        {!!errorMessage && (
          <View style={styles.errorCard}>
            <Text style={styles.errorTitle}>エラー</Text>
            <Text style={styles.errorText}>{errorMessage}</Text>
            {!!lastRequest && (
              <Pressable style={styles.retryButton} onPress={retryLastRequest} accessibilityRole="button" accessibilityLabel="前回リクエストを再送信">
                <Text style={styles.retryText}>再送信する</Text>
              </Pressable>
            )}
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F3EF',
  },
  content: {
    padding: 16,
    gap: 16,
  },
  title: {
    fontSize: 32,
    fontWeight: '800',
    color: '#0B3C49',
  },
  subtitle: {
    fontSize: 15,
    color: '#476268',
  },
  langWrap: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  langButton: {
    backgroundColor: '#DDE8EA',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 999,
  },
  langButtonActive: {
    backgroundColor: '#0B3C49',
  },
  langText: {
    color: '#0B3C49',
    fontWeight: '600',
  },
  langTextActive: {
    color: '#FFFFFF',
  },
  recordCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 16,
    gap: 12,
  },
  consentRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  checkbox: {
    width: 22,
    height: 22,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#6B7C80',
    alignItems: 'center',
    justifyContent: 'center',
  },
  checkboxActive: {
    backgroundColor: '#0B3C49',
    borderColor: '#0B3C49',
  },
  checkboxMark: {
    color: '#FFFFFF',
    fontWeight: '800',
  },
  consentText: {
    flex: 1,
    color: '#29444A',
    fontSize: 13,
    lineHeight: 18,
  },
  recordLabel: {
    color: '#29444A',
    fontSize: 16,
    fontWeight: '600',
  },
  micButton: {
    backgroundColor: '#1A7A87',
    borderRadius: 14,
    paddingVertical: 16,
    alignItems: 'center',
  },
  micButtonActive: {
    backgroundColor: '#B23C17',
  },
  micText: {
    color: '#FFFFFF',
    fontSize: 17,
    fontWeight: '700',
  },
  loading: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  loadingText: {
    color: '#29444A',
  },
  authCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 16,
    gap: 10,
  },
  pinInput: {
    borderWidth: 1,
    borderColor: '#BDD2D7',
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 10,
    color: '#162D32',
  },
  authButton: {
    backgroundColor: '#1A7A87',
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: 'center',
  },
  authButtonText: {
    color: '#FFFFFF',
    fontWeight: '700',
  },
  authStatus: {
    color: '#29444A',
    fontSize: 12,
  },
  manualCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 16,
    gap: 10,
  },
  manualInput: {
    minHeight: 96,
    borderWidth: 1,
    borderColor: '#BDD2D7',
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 10,
    color: '#162D32',
    textAlignVertical: 'top',
  },
  manualSendButton: {
    backgroundColor: '#0B3C49',
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: 'center',
  },
  manualSendText: {
    color: '#FFFFFF',
    fontWeight: '700',
  },
  resultCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 16,
    gap: 8,
  },
  sectionTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: '#0B3C49',
  },
  resultText: {
    color: '#162D32',
    lineHeight: 22,
  },
  stopSpeechButton: {
    marginTop: 8,
    backgroundColor: '#445B61',
    borderRadius: 10,
    paddingVertical: 10,
    alignItems: 'center',
  },
  stopSpeechText: {
    color: '#FFFFFF',
    fontWeight: '700',
  },
  errorCard: {
    backgroundColor: '#FFF3EF',
    borderRadius: 16,
    padding: 16,
    gap: 10,
    borderWidth: 1,
    borderColor: '#E3B09F',
  },
  errorTitle: {
    color: '#8B2C10',
    fontWeight: '700',
  },
  errorText: {
    color: '#6E2A16',
    lineHeight: 20,
  },
  retryButton: {
    backgroundColor: '#8B2C10',
    paddingVertical: 10,
    borderRadius: 10,
    alignItems: 'center',
  },
  retryText: {
    color: '#FFFFFF',
    fontWeight: '700',
  },
});
