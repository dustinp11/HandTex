import { useState, useRef, useEffect } from 'react';
import {
  StyleSheet,
  View,
  Text,
  TouchableOpacity,
  PanResponder,
  ScrollView,
  TextInput,
  Alert,
  Platform,
} from 'react-native';
import { SafeAreaView, SafeAreaProvider } from 'react-native-safe-area-context';
import Svg, { Path } from 'react-native-svg';
import { captureRef } from 'react-native-view-shot';
import * as FileSystem from 'expo-file-system';
import { StatusBar } from 'expo-status-bar';
import Constants from 'expo-constants';

const HISTORY_FILE = FileSystem.documentDirectory + 'handtex_history.json';
const MONO = Platform.OS === 'ios' ? 'Courier' : 'monospace';

// Auto-detect backend URL from the Expo dev server host (same machine)
function getDefaultBackendUrl() {
  const host =
    Constants.manifest2?.extra?.expoGo?.debuggerHost ??
    Constants.expoConfig?.hostUri ??
    '';
  const ip = host.split(':')[0];
  if (ip) return `http://${ip}:5000`;
  return process.env.EXPO_PUBLIC_BACKEND_URL ?? 'http://localhost:5000';
}

import { WebView } from "react-native-webview";

function LatexView({ latex, display = true }) {
  const safeLatex = (latex ?? "")
    .replace(/\\/g, "\\\\")
    .replace(/`/g, "\\`")
    .replace(/\$\{/g, "\\${"); // avoid template literal interpolation edge cases

  const html = `
  <!doctype html>
  <html>
    <head>
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
      <script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
      <style>
        html, body { margin: 0; padding: 0; background: transparent; }
        #wrap { padding: 10px; font-size: 22px; color: #111; }
      </style>
    </head>
    <body>
      <div id="wrap">Loading…</div>
      <script>
        function renderNow() {
          const el = document.getElementById("wrap");
          const expr = \`${safeLatex}\`;

          if (!window.katex) {
            el.textContent = "KaTeX failed to load (no internet / blocked CDN).";
            return;
          }

          try {
            window.katex.render(expr, el, {
              displayMode: ${display ? "true" : "false"},
              throwOnError: false
            });
          } catch (e) {
            el.textContent = "KaTeX error: " + e.message;
          }
        }

        // Wait until katex exists (CDN can be slightly delayed)
        (function waitForKatex() {
          let tries = 0;
          const t = setInterval(() => {
            tries++;
            if (window.katex || tries > 40) {
              clearInterval(t);
              renderNow();
            }
          }, 50);
        })();
      </script>
    </body>
  </html>`;

  return (
    <WebView
      originWhitelist={["*"]}
      source={{ html }}
      javaScriptEnabled
      domStorageEnabled
      mixedContentMode="always"
      scrollEnabled={false}
      style={{ flex: 1, backgroundColor: "transparent" }}
    />
  );
}

function pointsToSvgPath(points) {
  if (points.length === 0) return '';
  const [first, ...rest] = points;
  if (rest.length === 0) return `M${first.x},${first.y} l0.1,0.1`;
  return `M${first.x},${first.y} ` + rest.map(p => `L${p.x},${p.y}`).join(' ');
}

export default function App() {
  const [paths, setPaths] = useState([]);
  const [currentPoints, setCurrentPoints] = useState([]);
  const [latexResult, setLatexResult] = useState('');
  const [history, setHistory] = useState([]);
  const [backendUrl, setBackendUrl] = useState(getDefaultBackendUrl);
  const [loading, setLoading] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  const canvasRef = useRef(null);
  const currentPointsRef = useRef([]);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const info = await FileSystem.getInfoAsync(HISTORY_FILE);
      if (info.exists) {
        const content = await FileSystem.readAsStringAsync(HISTORY_FILE);
        setHistory(JSON.parse(content));
      }
    } catch (e) {
      console.log('Error loading history:', e);
    }
  };

  const saveHistory = async (newHistory) => {
    try {
      await FileSystem.writeAsStringAsync(HISTORY_FILE, JSON.stringify(newHistory));
    } catch (e) {
      console.log('Error saving history:', e);
    }
  };

  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: () => true,
      onPanResponderGrant: (evt) => {
        const { locationX, locationY } = evt.nativeEvent;
        currentPointsRef.current = [{ x: locationX, y: locationY }];
        setCurrentPoints([{ x: locationX, y: locationY }]);
      },
      onPanResponderMove: (evt) => {
        const { locationX, locationY } = evt.nativeEvent;
        currentPointsRef.current = [...currentPointsRef.current, { x: locationX, y: locationY }];
        setCurrentPoints([...currentPointsRef.current]);
      },
      onPanResponderRelease: () => {
        const completed = [...currentPointsRef.current];
        currentPointsRef.current = [];
        setCurrentPoints([]);
        if (completed.length > 0) {
          setPaths(prev => [...prev, completed]);
        }
      },
    })
  ).current;

  const clearCanvas = () => {
    setPaths([]);
    setCurrentPoints([]);
    currentPointsRef.current = [];
    setLatexResult('');
  };

  const predict = async () => {
    if (paths.length === 0 && currentPoints.length === 0) {
      Alert.alert('Empty canvas', 'Draw something first.');
      return;
    }
    setLoading(true);
    try {
      const uri = await captureRef(canvasRef, { format: 'png', quality: 1.0 });

      const formData = new FormData();
      formData.append('image', { uri, type: 'image/png', name: 'canvas.png' });

      const response = await fetch(`${backendUrl}/predict`, {
        method: 'POST',
        body: formData,
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      if (!response.ok) throw new Error(`Server error ${response.status}`);
      const data = await response.json();
      const latex = data.latex ?? data.error ?? 'No result';

      setLatexResult(latex);

      const entry = { id: Date.now(), latex, timestamp: new Date().toISOString() };
      const newHistory = [entry, ...history];
      setHistory(newHistory);
      await saveHistory(newHistory);
    } catch (e) {
      Alert.alert('Prediction failed', e.message);
    } finally {
      setLoading(false);
    }
  };

  const deleteEntry = async (id) => {
    const updated = history.filter(h => h.id !== id);
    setHistory(updated);
    await saveHistory(updated);
  };

  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.container}>
        <StatusBar style="dark" />

        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>HandTeX</Text>
          <TouchableOpacity onPress={() => setShowSettings(s => !s)} style={styles.settingsBtn}>
            <Text style={styles.settingsLabel}>Backend</Text>
          </TouchableOpacity>
        </View>

        {/* Backend URL input */}
        {showSettings && (
          <View style={styles.settingsRow}>
            <TextInput
              style={styles.urlInput}
              value={backendUrl}
              onChangeText={setBackendUrl}
              placeholder="http://192.168.x.x:5000"
              autoCapitalize="none"
              autoCorrect={false}
              keyboardType="url"
            />
          </View>
        )}

        {/* Drawing canvas */}
        <View
          ref={canvasRef}
          style={styles.canvas}
          collapsable={false}
          {...panResponder.panHandlers}
        >
          <Svg style={StyleSheet.absoluteFill}>
            {paths.map((pts, i) => (
              <Path
                key={i}
                d={pointsToSvgPath(pts)}
                stroke="#111"
                strokeWidth={3}
                fill="none"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            ))}
            {currentPoints.length > 0 && (
              <Path
                d={pointsToSvgPath(currentPoints)}
                stroke="#111"
                strokeWidth={3}
                fill="none"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            )}
          </Svg>
          {paths.length === 0 && currentPoints.length === 0 && (
            <Text style={styles.placeholder}>Draw your equation here</Text>
          )}
        </View>

        {/* Buttons */}
        <View style={styles.controls}>
          <TouchableOpacity style={styles.clearBtn} onPress={clearCanvas}>
            <Text style={styles.clearBtnText}>Clear</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.predictBtn, loading && styles.btnDisabled]}
            onPress={predict}
            disabled={loading}
          >
            <Text style={styles.predictBtnText}>{loading ? 'Predicting…' : 'Predict →'}</Text>
          </TouchableOpacity>
        </View>

        {/* LaTeX result */}
        {latexResult !== '' && (
          <View style={styles.resultBox}>
            <Text style={styles.resultLabel}>Rendered</Text>
              <View style={{ height: 140, width: "100%", backgroundColor: "#fff", borderRadius: 8, overflow: "hidden" }}>
                <LatexView latex={latexResult} display={true} />
              </View>
            <Text style={[styles.resultLabel, { marginTop: 10 }]}>Raw LaTeX</Text>
            <Text style={styles.resultText} selectable>{latexResult}</Text>
          </View>
        )}

        {/* History */}
        <ScrollView style={styles.historyScroll} contentContainerStyle={styles.historyContent}>
          <Text style={styles.historyTitle}>Saved predictions</Text>
          {history.length === 0 && (
            <Text style={styles.emptyText}>No predictions saved yet.</Text>
          )}
          {history.map(entry => (
            <View key={entry.id} style={styles.historyEntry}>
              <View style={styles.historyMeta}>
                <Text style={styles.historyTime}>
                  {new Date(entry.timestamp).toLocaleString()}
                </Text>
                <TouchableOpacity onPress={() => deleteEntry(entry.id)}>
                  <Text style={styles.deleteBtn}>✕</Text>
                </TouchableOpacity>
              </View>
              <Text style={styles.historyLatex} selectable>{entry.latex}</Text>
            </View>
          ))}
        </ScrollView>
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f3f4f6' },

  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: '#fff',
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#d1d5db',
  },
  title: { fontSize: 22, fontWeight: '700', color: '#111' },
  settingsBtn: {
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 6,
    backgroundColor: '#e5e7eb',
  },
  settingsLabel: { fontSize: 13, color: '#374151', fontWeight: '500' },

  settingsRow: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    backgroundColor: '#fff',
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#d1d5db',
  },
  urlInput: {
    borderWidth: 1,
    borderColor: '#d1d5db',
    borderRadius: 8,
    padding: 9,
    fontSize: 14,
    fontFamily: MONO,
    backgroundColor: '#f9fafb',
    color: '#111',
  },

  canvas: {
    height: 280,
    margin: 12,
    backgroundColor: '#fff',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#d1d5db',
    overflow: 'hidden',
    justifyContent: 'center',
    alignItems: 'center',
  },
  placeholder: { color: '#9ca3af', fontSize: 15 },

  controls: {
    flexDirection: 'row',
    marginHorizontal: 12,
    gap: 10,
    marginBottom: 10,
  },
  clearBtn: {
    flex: 1,
    backgroundColor: '#e5e7eb',
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: 'center',
  },
  clearBtnText: { fontSize: 16, fontWeight: '600', color: '#374151' },
  predictBtn: {
    flex: 2,
    backgroundColor: '#2563eb',
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: 'center',
  },
  predictBtnText: { fontSize: 16, fontWeight: '600', color: '#fff' },
  btnDisabled: { opacity: 0.55 },

  resultBox: {
    marginHorizontal: 12,
    marginBottom: 10,
    padding: 12,
    backgroundColor: '#eff6ff',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#bfdbfe',
  },
  resultLabel: { fontSize: 11, fontWeight: '600', color: '#3b82f6', marginBottom: 4, letterSpacing: 0.5 },
  resultText: { fontSize: 15, fontFamily: MONO, color: '#1e3a5f', lineHeight: 22 },

  historyScroll: { flex: 1 },
  historyContent: { paddingHorizontal: 12, paddingBottom: 20 },
  historyTitle: {
    fontSize: 13,
    fontWeight: '600',
    color: '#6b7280',
    marginBottom: 8,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  emptyText: { color: '#9ca3af', fontSize: 14 },
  historyEntry: {
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 12,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  historyMeta: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 4 },
  historyTime: { fontSize: 11, color: '#9ca3af' },
  deleteBtn: { fontSize: 14, color: '#9ca3af', paddingLeft: 8 },
  historyLatex: { fontSize: 14, fontFamily: MONO, color: '#111', lineHeight: 20 },
});
