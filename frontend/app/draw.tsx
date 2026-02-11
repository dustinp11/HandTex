import React, { useEffect, useRef, useState } from "react";
import {
  View,
  Text,
  Pressable,
  TextInput,
  Alert,
  Keyboard,
  TouchableWithoutFeedback,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useLocalSearchParams, useRouter } from "expo-router";
import DrawingCanvas from "../src/components/DrawingCanvas";
import type { Stroke } from "../src/components/DrawingCanvas";
import { getNote, upsertNote } from "../src/db";

import ViewShot from "react-native-view-shot";
import * as FileSystem from "expo-file-system/legacy";

export default function DrawScreen() {
  // ✅ Correct: capture the ViewShot itself
  const viewShotRef = useRef<any>(null);

  const FLASK_URL = "http://172.31.241.116:5000";

  const [latex, setLatex] = useState<string>("");
  const [predicting, setPredicting] = useState(false);

  const router = useRouter();
  const params = useLocalSearchParams();
  const idParam = params?.id;
  const noteId = idParam ? Number(idParam) : null;

  const [title, setTitle] = useState("Untitled");
  const [strokes, setStrokes] = useState<Stroke[]>([]);

  useEffect(() => {
    (async () => {
      if (!noteId) return;
      const note = await getNote(noteId);
      if (note) {
        setTitle(note.title);
        setStrokes(note.strokes);
      }
    })();
  }, [noteId]);

  async function onSave() {
    const savedId = await upsertNote({ id: noteId, title, strokes });
    Alert.alert("Saved", `Note #${savedId} saved.`);
    router.setParams({ id: String(savedId) });
  }

  async function onPredict() {
    try {
      if (!viewShotRef.current) throw new Error("ViewShot not ready");
      setPredicting(true);

      // (Optional) tiny delay so final stroke render commits
      await new Promise((r) => setTimeout(r, 50));

      // ✅ Correct capture call
      const uri: string = await viewShotRef.current.capture?.({
        format: "png",
        quality: 1,
        result: "tmpfile",
      });

      if (!uri) throw new Error("Capture failed (no uri)");

      const info = await FileSystem.getInfoAsync(uri);
      if (!info.exists) throw new Error("Captured file does not exist");

      const size = info.size ?? 0;
      if (size < 1000) {
        throw new Error(`Captured image looks empty (size=${size} bytes)`);
      }

      // ✅ Expo-compatible multipart upload
      const uploadRes = await FileSystem.uploadAsync(`${FLASK_URL}/predict`, uri, {
        httpMethod: "POST",
        uploadType: "multipart" as any,
        fieldName: "image", // MUST match Flask: request.files["image"]
        mimeType: "image/png",
      });

      let data: any = {};
      try {
        data = JSON.parse(uploadRes.body || "{}");
      } catch {
        throw new Error(`Server returned non-JSON: ${uploadRes.body}`);
      }

      if (uploadRes.status < 200 || uploadRes.status >= 300) {
        throw new Error(data?.error ?? `Prediction failed (status ${uploadRes.status})`);
      }

      if (data?.latex) {
        setLatex(data.latex);
        Alert.alert("LaTeX", data.latex || "(empty)");
      } else if (data?.saved_as) {
        Alert.alert("Uploaded", `Saved on laptop:\n${data.saved_as}`);
      } else {
        Alert.alert("Response", JSON.stringify(data, null, 2));
      }
    } catch (e: any) {
      Alert.alert("Predict failed", e?.message ?? String(e));
    } finally {
      setPredicting(false);
    }
  }

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: "white" }}>
      <TouchableWithoutFeedback onPress={Keyboard.dismiss} accessible={false}>
        <View style={{ flex: 1, backgroundColor: "white" }}>
          <View
            style={{
              padding: 12,
              gap: 10,
              borderBottomWidth: 1,
              backgroundColor: "white",
            }}
          >
            <Text style={{ fontSize: 18, fontWeight: "700", color: "black" }}>
              Draw
            </Text>

            <TextInput
              value={title}
              onChangeText={setTitle}
              placeholder="Title"
              placeholderTextColor="#888"
              style={{
                borderWidth: 1,
                borderRadius: 10,
                paddingHorizontal: 10,
                paddingVertical: 8,
                color: "black",
                backgroundColor: "white",
              }}
            />

            <View style={{ flexDirection: "row", gap: 10, flexWrap: "wrap" }}>
              <Pressable
                onPress={() => setStrokes([])}
                style={{
                  paddingVertical: 8,
                  paddingHorizontal: 12,
                  borderWidth: 1,
                  borderRadius: 10,
                  borderColor: "black",
                }}
              >
                <Text style={{ color: "black" }}>Clear</Text>
              </Pressable>

              <Pressable
                onPress={onSave}
                style={{
                  paddingVertical: 8,
                  paddingHorizontal: 12,
                  borderWidth: 1,
                  borderRadius: 10,
                  borderColor: "black",
                }}
              >
                <Text style={{ color: "black" }}>Save</Text>
              </Pressable>

              <Pressable
                onPress={onPredict}
                disabled={predicting}
                style={{
                  paddingVertical: 8,
                  paddingHorizontal: 12,
                  borderWidth: 1,
                  borderRadius: 10,
                  borderColor: "black",
                  opacity: predicting ? 0.5 : 1,
                }}
              >
                <Text style={{ color: "black" }}>
                  {predicting ? "Predicting..." : "Predict"}
                </Text>
              </Pressable>

              <Pressable
                onPress={() => router.back()}
                style={{
                  paddingVertical: 8,
                  paddingHorizontal: 12,
                  borderWidth: 1,
                  borderRadius: 10,
                  borderColor: "black",
                }}
              >
                <Text style={{ color: "black" }}>Back</Text>
              </Pressable>
            </View>

            {latex ? (
              <Text style={{ color: "black" }} numberOfLines={3}>
                LaTeX: {latex}
              </Text>
            ) : null}
          </View>

          {/* ✅ Capture THIS */}
          <ViewShot
            ref={viewShotRef}
            options={{ format: "png", quality: 1, result: "tmpfile" }}
            style={{ flex: 1, backgroundColor: "white" }}
          >
            <View collapsable={false} style={{ flex: 1, backgroundColor: "white" }}>
              <DrawingCanvas strokes={strokes} onChangeStrokes={setStrokes} />
            </View>
          </ViewShot>
        </View>
      </TouchableWithoutFeedback>
    </SafeAreaView>
  );
}
