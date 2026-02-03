import React, { useEffect, useState } from "react";
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
import DrawingCanvas, { Stroke } from "../src/components/DrawingCanvas";
import { getNote, upsertNote } from "../src/db";

export default function DrawScreen() {
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

            <View style={{ flexDirection: "row", gap: 10 }}>
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
          </View>

          <DrawingCanvas strokes={strokes} onChangeStrokes={setStrokes} />
        </View>
      </TouchableWithoutFeedback>
    </SafeAreaView>
  );
}
