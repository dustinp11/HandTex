import React, { useCallback, useEffect, useState } from "react";
import { View, Text, Pressable, FlatList } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter, type Href, useFocusEffect } from "expo-router";
import { initDb, listNotes, deleteNote } from "../../src/db";

type NoteRow = { id: number; title: string; updated_at: number };

export default function WritingScreen() {
  const router = useRouter();
  const [notes, setNotes] = useState<NoteRow[]>([]);
  const [err, setErr] = useState<string | null>(null);

  const toHref = (path: string) => (path as unknown) as Href;

  const refresh = useCallback(async () => {
    const rows = (await listNotes()) as NoteRow[];
    setNotes(rows);
  }, []);

  // runs once on mount: init DB + first load
  useEffect(() => {
    (async () => {
      try {
        setErr(null);
        await initDb();
        await refresh();
      } catch (e: any) {
        setErr(e?.message ?? String(e));
      }
    })();
  }, [refresh]);

  // runs every time you return to this screen/tab
  useFocusEffect(
    useCallback(() => {
      refresh();
    }, [refresh])
  );

  return (
    <SafeAreaView style={{ flex: 1, padding: 16, gap: 12, backgroundColor: "white" }}>
      <View style={{ flexDirection: "row", justifyContent: "space-between" }}>
        <Text style={{ fontSize: 22, fontWeight: "700", color: "black" }}>
          Writing
        </Text>

        <Pressable
          onPress={() => router.push(toHref("/draw"))}
          style={{
            paddingVertical: 8,
            paddingHorizontal: 12,
            borderWidth: 1,
            borderRadius: 10,
            borderColor: "black",
          }}
        >
          <Text style={{ color: "black" }}>New</Text>
        </Pressable>
      </View>

      {err && (
        <View style={{ padding: 12, borderWidth: 1, borderRadius: 12, borderColor: "crimson" }}>
          <Text style={{ color: "crimson", fontWeight: "600" }}>DB Error:</Text>
          <Text style={{ color: "crimson" }}>{err}</Text>
        </View>
      )}

      <FlatList
        data={notes}
        keyExtractor={(item) => String(item.id)}
        ItemSeparatorComponent={() => <View style={{ height: 10 }} />}
        renderItem={({ item }) => (
          <Pressable
            onPress={() => router.push(toHref(`/draw?id=${item.id}`))}
            style={{
              padding: 12,
              borderWidth: 1,
              borderRadius: 12,
              gap: 6,
              borderColor: "black",
            }}
          >
            <Text style={{ fontSize: 16, fontWeight: "600", color: "black" }}>
              {item.title}
            </Text>
            <Text style={{ opacity: 0.7, color: "black" }}>
              Updated: {new Date(item.updated_at).toLocaleString()}
            </Text>

            <Pressable
              onPress={async () => {
                await deleteNote(item.id);
                await refresh();
              }}
              style={{ alignSelf: "flex-start", marginTop: 8 }}
            >
              <Text style={{ color: "crimson" }}>Delete</Text>
            </Pressable>
          </Pressable>
        )}
      />
    </SafeAreaView>
  );
}
