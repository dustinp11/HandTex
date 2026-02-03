import * as SQLite from "expo-sqlite";

export type Note = {
  id: number;
  title: string;
  strokes: any;
  updated_at: number;
};

let dbPromise: Promise<SQLite.SQLiteDatabase> | null = null;

async function getDb() {
  if (!dbPromise) dbPromise = SQLite.openDatabaseAsync("writing.db");
  return dbPromise;
}

export async function initDb() {
  const db = await getDb();
  await db.execAsync(`
    PRAGMA journal_mode = WAL;
    CREATE TABLE IF NOT EXISTS notes (
      id INTEGER PRIMARY KEY NOT NULL,
      title TEXT NOT NULL,
      strokes_json TEXT NOT NULL,
      updated_at INTEGER NOT NULL
    );
  `);
}

export async function listNotes() {
  const db = await getDb();
  return await db.getAllAsync(
    "SELECT id, title, updated_at FROM notes ORDER BY updated_at DESC;"
  );
}

export async function getNote(id: number): Promise<Note | null> {
  const db = await getDb();
  const row = await db.getFirstAsync(
    "SELECT id, title, strokes_json, updated_at FROM notes WHERE id = ?;",
    [id]
  );

  if (!row) return null;

  return {
    id: (row as any).id,
    title: (row as any).title,
    strokes: JSON.parse((row as any).strokes_json),
    updated_at: (row as any).updated_at,
  };
}

export async function upsertNote(opts: { id: number | null; title: string; strokes: any }) {
  const db = await getDb();
  const now = Date.now();
  const strokes_json = JSON.stringify(opts.strokes);

  if (opts.id) {
    await db.runAsync(
      "UPDATE notes SET title = ?, strokes_json = ?, updated_at = ? WHERE id = ?;",
      [opts.title, strokes_json, now, opts.id]
    );
    return opts.id;
  } else {
    const result = await db.runAsync(
      "INSERT INTO notes (title, strokes_json, updated_at) VALUES (?, ?, ?);",
      [opts.title, strokes_json, now]
    );
    return (result as any).lastInsertRowId as number;
  }
}

export async function deleteNote(id: number) {
  const db = await getDb();
  await db.runAsync("DELETE FROM notes WHERE id = ?;", [id]);
}
