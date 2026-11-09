import fs from "node:fs";
import path from "node:path";
import { load as yamlLoad, JSON_SCHEMA } from "js-yaml";

// Reads the existing ../_data/cv.yml so the CV keeps a single source of truth.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function loadCv(): any {
  const file = path.resolve(process.cwd(), "../_data/cv.yml");
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (yamlLoad(fs.readFileSync(file, "utf8"), { schema: JSON_SCHEMA }) as any).cv;
}

export function fmtYm(v: unknown): string {
  const s = String(v ?? "");
  const m = s.match(/^(\d{4})-(\d{2})/);
  if (!m) return s;
  return new Date(Date.UTC(+m[1], +m[2] - 1, 1)).toLocaleDateString("en-US", { month: "short", year: "numeric", timeZone: "UTC" });
}
