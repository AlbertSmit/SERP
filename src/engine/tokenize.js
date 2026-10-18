const TOKEN_RE = /\b[a-z0-9_]{2,}\b/g;

// Code-aware normalization so "C++", "C#", ".env", and single-letter languages survive
export function tokenize(text) {
  const normalized = text
    .replace(/\bc\+\+/gi, " cpp ")
    .replace(/\bc#/gi, " csharp ")
    .replace(/\bf#/gi, " fsharp ")
    .replace(/(^|\s)\.([a-z0-9_]{2,})/gi, "$1dotenv_$2 ")
    .replace(/\b([RKC])\b(?=\s|$|\?|\.|,)/g, " lang_$1 ");

  return normalized.toLowerCase().match(TOKEN_RE) || [];
}

export function countTokens(tokens) {
  const counts = new Map();
  for (const t of tokens) {
    counts.set(t, (counts.get(t) || 0) + 1);
  }

  return counts;
}
