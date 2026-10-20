const TOKEN_RE = /\b[a-z0-9_]{2,}\b/g;

export const STOPWORDS = new Set([
  "a", "aan", "all", "als", "and", "are", "bij", "dat", "de", "der", "die", "dit", "een",
  "en", "er", "for", "from", "het", "in", "is", "met", "naar", "of", "om", "op", "or",
  "te", "that", "the", "this", "to", "uit", "van", "voor", "was", "were", "with", "wordt",
  "zijn", "ok", "okay", "thanks", "thank", "hey", "yes", "no", "good", "sounds", "cool",
  "sure", "great", "lol", "please",
]);

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

// In-word character 4-grams with word-boundary markers. The markers are
// alphanumeric, so FTS5 unicode61 keeps each gram as a single token.
const GRAM_SIZE = 4;
const WORD_START = "1";
const WORD_END = "2";

export function wordGrams(word) {
  const marked = `${WORD_START}${word}${WORD_END}`;
  if (marked.length <= GRAM_SIZE) return [marked];

  const grams = [];
  for (let i = 0; i + GRAM_SIZE <= marked.length; i++) {
    grams.push(marked.slice(i, i + GRAM_SIZE));
  }

  return grams;
}

export function gramTokens(text) {
  return tokenize(text).flatMap(wordGrams);
}

export function countTokens(tokens) {
  const counts = new Map();
  for (const t of tokens) {
    counts.set(t, (counts.get(t) || 0) + 1);
  }

  return counts;
}

export function isSubstantiveToken(token) {
  return token.length >= 2 && !STOPWORDS.has(token);
}
