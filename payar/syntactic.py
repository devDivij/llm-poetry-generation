"""Single-couplet Bengali Payār analysis, ready to copy into another codebase.

    from syntactic import analyze_couplet
    result = analyze_couplet("first Bengali line।\nsecond Bengali line।।")
    # Or analyze_couplet((first_line, second_line))

No CLI, file access, external lexicon, model loading, or batch processing.
The lexicon and rule-based analysis from syntactic/payar.py and lexicon.py are
embedded here. Input must contain exactly two verse lines. Invalid input shape
raises ValueError/TypeError; a well-shaped couplet that fails a rule returns
payar=False, together with diagnostics.

Rules: 14 mātrā per line, a word boundary after 8, 7 or 9 mātrā (in that order
of preference), readable Bengali, and matching rhyme endings. task_valid also
requires no recognized colloquial diction. This is a spelling/pronunciation
heuristic for the existing Sadhu/medieval convention, not a general Bengali
syntactic parser. Semantic completeness and devotional register are not tested;
absence of a known colloquial form does not prove Sadhu diction. Existing
contraction/expansion suggestions are optional and do not establish meaning.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from itertools import accumulate

__all__ = ["analyze_couplet"]

def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _lexicon_words(s: str) -> set[str]:
    return {_nfc(w) for w in s.split()}


# ---- Colloquial (Chalit) forms ------------------------------------------------
# Chalit verbs are a common stem + an ending that drops the Sadhu ি:
# কর+ল = করল (Sadhu করিল), বল+ছে = বলছে (Sadhu বলিতেছে). Sadhu forms keep the
# ি, so the generated table never matches them.
_STEMS = """
কর বল চল দেখ শুন শোন কহ পড় ধর রাখ লিখ লেখ ভাব জান মান ডাক থাক বস হাস কাঁদ
নাচ আস উঠ ওঠ শিখ শেখ ছাড় মার বুঝ বোঝ পার ফির ফের ভজ রহ ভুল ভোল কাট ঘুর
তোল খুল খোল ছুট গাহ সাজ বাজ মিল চিন চেন ফেল পাঠ খুঁজ খোঁজ বাঁচ ভাঙ পূজ
"""
_ENDINGS = """
ো ল লো লাম লি লে লেন ব বো বে বেন বি ত তো তাম তে তিস তেন
ছ ছি ছে ছো ছেন ছিস ছিল ছিলো ছিলাম ছিলে ছিলেন
েছ েছি েছে েছো েছেন েছিল েছিলো েছিলাম েছিলেন
"""
# Forms the stem table can't produce. Modern-only: progressive/perfect tenses,
# modern spellings, modern pronouns.
_MODERN = _lexicon_words("""
যাবো যাচ্ছে যাচ্ছি যেতো গেছে গেছি গেলো হবো হলো হতো হচ্ছে হয়েছে হয়েছি হয়েছিল
দেবো দিচ্ছে দিয়েছে নেবো নিচ্ছে নিয়েছে খাবো খাচ্ছে খেয়েছে খেলো পাবো পাচ্ছে পেয়েছে
পেলো এসো যেও এলো এসেছে আসছে চাইছে গাইছে নেই
তাদের যাদের এদের ওদের ওরা ওর ওকে তাকে যাকে এটা ওটা সেটা এগুলো ওগুলো সেগুলো
""")
# Not colloquial: short forms old texts use alongside the long Sadhu ones
# (Kāśīrām's Mahābhārata has হবে 131× next to হইবে 130×), and real Sadhu /
# devotional words the stem table would otherwise hit (দেব = god).
_NOT_CHALIT = _lexicon_words("""
যাব যাবে যাবেন যেত যেতে হব হবে হবেন হল হলাম হলে হলেন হত হতে নেব নেবে খাবে পাবে আসবে
ছেড়ে রেখে মেরে কেটে বেঁধে খেয়ে গেয়ে বেয়ে পেয়ে দিয়ে নিয়ে হয়ে গিয়ে এসে চেয়ে
মানব ভরত বসত ফেরত কহত দেব রহত পারত
""")

CHALIT = ({_nfc(s + e) for s in _STEMS.split() for e in _ENDINGS.split()}
          | _MODERN) - _NOT_CHALIT

_CHALIT_ENDING = re.compile(_nfc(
    r"(?<!ত)েছ(?:ে|ি|ো|েন|িল|িলাম)$"       # করেছে, but not Sadhu করিতেছে
    r"|(?<!র্)চ্ছ(?:ে|ি|ো|েন|িল|িলাম)$"     # যাচ্ছে, but not Sadhu মূর্চ্ছিল
    r"|(?<!আ)লো$|[^ি]বো$"))                 # বললো, করবো

# ---- Sadhu / CC-register markers ---------------------------------------------
_SADHU = _lexicon_words("""
যাহা তাহা ইহা উহা তাঁহার যাঁহার এবং
রহুক হউক হৌক থাকুক মোর মোরে তোর তোরে তাঁরে যাঁরে তারে যারে
হৈল কৈল লৈল হৈয়া লৈয়া হৈবে
""")
_SADHU_ENDING = re.compile(_nfc(r"(?:িয়া|িয়াছে|িতেছে|িল|িলা|িলেন|িব|িবে|িত|িতে|িলে)$"))


def _is_chalit(word: str) -> bool:
    return word in CHALIT or bool(_CHALIT_ENDING.search(word))


def _is_sadhu(token: str) -> bool:
    """token may end in an apostrophe: করি' / মাতি' mark an elided Sadhu -িয়া."""
    word = token.rstrip("'’")
    elided = token != word and word.endswith(("ি", "ই"))
    return elided or word in _SADHU or bool(_SADHU_ENDING.search(word))


# ---- Repair hints: attested CC contractions ----------------------------------
_PAIRS = """হইল:হৈল করিল:কৈল লইল:লৈল হইয়া:হৈয়া লইয়া:লৈয়া হইবে:হৈবে
            হউক:হৌক কহিল:কৈল হইতে:হৈতে লইতে:লৈতে কহিতে:কৈতে করিতে:কৈতে"""
_VARIANTS: dict[str, list[str]] = {}
for _pair in _PAIRS.split():
    _long, _short = map(_nfc, _pair.split(":"))
    _VARIANTS.setdefault(_long, []).append(_short)
    _VARIANTS.setdefault(_short, []).append(_long)


def _variants(word: str) -> list[str]:
    """Same word, one mātrā longer or shorter: হইল↔হৈল, করিয়া→করি'."""
    out = list(_VARIANTS.get(word, []))
    if word.endswith(_nfc("িয়া")) and len(word) > 4:
        out.append(word[:-3] + "'")
    return out


LINE = 14       # mātrā per line
BREAKS = (8, 7, 9)   # the pause may follow these mātrā, in order of preference

# ---------------------------------------------------------------- script

_CONSONANTS = "কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ"
_VOWEL_SIGN = {"া": "আ", "ি": "ই", "ী": "ঈ", "ু": "উ", "ূ": "ঊ", "ৃ": "ঋ",
               "ে": "এ", "ৈ": "ঐ", "ো": "ও", "ৌ": "ঔ"}
_C = f"[{_CONSONANTS}]়?"                      # a consonant, with optional nukta (ড় ঢ় য়)
_AKSHARA = re.compile(
    rf"(?:(?P<onset>(?:{_C}্)*{_C})(?P<sign>[{''.join(_VOWEL_SIGN)}্])?"   # (cluster)+sign/hasanta
    r"|(?P<vowel>[অআইঈউঊঋএঐওঔ])|ৎ)"                                   # or a vowel letter, or ৎ
    r"(?P<marks>[ংঃঁ]*)")
_LONG = set("আঈঊএঐওঔ")                              # guru vowels, per the spec

# Pronunciation. ঞ is "y": CC's ঞা (পাঞা, হঞা) is a nasalised -ইয়া and rhymes with it.
_PHONE = dict(zip(_CONSONANTS, "k kh g gh ŋ c ch j jh y ṭ ṭh ḍ ḍh n "
                               "t th d dh n p ph b bh m j r l ś ś ś h".split()))
_PHONE |= {"ড়": "ṛ", "ঢ়": "ṛ", "য়": "y"}
_VOWEL_PHONE = {"অ": "ɔ", "আ": "a", "ই": "i", "ঈ": "i", "উ": "u", "ঊ": "u", "ঋ": "r i",
                "এ": "e", "ঐ": "oi", "ও": "o", "ঔ": "ou"}
_VOWEL_PHONES = {"ɔ", "o", "a", "i", "u", "e", "oi", "ou"}


def normalize(text: str) -> str:
    # Scans often store ো ৌ in visual order (া+ে), which NFC doesn't repair.
    text = text.replace("াে", "ো").replace("ৗে", "ৌ")
    text = unicodedata.normalize("NFC", text)
    return text.translate(str.maketrans({"ৰ": "র", "ৱ": "ব", "‌": None, "‍": None}))


# ---------------------------------------------------------------- words

@dataclass
class Akshara:
    text: str
    onset: list[str]        # consonant letters; [] for a vowel letter
    vowel: str | None       # as an independent vowel; None for a bare consonant (্, ৎ)
    inherent: bool          # the vowel is the unwritten অ
    marks: str              # ং ঃ ঁ
    matra: int = 1
    sounded: bool = True    # is the vowel pronounced?
    weight: str = "L"       # L, G, or · (no vowel of its own)


@dataclass
class Word:
    text: str
    aksharas: list[Akshara]
    matra: int
    phones: list[str]
    errors: list[str]


def segment(word: str) -> tuple[list[Akshara], list[str]]:
    units, errors, pos = [], [], 0
    while pos < len(word):
        m = _AKSHARA.match(word, pos)
        if not m:
            errors.append(f"unreadable character {word[pos]!r} in {word!r}")
            pos += 1
            continue
        pos = m.end()
        if m["onset"]:
            vowel = None if m["sign"] == "্" else _VOWEL_SIGN.get(m["sign"], "অ")
            units.append(Akshara(m[0], m["onset"].split("্"), vowel, m["sign"] is None, m["marks"]))
        elif m["vowel"]:
            units.append(Akshara(m[0], [], m["vowel"], False, m["marks"]))
        else:                                                           # ৎ = bare ত
            units.append(Akshara(m[0], ["ত"], None, False, m["marks"]))
    return units, errors


def _final_vowel_sounded(units: list[Akshara]) -> bool:
    """Is a word-final unwritten অ pronounced (as o) in Sadhu recitation?"""
    last = units[-1]
    if len(units) == 1 or len(last.onset) > 1 or last.onset[0] == "হ":
        return True                                     # ত, নন্দ, ভক্ত, দেহ
    prev = units[-2]
    pv = {"ঈ": "ই", "ঊ": "উ"}.get(prev.vowel, prev.vowel)  # ি/ী sound alike: মিশ্রিত/প্রীত rhyme
    c = last.onset[0]
    return ((c == "ল" and pv in ("ই", "এ", "ঐ"))         # করিল, গেল, হৈল
            or (c == "ব" and pv == "ই")                   # করিব
            or (c == "ত" and (pv in ("ই", "ঋ") or prev.inherent)))  # করিত, পতিত, গত


def _weight(u: Akshara, nxt: Akshara | None) -> str:
    """Guru if the vowel is long, carries ং/ঃ, or its syllable is closed."""
    if not u.sounded:
        return "·"
    closed = nxt is not None and (len(nxt.onset) > 1 or not nxt.sounded)
    return "G" if u.vowel in _LONG or "ং" in u.marks or "ঃ" in u.marks or closed else "L"


def _pronounce(units: list[Akshara]) -> list[str]:
    phones = []
    for k, u in enumerate(units):
        phones += [_PHONE.get(c, c) for c in u.onset]
        if u.sounded:
            final_o = u.inherent and k == len(units) - 1        # নন্দ = nondo
            phones += ["o"] if final_o else _VOWEL_PHONE[u.vowel].split()
        if "ং" in u.marks:
            phones.append("ŋ")
    return phones


def analyse_word(text: str) -> Word:
    text = normalize(text)
    units, errors = segment(text)
    errors.extend(f"unsupported consonant {c!r} in {text!r}"
                  for u in units for c in u.onset if c not in _PHONE)
    for k, u in enumerate(units):
        final = k == len(units) - 1
        if u.vowel is None:                     # bare consonant: closes the syllable before it
            u.sounded, u.matra = False, int(final)
        else:
            u.matra = 1 + (final and "ং" in u.marks)
            if final and u.inherent:
                u.sounded = _final_vowel_sounded(units)
    for k, u in enumerate(units):
        u.weight = _weight(u, units[k + 1] if k + 1 < len(units) else None)
    return Word(text, units, sum(u.matra for u in units), _pronounce(units), errors)


# ---------------------------------------------------------------- lines

@dataclass
class Line:
    text: str
    words: list[Word]
    cut: int | None         # the caesura falls after words[:cut]
    note: str | None        # why the caesura fails, if it does

    @property
    def matra(self) -> int:
        return sum(w.matra for w in self.words)

    @property
    def left(self) -> int:
        return sum(w.matra for w in self.words[:self.cut or 0])

    @property
    def caesura_ok(self) -> bool:
        return self.note is None

    @property
    def errors(self) -> list[str]:
        return [e for w in self.words for e in w.errors]

    @property
    def weights(self) -> str:
        parts = ["".join(u.weight for u in w.aksharas) for w in self.words]
        if self.cut is not None:
            parts.insert(self.cut, "|")
        return " ".join(parts)


def _words(text: str) -> list[Word]:
    text = re.sub(r"[-‐‑–—―,;:!?()\[\]{}…“”\"'‘’`*_~/\\.।॥]", " ", text)
    return [analyse_word(w) for w in text.split()]


def _caesura(words: list[Word], marker: int | None) -> tuple[int | None, str | None]:
    """Where the pause falls (as an index into words) and, if it fails, why."""
    ends = list(accumulate(w.matra for w in words))
    total = ends[-1] if ends else 0
    if marker is not None:
        cut = marker
    else:
        cut = next((ends.index(p) + 1 for p in BREAKS if p in ends), None)
        if cut is None:
            for w, end in zip(words, ends):
                if end > min(BREAKS):
                    return None, (f"no word break after mātrā 7, 8 or 9: "
                                  f"{w.text!r} spans {end - w.matra + 1}–{end}")
            return None, f"line has only {total} mātrā"
    left = ends[cut - 1] if cut else 0
    if left not in BREAKS:
        return cut, f"marked pause falls after mātrā {left}; it must be 7, 8 or 9"
    if total != LINE:
        return cut, f"second half has {total - left} mātrā, needs {LINE - left}"
    return cut, None


def analyse_line(text: str) -> Line:
    before, bar, after = re.sub("[｜¦]", "|", text).partition("|")
    words = _words(before)
    marker = len(words) if bar else None
    words += _words(after)
    return Line(text, words, *_caesura(words, marker))


# ---------------------------------------------------------------- rhyme, diction, repairs

def rhyme_key(phones: list[str], exact: bool = False) -> tuple[str, ...]:
    """The last vowel and what follows it; if the word ends in that vowel, also
    the consonants before it (mon → on, korilo → lo, bhai → ai)."""
    vowels = [i for i, p in enumerate(phones) if p in _VOWEL_PHONES]
    if not vowels:
        return ()
    start = vowels[-1]
    if start == len(phones) - 1:
        while start > 0 and phones[start - 1] not in _VOWEL_PHONES:
            start -= 1
        if start == vowels[-1] and len(vowels) > 1:             # hiatus: ভাই → ai
            start = vowels[-2]
    key = ["o" if p == "ɔ" else p for p in phones[start:]]       # অ's ɔ/o isn't modelled
    if not exact:                                               # ফন্দি/সন্ধি: near rhyme
        key = [p[:-1] if len(p) > 1 and p.endswith("h") else p for p in key]
    return tuple(key)


def diction_check(text: str) -> dict:
    colloquial, sadhu = [], []
    for token in re.findall(r"[ঀ-৿]+['’]?", normalize(text)):
        if _is_chalit(token.rstrip("'’")):
            colloquial.append(token.rstrip("'’"))
        elif _is_sadhu(token):
            sadhu.append(token)
    return {"ok": not colloquial, "colloquial": colloquial, "sadhu_markers": sadhu}


def repair_hints(line: Line) -> list[str]:
    """One-word CC contractions/expansions that would give a valid 14-mātrā line."""
    if line.matra == LINE and line.caesura_ok:
        return []
    hints = []
    for i, w in enumerate(line.words):
        for alt in _variants(w.text):
            matras = [x.matra for x in line.words]
            matras[i] = analyse_word(alt.rstrip("'")).matra
            ends = list(accumulate(matras))
            if ends[-1] == LINE and any(p in ends for p in BREAKS):
                hints.append(f"{w.text} → {alt}  ({line.matra} → {LINE})")
    return hints


# ---------------------------------------------------------------- classification

def _line_dict(ln: Line) -> dict:
    return {
        "text": ln.text,
        "matra": ln.matra,
        "split": f"{ln.left}+{ln.matra - ln.left}" if ln.cut is not None else None,
        "segmentation": " ".join(f"{w.text}({w.matra})" for w in ln.words),
        "weights": ln.weights,
        "caesura": {"ok": ln.caesura_ok, "note": ln.note},
    }


def _classify_couplet(text1: str, text2: str, include_repair_hints: bool) -> dict:
    lines = [analyse_line(text1), analyse_line(text2)]
    ends = [ln.words[-1] if ln.words else analyse_word("") for ln in lines]
    keys = [rhyme_key(w.phones) for w in ends]
    exact = [rhyme_key(w.phones, exact=True) for w in ends]
    diction = diction_check(f"{text1} {text2}")
    checks = {
        "M14": all(ln.matra == LINE for ln in lines),
        "pause": all(ln.caesura_ok for ln in lines),
        "R_AA": bool(keys[0]) and keys[0] == keys[1],
        "P": all(ln.words and not ln.errors for ln in lines),
        "sadhu_diction": diction["ok"],
    }

    failures = []
    for n, ln in enumerate(lines, 1):
        if ln.matra != LINE:
            failures.append(f"line {n}: {ln.matra} mātrā, needs {LINE}")
        if ln.note:
            failures.append(f"line {n}: pause — {ln.note}")
        failures += [f"line {n}: {e}" for e in ln.errors]
    if not checks["R_AA"]:
        failures.append("rhyme: " + " ≠ ".join(f"{w.text} (-{''.join(k)})" for w, k in zip(ends, exact)))
    if not diction["ok"]:
        failures.append("diction: colloquial " + ", ".join(diction["colloquial"]))

    payar = checks["M14"] and checks["pause"] and checks["R_AA"] and checks["P"]
    quality = "none" if not checks["R_AA"] else "full" if exact[0] == exact[1] else "near (aspiration)"
    return {
        "payar": payar,
        "task_valid": payar and diction["ok"],
        "checks": checks | {"semantic_completeness": "not_checked",
                            "devotional_register": "not_checked"},
        "rhyme": {"words": [w.text for w in ends], "keys": ["".join(k) for k in exact],
                  "quality": quality},
        "diction": diction,
        "lines": [_line_dict(ln) for ln in lines],
        "failures": failures,
        "repair_hints": [f"line {n}: {h}" for n, ln in enumerate(lines, 1)
                         for h in repair_hints(ln)] if include_repair_hints else [],
    }


def _couplet_lines(couplet: str | tuple[str, str]) -> tuple[str, str]:
    """Accept two lines separated by a newline/danda, or an explicit pair.

    Whitespace-only pieces and repeated danda terminators are harmless. Extra
    content, including a third non-Bengali line, is never silently discarded.
    Digits inside a verse are retained and fail the readability check.
    """
    if isinstance(couplet, str):
        pieces = [part.strip() for part in re.split(r"[\r\n।॥]+", normalize(couplet))
                  if part.strip()]
    elif isinstance(couplet, tuple):
        if len(couplet) != 2:
            raise ValueError("Expected exactly two verse lines in the couplet")
        pieces = []
        for line in couplet:
            if not isinstance(line, str):
                raise TypeError("Each couplet line must be a string")
            parts = [part.strip() for part in re.split(r"[\r\n।॥]+", normalize(line))
                     if part.strip()]
            if len(parts) != 1:
                raise ValueError("Each tuple element must contain exactly one verse line")
            pieces.extend(parts)
    else:
        raise TypeError("Expected a two-line string or a tuple of two strings")
    if len(pieces) != 2:
        raise ValueError(f"Expected exactly two verse lines; received {len(pieces)}")
    return pieces[0], pieces[1]


def analyze_couplet(couplet: str | tuple[str, str], *, include_repair_hints: bool = False) -> dict:
    """Return one couplet's meter, rhyme, diction, and failure diagnostics.

    A single string requires newlines or Bengali danda separators; a space alone
    cannot establish a line boundary. In each line, an optional | marks the
    intended pause. Meter decisions preserve the original rules. When no pause
    is marked, the reported split is an inferred word boundary, not a claim
    about the performer's actual recitation. Laghu/guru weights are diagnostic.

    Result keys: payar, task_valid, checks, rhyme, diction, lines, failures,
    repair_hints. Neither payar nor task_valid implies semantic correctness.
    """
    if not isinstance(include_repair_hints, bool):
        raise TypeError("include_repair_hints must be a boolean")
    first, second = _couplet_lines(couplet)
    return _classify_couplet(first, second, include_repair_hints)
