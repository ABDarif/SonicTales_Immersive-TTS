"""Stage 2: raw chapters -> structured script (who speaks, how, and what ambience plays).

Two builders share one output model:

* `HeuristicBuilder` works fully offline with quote detection, "said X" attribution,
  pronoun resolution and a keyword lexicon for ambience.
* `LLMScriptBuilder` asks Claude for a richer annotation, validates the answer against
  the source text and falls back to the heuristic builder for any chunk that fails.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from app.config import Settings
from app.models import (
    NARRATOR,
    Book,
    Chapter,
    Character,
    Gender,
    RawChapter,
    Scene,
    Script,
    Segment,
)
from app.pdf_extract import SCENE_MARK

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]

# --------------------------------------------------------------------------- lexicons

AMBIENCE_TAGS = [
    "rain",
    "thunder",
    "wind",
    "night",
    "fire",
    "ocean",
    "forest",
    "river",
    "city",
    "crowd",
    "birds",
]

_START = {
    "rain": r"rain|raining|rained|rainy|raindrops?|drizzl\w+|downpour|rainstorm|storm\w*",
    "thunder": r"thunder\w*|lightning",
    "wind": r"winds?|windy|gusts?|gusting|breeze|gale|howling|blizzard",
    "night": r"night|nighttime|midnight|moonlit|moonlight|moon|starlit|dusk|nightfall|crickets?|darkness",
    "fire": r"fire|fireplace|campfire|bonfire|flames?|hearth|embers|crackling|torch(?:es)?",
    "ocean": r"ocean|sea|seas|waves|shore|seashore|beach|surf|harbou?r|seagulls?|tide",
    "forest": r"forest|woods|woodland|jungle|trees|grove|undergrowth|thicket",
    "river": r"river|stream|brook|creek|waterfall|rapids",
    "city": r"city|streets?|traffic|cars?|downtown|sidewalk|taxi|honking",
    "crowd": r"crowd|tavern|inn|pub|market|marketplace|festival|bustling|murmur\w*|chatter\w*|banquet|ballroom",
    "birds": r"birds?|birdsong|songbirds?|sparrows?|robins?|chirp\w*|dawn chorus",
}
_END = {
    "rain": r"\b(?:rain|storm)\b[^.!?]{0,30}\b(?:stopped|ceased|ended|cleared|passed|let up|eased)\b"
    r"|\b(?:stopped|ceased) raining\b|\bsun (?:came out|broke through)\b",
    "thunder": r"\bthunder\b[^.!?]{0,30}\b(?:faded|ceased|stopped)\b",
    "wind": r"\bwind\b[^.!?]{0,30}\b(?:died|dropped|stilled|ceased)\b",
    "night": r"\b(?:dawn|daybreak|sunrise|morning|daylight|sunlit|noon|afternoon)\b|\bsun (?:rose|was up)\b",
    "fire": r"\bfire\b[^.!?]{0,30}\b(?:died|went out|burnt out|burned out|dwindled)\b",
}
# How many paragraphs a cue keeps sounding after its last mention.
_PERSIST = {
    "rain": 10,
    "thunder": 3,
    "wind": 8,
    "night": 30,
    "fire": 10,
    "ocean": 12,
    "forest": 10,
    "river": 10,
    "city": 10,
    "crowd": 8,
    "birds": 6,
}
START_RE = {k: re.compile(rf"\b(?:{v})\b", re.IGNORECASE) for k, v in _START.items()}
END_RE = {k: re.compile(v, re.IGNORECASE) for k, v in _END.items()}

VERB_EMOTION: dict[str, str] = {
    "said": "neutral",
    "asked": "neutral",
    "replied": "neutral",
    "answered": "neutral",
    "added": "neutral",
    "called": "neutral",
    "told": "neutral",
    "continued": "neutral",
    "returned": "neutral",
    "observed": "neutral",
    "remarked": "neutral",
    "rejoined": "neutral",
    "inquired": "neutral",
    "enquired": "neutral",
    "suggested": "neutral",
    "explained": "neutral",
    "agreed": "neutral",
    "responded": "neutral",
    "announced": "neutral",
    "insisted": "neutral",
    "protested": "neutral",
    "persisted": "neutral",
    "demanded": "angry",
    "retorted": "angry",
    "thundered": "angry",
    "pleaded": "sad",
    "begged": "sad",
    "whispered": "whisper",
    "murmured": "whisper",
    "muttered": "whisper",
    "breathed": "whisper",
    "shouted": "angry",
    "yelled": "angry",
    "snapped": "angry",
    "growled": "angry",
    "barked": "angry",
    "hissed": "angry",
    "screamed": "afraid",
    "gasped": "afraid",
    "stammered": "afraid",
    "cried": "excited",  # in classic prose "cried" means exclaimed; weeping is wept/sobbed
    "sobbed": "sad",
    "sighed": "sad",
    "wept": "sad",
    "whimpered": "sad",
    "laughed": "happy",
    "giggled": "happy",
    "chuckled": "happy",
    "grinned": "happy",
    "cheered": "excited",
    "exclaimed": "excited",
    "declared": "excited",
    "yelped": "excited",
}
ADVERB_EMOTION = {
    "angrily": "angry",
    "impatiently": "angry",
    "indignantly": "angry",
    "tenderly": "happy",
    "eagerly": "excited",
    "furiously": "angry",
    "sharply": "angry",
    "coldly": "angry",
    "softly": "whisper",
    "quietly": "whisper",
    "gently": "whisper",
    "faintly": "whisper",
    "sadly": "sad",
    "miserably": "sad",
    "bitterly": "sad",
    "happily": "happy",
    "cheerfully": "happy",
    "warmly": "happy",
    "brightly": "happy",
    "nervously": "afraid",
    "fearfully": "afraid",
    "anxiously": "afraid",
    "timidly": "afraid",
    "excitedly": "excited",
}

_MALE_NAMES = set(
    [
        "john",
        "james",
        "robert",
        "michael",
        "william",
        "david",
        "richard",
        "joseph",
        "thomas",
        "charles",
        "daniel",
        "matthew",
        "mark",
        "paul",
        "peter",
        "harry",
        "ron",
        "george",
        "arthur",
        "henry",
        "edward",
        "jack",
        "tom",
        "sam",
        "ben",
        "luke",
        "jacob",
        "adam",
        "alex",
        "oliver",
        "noah",
        "liam",
        "ethan",
        "nathan",
        "frank",
        "walter",
        "albert",
        "hugh",
        "victor",
        "leo",
        "max",
        "simon",
        "tony",
        "bob",
        "bill",
    ]
)
_FEMALE_NAMES = set(
    [
        "mary",
        "elizabeth",
        "anna",
        "emma",
        "sarah",
        "jane",
        "lucy",
        "alice",
        "emily",
        "kate",
        "katherine",
        "catherine",
        "margaret",
        "hermione",
        "ginny",
        "helen",
        "laura",
        "julia",
        "rose",
        "grace",
        "lily",
        "sophie",
        "sophia",
        "olivia",
        "charlotte",
        "amy",
        "anne",
        "susan",
        "lisa",
        "clara",
        "eve",
        "eva",
        "ella",
        "mia",
        "nora",
        "ruth",
        "rachel",
        "hannah",
        "diana",
        "maria",
        "isabel",
        "isabella",
    ]
)
_STOP_NAMES = set(
    [
        "the",
        "a",
        "an",
        "he",
        "she",
        "it",
        "they",
        "we",
        "you",
        "i",
        "and",
        "but",
        "then",
        "his",
        "her",
        "their",
        "there",
        "this",
        "that",
        "when",
        "while",
        "as",
        "suddenly",
        "now",
        "just",
        "so",
        "well",
        "yes",
        "no",
        "after",
        "before",
        "if",
        "what",
        "who",
        "how",
        "why",
        "one",
        "some",
        "all",
        "each",
    ]
)

_VERBS = "|".join(sorted(VERB_EMOTION, key=len, reverse=True))


def _adv(n: int) -> str:
    return rf"(?:\s+(?P<adv{n}>\w+ly))?"


# "said his lady", "cried the old man": a role noun instead of a name. Nouns in the same group are
# treated as one character ("his lady" and "his wife" are the same person in the same scene).
_ROLE_GROUPS: dict[str, tuple[str, Gender, tuple[str, ...]]] = {
    "woman": ("Woman", "female", ("lady", "wife", "woman", "madam", "mistress")),
    "man": ("Man", "male", ("husband", "gentleman", "man")),
}
_ROLE_NOUNS: dict[str, tuple[str, Gender]] = {
    **{n: (name, g) for name, g, nouns in _ROLE_GROUPS.values() for n in nouns},
    **{
        n: (n.capitalize(), "female")
        for n in ("mother", "sister", "girl", "daughter", "queen", "maid", "princess", "aunt")
    },
    **{n: (n.capitalize(), "male") for n in ("father", "brother", "boy", "son", "king", "prince", "uncle", "master")},
    **{
        n: (n.capitalize(), "unknown")
        for n in ("stranger", "child", "doctor", "officer", "innkeeper", "servant", "guard")
    },
}
_ROLE_ALT = "|".join(sorted(_ROLE_NOUNS, key=len, reverse=True))
_ROLE = rf"(?P<role>(?:his|her|their|the|a|an)\s+(?:\w+\s+)?(?P<noun>{_ROLE_ALT}))\b"
_TITLE = r"(?:Mr|Mrs|Ms|Dr|Miss|Sir|Lady|Lord|Captain|Professor)\.?\s+"
_NAME = rf"(?-i:(?:{_TITLE})?[A-Z][a-z]+)"
_WHO = rf"(?:(?P<pron>he|she|I)|{_ROLE}|(?P<name>{_NAME}))"
_AFTER_RE = [
    re.compile(rf"^\s*[,.:]?\s*{_WHO}{_adv(1)}\s+(?P<verb>{_VERBS})\b{_adv(2)}", re.IGNORECASE),
    re.compile(rf"^\s*[,.:]?\s*(?P<verb>{_VERBS})\s+{_WHO}{_adv(1)}", re.IGNORECASE),
]
_BEFORE_RE = [
    re.compile(rf"\b{_WHO}{_adv(1)}\s+(?P<verb>{_VERBS})\b{_adv(2)}[,:;]?\s*$", re.IGNORECASE),
    re.compile(rf"\b(?P<verb>{_VERBS})\s+{_WHO}{_adv(1)}[,:;]?\s*$", re.IGNORECASE),
]
# Narration that names a speaker without a quote ("Mr. Bennet replied that he had not.")
_SPOKE_RE = re.compile(rf"\b(?P<name>{_NAME})\s+(?:\w+ly\s+)?(?:{_VERBS})\b", re.IGNORECASE)
_QUOTE_RE = re.compile(r"“([^”]+)”|\"([^\"]+)\"")
_PRON_VOTE_RE = re.compile(rf"(?P<name>{_NAME})\b[^.!?\"“”]{{0,60}}?\b(?P<pron>he|she|his|her|him)\b", re.IGNORECASE)


@dataclass
class _Attribution:
    speaker: str | None
    gender: Gender
    emotion: str


@dataclass
class _State:
    """Character memory that persists across paragraphs and chapters."""

    votes: dict[str, dict[str, int]] = field(default_factory=dict)
    last_by_gender: dict[str, str] = field(default_factory=dict)
    recent: list[str] = field(default_factory=list)  # most recent distinct speakers

    def gender_of(self, name: str) -> Gender:
        first = name.split()[-1].lower()
        title = name.split()[0].rstrip(".").lower()
        v = self.votes.get(name, {})
        score = v.get("male", 0) - v.get("female", 0)
        if title in {"mr", "sir", "lord"}:
            score += 3
        elif title in {"mrs", "ms", "miss", "lady"}:
            score -= 3
        if first in _MALE_NAMES:
            score += 2
        elif first in _FEMALE_NAMES:
            score -= 2
        return "male" if score > 0 else "female" if score < 0 else "unknown"

    def vote(self, name: str, gender: str, weight: int = 1) -> None:
        self.votes.setdefault(name, {}).setdefault(gender, 0)
        self.votes[name][gender] += weight

    def touch(self, name: str) -> None:
        if name in self.recent:
            self.recent.remove(name)
        self.recent.insert(0, name)
        del self.recent[6:]
        g = self.gender_of(name)
        if g != "unknown":
            self.last_by_gender[g] = name


def _valid_name(name: str) -> bool:
    return name.split()[-1].lower() not in _STOP_NAMES


def _emotion_from(verb: str | None, adv: str | None, text: str) -> str:
    if adv and adv.lower() in ADVERB_EMOTION:
        return ADVERB_EMOTION[adv.lower()]
    if verb and VERB_EMOTION.get(verb.lower(), "neutral") != "neutral":
        return VERB_EMOTION[verb.lower()]
    if text.rstrip('”" ').endswith("!"):
        return "excited"
    return "neutral"


class HeuristicBuilder:
    """Offline script builder; call `build_chapter` for each chapter in book order."""

    def __init__(self) -> None:
        self.state = _State()
        self.speakers: set[str] = set()

    # ---- speaker attribution

    def _match(self, patterns: list[re.Pattern[str]], text: str, near_end: bool) -> re.Match[str] | None:
        # Only look at a short window next to the quote: attributions are brief.
        window = text[-90:] if near_end else text[:90]
        for p in patterns:
            m = p.search(window) if near_end else p.match(window)
            if m:
                return m
        return None

    def _resolve(self, m: re.Match[str], dialogue: str) -> _Attribution | None:
        verb, adv = m.group("verb"), (m.groupdict().get("adv1") or m.groupdict().get("adv2"))
        emotion = _emotion_from(verb, adv, dialogue)
        pron, name, noun = m.group("pron"), m.group("name"), m.group("noun")
        if noun:
            role_name, role_gender = _ROLE_NOUNS[noun.lower()]
            if role_gender != "unknown":
                self.state.vote(role_name, role_gender, 5)
            return _Attribution(role_name, role_gender, emotion)
        if name:
            if not _valid_name(name):
                return None
            return _Attribution(name.strip(), self.state.gender_of(name.strip()), emotion)
        if pron and pron.lower() == "i":
            return _Attribution(NARRATOR, "unknown", emotion)
        g: Gender = "male" if pron and pron.lower() == "he" else "female"
        known = self.state.last_by_gender.get(g)
        return _Attribution(known or ("Man" if g == "male" else "Woman"), g, emotion)

    # ---- paragraph -> segments

    def _paragraph_segments(self, para: str, gap: int | None) -> tuple[list[Segment], bool]:
        """`gap` = paragraphs since the last dialogue paragraph (0 = the previous one), None if none yet."""
        in_conversation = gap is not None and gap <= 1
        pieces: list[tuple[str, str]] = []
        pos = 0
        for m in _QUOTE_RE.finditer(para):
            if m.start() > pos:
                pieces.append(("narr", para[pos : m.start()]))
            pieces.append(("dlg", (m.group(1) or m.group(2)).strip()))
            pos = m.end()
        if pos < len(para):
            pieces.append(("narr", para[pos:]))

        # Vote on genders from narration ("Mary ... she ...")
        for kind, text in pieces:
            if kind == "narr":
                for vm in _PRON_VOTE_RE.finditer(text):
                    if _valid_name(vm.group("name")):
                        p = vm.group("pron").lower()
                        self.state.vote(vm.group("name").strip(), "male" if p in {"he", "his", "him"} else "female")
                # people who speak in reported speech still count as participants in the conversation
                for sm in _SPOKE_RE.finditer(text):
                    if _valid_name(sm.group("name")):
                        self.state.touch(sm.group("name").strip())

        segments: list[Segment] = []
        has_dialogue = any(k == "dlg" for k, _ in pieces)
        last_para_speaker: str | None = None
        gap_words = 99
        for i, (kind, text) in enumerate(pieces):
            text = text.strip(" \t")
            if not text.strip():
                continue
            if kind == "narr":
                gap_words = len(text.split())
                if len(re.sub(r"[\s,.;:—–-]+", "", text)) == 0:
                    continue
                segments.append(Segment(speaker=NARRATOR, text=text.strip(), pause_after=0.2))
                continue
            attr = None
            if i + 1 < len(pieces) and pieces[i + 1][0] == "narr":
                m = self._match(_AFTER_RE, pieces[i + 1][1], near_end=False)
                attr = self._resolve(m, text) if m else None
            if attr is None and i > 0 and pieces[i - 1][0] == "narr":
                m = self._match(_BEFORE_RE, pieces[i - 1][1], near_end=True)
                attr = self._resolve(m, text) if m else None
            speaker, emotion = (attr.speaker, attr.emotion) if attr else (None, _emotion_from(None, None, text))
            if speaker is None:
                if last_para_speaker and gap_words <= 6:
                    speaker = last_para_speaker  # `"Hi," he smiled, "how are you?"`
                elif in_conversation and len(self.state.recent) >= 2:
                    speaker = self.state.recent[1]  # alternate in back-and-forth exchanges
                elif in_conversation and self.state.recent:
                    speaker = "Stranger"
                elif self.state.recent:
                    speaker = self.state.recent[0]
                else:
                    speaker = "Stranger"
            if speaker != NARRATOR:
                self.state.touch(speaker)
                self.speakers.add(speaker)
            last_para_speaker = speaker
            gap_words = 99
            segments.append(
                Segment(speaker=speaker, text=text, emotion=emotion, is_dialogue=True, pause_after=0.3)  # type: ignore[arg-type]
            )
        if segments:
            segments[-1].pause_after = 0.7
        return segments, has_dialogue

    # ---- ambience

    def _scene_runs(self, paragraphs: list[str]) -> list[tuple[frozenset[str], list[str]]]:
        """Group paragraphs into runs that share the same active ambience (thunder excluded from the key)."""
        last_seen: dict[str, int] = {}
        runs: list[tuple[frozenset[str], list[str]]] = []
        current: list[str] = []
        current_tags: set[str] = set()
        current_key: frozenset[str] | None = None

        def flush() -> None:
            nonlocal current, current_tags, current_key
            if current:
                runs.append((frozenset(current_tags), current))
            current, current_tags, current_key = [], set(), None

        for idx, para in enumerate(paragraphs):
            if para == SCENE_MARK:
                flush()
                last_seen.clear()
                continue
            for tag, end_re in END_RE.items():
                if end_re.search(para):
                    last_seen.pop(tag, None)
            ended = {t for t, r in END_RE.items() if r.search(para)}
            for tag, start_re in START_RE.items():
                if tag not in ended and start_re.search(para):
                    last_seen[tag] = idx
            active = {t for t, seen in last_seen.items() if idx - seen <= _PERSIST[t]}
            key = frozenset(active - {"thunder"})
            if current_key is not None and key != current_key:
                flush()
            current_key = key
            current.append(para)
            current_tags |= active
        flush()
        return runs

    def build_scenes(self, paragraphs: list[str]) -> list[Scene]:
        scenes: list[Scene] = []
        gap: int | None = None
        for tags, paras in self._scene_runs(paragraphs):
            segments: list[Segment] = []
            for para in paras:
                segs, has_dialogue = self._paragraph_segments(para, gap)
                gap = 0 if has_dialogue else (gap + 1 if gap is not None else None)
                segments.extend(segs)
            if not segments:
                continue
            ordered = [t for t in AMBIENCE_TAGS if t in tags]
            scenes.append(Scene(setting=", ".join(ordered) or "unspecified", ambience=ordered, segments=segments))
        return scenes

    def build_chapter(self, index: int, raw: RawChapter) -> Chapter:
        return Chapter(index=index, title=raw.title, scenes=self.build_scenes(raw.paragraphs))

    def characters(self) -> dict[str, Character]:
        return {n: Character(name=n, gender=self.state.gender_of(n)) for n in sorted(self.speakers)}


# --------------------------------------------------------------------------- LLM builder

LLM_SYSTEM = f"""You convert book text into a script for a multi-voice, immersive audiobook.
Return ONLY a JSON object, no prose, no code fences, with this exact shape:
{{"characters": [{{"name": str, "gender": "male"|"female"|"unknown"}}],
  "scenes": [{{"ambience": [tags], "segments": [{{"speaker": str, "text": str, "emotion": str}}]}}]}}

Rules:
- Cover the ENTIRE input text, in order, verbatim. Never summarise, skip, reorder or rewrite it.
- Narration and speech tags ("he said") are spoken by "{NARRATOR}". Quoted speech is spoken by the
  character who says it; remove the quotation marks from "text".
- Use one consistent name per character (reuse the known characters supplied by the user).
- emotion is one of: neutral, happy, sad, angry, afraid, whisper, excited.
- Start a new scene when the place, time of day or weather changes. "ambience" lists the background
  sounds that would realistically be heard during the scene, chosen ONLY from: {", ".join(AMBIENCE_TAGS)}.
  Use [] when nothing fits. Only include ambience the text supports (e.g. "rain" only if it is raining).
"""

CompleteFn = Callable[[str, str], str]


def anthropic_complete(settings: Settings) -> CompleteFn:
    import anthropic  # imported lazily: optional dependency path

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def complete(system: str, user: str) -> str:
        msg = client.messages.create(
            model=settings.llm_model,
            max_tokens=8000,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

    return complete


def _compact(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


def chunk_paragraphs(paragraphs: list[str], max_chars: int = 5000) -> list[list[str]]:
    chunks: list[list[str]] = []
    cur: list[str] = []
    size = 0
    for p in paragraphs:
        if cur and (size + len(p) > max_chars or p == SCENE_MARK):
            chunks.append(cur)
            cur, size = [], 0
        cur.append(p)
        size += len(p)
    if cur:
        chunks.append(cur)
    return chunks


class LLMScriptBuilder:
    def __init__(self, complete: CompleteFn, fallback: HeuristicBuilder | None = None) -> None:
        self.complete = complete
        self.fallback = fallback or HeuristicBuilder()
        self.known: dict[str, Character] = {}
        self.fallback_chunks = 0

    @staticmethod
    def _parse_json(raw: str) -> dict:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end < start:
            raise ValueError("no JSON object in model output")
        return json.loads(raw[start : end + 1])

    def _scenes_from_llm(self, paragraphs: list[str]) -> list[Scene]:
        body = "\n\n".join(p for p in paragraphs if p != SCENE_MARK)
        known = ", ".join(f"{c.name} ({c.gender})" for c in self.known.values()) or "none yet"
        data = self._parse_json(self.complete(LLM_SYSTEM, f"Known characters: {known}\n\nTEXT:\n{body}"))
        scenes: list[Scene] = []
        for s in data.get("scenes", []):
            segs = []
            for seg in s.get("segments", []):
                text = str(seg.get("text", "")).strip()
                if not text:
                    continue
                emotion = seg.get("emotion", "neutral")
                if emotion not in VERB_EMOTION.values():
                    emotion = "neutral"
                speaker = str(seg.get("speaker") or NARRATOR).strip() or NARRATOR
                segs.append(
                    Segment(
                        speaker=speaker,
                        text=text,
                        emotion=emotion,  # type: ignore[arg-type]
                        is_dialogue=speaker != NARRATOR,
                        pause_after=0.4,
                    )
                )
            if segs:
                segs[-1].pause_after = 0.7
                tags = [t for t in s.get("ambience", []) if t in AMBIENCE_TAGS]
                scenes.append(Scene(setting=", ".join(tags) or "unspecified", ambience=tags, segments=segs))
        covered = sum(len(_compact(g.text)) for sc in scenes for g in sc.segments)
        wanted = len(_compact(body))
        if wanted and covered < 0.9 * wanted:
            raise ValueError(f"model output covers only {covered}/{wanted} characters of the text")
        for c in data.get("characters", []):
            name = str(c.get("name", "")).strip()
            gender = c.get("gender", "unknown")
            if name and name != NARRATOR:
                self.known[name] = Character(name=name, gender=gender if gender in {"male", "female"} else "unknown")
        return scenes

    def build_chapter(self, index: int, raw: RawChapter) -> Chapter:
        scenes: list[Scene] = []
        for chunk in chunk_paragraphs(raw.paragraphs):
            try:
                scenes.extend(self._scenes_from_llm(chunk))
            except Exception as exc:  # network, JSON, schema, coverage: all degrade gracefully
                log.warning("LLM chunk failed (%s); using heuristic builder for it", exc)
                self.fallback_chunks += 1
                scenes.extend(self.fallback.build_scenes(chunk))
        return Chapter(index=index, title=raw.title, scenes=scenes)

    def characters(self) -> dict[str, Character]:
        return {**self.fallback.characters(), **self.known}


def build_script(
    book: Book,
    mode: str = "auto",
    settings: Settings | None = None,
    complete: CompleteFn | None = None,
    progress: ProgressFn | None = None,
) -> Script:
    """mode: 'heuristic' (offline), 'llm' (Claude), or 'auto' (llm when an API key exists)."""
    use_llm = mode == "llm" or (mode == "auto" and (complete or (settings and settings.anthropic_api_key)))
    if use_llm:
        if complete is None:
            if not (settings and settings.anthropic_api_key):
                raise ValueError("ANTHROPIC_API_KEY is not set; use builder 'heuristic' instead")
            complete = anthropic_complete(settings)
        builder: HeuristicBuilder | LLMScriptBuilder = LLMScriptBuilder(complete)
    else:
        builder = HeuristicBuilder()

    chapters: list[Chapter] = []
    for i, raw in enumerate(book.chapters, start=1):
        if progress:
            progress((i - 1) / max(1, len(book.chapters)), f"Analysing chapter {i}/{len(book.chapters)}")
        chapters.append(builder.build_chapter(i, raw))

    characters = builder.characters()
    for ch in chapters:  # make sure every speaker in the script has a character entry
        for sc in ch.scenes:
            for seg in sc.segments:
                if seg.speaker != NARRATOR:
                    characters.setdefault(seg.speaker, Character(name=seg.speaker))
    return Script(title=book.title, characters=characters, chapters=chapters)
