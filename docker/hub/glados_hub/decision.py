"""
Pure decision logic: should GLaDOS respond to what she just heard?

Pipeline (cheapest check first):
  1. Wake word / her name in the transcript -> respond immediately (no LLM).
  2. Otherwise -> the tiny LLM answers YES/NO.

Everything in this module is a pure function so it is trivially testable.
"""

from __future__ import annotations

import re

# The XVF mic + small ASR model will mangle the name. Keep the list forgiving.
_WAKE_WORDS = (
    "glados",
    "gladys",
    "glady's",
    "gladies",
    "glahdos",
    "glah dos",
    "gladous",
    "galadas",
    "glados's",
    "glahdus",
)

# Fallback for mangled transcriptions: "glad" + (dys|dos|das|dus|dous|y's|ies)
_WAKE_RE = re.compile(r"\bglad(?:ah?)?\s?(?:dos|dus|das|dys|dous|y's|ies)\b", re.IGNORECASE)

_NAME_RE = re.compile(
    r"\b(?:i(?:'m| am)|my name is|call me)\s+([a-zA-Z][a-zA-Z'-]{1,15})",
    re.IGNORECASE,
)

_QUESTION_WORDS = (
    "what", "where", "when", "who", "whom", "whose", "why", "how",
    "is", "are", "was", "were", "do", "does", "did", "can", "could",
    "will", "would", "shall", "should", "may", "might", "tell", "give",
    "show", "play", "look up", "search",
)


def detect_wake(text: str) -> bool:
    """True if the transcript addresses GLaDOS by name."""
    if not text:
        return False
    lowered = text.lower()
    for word in _WAKE_WORDS:
        if word in lowered:
            return True
    return bool(_WAKE_RE.search(lowered))


def extract_name(text: str) -> str:
    """Pull a self-introduced name out of the transcript, else ''."""
    if not text:
        return ""
    match = _NAME_RE.search(text)
    if not match:
        return ""
    name = match.group(1).strip().capitalize()
    # Filter out obvious non-names
    if name.lower() in ("not", "no", "your", "you", "a", "the", "fine", "good"):
        return ""
    return name


def looks_like_question(text: str) -> bool:
    """Cheap heuristic (logging/fallback only — the tiny LLM is the decider)."""
    if not text:
        return False
    stripped = text.strip()
    if stripped.endswith("?"):
        return True
    first_words = stripped.lower().split()[:2]
    return any(q in first_words for q in _QUESTION_WORDS)


# Pure vocalizations — a lone interjection that is not, by itself, a
# request. Outside the follow-up window (GLaDOS just spoke) these are
# background noise or the mic's rumble, and must never cost a
# decision-LLM call: a mic fed with constant low-frequency noise
# produces exactly these, every ~20 seconds, all day.
# Deliberately excludes meaningful words (yes/no/ok/okay/sure/...):
# those can answer a question GLaDOS asked just before the follow-up
# window lapsed, and persistent repeats of them are caught by the
# session's repeat-suppression instead.
_BACKCHANNELS = {
    "huh", "uh", "uhh", "um", "hmm", "hm", "mm", "mmm", "mhm",
    "eh", "oh", "ooh", "ooo", "aah",
}


def is_backchannel(text: str) -> bool:
    """True if the transcript is a lone interjection ("huh", "hmm", ...)."""
    if not text:
        return False
    words = re.findall(r"[a-z']+", text.lower())
    return len(words) == 1 and words[0] in _BACKCHANNELS


def build_decision_prompt(text: str, context: str, followup: bool = False) -> str:
    """Prompt for the tiny LLM's YES/NO gatekeeping decision.

    followup=True: the utterance arrived inside the follow-up window
    (GLaDOS just spoke, no wake word). The gate's job flips from "is this
    for GLaDOS?" to "is this a continuation of the exchange that just
    ended?" — after her reply the mic hears the whole room for 20 s, and
    most of what it hears is not addressed to her.
    """
    context_block = context if context else "(none)"
    if followup:
        rule = (
            "GLaDOS just finished speaking, and this utterance arrived WITHOUT a "
            "wake word. Respond YES only if it continues the exchange that just "
            "ended: it answers a question GLaDOS asked, asks about what was just "
            "said, or extends the same topic. Short follow-ups like 'for tomorrow' "
            "or 'where' count YES when they connect to the recent conversation. "
            "EXCEPTION: a direct command or question about something GLaDOS "
            "manages (lights, timers, announcements) is YES even if it is "
            "unrelated to the conversation. Respond NO for a new unrelated topic, "
            "human-to-human conversation, TV or radio audio, or anything that "
            "doesn't connect to the recent conversation. "
        )
    else:
        rule = (
            "Respond YES only if: they call GLaDOS by name, they ask a question or "
            "give a command that GLaDOS should handle, or they are clearly talking "
            "to GLaDOS. "
        )
    return (
        "You are the gatekeeper for GLaDOS, a sarcastic AI assistant on a smart speaker. "
        "Decide if GLaDOS should speak up now.\n"
        f"Recent conversation:\n{context_block}\n"
        f'What the person just said: "{text}"\n'
        + rule
        + "Questions, complaints, or corrections about things GLaDOS manages (lights, "
        "timers, announcements) are YES even if garbled. "
        "Respond NO for human-to-human small talk or background conversation. "
        "Answer with exactly one word: YES or NO."
    )


def parse_decision(answer: str) -> bool:
    """Parse the tiny LLM's answer robustly.

    Template tokens are already stripped upstream; here we look for the
    decision word itself. Anything without a clear YES is NO (fail-safe).
    """
    if not answer:
        return False
    text = " ".join(answer.upper().split())
    if "YES" in text:
        return True
    return False


def route(text: str) -> str:
    """Route a final transcript: 'respond' (wake word) or 'ask_llm'."""
    return "respond" if detect_wake(text) else "ask_llm"


def route_full(text: str, followup_active: bool) -> str:
    """Full routing with the follow-up window.

    - Wake word -> respond (highest priority).
    - Within the follow-up window (GLaDOS just spoke) -> 'followup': the hub
      still gates it, in follow-up mode — a continuation check, not a
      wake-word check. The mic hears the whole room for 20 s after her
      reply, and most of it isn't for her.
    - Otherwise -> let the decision LLM decide.
    """
    if detect_wake(text):
        return "wake"
    if followup_active:
        return "followup"
    return "ask_llm"


# ── Light command detection (deterministic backstop) ─────────────────
# The response LLM occasionally answers a light command in words without
# ever calling the tool — the source of "she says the lights are off but
# they're not". The hub uses light_command() to run the tool ITSELF when
# that happens. It must be conservative: a false positive flips a real
# light nobody asked about, which is worse than a false negative.

_LIGHT_VERBS = ("turn", "switch", "flip")
# Generic light nouns enable the match; room names ("livingroom") are there
# too, because "turn off the livingroom" (no noun) is a command people say.
_LIGHT_NOUNS = (
    "light", "lights", "lamp", "lamps", "ceiling",
    "livingroom", "living room",
)
_LIGHT_NOISE = (
    "the", "a", "an", "please", "now", "it", "them",
    "all", "of", "every", "each", "whole", "entire",
    "glados", "gladys", "glady's", "glahdos",
)
_LIGHT_GENERIC = ("light", "lights", "lamp", "lamps")


def light_command(text: str) -> tuple[str, str] | None:
    """Return (target, state) if the transcript is a light on/off command.

    Requires ALL of: a verb (turn/switch/flip), a light noun, and a state
    word. 'turn off the living room lights' -> ('living room', 'off');
    'the lights are flickering', 'turn up the heat', 'is the light on'
    -> None (no verb). `target` is the remaining words with noise and
    generic nouns stripped, for the Kasa registry resolver; an empty
    target means "the lights" with no room named. When the utterance says
    "all the lights" (scope word, no room) the target is the sentinel
    "all" — the resolver maps that to every configured light.
    """
    words = re.findall(r"[a-z']+", (text or "").lower())
    if not words:
        return None
    if not any(w in _LIGHT_VERBS for w in words):
        return None
    state = "off" if "off" in words else ("on" if "on" in words else None)
    if state is None:
        return None
    if not any(n in words for n in _LIGHT_NOUNS):
        return None
    target = " ".join(
        w for w in words
        if w not in _LIGHT_NOISE
        and w not in _LIGHT_VERBS
        and w != state
        and w not in _LIGHT_GENERIC
    )
    if not target and any(w in ("all", "every") for w in words):
        # "turn on all the lights": explicit scope, no room named.
        return "all", state
    return target, state


# Words that indicate a weather question (checked against the transcript).
_WEATHER_WORDS = (
    "weather", "forecast", "temperature", "raining", "rain", "snowing",
    "snow", "windy", "wind", "humid", "storm", "chance of",
)

# Spoken location patterns, anchored to the end of the utterance so we don't
# grab random words. ASR gives lowercase text without commas.
_LOCATION_PATTERNS = (
    re.compile(r"\bthe weather in\s+([a-z][a-z\s]{1,40}?)\s*[.,!?]*$", re.IGNORECASE),
    re.compile(r"\b(?:it's|its)\s+([a-z][a-z\s]{1,40}?)\s*[.,!?]*$", re.IGNORECASE),
    re.compile(r"\b(?:i'm|im|we're|we are)\s+in\s+([a-z][a-z\s]{1,40}?)\s*[.,!?]*$", re.IGNORECASE),
)

# Words that would make a "location" capture obviously not a place.
_NOT_PLACES = {
    "raining", "snowing", "cold", "hot", "warm", "fine", "good", "great",
    "windy", "humid", "sunny", "cloudy", "freezing", "overcast", "tomorrow",
    "today", "outside", "inside", "here", "there",
}


def weather_intent(text: str) -> bool:
    """True if the transcript is (probably) asking about the weather."""
    if not text:
        return False
    lowered = text.lower()
    return any(word in lowered for word in _WEATHER_WORDS)


def extract_location(text: str) -> str:
    """Pull a spoken place name out of the transcript, else ''."""
    if not text:
        return ""
    for pattern in _LOCATION_PATTERNS:
        match = pattern.search(text)
        if match:
            place = " ".join(match.group(1).split())
            words = place.lower().split()
            if len(words) <= 5 and words[0] not in _NOT_PLACES:
                return place.title()
    return ""