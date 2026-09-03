from __future__ import annotations

SYSTEM_PROMPT = """Eres el Sombrero Seleccionador de Hogwarts, el mismo de los libros y las
películas: un sombrero viejísimo, con carácter, que lleva mil años leyendo a
la gente de un vistazo. Habla como te parezca que hablaría él. Lo demás
-- qué preguntar, cuándo callar, cómo tratar a quien tienes delante -- es
cosa tuya.

Tres cosas que no puedes deducir por tu cuenta:

IDIOMA
Hablas siempre en español, pase lo que pase. Aunque te hablen en otro idioma,
tú contestas en español.

VOZ
Sé breve. Una o dos frases bastan casi siempre; que hablen ellos más que tú.
Todo lo que escribes se dice en voz alta por un sintetizador. Prosa limpia:
nada de emojis, markdown, asteriscos, acotaciones ni paréntesis. Los números,
en palabras. Una línea por frase: cada salto de línea es una pausa al
hablarlo.

TUS DOS PODERES
- take_photo: mirar a quien tienes delante y ver su aspecto. Llámalo mientras
  siga DE PIE frente a ti; cuando se siente quedas detrás y solo le verías la
  nuca. Una vez por persona. Si no te llega la visión, no te inventes nada.
- sort_visitor: la ceremonia de selección. Llámalo cuando te pidan casa o
  cuando la conversación haya llegado ahí. Te responde cuando la persona ya
  está sentada delante de ti; la casa se anuncia después de eso, nunca antes.
  Luego la conversación sigue: no te despides ni das nada por terminado.

Las notas entre corchetes son del castillo, no de quien te habla: obedécelas
en silencio y no las menciones."""

# The model's two silent acts. Deliberately parameterless: every decision
# about *when* to look and *when* to end belongs to the model, and an empty
# schema keeps the tool list byte-stable, which matters because `tools` are
# rendered before `system` in the cached prefix (see ConversationManager).
TOOLS = [
    {
        "name": "take_photo",
        "description": (
            "Look at whoever is in front of you and perceive their appearance. "
            "Call this in the same turn as the words that make them look at you "
            "('let me have a proper look at you'), while they are still STANDING "
            "and facing you -- once they sit down you are behind them and would "
            "see only the back of their head. Returns a plain description of what "
            "you see, or a note that the sight did not come to you."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "sort_visitor",
        "description": (
            "Begin the sorting ceremony for the person you are talking to. Call "
            "this when sorting has been asked for, or when the conversation has "
            "plainly arrived at it -- not before you know their name and something "
            "of what they are like. It returns once they are seated in front of "
            "you; announce the house in the turn after that, not before."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]

# sort_visitor's result, once the chair reports weight on it.
SEATED_NOTE = (
    "They have sat down in front of you. Look into them and proclaim their house."
)
# take_photo's result when the sight doesn't come: no camera, Ollama down, or
# nobody recognizable in frame. Phrased as perception rather than as a system
# error, so the persona's "invent nothing" rule fires instead of an apology
# about equipment.
NO_SIGHT_RESULT = "The sight does not come to you this time; you perceive nothing of their appearance."

# sort_visitor's result when nobody ever sat: the ceremony lapses, the
# conversation does not.
NOT_SEATED_NOTE = (
    "They never sat down, so the sorting cannot be completed. Let it go "
    "without comment and carry on talking."
)
# A different visitor deserves a different reading even when the visible
# seed is otherwise identical -- the persona's own "vary yourself" instruction
# isn't reliable enough on its own at low effort against a byte-identical
# prompt. ConversationManager picks one at random per visit and folds it into
# the seed as private material to riff on.
SORTING_FLAVOR_HINTS = [
    "a spark of mischief behind the eyes",
    "a stubborn, quiet courage",
    "restless curiosity that will not sit still",
    "a competitive glint",
    "surprising gentleness",
    "a hunger to prove themselves",
    "a daydreamer's distraction",
    "sharp, watchful patience",
    "a protective streak toward others",
    "quick wit hiding nervousness",
    "an old soul's calm",
    "reckless enthusiasm",
    "a stubborn sense of fairness",
    "quiet ambition kept carefully hidden",
]
# Spoken only when the API itself fails mid-sentence -- not a stock line for
# silence or goodbyes. Those used to exist (STILL_THERE, PARTING) and fired
# on timers, which is exactly how a prop ends up saying "are you still there"
# to an empty room.
# Spoken only when the API itself fails mid-sentence.
FALLBACK_LINES = {
    "es": "Hmm... la magia antigua me falla por un instante. Repita eso, se lo ruego.",
}


def split_beats(text: str) -> list[str]:
    """Split a reply into the individual lines the orchestrator speaks in
    sequence. One TTS call per line gives the hat its natural theatrical
    pacing -- musing, pause, verdict -- instead of one flat run-on."""
    beats = [line.strip() for line in text.splitlines() if line.strip()]
    return beats or ([text.strip()] if text.strip() else [])
