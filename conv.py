from dataclasses import dataclass, field
import json

MAX_HISTORY_TURNS = 4

@dataclass
class ConversationState:
    current_intent: str | None = None
    slots: dict = field(default_factory=dict)
    history: list = field(default_factory=list)

def append_history(state: ConversationState, role: str, content: str):
    state.history.append({"role": role, "content": content})
    state.history = state.history[-MAX_HISTORY_TURNS:]

# Slot handling
REQUIRED_FIELDS = {
    "calendar_create": ["title", "description", "start_time", "end_time", "location"],
    "calendar_update": ["event_id"],  # plus optional slots to update
    "calendar_delete": ["event_id"],
    "calendar_get": [],  # optional: "day"
    "weather": ["location", "day"]
}

CALENDAR_ACTIONS = [
    "calendar_create",
    "calendar_update",
    "calendar_delete",
    "calendar_get"
]

def merge_slots(state: ConversationState, extracted: dict):
    for key, value in extracted.items():
        if key != "intent" and value is not None:
            state.slots[key] = value

def validate_slots(state: ConversationState):
    required = REQUIRED_FIELDS.get(state.current_intent, [])
    return [f for f in required if not state.slots.get(f)]

# Follow-up question builder
def build_followup(missing_fields: list[str]) -> str:
    if len(missing_fields) == 1:
        return f"Please provide {missing_fields[0]}."
    return (
        "Please provide "
        + ", ".join(missing_fields[:-1])
        + " and "
        + missing_fields[-1]
        + "."
    )

# Main conversation turn
def handle_turn(text: str, state: ConversationState, extractor) -> dict:
    extracted = extractor.classify_and_extract(text, state)

    new_intent = extracted["intent"]
    if new_intent != state.current_intent and new_intent != "unknown":
        # Reset slots for new intent
        state.slots = {}
        state.current_intent = new_intent

    merge_slots(state, extracted)
    #state.slots['day'] = resolve_day(state.slots.get('day'))
    missing = validate_slots(state)
    append_history(state, "user", text)

    if missing:
        question = build_followup(missing)
        append_history(state, "assistant", question)
        return {"status": "missing_info", "question": question, "state": state, "intent": state.current_intent}
    else:
        complete = {"status": "complete", "intent": state.current_intent, "data": state.slots}
        append_history(state, "assistant", json.dumps(complete))
        return complete
