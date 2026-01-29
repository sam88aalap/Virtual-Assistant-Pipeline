import ollama
import json
import os
import re
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from weatherAPI import WeatherAPI
from CalendarAPI import CalendarAPI

SYSTEM_PROMPT = """
You are a concise, friendly assistant.
Rules:
- Answer simple factual questions directly.
- Use conversation history ONLY when necessary.
- Keep responses short and clear.
- Remember context for follow-up questions when appropriate.
- If you don't know the answer, say "I don't know".
"""

# --------------------
# WEATHER CONDITIONS
# --------------------
WEATHER_CONDITIONS = [
    "rain", "snow", "clear", "cloud", "cloudy",
    "mist", "fog", "sun", "sunny", "storm", "thunder"
]

MAX_CONTEXT_TURNS = 5  # Context expires after N turns


def normalize(text: str) -> str:
    return text.lower().strip()


def extract_location(text: str) -> Optional[str]:
    match = re.search(r"\bin\s+([A-Za-z\s]+)", text, re.IGNORECASE)
    if not match:
        return None
    loc = match.group(1)
    loc = re.sub(
        r"\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        "", loc, flags=re.IGNORECASE
    )
    loc = re.sub(
        r"\b(will|be|like|weather|forecast|on|at)\b",
        "", loc, flags=re.IGNORECASE
    )
    return loc.strip() or None


def extract_day(text: str) -> Optional[str]:
    t = normalize(text)
    if "today" in t:
        return "today"
    if "tomorrow" in t:
        return "tomorrow"
    for d in ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]:
        if d in t:
            return d
    return None


def resolve_day(day: str) -> str:
    if day == "today":
        return datetime.now().strftime("%A").lower()
    if day == "tomorrow":
        return (datetime.now() + timedelta(days=1)).strftime("%A").lower()
    return day


def extract_weather_condition(text: str) -> Optional[str]:
    t = normalize(text)
    for cond in WEATHER_CONDITIONS:
        if cond in t:
            return cond
    return None


def condition_matches(requested: str, actual: str) -> bool:
    actual = normalize(actual)
    mapping = {
        "clear": ["clear"],
        "sun": ["sun", "clear"],
        "sunny": ["sun", "clear"],
        "cloud": ["cloud"],
        "cloudy": ["cloud"],
        "rain": ["rain", "shower"],
        "snow": ["snow"],
        "storm": ["storm", "thunder"],
        "thunder": ["thunder", "storm"],
        "mist": ["mist", "fog"],
        "fog": ["fog", "mist"]
    }
    return any(k in actual for k in mapping.get(requested, []))


def is_weather_query(text: str) -> bool:
    return any(k in normalize(text) for k in [
        "weather", "forecast", "rain", "snow", "sun",
        "cloud", "temperature", "temp", "clear"
    ])


def is_temperature_query(text: str) -> bool:
    return any(k in normalize(text) for k in ["temperature", "temp"])


def is_calendar_query(text: str) -> bool:
    return any(k in normalize(text) for k in [
        "calendar", "appointment", "meeting", "schedule", "event"
    ])


def is_add_event(text: str) -> bool:
    return any(k in normalize(text) for k in [
        "add", "create", "schedule", "set up", "new appointment"
    ])


def is_delete_event(text: str) -> bool:
    return any(k in normalize(text) for k in ["delete", "remove", "cancel"])


def is_update_event(text: str) -> bool:
    return any(k in normalize(text) for k in ["change", "update", "edit"])


class LLM:
    def __init__(
        self,
        model="phi3:mini",
        history_file="conversation_history.json",
        facts_file="facts.json",
        max_exchanges=4,
    ):
        self.model = model
        self.history_file = history_file
        self.facts_file = facts_file
        self.max_exchanges = max_exchanges

        self.history: List[Dict] = []
        self.facts: Dict = {}

        self.weather_api = WeatherAPI()
        self.calendar_api = CalendarAPI()

        # Last weather context for follow-ups
        self.last_weather_context = {"place": None, "day": None, "turn": 0}
        # Last calendar event for follow-ups
        self.last_calendar_event_id = None
        self.last_calendar_turn = 0

        self._load_memory()

    # --------------------
    # MEMORY FUNCTIONS
    # --------------------
    def _load_memory(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    self.history = json.load(f)
            except json.JSONDecodeError:
                self.history = []

        if os.path.exists(self.facts_file):
            try:
                with open(self.facts_file, "r") as f:
                    self.facts = json.load(f)
            except json.JSONDecodeError:
                self.facts = {}

    def _save_memory(self):
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)
        with open(self.facts_file, "w") as f:
            json.dump(self.facts, f, indent=2)

    def _trim_history(self):
        exchanges = []
        buffer = []
        for msg in self.history:
            buffer.append(msg)
            if msg["role"] == "assistant":
                exchanges.append(buffer)
                buffer = []
        exchanges = exchanges[-self.max_exchanges:]
        self.history = [m for pair in exchanges for m in pair]

    # --------------------
    # WEATHER CONTEXT FUNCTIONS
    # --------------------
    def _update_weather_context(self, place, day):
        self.last_weather_context = {
            "place": place,
            "day": day,
            "turn": len(self.history) // 2
        }

    def _get_weather_context(self):
        if (len(self.history)//2 - self.last_weather_context.get("turn",0)) > MAX_CONTEXT_TURNS:
            self.last_weather_context = {"place": None, "day": None, "turn":0}
            return None, None
        return self.last_weather_context.get("place"), self.last_weather_context.get("day")

    # --------------------
    # GENERATE FUNCTION
    # --------------------
    def generate(self, user_text: str) -> str:
        user_norm = normalize(user_text)

        # Forget context command
        if user_norm == "/forget":
            self.last_weather_context = {"place": None, "day": None, "turn":0}
            self.last_calendar_event_id = None
            self.last_calendar_turn = 0
            return "Context forgotten."

        # --------------------
        # WEATHER INTENT
        # --------------------
        if is_weather_query(user_text):
            place = extract_location(user_text)
            day_key = extract_day(user_text)

            # Use last context if missing
            if not place or not day_key:
                last_place, last_day = self._get_weather_context()
                place = place or last_place or "Marburg"
                day_key = day_key or last_day or "today"

            resolved_day = resolve_day(day_key)
            self._update_weather_context(place, day_key)

            forecast = self.weather_api.get_forecast_day(place, resolved_day)
            if forecast.get("error"):
                return "I couldn't find that forecast."

            weather = forecast["weather"]
            tmin = forecast["temperature"]["min"]
            tmax = forecast["temperature"]["max"]

            requested_condition = extract_weather_condition(user_text)

            if requested_condition:
                yesno = "Yes" if condition_matches(requested_condition, weather) else "No"
                return (
                    f"{yesno}. The weather in {place} on {resolved_day} will be {weather}, "
                    f"with a temperature between {tmin}°C and {tmax}°C."
                )

            if is_temperature_query(user_text):
                return f"Today, the temperature in {place} will be between {tmin}°C and {tmax}°C."

            if day_key == "today":
                return f"Today, the weather in {place} will be {weather} with a temperature between {tmin}°C and {tmax}°C."
            return f"The weather in {place} on {resolved_day} will be {weather} with a temperature between {tmin}°C and {tmax}°C."

        # --------------------
        # CALENDAR INTENT
        # --------------------
        if is_calendar_query(user_text):
            events = self.calendar_api.list_events() or []

            # NEXT event
            if "next" in user_norm:
                if not events:
                    return "You have no upcoming events."
                return self.calendar_api.event_to_text(events[0])

            # ADD event
            if is_add_event(user_text):
                title = re.search(r"title(?:d)?\s+([A-Za-z0-9\s]+)", user_text, re.IGNORECASE)
                date = re.search(r"(\d{1,2}(?:th|st|nd|rd)?\s+of\s+[A-Za-z]+)", user_text, re.IGNORECASE)

                event = self.calendar_api.create_event(
                    title=title.group(1).strip() if title else "Untitled",
                    description="Created via assistant",
                    start_time=(date.group(1) if date else "2025-01-12") + " 10:00",
                    end_time=(date.group(1) if date else "2025-01-12") + " 11:00",
                    location=self.facts.get("location", "Marburg"),
                )
                self.last_calendar_event_id = event.get("id")
                self.last_calendar_turn = len(self.history)//2
                return f"Created event: {event.get('title', 'Untitled')}"

            # DELETE event
            if is_delete_event(user_text):
                if self.last_calendar_event_id:
                    self.calendar_api.delete_event(self.last_calendar_event_id)
                    self.last_calendar_event_id = None
                    self.last_calendar_turn = len(self.history)//2
                    return "Deleted the previously created appointment."
                elif events:
                    self.calendar_api.delete_event(events[-1]["id"])
                    return "Deleted the last event."
                else:
                    return "You have no events to delete."

            # UPDATE event
            if is_update_event(user_text):
                if self.last_calendar_event_id:
                    new_loc = extract_location(user_text) or "Marburg"
                    self.calendar_api.update_event(self.last_calendar_event_id, location=new_loc)
                    return f"Updated the previously created appointment location to {new_loc}."
                elif events:
                    new_loc = extract_location(user_text) or "Marburg"
                    self.calendar_api.update_event(events[0]["id"], location=new_loc)
                    return f"Updated the event location to {new_loc}."
                else:
                    return "You have no events to update."

            return "I can help with calendar events."

        # --------------------
        # FALLBACK
        # --------------------
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
            )
            return response["message"]["content"].strip()
        except Exception:
            return "Sorry, I'm having trouble responding right now."

if __name__ == "__main__":
    llm = LLM()
    print("Local assistant running. Type 'exit' to quit. Type '/forget' to clear context.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        print("Assistant:", llm.generate(user_input))
