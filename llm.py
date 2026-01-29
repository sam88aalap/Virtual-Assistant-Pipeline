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
- Do NOT restate past conversations unless asked.
- Keep responses short and clear.
- If you don't know the answer, say "I don't know" and do NOT hallucinate.
"""

client = ollama.Client(
    host=os.getenv("OLLAMA_HOST", "http://localhost:11434")
)


def normalize(text: str) -> str:
    return text.lower().strip()


def extract_location(text: str) -> Optional[str]:
    text = text.strip()

    # 1. Try "in <location>"
    match = re.search(r"\bin\s+([A-Za-z\s]+)", text, re.IGNORECASE)
    if match:
        loc = match.group(1).strip()

        # remove trailing keywords like today/tomorrow/weekdays
        loc = re.sub(
            r"\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            "",
            loc,
            flags=re.IGNORECASE,
        )
        loc = loc.strip()

        # remove any extra words like "will", "be", "like"
        loc = re.sub(
            r"\b(will|be|like|weather|forecast|on|at)\b",
            "",
            loc,
            flags=re.IGNORECASE,
        )
        loc = loc.strip()

        return loc if loc else None

    return None


def extract_day(text: str) -> Optional[str]:
    text = normalize(text)

    if "today" in text:
        return "today"
    if "tomorrow" in text:
        return "tomorrow"

    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for d in days:
        if d in text:
            return d
    return None


def resolve_day(day_keyword: str) -> str:
    if day_keyword == "today":
        return datetime.now().strftime("%A").lower()
    if day_keyword == "tomorrow":
        return (datetime.now() + timedelta(days=1)).strftime("%A").lower()
    return day_keyword


def is_weather_query(text: str) -> bool:
    return any(k in normalize(text) for k in ["weather", "forecast", "rain", "sunny", "temperature", "temp"])


def is_temperature_query(text: str) -> bool:
    return any(k in normalize(text) for k in ["temperature", "temp"])


def is_rain_query(text: str) -> bool:
    return "rain" in normalize(text)


def is_calendar_query(text: str) -> bool:
    return any(k in normalize(text) for k in ["calendar", "appointment", "meeting", "schedule", "event"])


def is_add_event(text: str) -> bool:
    return any(k in normalize(text) for k in ["add", "create", "schedule", "set up", "new appointment"])


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
        self.pending_fact: Optional[Dict] = None

        self.weather_api = WeatherAPI()
        self.calendar_api = CalendarAPI()

        self._load_memory()

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

    def generate(self, user_text: str) -> str:
        user_norm = normalize(user_text)

        # --------------------
        # WEATHER INTENT
        # --------------------
        if is_weather_query(user_text):
            place = extract_location(user_text) or self.facts.get("location", "Marburg")

            day_keyword = extract_day(user_text) or "today"
            day = resolve_day(day_keyword)

            forecast = self.weather_api.get_forecast_day(place, day)
            if forecast.get("error"):
                return "I couldn't find that day in the forecast."

            weather = forecast.get("weather", "unknown")
            temp = forecast.get("temperature", {})
            min_temp = temp.get("min", "?")
            max_temp = temp.get("max", "?")

            # Rain query YES/NO
            if is_rain_query(user_text):
                if "rain" in normalize(weather):
                    return f"Yes, it will rain on {forecast['day']} in {place}. Weather: {weather}, {min_temp}°C and {max_temp}°C."
                return f"No, it will not rain on {forecast['day']} in {place}. Weather: {weather}, {min_temp}°C and {max_temp}°C."

            # Temperature query (ONLY temperature)
            if is_temperature_query(user_text):
                return f"Today the temperature in {place} will be between {min_temp}°C and {max_temp}°C."

            # General weather query (with temperature)
            if day_keyword == "today":
                return f"The weather today in {place} will be {weather} with temperature {min_temp}°C and {max_temp}°C."
            return f"The weather in {place} on {forecast['day']} will be {weather} with temperature {min_temp}°C and {max_temp}°C."

        # --------------------
        # CALENDAR INTENT
        # --------------------
        if is_calendar_query(user_text):
            if "next" in user_norm:
                events = self.calendar_api.list_events()
                if not events:
                    return "You have no upcoming events."
                next_event = events[0]
                return self.calendar_api.event_to_text(next_event)

            if is_add_event(user_text):
                title_match = re.search(r"title(?:d)?\s+([A-Za-z0-9\s]+)", user_text, re.IGNORECASE)
                date_match = re.search(r"(\d{1,2}(?:th|st|nd|rd)?\s+of\s+[A-Za-z]+)", user_text, re.IGNORECASE)

                title = title_match.group(1).strip() if title_match else "Untitled"
                date = date_match.group(1).strip() if date_match else "2025-01-12"

                event = self.calendar_api.create_event(
                    title=title,
                    description="Created via voice assistant",
                    start_time=date + " 10:00",
                    end_time=date + " 11:00",
                    location=self.facts.get("location", "Marburg")
                )
                return f"Created event: {event.get('title', title)}"

            if is_delete_event(user_text):
                events = self.calendar_api.list_events()
                if not events:
                    return "You have no events to delete."
                last_id = events[-1]["id"]
                self.calendar_api.delete_event(last_id)
                return f"Deleted event #{last_id}."

            if is_update_event(user_text):
                events = self.calendar_api.list_events()
                if not events:
                    return "You have no events to update."

                tomorrow_event = events[0]
                new_loc = extract_location(user_text) or "Marburg"
                self.calendar_api.update_event(tomorrow_event["id"], location=new_loc)
                return f"Updated event #{tomorrow_event['id']} location to {new_loc}."

            return "I can help with calendar events. Ask me to list, create, delete, or update events."

        # --------------------
        # FALLBACK to LLM
        # --------------------
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append({"role": "user", "content": user_text})

        try:
            response = client.chat(model=self.model, messages=messages)
            assistant_text = response["message"]["content"].strip()
            if not assistant_text:
                raise RuntimeError("Empty response")
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            return "Sorry, I'm having trouble responding right now. Please try again."

        self._append_exchange(user_text, assistant_text)
        return assistant_text

    def _append_exchange(self, user_text, assistant_text):
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": assistant_text})
        self._trim_history()
        self._save_memory()

    def reset_memory(self):
        self.history = []
        self.facts = {}
        self.pending_fact = None
        self.last_list = None
        self._save_memory()


if __name__ == "__main__":
    llm = LLM()
    print("Local assistant running.")
    print("Commands: /reset | exit\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            break

        if user_input.lower() == "/reset":
            llm.reset_memory()
            print("Assistant: Memory cleared.")
            continue

        reply = llm.generate(user_input)
        print("Assistant:", reply)
