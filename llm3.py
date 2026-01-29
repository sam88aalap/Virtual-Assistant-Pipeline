import ollama
import json
from typing import Dict, Literal
from weatherAPI import WeatherAPI
from CalendarAPI import CalendarAPI
from conv import ConversationState, handle_turn
from datetime import datetime

class llm3:
    def __init__(self, model="phi3:mini"):
        self.model = model

    def _call_ollama(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}, 
                      {"role": "system", "content": "You extract structured information and reply ONLY in JSON.",}],
            format="json",
            options={"temperature": 0},
        )
        return response["message"]["content"]
    
    def classify_and_extract(self, text: str, state: ConversationState) -> Dict[str, str]:
        """
        Single LLM call to classify intent and extract calendar data if needed.
        """
        prompt = f"""
You are an assistant extracting structured data from a conversation.

Current intent: {state.current_intent or "none"}
Known fields: {state.slots}

Recent conversation:{state.history}

User input:"{text}"

Extract ONLY updated fields.
Respond ONLY in JSON with this schema:
{{
  "intent": "weather | calendar_create | calendar_update | calendar_delete | calendar_get | unknown",
  "title": string or null,
  "description": string or null,
  "start_time": string or null,
  "end_time": string or null,
  "location": string or null,
  "day": string or null,
  "event_id": int or null
}}

Rules:
- Use null for unknown fields
- title: Short event name user provides (e.g., "titled X", "called X").
- intent: ONLY weather or calendar_create or calendar_update or calendar_delete or calendar_get
- description: Event purpose or details.
- start_time: format as ISO 8601 (YYYY-MM-DDTHH:MM:SSZ)
- end_time: format as ISO 8601 (YYYY-MM-DDTHH:MM:SSZ). If only start_time, assume 1-hour duration.
- day: Today is {datetime.now()}. Convert relative dates to **weekday names** (Monday, Tuesday, etc.)
- event_id: get from slot history
- If intent is already set, do NOT change it unless the user explicitly switches tasks.
- Fill missing fields ONLY from history provided if possible.
- Never guess values.
"""
        return json.loads(self._call_ollama(prompt))
    
    
if __name__ == "__main__":

    state = ConversationState()
    llm = llm3()
    weather = WeatherAPI()
    calendar = CalendarAPI()

    STOP_WORDS = {"cancel", "never mind", "stop", "exit", "quit"}

    def is_stop(text: str) -> bool:
        return text.lower().strip() in STOP_WORDS

    def execute_calendar_action(calendar_api, state):
        intent = state.current_intent
        slots = state.slots
        if intent == "calendar_create":
            return calendar_api.create_event(slots["title"], slots["description"], slots["start_time"], slots["end_time"], slots["location"])
        elif intent == "calendar_update":
            event_id = slots.get("event_id")
            update_fields = {k: v for k, v in slots.items() if k != "event_id"}
            return calendar_api.update_event(event_id, **update_fields)
        elif intent == "calendar_delete":
            event_id = slots.get("event_id")
            return calendar_api.delete_event(slots.get("event_id"))
        elif intent == "calendar_get":
            return calendar_api.list_events()

    print("📝 Assistant ready. Ask about weather or calendar events.\nType Stop to stop")

    while True:
        text = input("User: ")

        if is_stop(text):
            break

        res = handle_turn(text, state, llm)
        print("Assistant:", res)

        if res["status"] == "complete":
            intent = res["intent"]
            if intent == None:
                print("Assistant: How may I help you?")
            elif intent.startswith("calendar"):
                result = execute_calendar_action(calendar, state)
                print("📅 Calendar API result:", result)
                if state.current_intent == "calendar_create":
                    state.slots["event_id"] = result["id"]
            elif state.current_intent == "weather":
                forecast = weather.get_forecast_text(state.slots["location"], state.slots["day"])
                print("🌤️ Forecast:", forecast)

            print("\n✅ You can continue with another query.\n")

    #print(llm.process_input("What is the weather in Marburg tomorrow?"))
    #print(llm.process_weather_input("What is the weather in Marburg tomorrow?"))
    #print(llm.process_input("Make doctor's appointment for 2pm till 3pm tomorrow in Marburg for tooth removal"))
    #print(llm.process_input("Set meeting for 5pm today in Room 12 for contract"))

    '''
    weather = WeatherAPI()
    fields = llm.process_input("What is the weather in Marburg tomorrow?")
    print(weather.get_forecast_text(fields["location"], fields["day"]))
    
    
    calendar = CalendarAPI()
    fields = llm.process_input("Make doctor's appointment for 2pm till 3pm tomorrow in Marburg for tooth removal")
    #print(fields)
    print(calendar.create_event(fields["title"], fields["description"], fields["start_time"], fields["end_time"], fields["location"]))
    '''
