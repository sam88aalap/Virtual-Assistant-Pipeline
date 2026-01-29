from llm3 import llm3
from weatherAPI import WeatherAPI
from CalendarAPI import CalendarAPI
from conv import ConversationState, handle_turn
from ASR import listen_once
from TTS import text_to_speech_stream

state = ConversationState()
llm = llm3()
weather = WeatherAPI()
calendar = CalendarAPI()

STOP_WORDS = {"cancel.", "never mind.", "stop.", "exit.", "quit."}

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


def run_voice_assistant():
    print("📝 Assistant ready. Ask about weather or calendar events.\nSay Stop to stop")

    while True:
        text = listen_once()
        if not text:
            continue

        if is_stop(text.lower()):
            break

        print("User:", text)
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


if __name__ == "__main__":
    run_voice_assistant()
