from llm import LLM
from weatherAPI import WeatherAPI
from CalendarAPI import CalendarAPI
from ASR import listen_once, listen_once_mic
from TTS import text_to_speech_stream, text_to_speech_file

llm = LLM()
weather_api = WeatherAPI()
calendar_api = CalendarAPI()


def is_weather_query(text: str) -> bool:
    text = text.lower()
    return any(k in text for k in ["weather", "forecast", "temperature", "rain", "sunny"])


def is_calendar_query(text: str) -> bool:
    text = text.lower()
    return any(k in text for k in ["calendar", "event", "schedule", "reminder", "meeting"])


def handle_weather(text: str) -> str:
    place = llm.facts.get("location", "Marburg")
    return weather_api.get_forecast_text(place)


def handle_calendar(text: str) -> str:
    text = text.lower()

    if "list" in text or "show" in text:
        events = calendar_api.list_events()
        return calendar_api.events_to_text(events)

    if "delete" in text:
        import re
        match = re.search(r"delete event (\d+)", text)
        if match:
            event_id = int(match.group(1))
            calendar_api.delete_event(event_id)
            return f"Event {event_id} deleted."
        return "Tell me the event ID to delete, like 'delete event 12'."

    if "create" in text or "add" in text:
        return "To create an event, please provide: title, start_time, end_time, location."

    return "I can help with calendar events. Ask me to list, create or delete events."


def run_voice_assistant():
    print("Voice assistant running. Say something... (or type 'exit')")

    while True:
        user_text = listen_once()
        #user_text = listen_once_mic()
        print("You: ", user_text)
        if not user_text:
            continue

        if user_text.lower() in {"exit", "quit"}:
            break

        if user_text.lower() == "/reset":
            llm.reset_memory()
            reply = "Memory cleared."
        else:
            reply = llm.generate(user_text)
        '''
        elif is_weather_query(user_text):
            reply = handle_weather(user_text)
        elif is_calendar_query(user_text):
            reply = handle_calendar(user_text)
        '''

        
        print("Assistant:", reply)

        try:
            #text_to_speech_stream(reply)
            text_to_speech_file(reply)
        except Exception as e:
            print("TTS error:", e)


if __name__ == "__main__":
    run_voice_assistant()
