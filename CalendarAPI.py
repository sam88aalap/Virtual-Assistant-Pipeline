import requests


class CalendarAPIError(Exception):
    """Custom exception for Calendar API errors."""
    pass


class CalendarAPI:
    def __init__(self, base_url="https://api.responsible-nlp.net/calendar.php", timeout=10, calenderid=54):
        self.base_url = base_url
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        self.calenderid = calenderid  # Corrected parameter name

    def _handle_response(self, response):
        try:
            response.raise_for_status()
            if response.text:
                return response.json()
            return {}
        except requests.RequestException as e:
            raise CalendarAPIError(f"Request failed: {e}")
        except ValueError:
            raise CalendarAPIError("Invalid JSON response from API")

    # ---------------------------
    # CRUD METHODS
    # ---------------------------
    def create_event(self, title, description, start_time, end_time, location) -> dict:
        payload = {
            "calenderid": self.calenderid,
            "title": title,
            "description": description,
            "start_time": start_time,
            "end_time": end_time,
            "location": location,
        }
        response = requests.post(
            self.base_url,
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        return self._handle_response(response)

    def list_events(self) -> dict:
        response = requests.get(
            self.base_url,
            params={"calenderid": self.calenderid},
            timeout=self.timeout,
        )
        return self._handle_response(response)

    def get_event(self, event_id: int) -> dict:
        response = requests.get(
            self.base_url,
            params={"calenderid": self.calenderid, "id": event_id},
            timeout=self.timeout,
        )
        return self._handle_response(response)

    def update_event(self, event_id: int, **updates) -> dict:
        response = requests.put(
            self.base_url,
            params={"calenderid": self.calenderid, "id": event_id},
            json=updates,
            headers=self.headers,
            timeout=self.timeout,
        )
        return self._handle_response(response)

    def delete_event(self, event_id: int) -> dict:
        response = requests.delete(
            self.base_url,
            params={"calenderid": self.calenderid, "id": event_id},
            timeout=self.timeout,
        )
        return self._handle_response(response)

    # ---------------------------
    # HELPER METHODS
    # ---------------------------
    def event_to_text(self, event: dict) -> str:
        if not event:
            return "Event not found."

        return (
            f"Event #{event.get('id', '?')}\n"
            f"Title: {event.get('title', 'N/A')}\n"
            f"Description: {event.get('description', 'N/A')}\n"
            f"Start: {event.get('start_time', 'N/A')}\n"
            f"End: {event.get('end_time', 'N/A')}\n"
            f"Location: {event.get('location', 'N/A')}"
        )

    def events_to_text(self, events) -> str:
        if not events:
            return "No calendar events found."

        lines = ["Calendar events:", "-" * 40]
        for event in events:
            lines.append(self.event_to_text(event))
            lines.append("-" * 40)

        return "\n".join(lines)


###### test #####

if __name__ == "__main__":
    api = CalendarAPI()
    print(api.list_events())
