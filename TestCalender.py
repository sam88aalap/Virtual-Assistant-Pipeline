from CalendarAPI import CalendarAPI

# Initialize the API
api = CalendarAPI(calenderid=54)

# --------------------
# 1. Create an event
# --------------------
print("Creating a new event...")
event = api.create_event(
    title="Test Meeting",
    description="This is a test event",
    start_time="2026-01-29 10:00",
    end_time="2026-01-29 11:00",
    location="Berlin"
)
print("Created event:")
print(event)
print("-" * 50)

# Get the ID of the new event
event_id = event.get("id")

# --------------------
# 2. List all events
# --------------------
print("Listing all events...")
events = api.list_events()
print(events)
print("-" * 50)

# --------------------
# 3. Get a specific event by ID
# --------------------
if event_id:
    print(f"Getting details of event ID {event_id}...")
    event_detail = api.get_event(event_id)
    print(event_detail)
    print("-" * 50)

# --------------------
# 4. Update the event
# --------------------
if event_id:
    print(f"Updating event ID {event_id}...")
    updated = api.update_event(event_id, title="Updated Test Meeting", location="Munich")
    print(updated)
    print("-" * 50)

# --------------------
# 5. Delete the event
# --------------------
if event_id:
    print(f"Deleting event ID {event_id}...")
    deleted = api.delete_event(event_id)
    print(deleted)
    print("-" * 50)

# --------------------
# 6. Verify deletion
# --------------------
print("Listing all events after deletion...")
events_after = api.list_events()
print(events_after)
