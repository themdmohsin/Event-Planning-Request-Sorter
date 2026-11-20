"""Event request sorter: a GPT-2 model fine-tuned to end each request with a label token like <venue_inquiry>."""

# --- Milestone 1: request types and a small dataset of example requests.
LABELS = ["venue_inquiry", "catering_request", "decor_request", "entertainment_booking",
          "budget_question", "schedule_change"]

DATASET = [
    ("Is your ballroom available on June 14th for 200 guests?", "venue_inquiry"),
    ("Do you have an outdoor garden space we could rent for a wedding?", "venue_inquiry"),
    ("What is the seating capacity of the conference hall?", "venue_inquiry"),
    ("Can we book the rooftop terrace for a corporate party?", "venue_inquiry"),
    ("Is there parking near the venue for our guests?", "venue_inquiry"),
    ("We need a buffet dinner for 150 people with vegetarian options.", "catering_request"),
    ("Can you provide gluten-free and vegan meals at the reception?", "catering_request"),
    ("Please quote us for a three-course plated lunch.", "catering_request"),
    ("We would like a coffee and pastry station in the morning.", "catering_request"),
    ("Do you offer an open bar package with cocktails?", "catering_request"),
    ("We want white and gold flowers on every table.", "decor_request"),
    ("Can you set up fairy lights and drapes along the walls?", "decor_request"),
    ("Please arrange balloon arches at the entrance.", "decor_request"),
    ("We need centerpieces and themed tablecloths for a winter theme.", "decor_request"),
    ("Could you decorate the stage with a floral backdrop?", "decor_request"),
    ("Can you book a live jazz band for the evening?", "entertainment_booking"),
    ("We are looking for a DJ who plays 80s music.", "entertainment_booking"),
    ("Is it possible to hire a magician for the kids?", "entertainment_booking"),
    ("Please find us a string quartet for the ceremony.", "entertainment_booking"),
    ("Can you arrange a photo booth and a host for the night?", "entertainment_booking"),
    ("What is the total cost for a 100 guest wedding?", "budget_question"),
    ("Our budget is 5000 dollars, what can we get for that?", "budget_question"),
    ("Do you require a deposit and how much is it?", "budget_question"),
    ("Are there any discounts for weekday events?", "budget_question"),
    ("Can you send a price breakdown of all services?", "budget_question"),
    ("We need to move our event from Friday to Saturday.", "schedule_change"),
    ("Can we push the start time back by two hours?", "schedule_change"),
    ("The party has been postponed to next month, please update.", "schedule_change"),
    ("Please cancel the rehearsal dinner on the 3rd.", "schedule_change"),
    ("Could we reschedule the tasting to next Tuesday?", "schedule_change"),
]


def label_token(label: str) -> str:
    return f"<{label}>"


def to_training_text(text: str, label: str) -> str:
    return f"Request: {text}\nType: {label_token(label)}"
