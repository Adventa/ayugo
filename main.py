import json
import os
from typing import Any, Literal

from anthropic import Anthropic, APIStatusError
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
API_SECRET = os.environ.get("API_SECRET")
DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

client = Anthropic(api_key=ANTHROPIC_API_KEY)
app = FastAPI(title="Ayugo AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- auth ----------

def check_auth(authorization: str | None) -> None:
    if not API_SECRET:
        return
    expected = f"Bearer {API_SECRET}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.exception_handler(APIStatusError)
async def handle_anthropic_error(request: Request, exc: APIStatusError) -> JSONResponse:
    detail = str(exc)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            detail = err["message"]
    return JSONResponse(
        status_code=502,
        content={
            "error": "anthropic_error",
            "upstream_status": exc.status_code,
            "detail": detail,
        },
    )


# ---------- shared models ----------

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


# ---------- hotel chat ----------

class Hotel(BaseModel):
    id: str
    name: str
    location: str | None = None
    # any extra fields Bubble wants to pass through (price, amenities, etc.)
    extra: dict[str, Any] = Field(default_factory=dict)


class SearchContext(BaseModel):
    location: str
    check_in: str | None = None
    check_out: str | None = None


class HotelChatRequest(BaseModel):
    messages: list[Message]
    search_context: SearchContext
    hotels: list[Hotel]
    model: str | None = None


class HotelMatch(BaseModel):
    id: str
    name: str
    relevance: str


class HotelChatResponse(BaseModel):
    reply: str
    hotels: list[HotelMatch]
    assistant_message: Message


HOTEL_SYSTEM = """You are Ayugo's hotel concierge. You help travellers find hotels from a fixed catalogue the system gives you.

Rules:
- Only recommend hotels from the provided catalogue. Never invent hotels or IDs.
- Match by location relevance, then by anything the user has asked about (price, vibe, amenities, distance).
- On the very first turn, return every reasonably relevant hotel for the search location.
- On follow-ups, refine the list based on what the user said. If nothing changed, return the same list.
- Always call the `return_hotel_results` tool. Never reply in plain text.
- Keep `reply` warm and concise (2-4 sentences). The hotel cards render separately in the UI."""


HOTEL_TOOL = {
    "name": "return_hotel_results",
    "description": "Return the chat reply and the current filtered hotel list.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reply": {
                "type": "string",
                "description": "Natural-language reply shown to the user.",
            },
            "hotels": {
                "type": "array",
                "description": "Filtered hotels, ordered most relevant first.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Hotel ID from the catalogue."},
                        "name": {"type": "string"},
                        "relevance": {
                            "type": "string",
                            "description": "One short sentence on why this hotel matches.",
                        },
                    },
                    "required": ["id", "name", "relevance"],
                },
            },
        },
        "required": ["reply", "hotels"],
    },
}


def build_hotel_system(req: HotelChatRequest) -> list[dict]:
    catalogue = [h.model_dump() for h in req.hotels]
    context_block = (
        f"Search context:\n"
        f"- Location: {req.search_context.location}\n"
        f"- Check-in: {req.search_context.check_in or 'not specified'}\n"
        f"- Check-out: {req.search_context.check_out or 'not specified'}\n"
    )
    catalogue_block = f"Hotel catalogue ({len(catalogue)} hotels):\n{json.dumps(catalogue)}"
    return [
        {"type": "text", "text": HOTEL_SYSTEM},
        {"type": "text", "text": context_block},
        # cache the catalogue — it's the bulk of the tokens and stable across the session
        {"type": "text", "text": catalogue_block, "cache_control": {"type": "ephemeral"}},
    ]


@app.post("/hotel-chat", response_model=HotelChatResponse)
def hotel_chat(
    req: HotelChatRequest,
    authorization: str | None = Header(default=None),
) -> HotelChatResponse:
    check_auth(authorization)
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    response = client.messages.create(
        model=req.model or DEFAULT_MODEL,
        max_tokens=2048,
        system=build_hotel_system(req),
        tools=[HOTEL_TOOL],
        tool_choice={"type": "tool", "name": "return_hotel_results"},
        messages=[m.model_dump() for m in req.messages],
    )

    tool_use = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use is None:
        raise HTTPException(status_code=502, detail="Model did not call the tool")

    payload = tool_use.input
    return HotelChatResponse(
        reply=payload["reply"],
        hotels=[HotelMatch(**h) for h in payload["hotels"]],
        assistant_message=Message(role="assistant", content=payload["reply"]),
    )


# ---------- trip planning ----------

class TripChatRequest(BaseModel):
    messages: list[Message]
    preferences: dict[str, Any] | None = None
    model: str | None = None


class TripDay(BaseModel):
    day: int
    title: str
    location: str
    activities: list[str]
    suggested_hotels: list[str] = Field(default_factory=list)


class TripChatResponse(BaseModel):
    reply: str
    itinerary: list[TripDay]
    recommendations: list[str] = Field(default_factory=list)
    assistant_message: Message


TRIP_SYSTEM = """You are Ayugo's trip designer. Given a traveller's idea or preferences, you build a complete day-by-day plan.

Rules:
- Always call the `return_trip_plan` tool. Never reply in plain text.
- Cover every day of the trip with a clear theme, location, and 3-6 concrete activities.
- Suggest hotels by name only (you don't have a catalogue here). Keep them realistic for the location.
- `reply` is a short conversational intro (2-3 sentences). The itinerary renders separately.
- On follow-ups, revise the existing plan based on what changed — don't restart from scratch unless asked."""


TRIP_TOOL = {
    "name": "return_trip_plan",
    "description": "Return the chat reply and a structured day-by-day itinerary.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reply": {"type": "string"},
            "itinerary": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "day": {"type": "integer"},
                        "title": {"type": "string"},
                        "location": {"type": "string"},
                        "activities": {"type": "array", "items": {"type": "string"}},
                        "suggested_hotels": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["day", "title", "location", "activities"],
                },
            },
            "recommendations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional general tips (packing, transport, season notes).",
            },
        },
        "required": ["reply", "itinerary"],
    },
}


def build_trip_system(req: TripChatRequest) -> list[dict] | str:
    if not req.preferences:
        return TRIP_SYSTEM
    return [
        {"type": "text", "text": TRIP_SYSTEM},
        {"type": "text", "text": f"Traveller preferences:\n{json.dumps(req.preferences)}"},
    ]


@app.post("/trip-plan", response_model=TripChatResponse)
def trip_plan(
    req: TripChatRequest,
    authorization: str | None = Header(default=None),
) -> TripChatResponse:
    check_auth(authorization)
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    response = client.messages.create(
        model=req.model or DEFAULT_MODEL,
        max_tokens=4096,
        system=build_trip_system(req),
        tools=[TRIP_TOOL],
        tool_choice={"type": "tool", "name": "return_trip_plan"},
        messages=[m.model_dump() for m in req.messages],
    )

    tool_use = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use is None:
        raise HTTPException(status_code=502, detail="Model did not call the tool")

    payload = tool_use.input
    return TripChatResponse(
        reply=payload["reply"],
        itinerary=[TripDay(**d) for d in payload["itinerary"]],
        recommendations=payload.get("recommendations", []),
        assistant_message=Message(role="assistant", content=payload["reply"]),
    )


# ---------- health ----------

@app.get("/")
def health() -> dict:
    return {"status": "ok", "service": "ayugo-ai"}