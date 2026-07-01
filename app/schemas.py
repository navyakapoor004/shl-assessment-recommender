"""
Strict Pydantic schemas — this is the non-negotiable contract for the API.
Every response MUST validate against ChatResponse before it goes out.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator


# ---------------------------------------------------------------------------
# Incoming conversation
# ---------------------------------------------------------------------------

class Role(str, Enum):
    user = "user"
    assistant = "assistant"


class Message(BaseModel):
    role: Role
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1)

    @field_validator("messages")
    @classmethod
    def last_message_must_be_user(cls, v: List[Message]) -> List[Message]:
        if v[-1].role != Role.user:
            raise ValueError("Last message in history must be from role='user'")
        return v


# ---------------------------------------------------------------------------
# Catalog (ground truth loaded from data/catalog.json)
# ---------------------------------------------------------------------------

class TestType(str, Enum):
    cognitive = "cognitive"
    personality = "personality"
    situational_judgement = "situational_judgement"
    skills = "skills"
    behavioral = "behavioral"
    coding = "coding"


class Level(str, Enum):
    entry = "entry"
    mid = "mid"
    senior = "senior"
    graduate = "graduate"
    all_levels = "all_levels"


class CatalogItem(BaseModel):
    name: str
    url: str
    test_type: TestType
    description: str
    level: Level
    duration_minutes: Optional[int] = None
    remote_testing: Optional[bool] = None


# ---------------------------------------------------------------------------
# Router LLM output (Call A) — internal, not returned to client directly
# ---------------------------------------------------------------------------

class Route(str, Enum):
    clarify = "CLARIFY"
    recommend = "RECOMMEND"
    refine = "REFINE"
    compare = "COMPARE"
    refuse = "REFUSE"


class Constraints(BaseModel):
    skills: List[str] = Field(default_factory=list)
    test_type: List[TestType] = Field(default_factory=list)
    level: Optional[Level] = None
    compare_items: List[str] = Field(default_factory=list)  # names to compare, for COMPARE route
    max_duration_minutes: Optional[int] = None


class RouterOutput(BaseModel):
    route: Route
    constraints: Constraints = Field(default_factory=Constraints)
    clarifying_question: Optional[str] = None


# ---------------------------------------------------------------------------
# Outgoing response — THE strict schema the assignment requires
# ---------------------------------------------------------------------------

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: TestType


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"
