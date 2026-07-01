"""
Orchestrates the full flow per request:

  messages -> route_conversation (Call A)
     CLARIFY -> return question, empty recs
     REFUSE  -> return canned reply, empty recs
     COMPARE -> compare grounded catalog items
     RECOMMEND/REFINE:
         hard filter -> hybrid retrieve (FAISS) -> grounding gate -> write_reply (Call B)
"""
from __future__ import annotations

from app.schemas import (
    ChatResponse,
    Constraints,
    Message,
    Recommendation,
    Route,
    RouterOutput,
)
from app.services import llm
from app.services.retrieval import retrieval_service

TOP_K = 10


def handle_conversation(messages: list[Message]) -> ChatResponse:
    router_output: RouterOutput = llm.route_conversation(messages)

    if router_output.route == Route.clarify:
        question = router_output.clarifying_question or (
            "Could you tell me more about the role and skills you're hiring for?"
        )
        return ChatResponse(
            reply=question,
            recommendations=[],
            end_of_conversation=False,
        )

    if router_output.route == Route.refuse:
        return ChatResponse(
            reply=llm.canned_refuse_reply(),
            recommendations=[],
            end_of_conversation=False,
        )

    if router_output.route == Route.compare:
        return _handle_compare(messages, router_output.constraints)

    return _handle_recommend(messages, router_output.constraints)


def _handle_recommend(
    messages: list[Message],
    constraints: Constraints,
) -> ChatResponse:

    query = _build_query_text(messages, constraints)

    retrieved_items = retrieval_service.search(
        query=query,
        test_type=[t.value for t in constraints.test_type] or None,
        level=constraints.level.value if constraints.level else None,
        max_duration_minutes=constraints.max_duration_minutes,
        top_k=TOP_K,
    )

    if not retrieved_items:
        return ChatResponse(
            reply=(
                "I couldn't find any assessments matching those constraints. "
                "Could you loosen one of them (for example duration or level)?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    writer_output = llm.write_reply(messages, retrieved_items)

    recommendations = [
        Recommendation(
            name=item.name,
            url=item.url,
            test_type=item.test_type,
        )
        for item in retrieved_items
    ]

    return ChatResponse(
        reply=writer_output.get(
            "reply",
            "Here are some assessments that match your requirements.",
        ),
        recommendations=recommendations,
        end_of_conversation=bool(
            writer_output.get("end_of_conversation", False)
        ),
    )


def _handle_compare(
    messages: list[Message],
    constraints: Constraints,
) -> ChatResponse:

    names_wanted = [n.lower() for n in constraints.compare_items]

    matched = []

    # Use retrieval instead of exact string matching
    for wanted in names_wanted:
        results = retrieval_service.search(
            query=wanted,
            top_k=1,
        )

        if results:
            matched.append(results[0])

    # Remove duplicates
    seen = set()
    matched = [
        item
        for item in matched
        if not (item.url in seen or seen.add(item.url))
    ]

    if len(matched) < 2:
        return ChatResponse(
            reply=(
                "I couldn't find both of those assessments in the catalog. "
                "Could you double-check the assessment names?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    writer_output = llm.write_reply(messages, matched)

    recommendations = [
        Recommendation(
            name=item.name,
            url=item.url,
            test_type=item.test_type,
        )
        for item in matched
    ]

    return ChatResponse(
        reply=writer_output.get(
            "reply",
            "Here's how the assessments compare.",
        ),
        recommendations=recommendations,
        end_of_conversation=bool(
            writer_output.get("end_of_conversation", False)
        ),
    )


def _build_query_text(
    messages: list[Message],
    constraints: Constraints,
) -> str:
    """Build a richer semantic query from the conversation."""

    last_user_msg = next(
        (
            m.content
            for m in reversed(messages)
            if m.role.value == "user"
        ),
        "",
    )

    parts = [last_user_msg]

    if constraints.skills:
        parts.append(
            "Skills: " + ", ".join(constraints.skills)
        )

    if constraints.test_type:
        parts.append(
            "Test types: "
            + ", ".join(t.value for t in constraints.test_type)
        )

    if constraints.level:
        parts.append(f"Level: {constraints.level.value}")

    return " | ".join(parts)