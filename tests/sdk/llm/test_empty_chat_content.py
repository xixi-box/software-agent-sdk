from openhands.sdk.llm.message import (
    ImageContent,
    Message,
    MessageToolCall,
    TextContent,
)


LIST_SERIALIZATION_OPTS = {
    "cache_enabled": True,
    "vision_enabled": False,
    "function_calling_enabled": False,
    "force_string_serializer": False,
    "send_reasoning_content": False,
}


def test_list_serializer_normalizes_empty_content_for_all_roles() -> None:
    messages = [
        Message(role="user", content=[]),
        Message(role="system", content=[TextContent(text="   \n\t")]),
        Message(role="assistant", content=[TextContent(text="")]),
        Message(
            role="tool",
            content=[TextContent(text="  ")],
            tool_call_id="call_1",
            name="echo",
        ),
    ]

    for message in messages:
        serialized = message.to_chat_dict(**LIST_SERIALIZATION_OPTS)
        assert serialized["content"] == ""


def test_list_serializer_drops_blank_text_but_preserves_other_content() -> None:
    message = Message(
        role="user",
        content=[
            TextContent(text="   "),
            TextContent(text="keep me"),
            ImageContent(image_urls=["https://example.com/image.png"]),
        ],
    )

    serialized = message.to_chat_dict(
        **{
            **LIST_SERIALIZATION_OPTS,
            "vision_enabled": True,
        }
    )

    assert serialized["content"] == [
        {"type": "text", "text": "keep me"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
    ]


def test_assistant_tool_call_still_omits_empty_content() -> None:
    message = Message(
        role="assistant",
        content=[TextContent(text=" \n")],
        tool_calls=[
            MessageToolCall(
                id="call_1",
                name="echo",
                arguments='{"value":"hello"}',
                origin="completion",
            )
        ],
    )

    serialized = message.to_chat_dict(**LIST_SERIALIZATION_OPTS)

    assert "content" not in serialized
    assert serialized["tool_calls"][0]["id"] == "call_1"
