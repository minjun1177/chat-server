"""Reading Server-Sent Events without waiting for data that has not been sent.

Both the MCP client and the cloud model providers stream SSE over `requests`,
and both hit the same trap, so the reader lives here rather than in either.
"""

def sse_chunks(response):
    """Yield bytes from a streaming response *as they arrive*.

    `iter_lines()` cannot be used here: it reads in fixed 512-byte blocks, and
    urllib3 1.x blocks until that many bytes exist. A server-sent event is far
    smaller than that, so an SSE stream would stall until enough later traffic
    happened to fill the block. `read1` (urllib3 2.x) returns whatever is
    already there; falling back to one byte at a time is slower but never waits
    for data that has not been sent.
    """
    read1 = getattr(getattr(response, "raw", None), "read1", None)
    if callable(read1):
        while True:
            chunk = read1(65536)
            if not chunk:
                return
            yield chunk
    else:
        for chunk in response.iter_content(chunk_size=1):
            if chunk:
                yield chunk


def iter_sse(response):
    """Yield (event, data) pairs from a text/event-stream response body."""
    import codecs

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    event = "message"
    data: list[str] = []
    buffer = ""

    for chunk in sse_chunks(response):
        buffer += decoder.decode(chunk)
        while "\n" in buffer:
            line, _, buffer = buffer.partition("\n")
            line = line.rstrip("\r")
            if not line:
                if data:
                    yield event, "\n".join(data)
                event, data = "message", []
                continue
            if line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "event":
                event = value
            elif field == "data":
                data.append(value)
    if data:
        yield event, "\n".join(data)


if __name__ == "__main__":
    print("This file can not run directly.")
