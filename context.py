import re
import sys
import asyncio
import itertools
import ollama
import config
import providers
from config import S, smrp


_CHARS_PER_TOKEN = 3.5

def _estimate_tokens(messages: list[dict]) -> int:
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return int(total_chars / _CHARS_PER_TOKEN)


def _get_ctx_budget() -> int:
    return int(config.NUM_CTX * 0.85) - config.NUM_PREDICT


def _get_summary_predict_tokens() -> int:
    # Sized off the local model's weights, which only Ollama reports; a hosted
    # model gets the middle setting.
    if providers.current().name != "ollama":
        return 400
    try:
        model_list = ollama.list()
        m_list = model_list.get("models", []) if isinstance(model_list, dict) else getattr(model_list, 'models', [])
        for m in m_list:
            name = m.get("model", m.get("name", "")) if isinstance(m, dict) else getattr(m, 'model', getattr(m, 'name', ''))
            if name == config.MODEL:
                size_gb = (m.get("size", 0) if isinstance(m, dict) else getattr(m, 'size', 0)) / (1024 ** 3)
                if size_gb >= 15.0: return 600
                if size_gb >= 7.0:  return 400
                return 250
    except Exception:
        pass
    return 300


def _trim_tool_results(messages: list[dict], max_chars: int = 3000) -> list[dict]:
    result = []
    for m in messages:
        content = m.get("content", "")
        if m["role"] == "user" and content.startswith("[Tool Result") and len(content) > max_chars:
            trimmed = content[:max_chars] + f"\n...[Tool result truncated – {len(content) - max_chars} chars omitted]"
            result.append({**m, "content": trimmed})
        else:
            result.append(m)
    return result


def _get_conv_pairs(messages: list[dict]) -> list[list[dict]]:
    pairs: list[list[dict]] = []
    current: list[dict] = []
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "user" and not m["content"].startswith("[Tool Result"):
            if current:
                pairs.append(current)
            current = [m]
        else:
            current.append(m)
    if current:
        pairs.append(current)
    return pairs


async def _compress_context(messages: list[dict]) -> bool:
    conv_msgs = [m for m in messages if m["role"] != "system"]
    if not conv_msgs:
        return False

    predict_tokens = _get_summary_predict_tokens()
    prompt = smrp()
    for m in conv_msgs:
        role = "User" if m["role"] == "user" else "Assistant"
        content = re.sub(r'<tool_call>.*?</tool_call>', '', m['content'], flags=re.DOTALL).strip()
        if content:
            if m["role"] == "user" and m["content"].startswith("[Tool Result"):
                content = content[:400] + ("..." if len(content) > 400 else "")
            prompt += f"[{role}]: {content}\n\n"

    summary_msg = [{"role": "user", "content": prompt}]

    async def spinner():
        frames = [
            f"{S.PURPLE}    ·  {S.R}",
            f"{S.PURPLE}   · · {S.R}",
            f"{S.PURPLE}  · · ·{S.R}",
            f"{S.PURPLE} · · · {S.R}",
            f"{S.PURPLE}· · ·  {S.R}",
            f"{S.PURPLE} · ·   {S.R}",
        ]
        cycle = itertools.cycle(frames)
        try:
            while True:
                frame = next(cycle)
                sys.stdout.write(f'\r  {frame} {S.GRAY}Compressing context…{S.R}  ')
                sys.stdout.flush()
                await asyncio.sleep(0.15)
        except asyncio.CancelledError:
            sys.stdout.write('\r\033[K')
            sys.stdout.flush()

    spin_task = asyncio.create_task(spinner())
    try:
        summary = await providers.complete(summary_msg, max_tokens=predict_tokens)
    except Exception as e:
        summary = ""
    finally:
        if not spin_task.done():
            spin_task.cancel()
            try: await spin_task
            except asyncio.CancelledError: pass

    if not summary or summary.startswith("(Summary failed"):
        return False

    sys_content = messages[0]["content"]
    sys_content = re.sub(r'\n\n<SUMMARY>.*?</SUMMARY>', '', sys_content, flags=re.DOTALL).strip()
    new_sys = f"{sys_content}\n\n<SUMMARY>\n{summary}\n</SUMMARY>"
    messages[0]["content"] = new_sys
    return True


def _sync_loaded_skills(messages: list[dict]) -> None:
    """Forget skills whose instructions were pruned out of the context.

    Without this the model is told a skill is "already loaded" long after the
    compressor dropped the message that carried its instructions.
    """
    if not config.LOADED_SKILLS:
        return
    blob = "\n".join(m.get("content", "") for m in messages)
    config.LOADED_SKILLS[:] = [n for n in config.LOADED_SKILLS if f"[Skill: {n}]\nSource:" in blob]


async def manage_context(messages: list[dict]) -> None:
    await _manage_context(messages)
    _sync_loaded_skills(messages)


async def _manage_context(messages: list[dict]) -> None:
    budget = _get_ctx_budget()

    trimmed = _trim_tool_results(messages)
    if _estimate_tokens(trimmed) <= budget:
        if trimmed != messages:
            messages[:] = trimmed
        return

    messages[:] = trimmed

    pairs = _get_conv_pairs(messages)
    n_pairs = len(pairs)

    if n_pairs <= 2:
        if _estimate_tokens(messages) > budget:
            print(f"\n  {S.WARN}⚠ Context limit approaching. Compressing…{S.R}")
            ok = await _compress_context(messages)
            if ok:
                latest = pairs[-1] if pairs else []
                messages[:] = [messages[0]] + latest
                print(f"  {S.OK}✓ Context compressed.{S.R}\n")
        return

    print(f"\n  {S.WARN}⚠ Context limit approaching. Compressing…{S.R}")
    ok = await _compress_context(messages)
    if ok:
        keep = pairs[-2:]
        keep_msgs = [msg for pair in keep for msg in pair]
        messages[:] = [messages[0]] + keep_msgs
        print(f"  {S.OK}✓ Context compressed ({n_pairs} → {len(keep)} pairs kept).{S.R}\n")
        return

    print(f"  {S.WARN}⚠ Summary failed. Dropping oldest turns…{S.R}")
    while _estimate_tokens(messages) > budget and len(_get_conv_pairs(messages)) > 2:
        cur_pairs = _get_conv_pairs(messages)
        keep_pairs = cur_pairs[1:]
        keep_msgs = [msg for pair in keep_pairs for msg in pair]
        messages[:] = [messages[0]] + keep_msgs

    dropped = n_pairs - len(_get_conv_pairs(messages))
    if dropped > 0:
        print(f"  {S.OK}✓ Dropped {dropped} oldest turn(s).{S.R}\n")
