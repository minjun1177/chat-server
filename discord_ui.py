"""The Discord end of `ui.Sink`.

Everything the terminal does with ANSI and `input()` this does with messages,
embeds and buttons:

- prose arrives as deltas and is written into a message that is edited about
  once a second, split across messages at 2000 characters and spilled into a
  file attachment when an answer is genuinely long;
- every tool call gets its own embed, posted the moment the call starts and
  edited in place when the result comes back;
- an approval, a question or a plan becomes a row of buttons, and the tool
  thread blocks on the click exactly the way it used to block on `input()`.

The thread part matters: tool handlers are synchronous, so they run in a worker
thread and reach back into the bot's loop with `run_coroutine_threadsafe`. That
is why the event loop stays free to answer other channels while `run_cmd` runs.
"""

import asyncio
import io
import os

import discord

import config
import permissions
import ui


RUNNING = discord.Color.blurple()
DONE = discord.Color.green()
FAILED = discord.Color.red()
WAITING = discord.Color.gold()
INFO = discord.Color.greyple()

_LEVEL_COLORS = {"warn": WAITING, "error": FAILED, "ok": DONE, "info": INFO}


def _fence_language(text: str) -> str | None:
    """The language of an unclosed ``` fence, or None when none is open."""
    language = None
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped.startswith("```"):
            continue
        if language is None:
            language = stripped[3:].strip()
        else:
            language = None
    return language


async def _wait_for_view(view: discord.ui.View, timeout: float) -> bool:
    """Wait for a click. True means nobody answered.

    `View.wait()` is driven by the view store, which only knows about the view
    once its message actually reached Discord - so a send that failed would
    leave the waiting tool thread hanging. The clock is kept here as well.
    """
    try:
        return await asyncio.wait_for(view.wait(), timeout + 5)
    except (asyncio.TimeoutError, TimeoutError):
        view.stop()
        return True


def _split_point(text: str, limit: int) -> int:
    """Where to break a message that is over Discord's length cap.

    A break lands on a line boundary, because a message cut mid-sentence - or
    worse, mid-word - is what makes long answers unreadable. A blank line is
    better than a single newline, a space is the last resort, and only a line
    longer than the whole limit is cut where it happens to fall.
    """
    window = text[:limit]
    # A paragraph break is the nicest place to stop, but only if it is late
    # enough to be worth a whole message - otherwise a lone blank line near the
    # top would strand three words in their own post.
    for separator, floor in (("\n\n", limit // 2), ("\n", 0)):
        index = window.rfind(separator)
        if index > floor:
            return index
    index = window.rfind(" ")
    if index > limit // 4:
        return index
    return limit


def _clip(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit - 20].rstrip() + f"\n… (+{len(text) - limit + 20} chars)"


# ---------------------------------------------------------------------------
# interactive views
# ---------------------------------------------------------------------------

class _Gated(discord.ui.View):
    """A view only the person who asked (or an owner) may press."""

    def __init__(self, allowed_ids: set[int], timeout: float):
        super().__init__(timeout=timeout)
        self.allowed_ids = allowed_ids

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if not self.allowed_ids or interaction.user.id in self.allowed_ids:
            return True
        await interaction.response.send_message(
            "Only the person who asked can use these buttons.", ephemeral=True)
        return False

    def _finish(self):
        for child in self.children:
            child.disabled = True
        self.stop()


class ApprovalView(_Gated):
    def __init__(self, allowed_ids: set[int], rule: str, can_save_rule: bool, timeout: float):
        super().__init__(allowed_ids, timeout)
        self.result = False
        self.saved_rule = ""
        self.rule = rule
        if not (rule and can_save_rule):
            self.remove_item(self.always)

    @discord.ui.button(label="Approve", style=discord.ButtonStyle.success, emoji="✅")
    async def approve(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.result = True
        self._finish()
        await interaction.response.edit_message(view=self)

    @discord.ui.button(label="Deny", style=discord.ButtonStyle.danger, emoji="✖")
    async def deny(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.result = False
        self._finish()
        await interaction.response.edit_message(view=self)

    @discord.ui.button(label="Always allow", style=discord.ButtonStyle.secondary, emoji="♾")
    async def always(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.result = True
        saved, where = permissions.add_rule(self.rule, "allow")
        self.saved_rule = f"{self.rule} → {where}" if saved else f"could not save the rule: {where}"
        self._finish()
        await interaction.response.edit_message(view=self)


class ChoiceView(_Gated):
    def __init__(self, allowed_ids: set[int], options: list[str], timeout: float):
        super().__init__(allowed_ids, timeout)
        self.answer: str | None = None
        for index, option in enumerate(options[:20]):
            self.add_item(_ChoiceButton(option, index))
        self.add_item(_CustomButton())


class _ChoiceButton(discord.ui.Button):
    def __init__(self, option: str, index: int):
        label = option if len(option) <= 78 else option[:75] + "..."
        super().__init__(label=f"{index + 1}. {label}", style=discord.ButtonStyle.primary,
                         row=min(index // 2, 3))
        self.option = option

    async def callback(self, interaction: discord.Interaction):
        self.view.answer = self.option
        self.view._finish()
        await interaction.response.edit_message(view=self.view)


class _CustomButton(discord.ui.Button):
    def __init__(self):
        super().__init__(label="Type an answer", style=discord.ButtonStyle.secondary, emoji="✏", row=4)

    async def callback(self, interaction: discord.Interaction):
        modal = _TextModal("Your answer", "Answer")
        await interaction.response.send_modal(modal)
        await modal.wait()
        if modal.value:
            self.view.answer = modal.value
            self.view._finish()


class _TextModal(discord.ui.Modal):
    def __init__(self, title: str, label: str):
        super().__init__(title=title[:45], timeout=config.DISCORD_QUESTION_TIMEOUT)
        self.value = ""
        self.field = discord.ui.TextInput(label=label[:45], style=discord.TextStyle.paragraph,
                                          required=True, max_length=2000)
        self.add_item(self.field)

    async def on_submit(self, interaction: discord.Interaction):
        self.value = str(self.field.value)
        await interaction.response.defer()


class PlanView(_Gated):
    def __init__(self, allowed_ids: set[int], timeout: float):
        super().__init__(allowed_ids, timeout)
        self.verdict = "reject"
        self.feedback = ""

    @discord.ui.button(label="Approve", style=discord.ButtonStyle.success, emoji="✅")
    async def approve(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.verdict = "approve"
        self._finish()
        await interaction.response.edit_message(view=self)

    @discord.ui.button(label="Deny", style=discord.ButtonStyle.danger, emoji="✖")
    async def reject(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.verdict = "reject"
        self._finish()
        await interaction.response.edit_message(view=self)

    @discord.ui.button(label="Revise", style=discord.ButtonStyle.secondary, emoji="✏")
    async def revise(self, interaction: discord.Interaction, button: discord.ui.Button):
        modal = _TextModal("Revise the plan", "What should change?")
        await interaction.response.send_modal(modal)
        await modal.wait()
        self.verdict = "revise"
        self.feedback = modal.value
        self._finish()


class StopView(discord.ui.View):
    """The one button that is not gated to a single person: anyone allowed to
    talk to the bot in this channel may stop a runaway turn."""

    def __init__(self, on_cancel, allowed_ids: set[int]):
        super().__init__(timeout=None)
        self.on_cancel = on_cancel
        self.allowed_ids = allowed_ids

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if not self.allowed_ids or interaction.user.id in self.allowed_ids:
            return True
        await interaction.response.send_message("You cannot use this.", ephemeral=True)
        return False

    @discord.ui.button(label="Stop", style=discord.ButtonStyle.danger, emoji="⏹")
    async def stop_turn(self, interaction: discord.Interaction, button: discord.ui.Button):
        button.disabled = True
        self.stop()
        await interaction.response.edit_message(view=self)
        self.on_cancel()


# ---------------------------------------------------------------------------
# the sink
# ---------------------------------------------------------------------------

class DiscordSink(ui.Sink):

    def __init__(self, client, channel, author, *, tool_policy=None, workspace="",
                 owner_ids=(), interaction=None, on_cancel=None):
        self.client = client
        self.channel = channel
        self.author = author
        self.interaction = interaction
        self.loop = asyncio.get_running_loop()
        self.tool_policy = tool_policy or ui.ALLOW_ALL
        self.workspace = workspace
        self.owner_ids = set(owner_ids)
        self.on_cancel = on_cancel

        self._chunk = ""              # text of the message being written
        self._pending = ""            # deltas not yet written out
        self._prefix = ""             # reopened code fence at the top of a chunk
        self._message = None          # the message currently being edited
        self._overflow = ""           # text past DISCORD_MAX_MESSAGES, sent as a file
        self._chunks_sent = 0
        self._flusher = None
        self._stopping = asyncio.Event()
        # The flusher task and the turn both call _flush, and _flush awaits an
        # HTTP request in the middle - without this the same text could go out
        # twice as two messages.
        self._writing = asyncio.Lock()
        self._last_written = ""
        self._thinking = ""
        self._used_original = False

        # These live for the whole exchange, not for one streamed reply: a turn
        # that calls three tools streams four times, and the user should still
        # see one typing indicator and one stop button.
        self._typing = None
        self._stop_message = None
        self._stop_view = None
        self._closed = False
        self._posted = False          # did this exchange put anything on screen?

    # -- plumbing ----------------------------------------------------------

    @property
    def _allowed_ids(self) -> set[int]:
        ids = set(self.owner_ids)
        if self.author is not None:
            ids.add(self.author.id)
        return ids

    def _call(self, coro, timeout: float):
        """Run a coroutine on the bot loop from the tool thread and wait."""
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result(timeout)

    async def _send(self, **kwargs):
        """Send into the conversation, following an interaction when there is one.

        A slash command's first message has to be the deferred response itself,
        or the "is thinking…" placeholder never resolves; everything after it is
        a followup. Empty keywords are dropped rather than passed as None -
        `Webhook.send` rejects `view=None`, where `Messageable.send` accepts it.
        """
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        if self.interaction is not None:
            if not self.interaction.response.is_done():
                await self.interaction.response.defer(thinking=True)
            if not self._used_original and "file" not in kwargs:
                self._used_original = True
                return await self.interaction.edit_original_response(**kwargs)
            return await self.interaction.followup.send(wait=True, **kwargs)
        if self.channel is None:
            raise RuntimeError("this conversation has no channel to write to")
        return await self.channel.send(**kwargs)

    # -- typing indicator --------------------------------------------------

    def _start_typing(self) -> None:
        """Show Discord's own "typing…" state instead of posting a placeholder.

        It clears itself the moment a message goes out and comes back while the
        next tool runs, which is exactly the shape of a turn.
        """
        if self._typing is not None or self.channel is None:
            return
        if not hasattr(self.channel, "typing"):
            return
        self._typing = asyncio.create_task(self._typing_loop())

    async def _typing_loop(self) -> None:
        try:
            async with self.channel.typing():
                while True:
                    await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise
        except Exception:                       # noqa: BLE001 - never break a turn over this
            pass

    def _claim_stop_view(self):
        """The stop button rides on the first thing this exchange posts.

        A turn that starts with a tool call has nothing else to hang it on, so
        the tool card carries it - and `close()` takes it away again.
        """
        if self._stop_message is not None or self.on_cancel is None:
            return None
        self._stop_view = StopView(self.on_cancel, self._allowed_ids)
        return self._stop_view

    async def _stop_typing(self) -> None:
        if self._typing is None:
            return
        self._typing.cancel()
        try:
            await self._typing
        except (asyncio.CancelledError, Exception):     # noqa: BLE001
            pass
        self._typing = None

    # -- streaming ---------------------------------------------------------

    async def turn_start(self) -> None:
        self._stopping = asyncio.Event()
        self._start_typing()
        self._flusher = asyncio.create_task(self._flush_loop())

    async def _flush_loop(self):
        """Edit the message every DISCORD_EDIT_INTERVAL until asked to stop.

        It is asked rather than cancelled: cancelling lands anywhere, including
        between "Discord accepted this message" and "we recorded which message
        it is", and the next flush would then post the text a second time.
        """
        while not self._stopping.is_set():
            try:
                await asyncio.wait_for(self._stopping.wait(),
                                       timeout=config.DISCORD_EDIT_INTERVAL)
            except (asyncio.TimeoutError, TimeoutError):
                pass
            await self._flush()

    async def text(self, delta: str) -> None:
        self._pending += delta

    async def thinking(self, delta: str) -> None:
        show = config.SHOW_THINKING if self.show_thinking is None else self.show_thinking
        if show:
            self._thinking += delta

    async def mark_tool_call(self) -> None:
        # Anything the model typed before the tag is worth showing; the tag is not.
        await self._flush()

    async def _flush(self, final: bool = False) -> None:
        async with self._writing:
            await self._flush_locked(final)

    async def _flush_locked(self, final: bool = False) -> None:
        if self._pending:
            self._chunk += self._pending
            self._pending = ""
        if not self._chunk.strip() and not final:
            return

        limit = config.DISCORD_MSG_LIMIT
        while len(self._chunk) > limit:
            cut = _split_point(self._chunk, limit)
            head, rest = self._chunk[:cut].rstrip(), self._chunk[cut:].lstrip("\n")
            language = _fence_language(head)
            if language is not None:
                head += "\n```"
            await self._write(head, close=True)
            self._prefix = f"```{language}\n" if language else ""
            self._chunk = self._prefix + rest

        if self._chunk.strip():
            await self._write(self._chunk, close=final)
        elif final and self._message is not None:
            self._message = None
            self._last_written = ""

    async def _write(self, content: str, close: bool = False) -> None:
        """Create or edit the message that holds `content`."""
        content = content.strip()
        if not content:
            if close:
                self._message = None
                self._chunk = ""
                self._last_written = ""
            return

        if self._chunks_sent >= config.DISCORD_MAX_MESSAGES and self._message is None:
            self._overflow += content + "\n"
            if close:
                self._chunk = ""
            return

        if content == self._last_written:
            # Nothing new since the last edit - do not spend a request on it.
            if close:
                self._message = None
                self._chunk = ""
                self._last_written = ""
            return
        self._last_written = content

        try:
            if self._message is None:
                view = self._claim_stop_view()
                self._message = await self._send(content=content, view=view)
                self._chunks_sent += 1
                self._posted = True
                if view is not None:
                    self._stop_message = self._message
            else:
                await self._message.edit(content=content)
        except Exception as error:              # noqa: BLE001 - a turn outlives a failed edit
            print(f"[discord] could not write a message: {error!r}")
            self._last_written = ""             # so the next flush tries again

        if close:
            self._message = None
            self._chunk = ""
            self._last_written = ""

    async def turn_end(self, prompt_tokens: int, completion_tokens: int,
                       total_seconds: float, eval_seconds: float) -> None:
        """One streamed reply is over - but the exchange may not be."""
        if self._flusher is not None:
            self._stopping.set()
            try:
                await self._flusher
            except Exception as error:          # noqa: BLE001 - a turn outlives a failed edit
                print(f"[discord] the flusher stopped badly: {error!r}")
            self._flusher = None

        await self._flush(final=True)

        if self._thinking.strip():
            preview = _clip(self._thinking.strip(), 1500)
            quoted = "\n".join("> " + line for line in preview.split("\n"))
            try:
                await self._send(content=f"**Reasoning**\n{quoted}")
            except discord.HTTPException:
                pass
            self._thinking = ""

        self._chunks_sent = 0

    async def close(self) -> None:
        """The whole exchange is over: stop typing and retire the stop button."""
        if self._closed:
            return
        self._closed = True

        await self._flush(final=True)
        await self._stop_typing()

        if self._overflow.strip():
            data = io.BytesIO(self._overflow.strip().encode("utf-8"))
            try:
                await self._send(content="The rest of the answer was too long for a message.",
                                 file=discord.File(data, filename="answer.md"))
            except discord.HTTPException as error:
                print(f"[discord] could not upload the overflow: {error}")
            self._overflow = ""

        if not self._posted and not self.aborted:
            # A turn that produced nothing at all would otherwise just stop
            # typing, which reads as the bot ignoring the message. A turn the
            # user stopped is different - it already said so.
            try:
                await self._send(content="(the model returned an empty answer)")
            except discord.HTTPException:
                pass

        if self._stop_message is not None:
            if self._stop_view is not None:
                self._stop_view.stop()
            try:
                await self._stop_message.edit(view=None)
            except discord.HTTPException:
                pass
            self._stop_message = None
            self._stop_view = None

    # -- tools -------------------------------------------------------------

    async def tool_call(self, name: str, arguments: dict):
        await self._flush(final=True)          # close the prose above the card
        # The message budget guards one answer against flooding a channel; a
        # turn that legitimately runs ten tools starts each segment fresh.
        self._chunks_sent = 0

        embed = discord.Embed(title=f"▸ {name}", color=RUNNING)
        if arguments:
            lines = []
            for key, value in arguments.items():
                text = value if isinstance(value, str) else repr(value)
                text = text.replace("```", "'''")
                lines.append(f"{key}: {_clip(text, 300)}")
            embed.description = "```\n" + _clip("\n".join(lines), config.DISCORD_TOOL_ARG_CHARS) + "\n```"
        embed.set_footer(text="running…")
        try:
            view = self._claim_stop_view()
            message = await self._send(embed=embed, view=view)
            self._posted = True
            if view is not None:
                self._stop_message = message
            return message
        except discord.HTTPException as error:
            print(f"[discord] could not post a tool card: {error}")
            return None

    async def tool_result(self, handle, name: str, result: str) -> None:
        result = result or ""
        failed = result.lstrip().startswith(("[Error]", "[System] User denied",
                                             "[System] The tool", "[System] '"))
        if handle is None:
            return
        embed = handle.embeds[0] if handle.embeds else discord.Embed(title=f"▸ {name}")
        embed.color = FAILED if failed else DONE
        body = _clip(result.replace("```", "'''"), config.DISCORD_TOOL_RESULT_CHARS)
        embed.clear_fields()
        embed.add_field(name="Result", value=f"```\n{body}\n```" if body else "(empty result)", inline=False)
        embed.set_footer(text=f"{'failed' if failed else 'done'} · {len(result)} chars")
        try:
            await handle.edit(embed=embed)
        except discord.HTTPException as error:
            print(f"[discord] could not update a tool card: {error}")

    async def run_tool(self, fn, name: str, arguments: dict):
        # Tool handlers are synchronous and some of them take minutes. Off the
        # loop they go, so the bot keeps answering everyone else meanwhile.
        return await asyncio.to_thread(fn, name, arguments)

    async def notice(self, text: str, level: str = "info") -> None:
        embed = discord.Embed(description=text, color=_LEVEL_COLORS.get(level, INFO))
        try:
            await self._send(embed=embed)
        except discord.HTTPException:
            pass

    # -- asking (called from the tool thread) -------------------------------

    def ask_approval(self, label: str, details: list[tuple[str, str]], rule: str = "") -> bool:
        timeout = config.DISCORD_APPROVAL_TIMEOUT
        try:
            return self._call(self._approval_flow(label, details, rule, timeout), timeout + 30)
        except Exception as error:                  # noqa: BLE001 - a failure must not approve
            print(f"[discord] approval failed: {error!r}")
            return False

    async def _approval_flow(self, label, details, rule, timeout) -> bool:
        await self._flush(final=True)
        is_owner = self.author is not None and self.author.id in self.owner_ids

        embed = discord.Embed(title=f"⚠ Approval needed · {label}", color=WAITING)
        for key, value in details[:10]:
            text = str(value).replace("```", "'''")
            embed.add_field(name=key, value=f"```\n{_clip(text, 900)}\n```", inline=False)
        if rule:
            embed.set_footer(text=f"'Always allow' saves the rule {rule} · {timeout}s to answer")
        else:
            embed.set_footer(text=f"No answer within {timeout}s counts as a refusal")

        view = ApprovalView(self._allowed_ids, rule, is_owner, timeout)
        message = await self._send(embed=embed, view=view)
        timed_out = await _wait_for_view(view, timeout)

        embed.color = DONE if (not timed_out and view.result) else FAILED
        verdict = "timed out · denied" if timed_out else ("approved" if view.result else "denied")
        if view.saved_rule:
            verdict += f" · {view.saved_rule}"
        embed.set_footer(text=verdict)
        try:
            await message.edit(embed=embed, view=None)
        except discord.HTTPException:
            pass
        return bool(view.result) and not timed_out

    def ask_choice(self, question: str, options: list[str], index: int, total: int) -> str | None:
        timeout = config.DISCORD_QUESTION_TIMEOUT
        try:
            return self._call(self._choice_flow(question, options, index, total, timeout), timeout + 30)
        except Exception as error:                  # noqa: BLE001
            print(f"[discord] question failed: {error!r}")
            return None

    async def _choice_flow(self, question, options, index, total, timeout) -> str | None:
        await self._flush(final=True)
        counter = f" ({index}/{total})" if total > 1 else ""
        embed = discord.Embed(title=f"❓ Question{counter}", description=question, color=WAITING)

        if not options:
            if self.channel is None or self.client is None:
                embed.set_footer(text="(this conversation cannot receive a typed answer)")
                await self._send(embed=embed)
                return None
            embed.set_footer(text=f"Write your answer in this channel ({timeout}s)")
            await self._send(embed=embed)

            def check(message):
                return (message.channel.id == self.channel.id
                        and message.author.id in self._allowed_ids
                        and not message.author.bot)
            try:
                reply = await self.client.wait_for("message", check=check, timeout=timeout)
            except asyncio.TimeoutError:
                return None
            return config.safe_text(reply.content.strip()) or None

        embed.set_footer(text=f"Pick one or type your own · {timeout}s")
        view = ChoiceView(self._allowed_ids, options, timeout)
        message = await self._send(embed=embed, view=view)
        await _wait_for_view(view, timeout)

        embed.set_footer(text=f"Answer: {view.answer}" if view.answer else "timed out")
        embed.color = DONE if view.answer else FAILED
        try:
            await message.edit(embed=embed, view=None)
        except discord.HTTPException:
            pass
        return view.answer

    def ask_plan(self, context_discovered: str, diff_blueprint: str, verification_steps: str) -> str:
        if config.AUTO_ALLOW:
            return ("[System] AUTOMODE is ON. Plan automatically approved. Proceed strictly "
                    "with execution and verification.")
        timeout = config.DISCORD_APPROVAL_TIMEOUT
        try:
            return self._call(
                self._plan_flow(context_discovered, diff_blueprint, verification_steps, timeout),
                timeout + 30)
        except Exception as error:                  # noqa: BLE001
            print(f"[discord] plan approval failed: {error!r}")
            return "[System] Plan Rejected by User. Abort the task."

    async def _plan_flow(self, context_discovered, diff_blueprint, verification_steps, timeout) -> str:
        await self._flush(final=True)
        embed = discord.Embed(title="📋 Plan approval needed", color=WAITING)
        for name, value in (("Context discovered", context_discovered),
                            ("Change blueprint", diff_blueprint),
                            ("Verification steps", verification_steps)):
            if value:
                embed.add_field(name=name, value=_clip(str(value).replace("```", "'''"), 1000),
                                inline=False)
        view = PlanView(self._allowed_ids, timeout)
        message = await self._send(embed=embed, view=view)
        timed_out = await _wait_for_view(view, timeout)

        embed.color = DONE if (not timed_out and view.verdict == "approve") else FAILED
        embed.set_footer(text={"approve": "approved", "reject": "denied", "revise": "revision requested"}
                         .get(view.verdict, "denied") if not timed_out else "timed out")
        try:
            await message.edit(embed=embed, view=None)
        except discord.HTTPException:
            pass

        if timed_out or view.verdict == "reject":
            return "[System] Plan Rejected by User. Abort the task."
        if view.verdict == "revise":
            return (f"[System] Plan Rejected with feedback: {view.feedback}\n"
                    "Please revise your plan and submit again.")
        return ("[System] Plan Approved by User. You may now execute the plan strictly within "
                "the approved blueprint. Conclude by executing the verification steps.")

    def confirm(self, question: str) -> bool:
        return self.ask_approval("Confirm", [("question", question)])


def workspace_path(base: str, filename: str) -> str:
    os.makedirs(base, exist_ok=True)
    safe = os.path.basename(filename).replace("..", "_") or "file"
    return os.path.join(base, safe)
