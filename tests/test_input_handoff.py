"""
Cross-`Application` input handoff tests.

These are *temporal* acceptance tests, not a table of modes: every test drives
at least two `Application.run_async` rounds (or an outer run with an inner
short run borrowing the same input) through a `PipeInput`, and asserts causal
ordering across the run boundary:

- residual ordinary keys at the ending state never execute a binding in the
  finished run, but show up at the start of the next run when carry is
  enabled, and are dropped when it is disabled;
- CPR responses are consumed by the run that requested them (also in the
  ending state) and never cross the boundary;
- candidates produced by a slow background-thread completer for a closed
  generation, or for a document that changed, never reach a later run's UI;
- chains are keyed by `Input` object identity (two identities in parallel
  never mix), and nested inner runs cannot drain or pollute the outer run's
  key sequence, even when the inner run crashes.

No real terminal and no network is involved.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Sequence

import pytest

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import (
    Completion,
    Completer,
    ThreadedCompleter,
)
from prompt_toolkit.filters import Always
from prompt_toolkit.input import InputHandoff, create_pipe_input
from prompt_toolkit.input.base import PipeInput
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.key_binding.defaults import load_key_bindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.renderer import CPR_Support

# c-^ is ignored by the default bindings; a custom binding earlier in the
# merge order overrides that ignore and exits the app deterministically.
EXIT_KEY = "\x1e"
# Cursor position report for row 5, column 1.
CPR_RESPONSE = "\x1b[5;1R"


class CPRDummyOutput(DummyOutput):
    "`DummyOutput` that participates in cursor position requests."

    @property
    def responds_to_cpr(self) -> bool:  # override
        return True


# ---------------------------------------------------------------------------
# Application / bindings builders
# ---------------------------------------------------------------------------


def make_app(
    inp: PipeInput,
    output: DummyOutput | None = None,
    buffer: Buffer | None = None,
    custom_bindings: KeyBindings | None = None,
    started_event: threading.Event | None = None,
) -> tuple[Application, Buffer]:
    "Build a minimal full-screen-less application backed by `inp`."
    if buffer is None:
        buffer = Buffer(multiline=False)

    registries: Sequence[KeyBindings] = (
        (custom_bindings,) if custom_bindings is not None else ()
    ) + (load_key_bindings(),)
    key_bindings = merge_key_bindings(list(registries))

    app = Application(
        layout=Layout(Window(BufferControl(buffer=buffer))),
        key_bindings=key_bindings,
        input=inp,
        output=output if output is not None else DummyOutput(),
        full_screen=False,
    )

    if started_event is not None:
        app.pre_run_callables.append(started_event.set)

    return app, buffer


def exit_bindings(
    on_exit: Callable[[], None] | None = None,
    markers: list[str] | None = None,
    marker_key: str = "Z",
    marker_name: str = "Z",
) -> KeyBindings:
    """
    Bindings that exit on c-^ and (optionally) record when a specific ordinary
    key is *executed* as a binding. The marker is how the tests observe
    whether an ending-state key was treated as a binding (it must not be).
    """
    kb = KeyBindings()

    @kb.add("c-^")
    def _exit(event) -> None:  # type: ignore[no-untyped-def]
        if on_exit is not None:
            on_exit()
        event.app.exit()

    if markers is not None:

        @kb.add(marker_key)
        def _marker(event) -> None:  # type: ignore[no-untyped-def]
            markers.append(marker_name)

    return kb


# ---------------------------------------------------------------------------
# Driving scripts
# ---------------------------------------------------------------------------


def run_script(inp: PipeInput, steps: list[tuple[threading.Event | None, float, str | None]]) -> None:
    """
    Send text into `inp` following (wait-event, pause, text) steps. Scripts
    run in a worker thread so they can interleave with `run_async` on the
    loop.
    """
    for event, pause, text in steps:
        if event is not None:
            assert event.wait(10), "timed out waiting for run state"
        if pause:
            time.sleep(pause)
        if text is not None:
            inp.send_text(text)


def start_script(
    inp: PipeInput, steps: list[tuple[threading.Event | None, float, str | None]]
) -> threading.Thread:
    thread = threading.Thread(target=run_script, args=(inp, steps), daemon=True)
    thread.start()
    return thread


def run_scenario(factory: Callable[[], object]) -> object:
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(factory())  # type: ignore[arg-type]
    finally:
        asyncio.set_event_loop(None)
        loop.close()


# ---------------------------------------------------------------------------
# 1. Same input, two runs: residual key fork + ending-state binding guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("carry", [True, False])
def test_residual_key_fork_at_run_boundary(carry: bool) -> None:
    exited = threading.Event()
    started1 = threading.Event()
    started2 = threading.Event()
    # Fires when the ending-state "Z" is *executed* as a binding.
    markers: list[str] = []

    async def scenario() -> None:
        with create_pipe_input() as inp:
            handoff = InputHandoff(inp, carry_residual_keys=carry)

            kb1 = exit_bindings(on_exit=exited.set, markers=markers)
            app1, buf1 = make_app(
                inp, output=CPRDummyOutput(), custom_bindings=kb1,
                started_event=started1,
            )

            # "ab" is normal input for run 1. c-^ ends it. After the result is
            # set, while run 1 is still answering its CPR, send "Z" (must not
            # execute) and then the CPR response run 1 is waiting for.
            driver1 = start_script(
                inp,
                [
                    (started1, 0.2, "ab"),
                    (None, 0.2, EXIT_KEY),
                    (exited, 0.15, "Z"),
                    (None, 0.15, CPR_RESPONSE),
                ],
            )

            await app1.run_async(input_handoff=handoff)
            driver1.join(10)

            # The ending-state "Z" never executed a binding in run 1, and the
            # buffer text is only what run 1 actually processed.
            assert buf1.text == "ab"
            assert markers == []

            carry_data = [k.data for k in handoff.pending_keys]
            if carry:
                assert carry_data == ["Z"]
            else:
                assert carry_data == []

            # Run 2: a fresh application borrowing the same input. The carried
            # "Z" is fed *before* run 2 reads its own pipe.
            kb2 = exit_bindings(markers=markers)
            app2, buf2 = make_app(inp, custom_bindings=kb2, started_event=started2)
            driver2 = start_script(inp, [(started2, 0.2, EXIT_KEY)])

            await app2.run_async(input_handoff=handoff)
            driver2.join(10)

            if carry:
                # The key really crossed the boundary and was processed as an
                # ordinary key at the start of run 2.
                assert markers == ["Z"]
            else:
                assert markers == []
            # The carry stash is consumed once.
            assert handoff.pending_keys == []

    run_scenario(scenario)


def test_ending_state_key_is_binding_side_effect_free() -> None:
    """
    Even when carry is enabled, the ending-state key must not run bindings in
    the finished round. This pins the causal direction explicitly: the key is
    *carried*, not *executed late*.
    """
    exited = threading.Event()
    started = threading.Event()
    fired: list[str] = []

    async def scenario() -> None:
        with create_pipe_input() as inp:
            handoff = InputHandoff(inp)
            kb = exit_bindings(on_exit=exited.set, markers=fired)
            app, buf = make_app(
                inp, output=CPRDummyOutput(), custom_bindings=kb,
                started_event=started,
            )

            start_script(
                inp,
                [
                    (started, 0.2, EXIT_KEY),
                    (exited, 0.15, "Z"),
                    (None, 0.15, CPR_RESPONSE),
                ],
            )

            await app.run_async(input_handoff=handoff)
            assert fired == []
            assert buf.text == ""
            assert [k.data for k in handoff.pending_keys] == ["Z"]

    run_scenario(scenario)


# ---------------------------------------------------------------------------
# 2. CPR is consumed in the ending state and never carried
# ---------------------------------------------------------------------------


def test_cpr_consumed_in_round_never_carried() -> None:
    exited = threading.Event()
    started1 = threading.Event()
    started2 = threading.Event()
    markers: list[str] = []

    async def scenario() -> None:
        with create_pipe_input() as inp:
            handoff = InputHandoff(inp)

            kb1 = exit_bindings(on_exit=exited.set)
            app1, _ = make_app(
                inp, output=CPRDummyOutput(), custom_bindings=kb1,
                started_event=started1,
            )

            # End run 1 first; *then* answer the CPR request while the run is
            # in its ending state, bundled after an ordinary residual key.
            start_script(
                inp,
                [
                    (started1, 0.2, EXIT_KEY),
                    (exited, 0.15, "Z" + CPR_RESPONSE),
                ],
            )

            await app1.run_async(input_handoff=handoff)

            # The CPR response was consumed by run 1 itself.
            assert app1.renderer.cpr_support == CPR_Support.SUPPORTED
            # Only the ordinary key crosses; no CPR key is in the carry.
            carry = handoff.pending_keys
            assert [k.data for k in carry] == ["Z"]
            assert all(k.key != Keys.CPRResponse for k in carry)

            kb2 = exit_bindings(markers=markers)
            app2, _ = make_app(inp, custom_bindings=kb2, started_event=started2)
            start_script(inp, [(started2, 0.2, EXIT_KEY)])

            await app2.run_async(input_handoff=handoff)

            # "Z" arrived at run 2's start; a leaked CPR answer would have
            # flipped run 2's renderer to SUPPORTED (its output says no CPR).
            assert markers == ["Z"]
            assert app2.renderer.cpr_support == CPR_Support.NOT_SUPPORTED
            assert handoff.pending_keys == []

    run_scenario(scenario)


# ---------------------------------------------------------------------------
# 3. Slow background-thread completer: dead generation cannot be committed
# ---------------------------------------------------------------------------


class SlowDocumentCompleter(Completer):
    """
    Threaded completer that blocks inside `get_completions` (on a worker
    thread) until a gate opens, then yields for one specific document text.
    """

    def __init__(
        self,
        gate: threading.Event,
        entered: threading.Event,
        only_for_text: str | None = None,
    ) -> None:
        self.gate = gate
        self.entered = entered
        self.only_for_text = only_for_text

    def get_completions(self, document, complete_event):  # type: ignore[no-untyped-def]
        self.entered.set()
        self.gate.wait(10)
        if self.only_for_text is None or document.text == self.only_for_text:
            yield Completion("STALE_MENU_ITEM", start_position=0)


def test_inflight_threaded_completion_dies_with_boundary() -> None:
    gate = threading.Event()
    entered = threading.Event()
    started = threading.Event()

    async def scenario() -> None:
        with create_pipe_input() as inp:
            handoff = InputHandoff(inp)

            completer = ThreadedCompleter(SlowDocumentCompleter(gate, entered))
            buf = Buffer(
                completer=completer,
                complete_while_typing=Always(),
                multiline=False,
            )
            kb = KeyBindings()

            @kb.add("c-^")
            def _exit(event) -> None:  # type: ignore[no-untyped-def]
                event.app.exit()

            app, _ = make_app(
                inp, custom_bindings=kb, buffer=buf, started_event=started,
            )

            # Type "a": a completion starts on a background thread and blocks
            # in the worker before producing anything. End the run, and only
            # then open the gate (the worker is still in flight).
            start_script(
                inp,
                [
                    (started, 0.2, "a"),
                    (entered, 0.2, EXIT_KEY),
                    (None, 0.3, None),  # let the run reach its closed boundary
                ],
            )

            run_task = asyncio.ensure_future(app.run_async(input_handoff=handoff))

            # Wait until the run is truly ending before releasing the worker.
            await _wait_until(lambda: app.future is not None and app.future.done())
            gate.set()

            await run_task

            # The worker may have finished (it was allowed to), and the local
            # completion object even received the item, but it was never
            # committed to visible state: the boundary detached the
            # generation.
            assert buf.complete_state is None

            # Run 2 reuses the same buffer (text "a" survives between runs):
            # the stale menu must not reappear.
            started2 = threading.Event()
            app2, _ = make_app(
                inp, custom_bindings=kb, buffer=buf, started_event=started2,
            )
            start_script(inp, [(started2, 0.2, EXIT_KEY)])
            await app2.run_async(input_handoff=handoff)

            assert buf.text == "a"
            assert buf.complete_state is None

    run_scenario(scenario)


def test_stale_completion_after_document_change_rejected() -> None:
    """
    Completion requested for document "a"; document becomes "ab" before the
    threaded worker answers. The item for "a" must not be committed; the
    restart for "ab" yields nothing, so no menu remains.
    """
    gate = threading.Event()
    entered = threading.Event()
    started = threading.Event()

    async def scenario() -> None:
        with create_pipe_input() as inp:
            handoff = InputHandoff(inp)

            completer = ThreadedCompleter(
                SlowDocumentCompleter(gate, entered, only_for_text="a")
            )
            buf = Buffer(
                completer=completer,
                complete_while_typing=Always(),
                multiline=False,
            )
            kb = KeyBindings()

            @kb.add("c-^")
            def _exit(event) -> None:  # type: ignore[no-untyped-def]
                event.app.exit()

            app, _ = make_app(
                inp, custom_bindings=kb, buffer=buf, started_event=started,
            )

            start_script(
                inp,
                [
                    (started, 0.2, "a"),
                    (entered, 0.2, "b"),
                    (None, 0.1, None),
                ],
            )

            run_task = asyncio.ensure_future(app.run_async(input_handoff=handoff))
            await _wait_until(lambda: buf.text == "ab")
            gate.set()

            # Give the async generator machinery time to settle.
            await _wait_until(lambda: buf.complete_state is None)

            inp.send_text(EXIT_KEY)
            await run_task

            assert buf.text == "ab"
            assert buf.complete_state is None

    run_scenario(scenario)


async def _wait_until(
    predicate: Callable[[], bool], timeout: float = 5.0
) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.02)


# ---------------------------------------------------------------------------
# 4. Identity isolation: two Input objects, two chains, parallel runs
# ---------------------------------------------------------------------------


def test_two_input_identities_run_in_parallel_never_mix() -> None:
    results: dict[str, object] = {}

    def worker(name: str, text: str, residue: str) -> None:
        started1 = threading.Event()
        started2 = threading.Event()
        exited = threading.Event()
        markers: list[str] = []

        async def scenario() -> None:
            with create_pipe_input() as inp:
                handoff = InputHandoff(inp)

                kb1 = exit_bindings(on_exit=exited.set)
                app1, buf1 = make_app(
                    inp, output=CPRDummyOutput(), custom_bindings=kb1,
                    started_event=started1,
                )

                start_script(
                    inp,
                    [
                        (started1, 0.2, text),
                        (None, 0.2, EXIT_KEY),
                        (exited, 0.15, residue + CPR_RESPONSE),
                    ],
                )

                await app1.run_async(input_handoff=handoff)

                pending = [k.data for k in handoff.pending_keys]

                kb2 = exit_bindings(markers=markers)
                app2, _ = make_app(
                    inp, custom_bindings=kb2, started_event=started2,
                )
                start_script(inp, [(started2, 0.2, EXIT_KEY)])
                await app2.run_async(input_handoff=handoff)

                results[name] = (buf1.text, pending, markers)

        run_scenario(scenario)

    thread_a = threading.Thread(target=worker, args=("A", "a", "P"))
    thread_b = threading.Thread(target=worker, args=("B", "b", "Q"))
    thread_a.start()
    thread_b.start()
    thread_a.join(30)
    thread_b.join(30)

    text_a, pending_a, markers_a = results["A"]  # type: ignore[misc]
    text_b, pending_b, markers_b = results["B"]  # type: ignore[misc]

    assert text_a == "a"
    assert text_b == "b"
    # Each identity carries exactly its own residual key.
    assert pending_a == ["P"]
    assert pending_b == ["Q"]
    # Round 2 on each identity consumed exactly its own carried key.
    assert markers_a == ["P"]
    assert markers_b == ["Q"]


# ---------------------------------------------------------------------------
# 5. Nesting: an inner short run borrows the input while the outer is alive
# ---------------------------------------------------------------------------


def test_nested_inner_run_does_not_touch_outer_sequence() -> None:
    outer_started = threading.Event()
    inner_started = threading.Event()
    inner_exiting = threading.Event()
    inner_finished = threading.Event()
    outer_exited = threading.Event()

    nested_observation: dict[str, object] = {}

    async def scenario() -> None:
        with create_pipe_input() as inp:
            handoff = InputHandoff(inp)

            inner_kb = KeyBindings()

            @inner_kb.add("c-^")
            def _inner_exit(event) -> None:  # type: ignore[no-untyped-def]
                inner_exiting.set()
                event.app.exit()
                # Deterministic ending-state ordinary key: queued after the
                # result is set, so it must not be executed.
                from prompt_toolkit.key_binding import KeyPress

                event.app.key_processor.feed(KeyPress("Q", "Q"))

            def run_inner() -> None:
                inner_app, inner_buf = make_app(
                    inp,
                    output=CPRDummyOutput(),
                    custom_bindings=inner_kb,
                    started_event=inner_started,
                )
                inner_app.run(
                    in_thread=True,
                    input_handoff=handoff,
                    set_exception_handler=False,
                )
                # The nested scope published nothing, even though an
                # ending-state ordinary key was drained.
                nested_observation["pending_after_inner"] = [
                    k.data for k in handoff.pending_keys
                ]
                nested_observation["inner_text"] = inner_buf.text
                inner_finished.set()

            outer_kb = KeyBindings()
            inner_launched = {"done": False}

            @outer_kb.add("c-^")
            def _outer(event) -> None:  # type: ignore[no-untyped-def]
                if not inner_launched["done"]:
                    inner_launched["done"] = True
                    run_inner()
                else:
                    outer_exited.set()
                    event.app.exit()
                    # Deterministic root residue for the next root run.
                    from prompt_toolkit.key_binding import KeyPress

                    event.app.key_processor.feed(KeyPress("Z", "Z"))

            outer_app, outer_buf = make_app(
                inp, custom_bindings=outer_kb, started_event=outer_started,
            )

            # "a" -> outer; c-^ launches inner; inside inner, "i" then c-^;
            # an external key is also sent while the inner run is in its CPR
            # ending window; then "b", c-^ ends the outer run.
            start_script(
                inp,
                [
                    (outer_started, 0.2, "a"),
                    (None, 0.2, EXIT_KEY),
                    (inner_started, 0.2, "i"),
                    (None, 0.2, EXIT_KEY),
                    (inner_exiting, 0.15, "X" + CPR_RESPONSE),
                    (inner_finished, 0.2, "b"),
                    (None, 0.2, EXIT_KEY),
                ],
            )

            await outer_app.run_async(input_handoff=handoff)

            # Outer key sequence was fully preserved: neither "i", "Q" nor "X"
            # were inserted into the outer buffer.
            assert outer_buf.text == "ab"
            assert nested_observation["inner_text"] == "i"
            assert nested_observation["pending_after_inner"] == []

            # Root residue ("Z") now waits for the next root run. The nested
            # run did not consume or overwrite it.
            assert [k.data for k in handoff.pending_keys] == ["Z"]

            # A later root run gets "Z" and nothing from the inner scope.
            markers: list[str] = []
            run2_started = threading.Event()
            kb2 = exit_bindings(markers=markers)
            app2, _ = make_app(
                inp, custom_bindings=kb2, started_event=run2_started,
            )
            start_script(inp, [(run2_started, 0.2, EXIT_KEY)])
            await app2.run_async(input_handoff=handoff)

            assert markers == ["Z"]
            assert handoff.pending_keys == []

    run_scenario(scenario)


def test_nested_inner_crash_pops_its_own_side_effects() -> None:
    outer_started = threading.Event()
    inner_started = threading.Event()
    inner_exiting = threading.Event()
    inner_finished = threading.Event()

    nested_observation: dict[str, object] = {}

    async def scenario() -> None:
        with create_pipe_input() as inp:
            handoff = InputHandoff(inp)

            inner_kb = KeyBindings()

            @inner_kb.add("c-^")
            def _inner_boom(event) -> None:  # type: ignore[no-untyped-def]
                inner_exiting.set()
                from prompt_toolkit.key_binding import KeyPress

                event.app.key_processor.feed(KeyPress("Q", "Q"))
                raise RuntimeError("inner run exploded")

            def run_inner() -> None:
                inner_app, _ = make_app(
                    inp,
                    output=CPRDummyOutput(),
                    custom_bindings=inner_kb,
                    started_event=inner_started,
                )
                try:
                    inner_app.run(
                        in_thread=True,
                        input_handoff=handoff,
                        set_exception_handler=False,
                    )
                except RuntimeError:
                    # The aborted nested scope published nothing and was
                    # popped off the identity chain.
                    nested_observation["pending_after_crash"] = [
                        k.data for k in handoff.pending_keys
                    ]
                    inner_finished.set()
                else:
                    raise AssertionError("inner run was expected to raise")

            outer_kb = KeyBindings()
            inner_launched = {"done": False}

            @outer_kb.add("c-^")
            def _outer(event) -> None:  # type: ignore[no-untyped-def]
                if not inner_launched["done"]:
                    inner_launched["done"] = True
                    run_inner()
                else:
                    event.app.exit()
                    from prompt_toolkit.key_binding import KeyPress

                    event.app.key_processor.feed(KeyPress("Z", "Z"))

            outer_app, outer_buf = make_app(
                inp, custom_bindings=outer_kb, started_event=outer_started,
            )

            # External ending-state key into the dying inner run, answered
            # CPR, then the outer resumes and ends with root residue "Z".
            start_script(
                inp,
                [
                    (outer_started, 0.2, EXIT_KEY),
                    (inner_started, 0.2, EXIT_KEY),
                    (inner_exiting, 0.15, "X" + CPR_RESPONSE),
                    (inner_finished, 0.2, EXIT_KEY),
                ],
            )

            await outer_app.run_async(input_handoff=handoff)

            assert nested_observation["pending_after_crash"] == []
            assert outer_buf.text == ""
            # If the crashed scope had stayed on the chain, the outer commit
            # would be treated as nested and drop this key.
            assert [k.data for k in handoff.pending_keys] == ["Z"]

    run_scenario(scenario)


# ---------------------------------------------------------------------------
# 6. No handoff: legacy process-global typeahead path is unchanged
# ---------------------------------------------------------------------------


def test_without_handoff_legacy_typeahead_path_unchanged() -> None:
    from prompt_toolkit.input.typeahead import get_typeahead

    exited = threading.Event()
    started1 = threading.Event()
    started2 = threading.Event()

    async def scenario() -> None:
        with create_pipe_input() as inp:
            kb1 = exit_bindings(on_exit=exited.set)
            app1, buf1 = make_app(
                inp, output=CPRDummyOutput(), custom_bindings=kb1,
                started_event=started1,
            )

            start_script(
                inp,
                [
                    (started1, 0.2, "ab"),
                    (None, 0.2, EXIT_KEY),
                    (exited, 0.15, "Z" + CPR_RESPONSE),
                ],
            )

            await app1.run_async()

            assert buf1.text == "ab"
            # No handoff machinery was ever attached to this pump.
            assert app1.key_processor._handoff_scope is None
            # Legacy global typeahead got the residual key.
            assert [k.data for k in get_typeahead(inp)] == ["Z"]

            app2, buf2 = make_app(inp, started_event=started2)
            start_script(inp, [(started2, 0.2, EXIT_KEY)])
            await app2.run_async()

            # The legacy store fed "Z" to run 2 (self-insert), and is drained.
            assert buf2.text == "Z"
            assert get_typeahead(inp) == []
            assert app2.key_processor._handoff_scope is None

    run_scenario(scenario)
