r"""
Cross-`Application` keystroke handoff, bound to the *identity* of an `Input`.

The legacy :mod:`prompt_toolkit.input.typeahead` store is a process-global,
hash-keyed bucket: at the end of every `Application.run_async`, every
unprocessed key ends up in one dictionary slot (keyed by
``input.typeahead_hash()``), and the next run unconditionally drains it. That
answers the simple "two consecutive prompts" case, but it has no notion of:

- a *decision* at a run boundary about whether residual ordinary keys are
  allowed to enter the next run;
- the *identity* of the `Input` object (a hash string can be shared by several
  distinct `Input` objects wrapping the same file descriptor);
- *nesting* (an inner short-running `Application` borrowing an `Input` while
  an outer `Application` is still running must not drain or overwrite the
  key sequence the outer run still depends on);
- abnormal exits of an inner run that have to clean up their own boundary
  side effects.

This module adds an explicit, opt-in handoff chain per `Input` object. An
:class:`InputHandoff` is a user-facing handle bound to exactly one `Input`
identity. It is passed to `Application.run_async` / `Application.run`; each run
opens a :class:`HandoffScope` on that identity's chain:

- The *outermost* (root) active scope of a chain owns the chain's carry
  stash. It drains the stash when the run starts and (depending on the
  boundary decision) repopulates it when the run ends.
- A *nested* scope (an inner run that borrows the same `Input` while an outer
  run is still in progress) starts with an empty queue and always discards
  its own residue on close. It can therefore neither drain nor pollute the
  outer run's chain.
- ``CPRResponse`` keys are never part of a handoff: they belong to the run
  that requested them. They are structurally excluded both when the key pump
  drains the boundary (see `KeyProcessor.drain_handoff_residue`) and again
  when the residue is committed here.
- Each scope is also a *generation* for background completion: once it is
  closed (normal commit or abort), candidates that were requested in that
  generation may no longer be committed (see `Buffer`).

The registry is keyed by object identity (a ``WeakKeyDictionary``), never by
the ``typeahead_hash()`` string, so two different `Input` objects can never
share a chain, even if their hashes collide. The legacy typeahead store is
left completely untouched when no :class:`InputHandoff` is in use.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Callable

from ..key_binding import KeyPress
from ..keys import Keys
from .base import Input

__all__ = [
    "InputHandoff",
    "HandoffScope",
]


class _IdentityChain:
    """
    Handoff state for exactly one `Input` object.

    :param carry: Residual ordinary key presses committed by the last root
        run, waiting to be fed to the next root run on this same identity.
    :param stack: Currently active scopes, outermost first. More than one
        entry means an inner `Application` borrowed the `Input` while an outer
        one was still running.
    """

    def __init__(self) -> None:
        self.carry: list[KeyPress] = []
        self.stack: list[HandoffScope] = []


# Process-level registry. Keyed by the `Input` object itself, so different
# `Input` identities can never share a chain. Weak references mean a registry
# entry disappears together with its `Input`.
_chains: weakref.WeakKeyDictionary[Input, _IdentityChain] = weakref.WeakKeyDictionary()

# Runs can be nested across threads (`Application.run(in_thread=True)`), so
# registry mutation is serialized. The protected sections only touch registry
# bookkeeping; they never call into a key pump.
_registry_lock = threading.RLock()


def _chain_for(input_obj: Input) -> _IdentityChain:
    "Return (creating if needed) the handoff chain for an `Input` identity."
    with _registry_lock:
        chain = _chains.get(input_obj)
        if chain is None:
            chain = _IdentityChain()
            _chains[input_obj] = chain
        return chain


def _ordinary_keys(key_presses: list[KeyPress]) -> list[KeyPress]:
    """
    CPR responses are answered to the run that requested them and must never
    cross a run boundary. Everything else is an ordinary residual key.
    """
    return [k for k in key_presses if k.key != Keys.CPRResponse]


class HandoffScope:
    """
    One run boundary on an :class:`Input` identity chain.

    Instances are created by :meth:`InputHandoff.begin_run` and driven by
    `Application.run_async` (``take_incoming`` on start, ``commit`` /
    ``abort`` on close). User code doesn't instantiate this directly.

    A scope is simultaneously the *generation token* for background
    completions started during its run: :attr:`closed` flips to ``True`` at
    the run boundary, after which candidates produced for that generation
    must be rejected.
    """

    def __init__(
        self,
        chain: _IdentityChain,
        input_obj: Input,
        carry_decider: Callable[[], bool],
    ) -> None:
        self._chain = chain
        self.input = input_obj
        self._carry_decider = carry_decider

        #: True when another scope for the same identity is already active.
        self.nested: bool = bool(chain.stack)

        #: Set at the run boundary. This is the gate for stale completions.
        self.closed: bool = False

        #: True when the run ended abnormally; its residue is never carried.
        self.aborted: bool = False

        #: Keys fed at the start of this run (taken from the carry stash for
        #: a root scope; always empty for a nested scope).
        self.incoming_keys: list[KeyPress] = []

        #: Ordinary residual keys drained at the run boundary.
        self.residual_keys: list[KeyPress] = []

        self._taken = False
        self._popped = False

    def take_incoming(self) -> list[KeyPress]:
        """
        Return the keys this run should be started with.

        Only the outermost active scope for an identity drains that
        identity's carry stash. A nested (inner) run always starts empty, so
        it can never drain key presses the outer run chain still depends on.
        """
        assert not self._taken, "HandoffScope.take_incoming called twice"
        self._taken = True

        if self.nested:
            self.incoming_keys = []
        else:
            # Take ownership of the chain's carry stash.
            self.incoming_keys = self._chain.carry
            self._chain.carry = []

        return list(self.incoming_keys)

    def commit(self, residual_keys: list[KeyPress]) -> None:
        """
        Normal end of run: hand the residual ordinary keys to the boundary.

        Residual keys of a root scope are carried to the next root run when
        the boundary decision allows it. Residual keys of a nested scope are
        always dropped; they never flow back into the outer run.
        """
        if self.closed:
            return

        ordinary = _ordinary_keys(residual_keys)
        self.residual_keys = ordinary
        self.closed = True

        if not self.nested and self._carry_decider():
            # Root run with carry enabled: publish exactly once for the next
            # root run. CPR keys have already been removed.
            self._chain.carry.extend(ordinary)

    def abort(self) -> None:
        """
        Abnormal end of run.

        Close the generation and publish nothing. In particular an inner run
        that crashes leaves the outer run's chain untouched.
        """
        if self.closed:
            return

        self.closed = True
        self.aborted = True
        self.residual_keys = []

    def pop(self) -> None:
        """Remove this scope from its identity chain (idempotent)."""
        if self._popped:
            return
        self._popped = True
        with _registry_lock:
            try:
                self._chain.stack.remove(self)
            except ValueError:
                pass


class InputHandoff:
    """
    Identity-bound, nestable handoff of residual key presses across
    consecutive `Application` runs.

    Pass an instance to ``Application.run_async(..., input_handoff=...)`` (or
    ``Application.run(..., input_handoff=...)``). Reuse the same instance for
    the consecutive runs that should share a boundary. Each `Input` object
    has its own independent chain.

    :param input_obj: The `Input` whose runs are handed off. The binding is
        to this object's identity, not to its ``typeahead_hash()`` string.
    :param carry_residual_keys: The boundary decision. When true (default),
        ordinary keys left over after a root run finishes are fed to the next
        run on the same identity. When false, they are discarded. The flag
        is read at the end of each run, so it can be changed between runs.

    The carry decision only concerns *ordinary* keys: ending-state ordinary
    keys never execute key bindings, and ``CPRResponse`` keys are always
    consumed by the run that requested them and never enter the carry stash.
    """

    def __init__(
        self,
        input_obj: Input,
        *,
        carry_residual_keys: bool = True,
    ) -> None:
        self.input = input_obj
        self.carry_residual_keys = carry_residual_keys

    def begin_run(self) -> HandoffScope:
        """
        Open a new run scope on this identity's chain. Called by
        `Application.run_async`; normally not used directly.
        """
        chain = _chain_for(self.input)
        with _registry_lock:
            scope = HandoffScope(
                chain,
                self.input,
                carry_decider=lambda: self.carry_residual_keys,
            )
            chain.stack.append(scope)
        return scope

    @property
    def pending_keys(self) -> list[KeyPress]:
        """
        Ordinary keys currently waiting to be fed to the next root run on
        this identity. (Observability/testing aid.)
        """
        chain = _chain_for(self.input)
        with _registry_lock:
            return list(chain.carry)

    def discard_pending(self) -> None:
        """
        Explicitly drop every key currently waiting on this identity's carry
        stash. Useful for a host that wants a "clean next prompt" without
        abandoning the handoff itself.
        """
        chain = _chain_for(self.input)
        with _registry_lock:
            chain.carry.clear()
