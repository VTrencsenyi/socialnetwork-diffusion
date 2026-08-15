"""What a household agent remembers, and how.

`AgentState` is the interface the simulation loop drives; `DefaultState` is the
one implementation so far. They live apart from `agent.py` because the memory
model is the part most likely to be replaced: the default keeps every message
verbatim, so prompt cost grows linearly in `t`. That is the right thing for a
7-trimester run over ~174 households and the wrong thing for anything longer,
and a later `SummaryState` (rolling summary instead of transcript) or
`BeliefState` (a scalar posterior, no text) should be swappable without touching
the agent or the loop. `STATE_VERSIONS` maps a name to a class so a run config
can name its state, and `to_dict()` tags the version so a saved run reloads into
whatever wrote it.

Two invariants carry the design; both are checked by `DefaultState.validate()`.

**`t` indexes the ledger, so the ledger is rectangular.** Every list in
`received` / `sent` has length `t + 1` at all times, with `NO_MESSAGE` ("") in
every round nothing was said. `received[j][t]` is therefore always safe to read,
no branch has to distinguish "no key" from "no message", and the whole ledger
drops straight into a (peers x rounds) matrix for analysis.

**Adoption is monotone.** `adopted` and `adopted_t` are set together by
`adopt()` and are never unset. That matches the data — BCDJ record a household
that joined as joined, with no exit — and `adopted_t` is the only per-household
timing the simulation produces, so corrupting it would cost the village-level
adoption curve that `docs/household_design.md` §2 says timing is validated
against.

This module has no dependency on `agent.py`, `data_loader.py` or pandas: it is
plain containers with rules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar, NamedTuple

# The "nothing was said this round" sentinel. Empty rather than None because it
# is the neutral element for the text these slots hold, it serialises to CSV and
# JSON without a special case, and `if row[t]:` is the natural test.
NO_MESSAGE = ""

# Sender id for the MFI's own initial disclosure to a leader household. Seeding
# is just a message from a non-household source, which keeps `informed_at`
# derivable from the ledger alone instead of needing a separate seed flag that
# could disagree with it. hhids in this bundle are village-prefixed 5-digit
# integers (73001, ...), so 0 is free.
SEED_SOURCE = 0


class AgentError(RuntimeError):
    """Misuse of an agent or its state: double-speaking in a round, re-adopting, drift.

    Base class for everything this package raises at simulation time, so a loop
    can catch one type. `agent.ContextMissing` is a subclass.
    """


class Turn(NamedTuple):
    """One line of an agent's history, in the order it happened.

    `direction` is "in" (heard) or "out" (said); `other` is the counterparty's
    hh_id, or `SEED_SOURCE` for the MFI's initial disclosure.
    """

    t: int
    direction: str
    other: int
    text: str


class AgentState(ABC):
    """What an agent remembers. Subclass to change *how* it remembers.

    The base fixes only the interface the simulation loop calls, so variants are
    free to store something other than a verbatim transcript. Two attributes are
    part of that contract because the loop reads them directly:

    `t`         current round, 0-based.
    `adopted`   monotone: once true, never false again.

    A variant that does not keep per-peer channels leaves `open_channels()` as
    the no-op it is here.
    """

    version: ClassVar[str] = "abstract"

    t: int
    adopted: bool
    adopted_t: int | None

    def open_channels(self, others: Iterable[int]) -> None:
        """Pre-create ledger slots for peers, so absence means silence, not ignorance.

        Optional: `record_received` / `record_sent` create slots on demand
        anyway. Calling it up front makes the ledger rectangular over *known*
        peers from t=0, which is what lets it be read as a matrix.
        """

    @abstractmethod
    def record_received(self, sender: int, text: str) -> None:
        """Log `text` as heard from `sender` during the current round."""

    @abstractmethod
    def record_sent(self, recipient: int, text: str) -> None:
        """Log `text` as said to `recipient` during the current round."""

    @abstractmethod
    def adopt(self) -> None:
        """Mark adoption in the current round. Not idempotent — see `DefaultState`."""

    @abstractmethod
    def advance(self) -> int:
        """End the current round and return the new `t`."""

    @abstractmethod
    def transcript(self, *, since: int = 0, direction: str = "both") -> list[Turn]:
        """History from round `since` onward, chronological. The prompt's input."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """JSON-safe snapshot, tagged with `version`."""

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentState:
        """Inverse of `to_dict`, for the same `version`."""


@dataclass
class DefaultState(AgentState):
    """Transcript-carrying default: every message kept verbatim, indexed by round.

    `received[j]` and `sent[j]` are lists of length `t + 1` whose index *is* the
    round, `NO_MESSAGE` where nothing passed. So `received[j][3]` is what
    household j told this one in round 3, and `[NO_MESSAGE] * 4` is a peer that
    has been silent for four rounds — the same shape either way.
    """

    version: ClassVar[str] = "default"

    t: int = 0
    received: dict[int, list[str]] = field(default_factory=dict)
    sent: dict[int, list[str]] = field(default_factory=dict)
    adopted: bool = False
    adopted_t: int | None = None

    def __post_init__(self) -> None:
        if self.t < 0:
            raise ValueError(f"t must be >= 0, got {self.t}")
        for ledger in (self.received, self.sent):
            for other in ledger:
                self._channel(ledger, other)
        self.validate()

    # -- writes ----------------------------------------------------------

    def open_channels(self, others: Iterable[int]) -> None:
        for other in others:
            self._channel(self.received, other)
            self._channel(self.sent, other)

    def record_received(self, sender: int, text: str) -> None:
        self._write(self.received, sender, text, "heard from")

    def record_sent(self, recipient: int, text: str) -> None:
        self._write(self.sent, recipient, text, "said to")

    def adopt(self) -> None:
        """Adopt in the current round.

        Raises rather than no-ops on a second call: two adoptions is a loop bug
        (an agent decided twice in one round, or was stepped twice), and
        swallowing it would corrupt `adopted_t` while looking like it worked.
        """
        if self.adopted:
            raise AgentError(f"already adopted at t={self.adopted_t}; cannot adopt again at t={self.t}")
        self.adopted = True
        self.adopted_t = self.t

    def advance(self) -> int:
        self.t += 1
        for ledger in (self.received, self.sent):
            for other in ledger:
                self._channel(ledger, other)
        return self.t

    # -- reads -----------------------------------------------------------

    @property
    def informed_at(self) -> int | None:
        """First round in which anything was heard, seed disclosure included.

        Derived rather than stored: a separate `informed` flag is one more thing
        that can disagree with the ledger it is supposed to summarise.
        """
        rounds = [i for row in self.received.values() for i, text in enumerate(row) if text]
        return min(rounds) if rounds else None

    @property
    def informed(self) -> bool:
        return self.informed_at is not None

    def heard_at(self, t: int) -> dict[int, str]:
        """Who said what to this agent in round `t`."""
        return {j: row[t] for j, row in self.received.items() if t < len(row) and row[t]}

    def said_at(self, t: int) -> dict[int, str]:
        """What this agent said to whom in round `t`."""
        return {j: row[t] for j, row in self.sent.items() if t < len(row) and row[t]}

    def transcript(self, *, since: int = 0, direction: str = "both") -> list[Turn]:
        """Chronological history — storage is by round, prompts want by time.

        Within a round, heard comes before said: that is the order the round
        protocol imposes (deliver, then respond), and rendering it the other way
        would show an agent replying to something it had not yet been told.
        """
        if direction not in ("both", "in", "out"):
            raise ValueError(f"direction must be 'both', 'in' or 'out', got {direction!r}")
        turns: list[Turn] = []
        for r in range(max(0, since), self.t + 1):
            if direction in ("both", "in"):
                turns += [Turn(r, "in", j, txt) for j, txt in sorted(self.heard_at(r).items())]
            if direction in ("both", "out"):
                turns += [Turn(r, "out", j, txt) for j, txt in sorted(self.said_at(r).items())]
        return turns

    @property
    def peers(self) -> list[int]:
        """Every counterparty with a channel open, in either direction."""
        return sorted(set(self.received) | set(self.sent))

    # -- integrity -------------------------------------------------------

    def validate(self) -> None:
        """Assert the invariants this class exists to maintain.

        Cheap, so the loop can call it every round while a run is being trusted
        for the first time.
        """
        for name, ledger in (("received", self.received), ("sent", self.sent)):
            for other, row in ledger.items():
                if len(row) != self.t + 1:
                    raise AgentError(
                        f"{name}[{other}] has {len(row)} slots at t={self.t}; expected {self.t + 1}"
                    )
        if self.adopted != (self.adopted_t is not None):
            raise AgentError(f"adopted={self.adopted} disagrees with adopted_t={self.adopted_t}")
        if self.adopted_t is not None and not 0 <= self.adopted_t <= self.t:
            raise AgentError(f"adopted_t={self.adopted_t} outside 0..{self.t}")

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        # Keys go out as strings: JSON has no integer keys, and pretending
        # otherwise means a round-trip silently changes the key type.
        return {
            "version": self.version,
            "t": self.t,
            "received": {str(k): v for k, v in self.received.items()},
            "sent": {str(k): v for k, v in self.sent.items()},
            "adopted": self.adopted,
            "adopted_t": self.adopted_t,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DefaultState:
        got = d.get("version", cls.version)
        if got != cls.version:
            raise ValueError(f"{cls.__name__} cannot load state version {got!r}")
        return cls(
            t=int(d["t"]),
            received={int(k): list(v) for k, v in d.get("received", {}).items()},
            sent={int(k): list(v) for k, v in d.get("sent", {}).items()},
            adopted=bool(d.get("adopted", False)),
            adopted_t=None if d.get("adopted_t") is None else int(d["adopted_t"]),
        )

    # -- internals -------------------------------------------------------

    def _channel(self, ledger: dict[int, list[str]], other: int) -> list[str]:
        """The row for `other`, created and/or back-filled to length t + 1."""
        row = ledger.get(other)
        if row is None:
            row = [NO_MESSAGE] * (self.t + 1)
            ledger[other] = row
        elif len(row) < self.t + 1:
            row.extend([NO_MESSAGE] * (self.t + 1 - len(row)))
        return row

    def _write(self, ledger: dict[int, list[str]], other: int, text: str, verb: str) -> None:
        if not text:
            raise ValueError(
                f"empty text is the {NO_MESSAGE!r} sentinel for 'nothing was said'; "
                "pass a real message or do not call this"
            )
        row = self._channel(ledger, other)
        if row[self.t]:
            # One message per (counterparty, direction, round) is the model. A
            # second write is either a duplicated loop pass or a state shared
            # between two agents; both are bugs worth stopping on.
            raise AgentError(f"already {verb} {other} at t={self.t}: {row[self.t]!r}")
        row[self.t] = text


# Version name -> class, so a run config can name its state and a saved run can
# be reloaded into whatever wrote it. Register new variants here.
STATE_VERSIONS: dict[str, type[AgentState]] = {DefaultState.version: DefaultState}


def make_state(version: str = "default", **kwargs: Any) -> AgentState:
    """Construct a state by version name."""
    try:
        cls = STATE_VERSIONS[version]
    except KeyError:
        raise ValueError(f"unknown state version {version!r}; have {sorted(STATE_VERSIONS)}") from None
    return cls(**kwargs)  # type: ignore[call-arg]


def state_from_dict(d: dict[str, Any]) -> AgentState:
    """Rebuild a state from `to_dict()` output, dispatching on its version tag."""
    version = d.get("version", "default")
    try:
        cls = STATE_VERSIONS[version]
    except KeyError:
        raise ValueError(f"unknown state version {version!r}; have {sorted(STATE_VERSIONS)}") from None
    return cls.from_dict(d)
