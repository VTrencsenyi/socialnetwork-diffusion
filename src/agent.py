"""`HH_Agent`: one household in the microfinance diffusion simulation.

An agent is deliberately thin: an identity (`hh_id`), a frozen persona read from
`context_<hh_id>.txt`, the neighbours it may talk to, and a mutable
`state.AgentState`. Nothing else. Construction from the built data lives in
`tools.build_agents()`; the memory model lives in `state.py`.

**The agent never sees the household feature table.** `docs/household_design.md`
§1 makes it non-negotiable that the ground-truth outcome is *structurally*
unable to reach a prompt, not merely omitted by convention. The strongest
available form of that is for the agent to hold no reference to the table at
all: it holds a rendered context file, produced upstream from an explicit
allowlist, and an id. `_adopted` cannot leak through an object that has no route
to it, and evaluation joins on `hh_id` from the CSV, outside the agent. The same
applies to a neighbour's *real* adoption (§4.7, "privileged by proxy") — an
agent sees neighbours only through messages that arrive in its own ledger, and
those are written by the simulation from *simulated* states.

Round protocol, which the simulation loop owns but which `state.py`'s invariants
assume:

    for t in 0 .. T-1:
        phase 1  every agent that has something to say calls send()/receive(),
                 all of which land in slot t
        phase 2  every agent reads slot t and may call adopt()
        phase 3  every agent calls advance() -- in lockstep, no exceptions

Splitting speak from decide is what makes a round order-independent: with a
single pass, whether B hears A at t or t+1 would depend on iteration order.
Advancing in lockstep is what keeps slot `t` meaning the same thing for
everyone. An agent that falls a round behind will silently misalign its own
history, which is why `advance()` returns the new `t` and the loop should assert
they all agree.

What is *not* here, on purpose: the prompt, the LLM client, and the decision
rule. `docs/household_design.md` leaves those open, and they are a policy that
reads an agent, not a property of one. The seam is `context` + `transcript()`
in, `receive` / `send` / `adopt` out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from .state import (
        SEED_SOURCE,
        AgentError,
        AgentState,
        DefaultState,
        Turn,
        state_from_dict,
    )
except ImportError:  # running as a script, not a package
    from state import (  # type: ignore[no-redef]
        SEED_SOURCE,
        AgentError,
        AgentState,
        DefaultState,
        Turn,
        state_from_dict,
    )

DEFAULT_CONTEXT_DIR = Path(__file__).resolve().parent.parent / "output" / "context"


class ContextMissing(AgentError):
    """`context_<hh_id>.txt` is absent. Raised on access, not on construction."""


@dataclass
class HH_Agent:
    """One household.

    Parameters
    ----------
    hh_id
        The bundle's `hhid` (village-prefixed, e.g. 73001). Identity, join key,
        and the name of the context file. Never rendered into a prompt —
        `docs/household_design.md` §4.6: a numeric id invites the model to
        invent orderings that do not exist.
    village, row
        Provenance for the join back to `output/hh_features_<village>.csv` and
        to adjacency row `row` (1-based, i.e. `adjmatrix_key`). Optional, and
        evaluation-side only.
    neighbours
        hhids this agent may exchange messages with, from the real network.
        Mechanics only. §5 is explicit that degree must not be stated
        numerically to the LLM — "you have 17 friends" reads as an instruction
        to be influential and manufactures the network effect being measured.
        Network position reaches the agent only through who actually talks to
        it, so this list must not be summarised into the persona.
    context_dir
        Where `context_<hh_id>.txt` lives.
    state
        Defaults to a fresh `DefaultState`; pass another `AgentState` to change
        the memory model.
    """

    hh_id: int
    village: int | None = None
    row: int | None = None
    neighbours: tuple[int, ...] = ()
    context_dir: Path = DEFAULT_CONTEXT_DIR
    state: AgentState = field(default_factory=DefaultState)
    _context: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.context_dir = Path(self.context_dir)
        self.neighbours = tuple(int(j) for j in self.neighbours)
        if self.hh_id in self.neighbours:
            raise ValueError(f"household {self.hh_id} is its own neighbour")
        self.state.open_channels(self.neighbours)

    # -- persona ---------------------------------------------------------

    @property
    def context_path(self) -> Path:
        return self.context_dir / f"context_{self.hh_id}.txt"

    @property
    def has_context(self) -> bool:
        return self.context_path.is_file()

    @property
    def context(self) -> str:
        """The persona, read once and cached for the run.

        Immutable by construction (§5), and caching it is the main lever on LLM
        cost: only the dynamic block — who told me what, which round — varies
        per call, so this string is what a provider-side prompt cache should be
        keyed on.

        Loaded lazily because context files are generated after the agents are
        constructed. A runner should still touch `.context` on every agent
        before the first API call: discovering a missing file at t=4, several
        hundred calls in, is an expensive way to learn it.
        """
        if self._context is None:
            path = self.context_path
            if not path.is_file():
                raise ContextMissing(f"no context file for household {self.hh_id}: {path}")
            self._context = path.read_text(encoding="utf-8")
        return self._context

    def set_context(self, text: str) -> None:
        """Override the persona in memory, for tests and dry runs."""
        self._context = text

    # -- round mechanics -------------------------------------------------

    @property
    def t(self) -> int:
        return self.state.t

    @property
    def adopted(self) -> bool:
        return self.state.adopted

    @property
    def adopted_t(self) -> int | None:
        return self.state.adopted_t

    def receive(self, sender: int, text: str) -> None:
        """Log a message heard this round. Called by the loop, not by the agent."""
        self.state.record_received(sender, text)

    def send(self, recipient: int, text: str) -> None:
        """Log a message said this round.

        Logging only: actually delivering it to `recipient` is the loop's job,
        because only the loop holds the other agents. Keeping the two halves
        separate is what allows a message to be dropped (an unreachable or
        pruned neighbour) without the sender's own history lying about it.
        """
        if self.neighbours and recipient not in self.neighbours:
            raise AgentError(f"household {self.hh_id} has no edge to {recipient}")
        self.state.record_sent(recipient, text)

    def seed(self, text: str) -> None:
        """Record the MFI's initial disclosure — a message from `SEED_SOURCE`.

        `has_leader` is the seed set and is *not* privileged (§4.7): a household
        legitimately knows the MFI spoke to it. BCDJ's caveat travels with it —
        `leader` marks who the MFI *could* have informed, not who verifiably was.
        """
        self.state.record_received(SEED_SOURCE, text)

    def adopt(self) -> None:
        self.state.adopt()

    def advance(self) -> int:
        return self.state.advance()

    def transcript(self, *, since: int = 0, direction: str = "both") -> list[Turn]:
        return self.state.transcript(since=since, direction=direction)

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Run log for this agent. Deliberately excludes the persona text.

        The context is large, identical every round, and already on disk at
        `context_path`; writing it into every snapshot would multiply the log
        size by the number of rounds for no information.
        """
        return {
            "hh_id": self.hh_id,
            "village": self.village,
            "row": self.row,
            "neighbours": list(self.neighbours),
            "state": self.state.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], context_dir: Path | str = DEFAULT_CONTEXT_DIR) -> HH_Agent:
        return cls(
            hh_id=int(d["hh_id"]),
            village=d.get("village"),
            row=d.get("row"),
            neighbours=tuple(d.get("neighbours", ())),
            context_dir=Path(context_dir),
            state=state_from_dict(d["state"]),
        )

    def __repr__(self) -> str:  # the default would dump the whole ledger
        where = f"v{self.village}" if self.village is not None else "?"
        mark = f"adopted@{self.adopted_t}" if self.adopted else "not adopted"
        return f"HH_Agent({self.hh_id}, {where}, deg={len(self.neighbours)}, t={self.t}, {mark})"
