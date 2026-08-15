"""Invariants for `agent.HH_Agent` and `state.DefaultState`.

    pytest tests/

`docs/household_design.md` §1: "'We were careful' is not a control; a test is."
The leakage tests at the bottom are that control for the agent half — the
persona-rendering half arrives with `to_persona()`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import agent as ag  # noqa: E402
import state as stt  # noqa: E402


@pytest.fixture
def a() -> ag.HH_Agent:
    """A household two rounds into a run: seeded at t=0, talking and adopting at t=1."""
    h = ag.HH_Agent(hh_id=73001, village=73, row=1, neighbours=(73003, 73044))
    h.set_context("(persona would go here)")
    h.seed("A representative from BSS explained a group loan scheme.")
    h.advance()
    h.receive(73003, "My wife's group took a loan.")
    h.send(73044, "Have you heard about the BSS meeting?")
    h.adopt()
    h.advance()
    h.advance()
    return h


# -- the ledger ---------------------------------------------------------


def test_ledger_is_rectangular(a):
    """Every row is length t + 1, so `received[j][t]` never needs a bounds check."""
    st = a.state
    st.validate()
    assert st.t == 3
    assert all(len(row) == 4 for row in st.received.values())
    assert all(len(row) == 4 for row in st.sent.values())


def test_index_is_the_round(a):
    assert a.state.received[73003] == ["", "My wife's group took a loan.", "", ""]
    assert a.state.sent[73044] == ["", "Have you heard about the BSS meeting?", "", ""]


def test_silent_peer_still_has_a_full_row(a):
    """Absence of a message is a slot, not a missing key."""
    assert a.state.received[73044] == [stt.NO_MESSAGE] * 4


def test_channel_opened_late_is_backfilled():
    h = ag.HH_Agent(hh_id=73001)
    h.advance()
    h.advance()
    h.receive(73009, "late arrival")
    assert h.state.received[73009] == ["", "", "late arrival"]
    h.state.validate()


def test_informed_at_is_derived_from_the_ledger(a):
    assert a.state.informed_at == 0  # the seed disclosure counts
    assert a.state.informed
    assert not ag.HH_Agent(hh_id=73002).state.informed


def test_adoption_time(a):
    assert a.adopted and a.adopted_t == 1


# -- misuse -------------------------------------------------------------


def test_double_speak_in_one_round_raises():
    h = ag.HH_Agent(hh_id=73002, neighbours=(73003,))
    h.receive(73003, "once")
    with pytest.raises(stt.AgentError):
        h.receive(73003, "twice")


def test_same_peer_may_speak_again_next_round():
    h = ag.HH_Agent(hh_id=73002, neighbours=(73003,))
    h.receive(73003, "once")
    h.advance()
    h.receive(73003, "again")
    assert h.state.received[73003] == ["once", "again"]


def test_readopting_raises(a):
    with pytest.raises(stt.AgentError):
        a.adopt()


def test_empty_text_is_rejected(a):
    """The sentinel means "nothing was said"; writing it explicitly is a bug."""
    with pytest.raises(ValueError):
        a.receive(73003, stt.NO_MESSAGE)


def test_sending_off_the_network_raises(a):
    with pytest.raises(stt.AgentError):
        a.send(99999, "hello")


def test_self_edge_rejected():
    with pytest.raises(ValueError):
        ag.HH_Agent(hh_id=73001, neighbours=(73001,))


def test_validate_catches_hand_broken_state(a):
    a.state.adopted_t = None  # adopted flag now disagrees
    with pytest.raises(stt.AgentError):
        a.state.validate()


# -- transcript ---------------------------------------------------------


def test_transcript_is_chronological_and_hears_before_speaking(a):
    turns = a.transcript()
    assert [t.t for t in turns] == [0, 1, 1]
    assert [t.direction for t in turns] == ["in", "in", "out"]
    assert turns[0].other == stt.SEED_SOURCE


def test_transcript_slicing(a):
    assert [t.direction for t in a.transcript(since=1)] == ["in", "out"]
    assert [t.other for t in a.transcript(direction="out")] == [73044]
    with pytest.raises(ValueError):
        a.transcript(direction="sideways")


# -- persistence --------------------------------------------------------


def test_run_log_round_trips_through_json(a):
    back = ag.HH_Agent.from_dict(json.loads(json.dumps(a.to_dict())))
    assert back.to_dict() == a.to_dict()
    assert back.state.received[73003][1] == "My wife's group took a loan."
    assert back.adopted_t == 1


def test_state_version_is_recorded_and_dispatched(a):
    d = a.state.to_dict()
    assert d["version"] == "default"
    assert isinstance(stt.state_from_dict(d), stt.DefaultState)
    with pytest.raises(ValueError):
        stt.state_from_dict({**d, "version": "no-such-model"})
    with pytest.raises(ValueError):
        stt.make_state("no-such-model")


def test_run_log_excludes_the_persona(a):
    """Context is large, identical every round, and already on disk."""
    assert "persona" not in json.dumps(a.to_dict())


# -- leakage and cost ---------------------------------------------------


def test_agent_holds_no_route_to_the_feature_table(a):
    """§1's non-negotiable, structurally: no dataframe, no row, no `_adopted`.

    `row` is an integer index for joining back, not the data itself.
    """
    held = vars(a)
    assert set(held) == {"hh_id", "village", "row", "neighbours", "context_dir", "state", "_context"}
    assert not any(hasattr(v, "columns") for v in held.values())
    assert not any("adopt" in k for k in vars(a.state) if k != "adopted" and k != "adopted_t")


def test_context_is_read_once(tmp_path):
    """Caching the immutable persona is the main lever on LLM cost (§5)."""
    p = tmp_path / "context_73001.txt"
    p.write_text("first", encoding="utf-8")
    h = ag.HH_Agent(hh_id=73001, context_dir=tmp_path)
    assert h.context == "first"
    p.write_text("second", encoding="utf-8")
    assert h.context == "first"


def test_missing_context_raises_on_access_not_construction(tmp_path):
    h = ag.HH_Agent(hh_id=99999, context_dir=tmp_path)  # constructs fine
    assert not h.has_context
    with pytest.raises(ag.ContextMissing):
        h.context
