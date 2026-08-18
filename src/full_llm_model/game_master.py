"""Full LLM diffusion model: LLM adoption and edge-level LLM transmission.

The BCDJ timing is retained. Leaders are informed by the MFI at ``t=0``. In
each round, newly informed households decide once whether to join; then each
informed household is asked whether to tell each *previously uninformed*
neighbour. When several successful transmissions target one household, exactly
one is selected uniformly and reaches it for the following round.

	python -m src.full_llm_model.game_master --adoption A1B0C1D0 --transmission A1B1D0
	python -m src.full_llm_model.game_master --adoption A1B0C1D0 --transmission A1B1D0 --live

Logs land in ``output/full_llm/<agent>/<adoption>-<transmission>/``. The design
pair is one folder, not two nested ones: the pair is the unit that was run, and
a transmission design under two adoption designs is two runs with nothing in
common but a name. ``src/full_llm_model/analysis.py`` mirrors this path exactly
under ``figures/full_llm/``, so a folder of logs and its folder of figures are
the same three components in the same order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
	from ..hybrid_model import game_master as hybrid
except ImportError:  # running as a script, not a package
	import sys

	sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
	from hybrid_model import game_master as hybrid  # type: ignore[no-redef]


VILLAGE = hybrid.VILLAGE
OUTPUT_DIR = Path("output/full_llm")

# The transmission pilot's instrument. Adoption continues to use the hybrid
# instrument, which is itself the adoption pilot's instrument in a loop.
TRANSMISSION_BASE = """
An institution providing microfinance services has started a new programme in villages across Karnataka, India.
You have been informed that their services are now available in your village too.
You are the head of a household in this village, and as the head you represent your household and its interests.
"""
TRANSMISSION_PROFILE = """
Your household has the following characteristics:
- Religion: {religion}
- Caste: {caste}
- Household size: {hh_size}
- Number of rooms: {num_rooms}
- Number of beds: {num_beds}
- Access to electricity: {electricity}
- Access to a private latrine: {latrine}
- Member of a savings self-help group: {savings_group}
- Access to a bank account: {bank_account}
"""
TRANSMISSION_NARRATIVE = """
Your household has been described as follows:
{narrative}
"""
LEADER_MESSAGE = "The organisation's staff identified your household as a leader through occupation and asked you to spread information."
SENDER_JOINED = "Your household has decided to join the programme."
SENDER_NOT_JOINED = "Your household has decided not to join the programme."
TARGET = "You can now tell your neighbour about the microfinance services available in the village if you see fit."
TARGET_PROFILE = """
Your neighbour has the following characteristics:
- Religion: {religion}
- Caste: {caste}
- Household size: {hh_size}
- Number of rooms: {num_rooms}
- Number of beds: {num_beds}
- Access to electricity: {electricity}
- Access to a private latrine: {latrine}
- Member of a savings self-help group: {savings_group}
- Access to a bank account: {bank_account}
- Occupation: {occupation}
"""
TARGET_NARRATIVE = """
Your neighbour has been described as follows:
{narrative}
"""
TRANSMISSION_FORMAT = "Do you wish to say something to your neighbour? End your response on a new line with {Y} for yes or {N} for no, and nothing else."
TRANSMISSION_MOA = """
Do you wish to say something to your neighbour? Answer these questions:
1. What kind of situation is this?
2. What kind of person am I?
3. What would a person like me do in this situation?

End your response on a new line with {Y} for yes or {N} for no, and nothing else.
"""
TRANSMISSION_DT = """
You should decide whether you inform your neighbour by conducting a decision-theoretic analysis.

Use everything you have been told and your own subjective judgement to fill out a decision matrix over two actions -- informing your neighbour about the programme or not -- and two states of nature describing what they will do: they end up joining the programme or they end up not joining.

For each state, estimate the probability that it is the state you are in, give the utility you would receive under that state from each of the two actions, and state the evidence that justifies those numbers. The two probabilities must sum to 1.

Then give your decision: Y if you decide to inform your neighbour, N if you decide not to inform them.
"""

TRANSMISSION_PROFILE_LEVELS = ("", "DEMOGRAPHIC", "NARRATIVE")
TRANSMISSION_TARGET_LEVELS = ("", "DEMOGRAPHIC", "NARRATIVE")
TRANSMISSION_INSTRUCTION_LEVELS = ("", "MOA", "DT")

TRANSMISSION_DT_STATES = ("they_join", "they_dont_join")
_TRANSMISSION_DT_STATE_SCHEMA = {
	"type": "object",
	"additionalProperties": False,
	"required": ["probability", "Y_utility", "N_utility", "evidence"],
	"properties": {
		"probability": {"type": "number"},
		"Y_utility": {"type": "number"},
		"N_utility": {"type": "number"},
		"evidence": {"type": "array", "items": {"type": "string"}},
	},
}
TRANSMISSION_DT_SCHEMA = {
	"type": "object",
	"additionalProperties": False,
	"required": ["states", "decision"],
	"properties": {
		"states": {
			"type": "object",
			"additionalProperties": False,
			"required": list(TRANSMISSION_DT_STATES),
			"properties": {state: _TRANSMISSION_DT_STATE_SCHEMA for state in TRANSMISSION_DT_STATES},
		},
		"decision": {"type": "string", "enum": ["Y", "N"]},
	},
}
TRANSMISSION_DT_FORMAT = {
	"format": {"type": "json_schema", "name": "transmission_dt_analysis", "strict": True, "schema": TRANSMISSION_DT_SCHEMA}
}


def transmission_label(design: tuple[str, str, str]) -> str:
	"""The full model's ``A#B#D#`` transmission design notation."""
	axes = (
		TRANSMISSION_PROFILE_LEVELS,
		TRANSMISSION_TARGET_LEVELS,
		TRANSMISSION_INSTRUCTION_LEVELS,
	)
	try:
		return "".join(f"{letter}{levels.index(level)}" for letter, levels, level in zip("ABD", axes, design))
	except ValueError as exc:
		raise ValueError(f"invalid transmission design: {design}") from exc


def parse_transmission_design(label: str) -> tuple[str, str, str]:
	"""Parse ``A1B1D0`` into the three transmission prompt levels."""
	match = re.fullmatch(r"A(\d)B(\d)D(\d)", label.strip(), flags=re.IGNORECASE)
	if not match:
		raise ValueError(f"not a transmission design: {label!r}; expected e.g. A1B1D0")
	axes = (
		TRANSMISSION_PROFILE_LEVELS,
		TRANSMISSION_TARGET_LEVELS,
		TRANSMISSION_INSTRUCTION_LEVELS,
	)
	values = []
	for digit, levels in zip(match.groups(), axes):
		index = int(digit)
		if index >= len(levels):
			raise ValueError(f"{label}: level {index} is unavailable")
		values.append(levels[index])
	return tuple(values)  # type: ignore[return-value]


PAIR_SEPARATOR = "-"
_PAIR = re.compile(r"^(?P<adoption>A\dB\dC\dD\d)-(?P<transmission>A\dB\dD\d)$", flags=re.IGNORECASE)


def pair_slug(adoption: str, transmission: str) -> str:
	"""``("A1B0C0D2", "A0B0D2") -> "A1B0C0D2-A0B0D2"`` -- one design pair, one folder name.

	Both labels start with ``A`` and only the adoption one carries a ``C`` axis,
	so the separator is unambiguous in both directions.
	"""
	return f"{adoption}{PAIR_SEPARATOR}{transmission}"


def parse_pair_slug(slug: str) -> tuple[str, str]:
	"""Split a folder name back into ``(adoption_design, transmission_design)``."""
	match = _PAIR.match(slug.strip())
	if not match:
		raise ValueError(f"not a design-pair folder: {slug!r}; expected e.g. A1B0C0D2-A0B0D2")
	return match["adoption"].upper(), match["transmission"].upper()


def transmission_prompt(
	sender: hybrid.HH_Agent,
	target: hybrid.HH_Agent,
	sender_adopted: bool,
	design: tuple[str, str, str],
) -> str:
	"""The prompt for one directed, eligible sender-target edge.

	A sender's leader framing is fixed by its empirical seed status. It is not a
	design axis: agents inherit it from the household they represent.
	"""
	sender_profile, target_profile, instruction = design
	if sender_profile not in TRANSMISSION_PROFILE_LEVELS or target_profile not in TRANSMISSION_TARGET_LEVELS:
		raise ValueError("invalid transmission profile level")
	if instruction not in TRANSMISSION_INSTRUCTION_LEVELS:
		raise ValueError("invalid transmission instruction level")

	parts = [TRANSMISSION_BASE]
	if sender_profile:
		template = TRANSMISSION_PROFILE if sender_profile == "DEMOGRAPHIC" else TRANSMISSION_NARRATIVE
		parts.append(template.format(**sender.fields))
	if sender.is_leader:
		parts.append(LEADER_MESSAGE)
	parts.append(SENDER_JOINED if sender_adopted else SENDER_NOT_JOINED)
	parts.append(TARGET)
	if target_profile:
		template = TARGET_PROFILE if target_profile == "DEMOGRAPHIC" else TARGET_NARRATIVE
		parts.append(template.format(**target.fields))
	template = {"": TRANSMISSION_FORMAT, "MOA": TRANSMISSION_MOA, "DT": TRANSMISSION_DT}[instruction]
	parts.append(template.format(Y=hybrid.YES_TOKEN, N=hybrid.NO_TOKEN))
	return "\n\n".join(part.strip() for part in parts if part.strip())


def parse_transmission_dt(response: str) -> dict | None:
	"""Return a usable two-state transmission decision matrix, if one was returned."""
	text = (response or "").strip()
	try:
		payload = json.loads(text)
	except json.JSONDecodeError:
		start, end = text.find("{"), text.rfind("}")
		if start < 0 or end <= start:
			return None
		try:
			payload = json.loads(text[start : end + 1])
		except json.JSONDecodeError:
			return None

	if not isinstance(payload, dict) or payload.get("decision") not in ("Y", "N"):
		return None
	states = payload.get("states")
	if not isinstance(states, dict) or set(states) != set(TRANSMISSION_DT_STATES):
		return None
	for state in TRANSMISSION_DT_STATES:
		block = states[state]
		if not isinstance(block, dict):
			return None
		try:
			values = [float(block[key]) for key in ("probability", "Y_utility", "N_utility")]
		except (KeyError, TypeError, ValueError):
			return None
		if not all(math.isfinite(value) for value in values):
			return None
	return payload


def get_transmission_response(
	llm: hybrid.LLMs,
	prompt: str,
	instruction: str = "",
	max_parse_attempts: int = 2,
) -> hybrid.Response:
	"""One edge-level LLM decision, using the transmission-specific DT schema."""
	if instruction not in TRANSMISSION_INSTRUCTION_LEVELS:
		raise ValueError(f"invalid transmission instruction: {instruction!r}")
	if llm not in hybrid.WIRED_UP:
		raise NotImplementedError(f"{llm.value} is not wired up yet")
	if max_parse_attempts < 1:
		raise ValueError("max_parse_attempts must be >= 1")

	request: dict[str, object] = {"model": llm.value, "input": prompt}
	if llm in hybrid.REASONING_EFFORT:
		request["reasoning"] = {"effort": hybrid.REASONING_EFFORT[llm]}
	if instruction == "DT":
		request["text"] = TRANSMISSION_DT_FORMAT

	text, decision, usage = "", hybrid.PARSING_ERROR, {}
	for attempt in range(1, max_parse_attempts + 1):
		text, usage = hybrid.one_call(
			hybrid._client(hybrid.PROVIDERS[llm]), request, max_output_tokens=hybrid.MAX_OUTPUT_TOKENS
		)
		if instruction == "DT":
			payload = parse_transmission_dt(text)
			decision = hybrid.PARSING_ERROR if payload is None else (
				hybrid.YES_TOKEN if payload["decision"] == "Y" else hybrid.NO_TOKEN
			)
		else:
			try:
				decision = hybrid.extract_decision(text)
			except ValueError:
				decision = hybrid.PARSING_ERROR
		if decision != hybrid.PARSING_ERROR:
			return hybrid.Response(text=text, decision=decision, usage=usage, attempts=attempt)
	return hybrid.Response(text=text, decision=hybrid.PARSING_ERROR, usage=usage, attempts=max_parse_attempts)


@dataclass
class TransmissionDecision:
	round: int
	sender_id: int
	target_id: int
	sender_adopted: bool
	prompt: str
	response: str
	decision: str
	transmitted: bool
	landed: bool = False
	attempts: int = 0
	usage: dict[str, int] = field(default_factory=dict, repr=False)
	error: str = ""

	def to_row(self) -> dict[str, object]:
		return {
			"round": self.round,
			"sender_hh_id": self.sender_id,
			"target_hh_id": self.target_id,
			"sender_adopted": int(self.sender_adopted),
			"prompt": self.prompt,
			"response": self.response,
			"decision": self.decision,
			"transmitted": int(self.transmitted),
			"landed": int(self.landed),
			"attempts": self.attempts,
			"input_tokens": self.usage.get("input_tokens", ""),
			"output_tokens": self.usage.get("output_tokens", ""),
			"total_tokens": self.usage.get("total_tokens", ""),
			"error": self.error,
		}


def decide_transmissions(
	pop: list[hybrid.HH_Agent],
	A: np.ndarray,
	informed: np.ndarray,
	adopted: np.ndarray,
	design: tuple[str, str, str],
	llm: hybrid.LLMs,
	round_r: int,
	landing_rng: np.random.Generator,
	max_workers: int = 8,
	responder=get_transmission_response,
	progress: tqdm | None = None,
) -> tuple[np.ndarray, list[TransmissionDecision]]:
	"""Elicit all eligible edges and choose one successful sender per target.

	Only targets that have never been informed are eligible. Thus no household
	receives a repeat pitch. The returned matrix has at most one ``True`` in
	each target column, which preserves the one-informer adoption treatment.
	"""
	n = len(pop)
	if A.shape != (n, n):
		raise ValueError(f"adjacency is {A.shape} but population has {n} agents")
	if np.any(adopted & ~informed):
		raise RuntimeError("a household adopted without being informed")

	eligible = A & informed[:, None] & ~informed[None, :]
	edges = [(int(sender), int(target)) for sender, target in zip(*np.nonzero(eligible))]
	work: list[TransmissionDecision] = []
	for sender_idx, target_idx in edges:
		work.append(
			TransmissionDecision(
				round=round_r,
				sender_id=pop[sender_idx].hh_id,
				target_id=pop[target_idx].hh_id,
				sender_adopted=bool(adopted[sender_idx]),
				prompt=transmission_prompt(pop[sender_idx], pop[target_idx], bool(adopted[sender_idx]), design),
				response="",
				decision=hybrid.PARSING_ERROR,
				transmitted=False,
			)
		)

	def ask(record: TransmissionDecision) -> TransmissionDecision:
		try:
			reply = responder(llm, record.prompt, design[2])
		except Exception as exc:  # one failed edge becomes a logged non-transmission
			record.error = f"{type(exc).__name__}: {exc}"
			return record
		record.response = reply.text
		record.decision = reply.decision
		record.transmitted = reply.joined
		record.attempts = reply.attempts
		record.usage = reply.usage
		return record

	if work:
		with ThreadPoolExecutor(max_workers=max_workers) as pool:
			decisions = []
			for future in as_completed([pool.submit(ask, record) for record in work]):
				decisions.append(future.result())
				if progress is not None:
					progress.update(1)
	else:
		decisions = []

	# Uniformly resolve competing successful transmissions for each target.
	hit = np.zeros((n, n), dtype=bool)
	by_id = {agent.hh_id: agent.idx for agent in pop}
	for target_idx in range(n):
		candidates = [d for d in decisions if d.target_id == pop[target_idx].hh_id and d.transmitted]
		if candidates:
			winner = candidates[int(landing_rng.integers(len(candidates)))]
			winner.landed = True
			hit[by_id[winner.sender_id], target_idx] = True
	return hit, sorted(decisions, key=lambda d: (d.sender_id, d.target_id))


@dataclass
class FullRunResult:
	run: hybrid.RunResult
	transmission_design: str
	transmissions: list[TransmissionDecision]

	def transmission_summary(self) -> str:
		"""Transmission decisions by the sender's simulated adoption state.

		``transmitted`` is the LLM's edge-level decision. ``landed`` is the
		one-success-per-target result after simultaneous senders are resolved.
		"""
		def rate(decisions: list[TransmissionDecision], attribute: str) -> str:
			if not decisions:
				return "n/a"
			return f"{sum(bool(getattr(d, attribute)) for d in decisions) / len(decisions):.1%} ({sum(bool(getattr(d, attribute)) for d in decisions)}/{len(decisions)})"

		adopters = [decision for decision in self.transmissions if decision.sender_adopted]
		non_adopters = [decision for decision in self.transmissions if not decision.sender_adopted]
		return (
			f"transmitted all {rate(self.transmissions, 'transmitted')}; "
			f"adopters {rate(adopters, 'transmitted')}; "
			f"non-adopters {rate(non_adopters, 'transmitted')}; "
			f"landed {rate(self.transmissions, 'landed')}"
		)

	@property
	def errors(self) -> list[object]:
		return [*self.run.errors, *(d for d in self.transmissions if d.error)]

	@property
	def parse_failures(self) -> list[object]:
		return [*self.run.parse_failures, *(d for d in self.transmissions if d.decision == hybrid.PARSING_ERROR and not d.error)]


def full_llm_run(
	pop: list[hybrid.HH_Agent],
	A: np.ndarray,
	adoption_design: tuple[str, str, str, str],
	transmission_design: tuple[str, str, str],
	llm: hybrid.LLMs,
	village: int = VILLAGE,
	rounds: int | None = None,
	seed: int = 0,
	replicate: int = 0,
	final_sweep: bool = False,
	max_workers: int = 8,
	adoption_responder=hybrid.get_response,
	transmission_responder=get_transmission_response,
	progress: tqdm | None = None,
) -> FullRunResult:
	"""One fresh full-LLM replicate under the BCDJ one-shot adoption timing."""
	n = len(pop)
	if A.shape != (n, n):
		raise ValueError(f"adjacency is {A.shape} but population has {n} agents")
	rounds = rounds if rounds is not None else hybrid.default_rounds(village)
	if rounds < 1:
		raise ValueError("rounds must be >= 1")

	is_leader = np.array([agent.is_leader for agent in pop], dtype=bool)
	informed = is_leader.copy()
	asked = np.zeros(n, dtype=bool)
	adopted = np.zeros(n, dtype=bool)
	informed_round = np.where(is_leader, hybrid.SEEDED, hybrid.NEVER)
	adopted_round = np.full(n, hybrid.NEVER, dtype=int)
	curve, info_curve = np.zeros(rounds), np.zeros(rounds)
	adoption_decisions: list[hybrid.Decision] = []
	transmission_decisions: list[TransmissionDecision] = []
	hit: np.ndarray | None = None
	landing_rng, informer_rng = (np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(2))

	for round_r in range(1, rounds + 1):
		deciding = informed & ~asked
		if deciding.any():
			batch = hybrid.decide_round(
				pop, deciding, adopted, hit, adoption_design, llm, round_r, informer_rng,
				max_workers=max_workers, responder=adoption_responder, progress=progress,
			)
			for decision in batch:
				if decision.joined:
					adopted[decision.idx] = True
					adopted_round[decision.idx] = round_r
			adoption_decisions.extend(batch)
			asked |= deciding
		else:
			informer_rng.random(n)

		hit, batch = decide_transmissions(
			pop, A, informed, adopted, transmission_design, llm, round_r, landing_rng,
			max_workers=max_workers, responder=transmission_responder, progress=progress,
		)
		transmission_decisions.extend(batch)
		newly = hit.any(axis=0) & ~informed
		informed_round[newly] = round_r
		informed |= newly
		curve[round_r - 1] = adopted.mean()
		info_curve[round_r - 1] = informed.mean()

	swept = 0
	if final_sweep:
		deciding = informed & ~asked
		if deciding.any():
			batch = hybrid.decide_round(
				pop, deciding, adopted, hit, adoption_design, llm, rounds + 1, informer_rng,
				max_workers=max_workers, responder=adoption_responder, progress=progress,
			)
			for decision in batch:
				if decision.joined:
					adopted[decision.idx] = True
					adopted_round[decision.idx] = rounds + 1
			adoption_decisions.extend(batch)
			asked |= deciding
			swept = int(deciding.sum())
			curve[-1] = adopted.mean()

	result = hybrid.RunResult(
		village=village,
		arm="full_llm",
		design=hybrid.design_label(*adoption_design),
		llm=llm.value,
		treatment=hybrid.TREATMENT_ONCE,
		rounds=rounds,
		seed=seed,
		replicate=replicate,
		hh_ids=np.array([agent.hh_id for agent in pop], dtype=int),
		is_leader=is_leader,
		adopted=adopted,
		informed=informed,
		asked=asked,
		adopted_round=adopted_round,
		informed_round=informed_round,
		curve=curve,
		info_curve=info_curve,
		decisions=adoption_decisions,
		swept=swept,
	)
	return FullRunResult(result, transmission_label(transmission_design), transmission_decisions)


def _stub_responder(llm: hybrid.LLMs, prompt: str, instruction: str = "") -> hybrid.Response:
	digest = hashlib.sha256(prompt.encode("utf-8")).digest()
	decision = hybrid.YES_TOKEN if digest[0] < 128 else hybrid.NO_TOKEN
	return hybrid.Response("DRY RUN -- no call was made.", decision, {}, attempts=0)


def write_result(result: FullRunResult, output_dir: Path | str = OUTPUT_DIR) -> tuple[Path, Path]:
	"""Write separate adoption and transmission audit logs for one replicate.

	Into ``<output_dir>/<agent>/<adoption>-<transmission>/`` -- the design pair
	is one folder, matching where `analysis.py` puts that pair's figures.
	"""
	run = result.run
	root = Path(output_dir) / run.llm.replace("-", "_") / pair_slug(run.design, result.transmission_design)
	root.mkdir(parents=True, exist_ok=True)
	stem = f"v{run.village}_rep{run.replicate}"
	adoption_path, transmission_path = root / f"{stem}_adoption.csv", root / f"{stem}_transmission.csv"
	pd.DataFrame([decision.to_row() for decision in run.decisions]).to_csv(adoption_path, index=False)
	pd.DataFrame([decision.to_row() for decision in result.transmissions]).to_csv(transmission_path, index=False)
	return adoption_path, transmission_path


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--village", type=int, default=VILLAGE)
	parser.add_argument("--adoption", required=True, help="adoption design, e.g. A1B0C1D0")
	parser.add_argument("--transmission", required=True, help="transmission design, e.g. A1B1D0")
	parser.add_argument("--model", default=hybrid.LLMs.GPT_5_4_NANO.value)
	parser.add_argument("--reps", type=int, default=1)
	parser.add_argument("--first-rep", type=int, default=0)
	parser.add_argument("--live", action="store_true")
	parser.add_argument("--workers", type=int, default=8)
	args = parser.parse_args(argv)
	if args.reps < 1:
		parser.error("--reps must be >= 1")

	adoption_design = hybrid.parse_design(args.adoption)
	tx_design = parse_transmission_design(args.transmission)
	llm = hybrid.get_llm(args.model)
	leaders, households = hybrid.build_village(args.village)
	pop = hybrid.population(leaders, households)
	A = hybrid.adjacency_matrix(pop)
	adoption_responder = hybrid.get_response if args.live else _stub_responder
	transmission_responder = get_transmission_response if args.live else _stub_responder
	for replicate in range(args.first_rep, args.first_rep + args.reps):
		result = full_llm_run(
			pop, A, adoption_design, tx_design, llm, village=args.village, seed=replicate,
			replicate=replicate, max_workers=args.workers, adoption_responder=adoption_responder,
			transmission_responder=transmission_responder,
		)
		adoption_path, transmission_path = write_result(result)
		print(result.run.summary())
		print(result.transmission_summary())
		print(f"wrote {adoption_path} and {transmission_path}; transmission calls {len(result.transmissions)}")
		if result.errors or result.parse_failures:
			print(f"WARNING: {len(result.errors)} API errors, {len(result.parse_failures)} parsing errors")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

