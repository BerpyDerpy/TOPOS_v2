# integration.py - final integration layer for topos
#
# wires the curiosity mechanism (ensemble forward model, curiosity signal,
# adaptive threshold) and the state encoder into the existing cognitive loop
# without modifying workspace.py
#
# two components:
#
#   QuestionGenerator
#     when epistemic uncertainty is high and no user input has arrived,
#     generates candidate introspective questions via the slm, scores them
#     by ensemble disagreement, and returns the highest epistemic candidate
#
#   WorkspaceIntegration
#     thin orchestration class that wraps GlobalWorkspace.process() and
#     adds state encoding, curiosity tracking, autonomous exploration,
#     and jsonl turn logging

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Protocol, Callable

import numpy as np
import ollama

import config
from embedder import Embedder
from workspace import GlobalWorkspace
from state_encoder import StateEncoder, Z_DIM
from forward_model import (
    EnsembleForwardModel,
    CuriositySignal,
    CuriosityState,
    AdaptiveThreshold,
    _ACTION_DIM,
)


# =====================
# action projection
# =====================
# the forward model works with a_t in r^128, but the embedder gives r^384
# this learned linear projection compresses the embedding into action space
# right now its a random orthogonal matrix (no training yet -- the forward
# model learns to predict z_{t+1} from whatever projection, so consistency
# matters more than optimality)

import torch
import torch.nn as nn


class ActionProjection(nn.Module):
    # project a 384d minilm embedding into 128d action space

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(config.WORKSPACE_DIM, _ACTION_DIM, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    @torch.no_grad()
    def project(self, embedding):
        # embedding: (384,) -> action: (128,)
        t = torch.from_numpy(embedding.astype(np.float32)).unsqueeze(0)
        out = self.proj(t).squeeze(0)
        return out.numpy()


# =====================
# exploration generator
# =====================

# R2: open-ended prompt — no instruction to produce questions.
# the slm generates freely from workspace state. if it asks questions,
# that's genuine curiosity. if it doesn't, the state doesn't encode
# real uncertainty.  either outcome is informative.
_EXPLORATION_PROMPT = """\
Below is your current internal state.

{context_string}

Continue from this state. Follow whatever feels most unresolved
or incomplete. One to two paragraphs.
"""


class ExplorationGenerator:
    # generates free-form exploration text when the epistemic gate opens.
    #
    # key change from QuestionGenerator (R2):
    #   - does NOT instruct the slm to produce questions
    #   - lets the slm generate whatever naturally follows from the
    #     workspace state
    #   - detects whether the output contains questions (for logging,
    #     not for control flow)
    #
    # this means "curiosity" is measured by whether the system
    # spontaneously asks questions, not by whether it complies
    # with a prompt that demands them.

    def __init__(self, model_name=config.OLLAMA_MODEL):
        self._model_name = model_name

    def generate(self, context_string):
        # produce free-form exploration text from the current workspace state.
        #
        # context_string: the current workspace context string
        #
        # returns (exploration_text, has_questions) or None if generation fails.
        text = self._call_slm(context_string)
        if text is None:
            return None

        has_questions = self._detect_questions(text)
        return text, has_questions

    # internal stuff

    def _call_slm(self, context_string):
        # single slm call with open-ended prompt
        prompt = _EXPLORATION_PROMPT.format(context_string=context_string)

        try:
            response = ollama.chat(
                model=self._model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response["message"]["content"].strip()
        except Exception:
            return None

        if len(raw) < 10:
            return None
        return raw

    @staticmethod
    def _detect_questions(text):
        # detect whether the exploration text contains interrogative forms.
        # this is a measurement, not a filter — we log it but don't act on it.
        # checks for: sentences ending in '?', or starting with wh-words/how/do/is/are/can
        if "?" in text:
            return True
        interrogative = re.compile(
            r"(?:^|(?<=\.\s))(?:what|why|how|where|when|who|which|do|does|is|are|can|could|would|should)\s",
            re.IGNORECASE | re.MULTILINE,
        )
        return bool(interrogative.search(text))


# =====================
# turn result
# =====================

@dataclass
class TurnResult:
    # result of a single turn through WorkspaceIntegration
    response: str
    curiosity_state: CuriosityState
    agent_initiated: bool
    question_used: Optional[str]       # exploration text (may or may not contain questions)
    has_questions: bool = False         # whether the exploration text contained interrogative forms
    response_integrated: bool = False   # R1: whether the slm response was fed back through workspace
    probe_response: Optional[str] = None  # R5: response to periodic probe question, if this was a probe turn


# =====================
# workspace integration
# =====================

_DEFAULT_LOG_PATH = Path(__file__).parent / "curiosity_log.jsonl"


class WorkspaceIntegration:
    # orchestration layer wrapping GlobalWorkspace with curiosity tracking
    #
    # wraps the existing process() -> response loop and adds:
    #   - state encoder calls before and after process()
    #   - curiosity signal tracking (epistemic, error, progress)
    #   - adaptive threshold gating for autonomous exploration
    #   - question generator for agent initiated turns
    #   - jsonl logging of every turn's curiosity signals

    def __init__(self, workspace, state_encoder, curiosity, threshold,
                 exploration_gen, action_proj, log_path=_DEFAULT_LOG_PATH,
                 probe_questions=None, probe_interval=5):
        self._ws = workspace
        self._encoder = state_encoder
        self._curiosity = curiosity
        self._threshold = threshold
        self._exploration_gen = exploration_gen
        self._action_proj = action_proj
        self._log_path = log_path

        # R5: periodic probe questions for behavioral change measurement
        # probe_questions: list of fixed questions to cycle through
        # probe_interval: every N autonomous turns, inject a probe instead
        self._probe_questions = probe_questions or []
        self._probe_interval = probe_interval
        self._autonomous_count = 0  # counts autonomous turns for probe scheduling

        # need a previous z_t for the first curiosity step
        # on the very first turn, encode the empty workspace state
        self._last_z = None
        self._turn_index = 0

    def turn(self, user_input=None, timeout_seconds=30.0, process_only=False):
        # execute one full turn with curiosity tracking
        #
        # user_input: the users message, or none for autonomous exploration
        # timeout_seconds: reserved for future async stuff, currently unused
        # process_only: if true, call ws.process() instead of ws.generate()
        #               (no SLM call). used for corpus-driven priming where
        #               we want the forward model and threshold to track state
        #               but don't need language output.
        #
        # returns a TurnResult
        agent_initiated = False
        question_used = None
        has_questions = False

        # encode current state (this z_t is the reference point for ALL
        # curiosity measurement in this turn — R4)
        z_t = self._encode_current()

        # figure out what input to use
        if user_input is not None:
            effective_input = user_input
        else:
            # no user input — autonomous exploration
            self._autonomous_count += 1

            # R5: check if this is a probe turn
            probe_response = None
            if (self._probe_questions and
                    self._autonomous_count % self._probe_interval == 0):
                probe_idx = (
                    (self._autonomous_count // self._probe_interval) - 1
                ) % len(self._probe_questions)
                probe_q = self._probe_questions[probe_idx]
                probe_response = self._ws.generate(probe_q)
                # don't mutate state further — probe is observation only

            # recall a memory using a random top concept as query
            concepts = self._ws.graph.top_concepts(10)
            if concepts:
                query_concept = concepts[
                    self._turn_index % len(concepts)
                ]
            else:
                query_concept = "uncertainty"

            memories = self._ws.memory.recall(query_concept, n=1)
            if memories:
                recall_text = memories[0]["text"]
            else:
                recall_text = query_concept

            # process the recalled memory (no SLM, just workspace update)
            self._ws.process(recall_text)

            # R4: gate decision uses predict() only — no training, no
            # curiosity.step(). the forward model should not train on the
            # recall transition separately. training happens once in the
            # main block below, on the full turn transition.
            recall_emb = self._ws.embedder.embed(recall_text)
            a_recall = self._action_proj.project(recall_emb)
            z_post_recall = self._encode_current()
            _, gate_epistemic = self._curiosity._model.predict(
                z_t, a_recall,
            )
            # update threshold with the gate reading (but don't train)
            self._threshold.update(gate_epistemic)

            # evaluate gate
            if self._threshold.should_explore(gate_epistemic):
                # R2: generate free-form exploration via SLM
                result = self._exploration_gen.generate(
                    self._ws.context_string(),
                )
                if result is not None:
                    question_used, has_questions = result
                    effective_input = question_used
                    agent_initiated = True
                else:
                    # R3: generation failed — treat as idle turn
                    # R4: still need one curiosity.step() for the recall
                    # transition so the forward model learns from it
                    curiosity_state = self._curiosity.step(
                        z_t, a_recall, z_post_recall,
                    )
                    self._last_z = z_post_recall
                    self._log_turn(
                        turn_index=self._turn_index,
                        agent_initiated=False,
                        question_used=None,
                        input_text=f"[recall:{query_concept}] {recall_text[:80]} [gen_failed]",
                        curiosity_state=curiosity_state,
                        domain=getattr(self, '_current_domain', None),
                    )
                    self._turn_index += 1
                    return TurnResult(
                        response="",
                        curiosity_state=curiosity_state,
                        agent_initiated=False,
                        question_used=None,
                        probe_response=probe_response,
                    )
                # fall through to the main processing block below
            else:
                # gate closed — train on the recall transition (R4: single
                # curiosity.step for idle turns)
                curiosity_state = self._curiosity.step(
                    z_t, a_recall, z_post_recall,
                )
                self._last_z = z_post_recall
                self._log_turn(
                    turn_index=self._turn_index,
                    agent_initiated=False,
                    question_used=None,
                    input_text=f"[recall:{query_concept}] {recall_text[:80]}",
                    curiosity_state=curiosity_state,
                    domain=getattr(self, '_current_domain', None),
                    probe_response=probe_response,
                )
                self._turn_index += 1
                return TurnResult(
                    response="",
                    curiosity_state=curiosity_state,
                    agent_initiated=False,
                    question_used=None,
                    probe_response=probe_response,
                )

        # run the cognitive loop
        if process_only:
            # corpus-driven priming: process only, no SLM
            context = self._ws.process(effective_input)
            response = ""
        else:
            response = self._ws.generate(effective_input)

        # R1: close the learning loop — feed the SLM's response back
        # through the workspace so the "answer" shapes understanding.
        response_integrated = False
        if agent_initiated and response and not process_only:
            self._ws.process(response)
            response_integrated = True

        # encode action and project to action space
        if process_only:
            input_emb = self._ws.embedder.embed(effective_input)
            a_t = self._action_proj.project(input_emb)
        else:
            response_emb = self._ws.embedder.embed(response)
            a_t = self._action_proj.project(response_emb)

        # encode post-turn state (captures state AFTER response integration)
        z_t1 = self._encode_current()

        # R4: single curiosity.step() per turn — measures the full transition
        # from pre-recall/pre-input z_t to post-response z_t1, and trains
        # the forward model exactly once on this transition.
        curiosity_state = self._curiosity.step(z_t, a_t, z_t1)

        # update threshold
        self._threshold.update(curiosity_state.epistemic)

        # save z for next turn
        self._last_z = z_t1

        # R5: attach probe response if this was a probe turn
        probe_response = locals().get('probe_response', None)

        # log it
        _has_q = has_questions if agent_initiated else False
        self._log_turn(
            turn_index=self._turn_index,
            agent_initiated=agent_initiated,
            question_used=question_used,
            input_text=effective_input,
            curiosity_state=curiosity_state,
            domain=getattr(self, '_current_domain', None),
            has_questions=_has_q,
            response_integrated=response_integrated,
            probe_response=probe_response,
        )
        self._turn_index += 1

        return TurnResult(
            response=response,
            curiosity_state=curiosity_state,
            agent_initiated=agent_initiated,
            question_used=question_used,
            has_questions=_has_q,
            response_integrated=response_integrated,
            probe_response=probe_response,
        )

    # internal helpers

    def _encode_current(self):
        # encode the current workspace module states into z_t
        return self._encoder.encode(
            graph=self._ws.graph,
            affect=self._ws.affect,
            memory=self._ws.memory,
            input_embedding=self._ws.state,
            current_turn=self._ws.turn,
        )

    def _last_epistemic(self):
        # return the most recent epistemic value, or a high default
        if self._threshold._values:
            return self._threshold._values[-1]
        # cold start, return something that guarantees exploration
        return float("inf")

    def _log_turn(self, turn_index, agent_initiated, question_used,
                  input_text, curiosity_state, domain=None,
                  has_questions=False, response_integrated=False,
                  probe_response=None):
        # append one json object to the jsonl log
        record = {
            "turn_index": turn_index,
            "timestamp": time.time(),
            "autonomous_question_fired": agent_initiated,
            "question_used": question_used,
            "input": input_text,
            "epistemic_uncertainty": curiosity_state.epistemic,
            "prediction_error": curiosity_state.error,
            "progress_signal": curiosity_state.progress,
            "top_5_concepts": self._ws.graph.top_concepts(5),
            "arousal": self._ws.affect.arousal(),
            "valence": self._ws.affect.valence(),
            "workspace_turn": self._ws.turn,
            "threshold": self._threshold.threshold,
            "has_questions": has_questions,
            "response_integrated": response_integrated,
        }
        if domain is not None:
            record["domain"] = domain
        if probe_response is not None:
            record["probe_response"] = probe_response  # R5
        with open(self._log_path, "a") as f:
            f.write(json.dumps(record) + "\n")


# =====================
# dry run test
# =====================
if __name__ == "__main__":
    # mock based dry run: 10 turns (7 user, 3 agent), validates jsonl

    import sys
    import tempfile
    from unittest.mock import patch, MagicMock

    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 65)
    print("WorkspaceIntegration dry-run test (mocked SLM + StateEncoder)")
    print("R1: learning loop closure, R2: emergent exploration, R3: no fallback")
    print("=" * 65)

    # mock the slm (ollama.chat)
    # return a fixed response for generate() and free-form exploration
    # for ExplorationGenerator calls
    _call_count = 0

    def mock_ollama_chat(model, messages, **kwargs):
        global _call_count
        _call_count += 1

        content = messages[-1]["content"]

        # detect exploration prompt (R2: open-ended, no "list 5 questions")
        if "Follow whatever feels most unresolved" in content:
            return {"message": {"content": (
                "The relationship between memory and attention keeps surfacing. "
                "When a system retrieves something from storage, it changes what "
                "it attends to next — but does the act of attending also reshape "
                "what gets stored? There's a circularity here that feels unresolved. "
                "What would it mean for retrieval itself to be a form of learning?"
            )}}

        # normal generation, short response
        return {"message": {"content": (
            f"[Mock response #{_call_count}] "
            "The architecture of attention and memory interweave."
        )}}

    # build everything with ollama mocked out
    with patch("ollama.chat", side_effect=mock_ollama_chat), \
         patch("workspace.ollama.chat", side_effect=mock_ollama_chat):

        ws = GlobalWorkspace()
        embedder = ws.embedder  # reuse the workspace's embedder

        encoder = StateEncoder(embedder)
        forward_model = EnsembleForwardModel(n_members=3, lr=1e-3)
        curiosity = CuriositySignal(forward_model, window_k=10)
        threshold = AdaptiveThreshold(window_size=20, percentile=85.0)
        action_proj = ActionProjection()
        exploration_gen = ExplorationGenerator(
            model_name=config.OLLAMA_MODEL,
        )

        # use a temporary log file
        log_path = Path(__file__).parent / "_test_curiosity_log.jsonl"
        if log_path.exists():
            log_path.unlink()

        integration = WorkspaceIntegration(
            workspace=ws,
            state_encoder=encoder,
            curiosity=curiosity,
            threshold=threshold,
            exploration_gen=exploration_gen,
            action_proj=action_proj,
            log_path=log_path,
        )

        # user initiated inputs
        user_inputs = [
            "I've been thinking about how memory shapes perception",
            "The kernel scheduler makes decisions faster than we can notice",
            "What does it mean to understand something deeply?",
            "Cache coherence feels like negotiation between processors",
            "There's a rhythm to how ideas connect across domains",
            "Improvisation requires real-time uncertainty resolution",
            "The most reliable systems know exactly how they'll fail",
        ]

        # schedule: turns 0-6 are user, turns 7-9 are agent initiated
        print("\n-- Running 10 turns --")
        results = []

        for i in range(10):
            if i < 7:
                # user turn
                result = integration.turn(user_input=user_inputs[i])
                tag = "USER"
            else:
                # agent turn (no user input)
                result = integration.turn(user_input=None)
                tag = "AGENT"

            results.append(result)
            extra = ""
            if result.agent_initiated:
                extra = f"  has_q={result.has_questions}  R1={result.response_integrated}"
            print(
                f"  turn {i:2d} [{tag:5s}]  "
                f"epistemic={result.curiosity_state.epistemic:.6f}  "
                f"error={result.curiosity_state.error:.4f}  "
                f"progress={result.curiosity_state.progress:+.4f}  "
                f"initiated={result.agent_initiated}{extra}"
            )

        # validate results
        print("\n-- Validation --")

        # 1. turn count
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"
        print(f"  ok  10 turns completed")

        # 2. user initiated turns
        user_turns = [r for r in results[:7]]
        assert all(not r.agent_initiated for r in user_turns), \
            "First 7 turns should be user-initiated"
        assert all(r.question_used is None for r in user_turns), \
            "User turns should have question_used=None"
        assert all(not r.has_questions for r in user_turns), \
            "User turns should have has_questions=False"
        assert all(not r.response_integrated for r in user_turns), \
            "User turns should have response_integrated=False"
        print(f"  ok  Turns 0-6: user-initiated, no question_used, no R1")

        # 3. agent initiated turns — R3 means some may NOT fire if gen fails
        agent_turns = results[7:]
        fired = [r for r in agent_turns if r.agent_initiated]
        idle = [r for r in agent_turns if not r.agent_initiated]
        print(f"  info  Agent turns: {len(fired)} fired, {len(idle)} idle")

        # R1: all fired turns should have response_integrated=True
        for r in fired:
            assert r.response_integrated, \
                "Fired agent turns should have response_integrated=True (R1)"
            assert r.question_used is not None, \
                "Fired agent turns should have question_used set"
        if fired:
            print(f"  ok  All fired turns have response_integrated=True (R1)")

        # R3: idle agent turns should NOT have question_used set
        for r in idle:
            assert r.question_used is None, \
                "Idle agent turns should have question_used=None (R3)"
        if idle:
            print(f"  ok  Idle agent turns have question_used=None (R3)")

        # print fired exploration text
        for i, r in enumerate(fired):
            print(f"       exploration [{i}]: \"{r.question_used[:80]}...\"")
            print(f"       has_questions={r.has_questions}")

        # 4. all responses are strings (may be empty for idle turns)
        assert all(isinstance(r.response, str) for r in results), \
            "All responses should be strings"
        print(f"  ok  All responses are strings")

        # 5. curiosity state fields are all floats
        for r in results:
            cs = r.curiosity_state
            assert isinstance(cs.epistemic, float), f"epistemic not float: {type(cs.epistemic)}"
            assert isinstance(cs.error, float), f"error not float: {type(cs.error)}"
            assert isinstance(cs.progress, float), f"progress not float: {type(cs.progress)}"
        print(f"  ok  All CuriosityState fields are floats")

        # validate jsonl log
        print("\n-- JSONL log validation --")

        assert log_path.exists(), f"Log file not found: {log_path}"
        with open(log_path) as f:
            lines = f.readlines()

        assert len(lines) == 10, f"Expected 10 log lines, got {len(lines)}"
        print(f"  ok  10 log lines written")

        required_keys = {
            "turn_index", "timestamp", "autonomous_question_fired",
            "question_used", "input", "epistemic_uncertainty",
            "prediction_error", "progress_signal", "top_5_concepts",
            "arousal", "valence", "workspace_turn", "threshold",
            "has_questions", "response_integrated",
        }

        for i, line in enumerate(lines):
            record = json.loads(line)

            # check top level keys
            missing = required_keys - set(record.keys())
            assert not missing, f"Line {i}: missing keys {missing}"

            # check types
            assert isinstance(record["turn_index"], int)
            assert isinstance(record["timestamp"], float)
            assert isinstance(record["autonomous_question_fired"], bool)
            assert isinstance(record["epistemic_uncertainty"], float)
            assert isinstance(record["prediction_error"], float)
            assert isinstance(record["progress_signal"], float)
            assert isinstance(record["top_5_concepts"], list)
            assert isinstance(record["has_questions"], bool)
            assert isinstance(record["response_integrated"], bool)

        print(f"  ok  All log records have correct schema and types (incl R1/R2 fields)")

        # R1 validation: response_integrated should be True for fired turns
        for i, line in enumerate(lines):
            record = json.loads(line)
            if record["autonomous_question_fired"]:
                assert record["response_integrated"], \
                    f"Line {i}: fired turn should have response_integrated=True"
        print(f"  ok  response_integrated=True for all fired turns in log (R1)")

        # print a sample log record from an agent turn
        for i, line in enumerate(lines):
            record = json.loads(line)
            if record["autonomous_question_fired"]:
                print(f"\n  Sample log record (turn {i}, agent-initiated):")
                print(f"    {json.dumps(record, indent=4)}")
                break

        # clean up
        log_path.unlink()
        print(f"\n  ok  Test log file cleaned up")

    print("\n" + "-" * 65)
    print("All assertions passed.")
    print("=" * 65)
