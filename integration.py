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
# question generator
# =====================

# prompt for candidate generation. the workspace context_string does all
# the steering, this just asks the model to surface its own uncertainty
_CANDIDATE_PROMPT = """\
Below is your current internal state.

{context_string}

Given this state, what are you genuinely uncertain about right now?
List exactly 5 questions things you find yourself wanting to know,
not questions for anyone else.  Each question should be a single
sentence.  Number them 1-5.
"""


class QuestionGenerator:
    # generates an autonomous introspective question when exploration is warranted
    #
    # strategy (option b, slm scored candidates):
    #   1. prompt the slm with current workspace context to produce 5 candidates
    #   2. embed each candidate, project into action space
    #   3. score each by epistemic uncertainty from the ensemble
    #   4. return the candidate with highest epistemic score

    def __init__(self, forward_model, embedder, action_proj, model_name=config.OLLAMA_MODEL):
        self._model = forward_model
        self._embedder = embedder
        self._action_proj = action_proj
        self._model_name = model_name

    def generate(self, z_t, context_string, workspace):
        # produce the highest epistemic introspective question
        #
        # z_t: shape (448,) current state vector
        # context_string: the current workspace context string
        # workspace: the workspace instance (not used right now but available)
        #
        # returns (question, epistemic_score) or None if parsing fails
        candidates = self._generate_candidates(context_string)
        if not candidates:
            return None

        best_q = None
        best_score = -1.0

        for q in candidates:
            emb = self._embedder.embed(q)
            a_i = self._action_proj.project(emb)
            _, epistemic_i = self._model.predict(z_t, a_i)
            if epistemic_i > best_score:
                best_score = epistemic_i
                best_q = q

        if best_q is None:
            return None
        return best_q, best_score

    # internal stuff

    def _generate_candidates(self, context_string):
        # calls the slm and parses numbered questions out of it
        prompt = _CANDIDATE_PROMPT.format(context_string=context_string)

        try:
            response = ollama.chat(
                model=self._model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response["message"]["content"]
        except Exception:
            return []

        return self._parse_candidates(raw)

    @staticmethod
    def _parse_candidates(raw):
        # extract numbered questions from slm output
        # handles formats like: 1. What is... / 1) What is... / 1: What is...
        pattern = re.compile(r"^\s*\d+[.):\-]\s*(.+)", re.MULTILINE)
        matches = pattern.findall(raw)

        questions = []
        for m in matches:
            q = m.strip().rstrip(".")
            if len(q) > 10:  # filter trivially short stuff
                questions.append(q)

        return questions[:5]  # cap at 5


# =====================
# turn result
# =====================

@dataclass
class TurnResult:
    # result of a single turn through WorkspaceIntegration
    response: str
    curiosity_state: CuriosityState
    agent_initiated: bool
    question_used: Optional[str]


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
                 question_gen, action_proj, log_path=_DEFAULT_LOG_PATH):
        self._ws = workspace
        self._encoder = state_encoder
        self._curiosity = curiosity
        self._threshold = threshold
        self._question_gen = question_gen
        self._action_proj = action_proj
        self._log_path = log_path

        # need a previous z_t for the first curiosity step
        # on the very first turn, encode the empty workspace state
        self._last_z = None
        self._turn_index = 0

    def turn(self, user_input=None, timeout_seconds=30.0):
        # execute one full turn with curiosity tracking
        #
        # user_input: the users message, or none for autonomous exploration
        # timeout_seconds: reserved for future async stuff, currently unused
        #
        # returns a TurnResult
        agent_initiated = False
        question_used = None

        # encode current state
        z_t = self._encode_current()

        # figure out what input to use
        if user_input is not None:
            effective_input = user_input
        else:
            # no user input, check exploration gate
            last_epistemic = self._last_epistemic()
            if self._threshold.should_explore(last_epistemic):
                result = self._question_gen.generate(
                    z_t, self._ws.context_string(), self._ws,
                )
                if result is not None:
                    question_used, _ = result
                    effective_input = question_used
                    agent_initiated = True
                else:
                    # slm failed to produce candidates, use a generic probe
                    effective_input = "What am I uncertain about right now?"
                    question_used = effective_input
                    agent_initiated = True
            else:
                # gate closed and no user input, nothing to do
                idle_state = CuriosityState(
                    epistemic=0.0, error=0.0, progress=0.0,
                )
                return TurnResult(
                    response="",
                    curiosity_state=idle_state,
                    agent_initiated=False,
                    question_used=None,
                )

        # run the cognitive loop
        response = self._ws.generate(effective_input)

        # encode action (embed response then project to action space)
        response_emb = self._ws.embedder.embed(response)
        a_t = self._action_proj.project(response_emb)

        # encode post-turn state
        z_t1 = self._encode_current()

        # curiosity step
        curiosity_state = self._curiosity.step(z_t, a_t, z_t1)

        # update threshold
        self._threshold.update(curiosity_state.epistemic)

        # save z for next turn
        self._last_z = z_t1

        # log it
        self._log_turn(
            turn_index=self._turn_index,
            agent_initiated=agent_initiated,
            question_used=question_used,
            input_text=effective_input,
            curiosity_state=curiosity_state,
        )
        self._turn_index += 1

        return TurnResult(
            response=response,
            curiosity_state=curiosity_state,
            agent_initiated=agent_initiated,
            question_used=question_used,
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
                  input_text, curiosity_state):
        # append one json object to the jsonl log
        record = {
            "turn": turn_index,
            "timestamp": time.time(),
            "agent_initiated": agent_initiated,
            "question_used": question_used,
            "input": input_text,
            "curiosity": {
                "epistemic": curiosity_state.epistemic,
                "error": curiosity_state.error,
                "progress": curiosity_state.progress,
            },
            "workspace_turn": self._ws.turn,
            "threshold": self._threshold.threshold,
        }
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
    print("=" * 65)

    # mock the slm (ollama.chat)
    # return a fixed response for generate() and a numbered list for
    # question generator candidate calls
    _call_count = 0

    def mock_ollama_chat(model, messages, **kwargs):
        global _call_count
        _call_count += 1

        content = messages[-1]["content"]

        # detect candidate generation prompt
        if "List exactly 5" in content:
            return {"message": {"content": (
                "1. What patterns connect rhythm and computation?\n"
                "2. How does memory decay shape what I attend to?\n"
                "3. Is there a structure to the silence between concepts?\n"
                "4. What does it mean for an abstraction to leak?\n"
                "5. Why do some ideas feel inevitable in retrospect?\n"
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
        question_gen = QuestionGenerator(
            forward_model, embedder, action_proj,
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
            question_gen=question_gen,
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
            print(
                f"  turn {i:2d} [{tag:5s}]  "
                f"epistemic={result.curiosity_state.epistemic:.6f}  "
                f"error={result.curiosity_state.error:.4f}  "
                f"progress={result.curiosity_state.progress:+.4f}  "
                f"initiated={result.agent_initiated}"
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
        print(f"  ok  Turns 0-6: user-initiated, no question_used")

        # 3. agent initiated turns
        agent_turns = [r for r in results[7:]]
        assert all(r.agent_initiated for r in agent_turns), \
            "Last 3 turns should be agent-initiated"
        assert all(r.question_used is not None for r in agent_turns), \
            "Agent turns should have question_used set"
        print(f"  ok  Turns 7-9: agent-initiated, question_used set")
        for i, r in enumerate(agent_turns, 7):
            print(f"       turn {i} question: \"{r.question_used}\"")

        # 4. all responses are non empty strings
        assert all(isinstance(r.response, str) and len(r.response) > 0
                    for r in results), "All responses should be non-empty strings"
        print(f"  ok  All responses are non-empty strings")

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
            "turn", "timestamp", "agent_initiated", "question_used",
            "input", "curiosity", "workspace_turn", "threshold",
        }
        curiosity_keys = {"epistemic", "error", "progress"}

        for i, line in enumerate(lines):
            record = json.loads(line)

            # check top level keys
            missing = required_keys - set(record.keys())
            assert not missing, f"Line {i}: missing keys {missing}"

            # check curiosity sub object
            c = record["curiosity"]
            c_missing = curiosity_keys - set(c.keys())
            assert not c_missing, f"Line {i}: curiosity missing {c_missing}"

            # check types
            assert isinstance(record["turn"], int)
            assert isinstance(record["timestamp"], float)
            assert isinstance(record["agent_initiated"], bool)
            assert isinstance(record["curiosity"]["epistemic"], float)
            assert isinstance(record["curiosity"]["error"], float)
            assert isinstance(record["curiosity"]["progress"], float)

        print(f"  ok  All log records have correct schema and types")

        # verify agent_initiated flags in log match results
        for i, line in enumerate(lines):
            record = json.loads(line)
            expected = i >= 7
            assert record["agent_initiated"] == expected, (
                f"Line {i}: expected agent_initiated={expected}, "
                f"got {record['agent_initiated']}"
            )
        print(f"  ok  agent_initiated flags correct in log")

        # verify question_used in log
        for i, line in enumerate(lines):
            record = json.loads(line)
            if i < 7:
                assert record["question_used"] is None, \
                    f"Line {i}: user turn should have question_used=None"
            else:
                assert record["question_used"] is not None, \
                    f"Line {i}: agent turn should have question_used set"
        print(f"  ok  question_used fields correct in log")

        # print a sample log record
        sample = json.loads(lines[8])
        print(f"\n  Sample log record (turn 8, agent-initiated):")
        print(f"    {json.dumps(sample, indent=4)}")

        # clean up
        log_path.unlink()
        print(f"\n  ok  Test log file cleaned up")

    print("\n" + "-" * 65)
    print("All assertions passed.")
    print("=" * 65)
