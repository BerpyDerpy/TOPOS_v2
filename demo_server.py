# demo_server.py — TOPOS live demo backend
#
# FastAPI server that runs the full TOPOS cognitive loop and streams
# workspace state to a browser frontend via WebSocket.
#
# Usage:
#   python demo_server.py
#   # then open http://localhost:8000 in a browser

import asyncio
import json
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# topos imports
import config
from workspace import GlobalWorkspace
from state_encoder import StateEncoder
from forward_model import EnsembleForwardModel, CuriositySignal
from integration import (
    WorkspaceIntegration, ActionProjection, ExplorationGenerator,
    AdaptiveThreshold,
)
from corpus_loader import SingleFileCorpusLoader

app = FastAPI()

# serve the demo frontend
demo_dir = Path(__file__).parent / "demo"
if demo_dir.exists():
    app.mount("/static", StaticFiles(directory=str(demo_dir)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(demo_dir / "index.html"))


def _serialize_graph(graph, max_nodes=60):
    """Extract nodes and edges from the concept graph for D3."""
    g = graph._graph
    # get top nodes by weight
    node_weights = [(n, d.get("weight", 0)) for n, d in g.nodes(data=True)]
    node_weights.sort(key=lambda x: x[1], reverse=True)
    top_nodes = node_weights[:max_nodes]
    top_names = {n for n, _ in top_nodes}

    nodes = [{"id": n, "weight": round(w, 3)} for n, w in top_nodes]
    edges = []
    for u, v, d in g.edges(data=True):
        if u in top_names and v in top_names:
            w = d.get("weight", 0)
            if w > 0.01:
                edges.append({
                    "source": u,
                    "target": v,
                    "weight": round(w, 3),
                })

    return nodes, edges


def _build_turn_msg(ws, curiosity_state, threshold, turn_idx, phase,
                    domain, input_text, response=""):
    """Build a turn state message for the frontend."""
    nodes, edges = _serialize_graph(ws.graph)
    memories = ws.memory.recall(ws._last_input or "", n=5) if ws._last_input else []

    return {
        "type": "turn",
        "data": {
            "turn": turn_idx,
            "phase": phase,
            "domain": domain,
            "concepts": nodes,
            "edges": edges,
            "arousal": round(ws.affect.arousal(), 4),
            "valence": round(ws.affect.valence(), 4),
            "surprise": round(ws._last_surprise, 4),
            "epistemic": round(float(curiosity_state.epistemic), 6),
            "prediction_error": round(float(curiosity_state.error), 4),
            "progress": round(float(curiosity_state.progress), 4),
            "threshold": round(float(threshold.threshold or 0.0), 6),
            "memory_count": ws.turn,
            "recent_memories": [
                {"text": m["text"][:100], "turn": m["turn"]}
                for m in memories[:3]
            ],
            "input_text": input_text[:120] if input_text else "",
            "response": response[:200] if response else "",
            "top_concepts": ws.graph.top_concepts(8),
        },
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        # wait for start message
        raw = await websocket.receive_text()
        start_msg = json.loads(raw)

        priming_per_domain = start_msg.get("priming_turns", 15)
        autonomous_turns = start_msg.get("autonomous_turns", 20)
        delay_ms = start_msg.get("delay_ms", 300)

        # initialize topos stack
        await websocket.send_json({"type": "status", "text": "Initializing TOPOS..."})

        ws, encoder, curiosity, threshold, exploration_gen, action_proj = \
            await asyncio.to_thread(_init_topos)

        # build integration (no log file for demo)
        log_path = Path("runs") / "demo_latest.jsonl"
        log_path.parent.mkdir(exist_ok=True)
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

        await websocket.send_json({"type": "status", "text": "Loading corpus..."})

        # load corpus
        everyday_loader = SingleFileCorpusLoader(config.CORPUS_EVERYDAY_PATH)

        # ========== PHASE 1: PRIMING ==========
        priming_turns = min(priming_per_domain, everyday_loader.total_turns)
        await websocket.send_json({
            "type": "status",
            "text": f"Priming: {priming_turns} turns",
        })

        total_priming = priming_turns
        turn_idx = 0
        priming_epistemics = []  # track for threshold seeding

        integration._current_domain = "everyday"
        for i in range(priming_turns):
            text = everyday_loader.next()
            if text is None:
                break
            result = await asyncio.to_thread(
                integration.turn, user_input=text, process_only=True,
            )
            turn_idx += 1
            priming_epistemics.append(result.curiosity_state.epistemic)

            msg = _build_turn_msg(
                ws, result.curiosity_state, threshold,
                turn_idx, "priming", "everyday", text,
            )
            msg["data"]["total_turns"] = total_priming + autonomous_turns
            await websocket.send_json(msg)

            # check for speed change
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(), timeout=delay_ms / 1000.0,
                )
                client_msg = json.loads(raw)
                if client_msg.get("type") == "set_speed":
                    delay_ms = client_msg["delay_ms"]
            except asyncio.TimeoutError:
                pass

            # gate check: did this priming turn make the agent curious enough
            # to interrupt? skip first 15 turns so threshold can calibrate.
            interject = await asyncio.to_thread(integration.maybe_interject, 15)
            if interject is not None:
                question, has_questions, epistemic = interject
                await websocket.send_json({
                    "type": "gate_fired",
                    "data": {
                        "turn": turn_idx,
                        "exploration_text": question,
                        "has_questions": has_questions,
                        "epistemic": round(float(epistemic), 6),
                        "response": "",
                        "concepts": [n["id"] for n in _serialize_graph(ws.graph)[0][:10]],
                        "arousal": round(ws.affect.arousal(), 4),
                        "valence": round(ws.affect.valence(), 4),
                    },
                })

                # pause priming, wait for the user's answer (90s timeout)
                try:
                    raw = await asyncio.wait_for(websocket.receive_text(), timeout=90.0)
                except asyncio.TimeoutError:
                    raw = '{"type":"answer","text":""}'
                answer_msg = json.loads(raw)
                if answer_msg.get("type") == "answer" and answer_msg.get("text"):
                    answer_text = answer_msg["text"]
                    pre_concepts = set(ws.graph.top_concepts(10))
                    await asyncio.to_thread(ws.process, answer_text)
                    post_concepts = set(ws.graph.top_concepts(10))
                    new_concepts = list(post_concepts - pre_concepts)

                    nodes, edges = _serialize_graph(ws.graph)
                    await websocket.send_json({
                        "type": "answer_processed",
                        "data": {
                            "answer": answer_text[:200],
                            "new_concepts": new_concepts,
                            "concepts": nodes,
                            "edges": edges,
                            "arousal": round(ws.affect.arousal(), 4),
                            "valence": round(ws.affect.valence(), 4),
                        },
                    })

        # ========== PHASE 2: AUTONOMOUS ==========
        await websocket.send_json({
            "type": "status",
            "text": f"Autonomous exploration: {autonomous_turns} turns. Gate active.",
        })

        # Reset threshold seeded with the top half of priming epistemics.
        # This anchors the 85th-percentile gate to the actual range seen
        # during priming — not the collapsed end-of-priming values — so
        # the gate has a realistic chance of firing during autonomous turns.
        priming_epistemics.sort(reverse=True)
        seed = priming_epistemics[:max(10, len(priming_epistemics) // 2)]
        threshold.reset(seed_values=seed)
        integration._current_domain = "autonomous"

        for i in range(autonomous_turns):
            result = await asyncio.to_thread(
                integration.turn, user_input=None, process_only=False,
            )
            turn_idx += 1

            if result.agent_initiated:
                # gate fired — send exploration and wait for answer
                await websocket.send_json({
                    "type": "gate_fired",
                    "data": {
                        "turn": turn_idx,
                        "exploration_text": result.question_used or "",
                        "has_questions": result.has_questions,
                        "epistemic": round(float(result.curiosity_state.epistemic), 6),
                        "response": result.response[:300] if result.response else "",
                        "concepts": [n["id"] for n in _serialize_graph(ws.graph)[0][:10]],
                        "arousal": round(ws.affect.arousal(), 4),
                        "valence": round(ws.affect.valence(), 4),
                    },
                })

                # wait for user answer (90s timeout so done always sends)
                try:
                    raw = await asyncio.wait_for(websocket.receive_text(), timeout=90.0)
                except asyncio.TimeoutError:
                    raw = '{"type":"answer","text":""}'
                answer_msg = json.loads(raw)

                if answer_msg.get("type") == "answer" and answer_msg.get("text"):
                    answer_text = answer_msg["text"]

                    # process answer through workspace (genuine learning)
                    pre_concepts = set(ws.graph.top_concepts(10))
                    await asyncio.to_thread(ws.process, answer_text)
                    post_concepts = set(ws.graph.top_concepts(10))
                    new_concepts = list(post_concepts - pre_concepts)

                    nodes, edges = _serialize_graph(ws.graph)
                    await websocket.send_json({
                        "type": "answer_processed",
                        "data": {
                            "answer": answer_text[:200],
                            "new_concepts": new_concepts,
                            "concepts": nodes,
                            "edges": edges,
                            "arousal": round(ws.affect.arousal(), 4),
                            "valence": round(ws.affect.valence(), 4),
                        },
                    })
            else:
                # idle turn
                msg = _build_turn_msg(
                    ws, result.curiosity_state, threshold,
                    turn_idx, "autonomous", "autonomous",
                    f"[thinking: {ws.graph.top_concepts(3)}]",
                    result.response,
                )
                msg["data"]["total_turns"] = total_priming + autonomous_turns
                await websocket.send_json(msg)

            # delay between autonomous turns (longer than priming)
            auto_delay = max(delay_ms, 1000)
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(), timeout=auto_delay / 1000.0,
                )
                client_msg = json.loads(raw)
                if client_msg.get("type") == "set_speed":
                    delay_ms = client_msg["delay_ms"]
            except asyncio.TimeoutError:
                pass

        # ========== DONE ==========
        nodes, edges = _serialize_graph(ws.graph)
        await websocket.send_json({"type": "status", "text": "Reflecting on what it became..."})
        summary = await asyncio.to_thread(
            _generate_personality_summary, ws
        )
        await websocket.send_json({
            "type": "done",
            "data": {
                "total_turns": turn_idx,
                "final_concepts": ws.graph.top_concepts(15),
                "final_arousal": round(ws.affect.arousal(), 4),
                "final_valence": round(ws.affect.valence(), 4),
                "concepts": nodes,
                "edges": edges,
                "personality_summary": summary,
            },
        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "text": str(e)})
        except Exception:
            pass


def _generate_personality_summary(ws):
    """Ask the SLM to describe the personality that formed, in 2-3 sentences."""
    top = ws.graph.top_concepts(10)
    context = ws.context_string()

    prompt = (
        f"{context}\n\n"
        f"You have just finished forming. Your dominant concepts are: {', '.join(top)}.\n\n"
        "In two or three plain sentences, describe the personality that emerged — "
        "what you seem to care about, what kind of thinker you became. "
        "Speak in first person. No hedging. No technical language. "
        "Write as if telling someone who just met you who you are."
    )
    try:
        import ollama as _ollama
        response = _ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"].strip()
    except Exception:
        return f"I kept returning to {', '.join(top[:3])}. Something settled, though I can't fully name it yet."


def _init_topos():
    """Initialize the full TOPOS stack. Runs in a thread."""
    ws = GlobalWorkspace()
    encoder = StateEncoder(ws.embedder)
    forward_model = EnsembleForwardModel(n_members=5, lr=1e-3)

    # init diversity
    scales = [0.5, 0.8, 1.0, 1.5, 2.0]
    for i, mlp in enumerate(forward_model._members):
        scale = scales[i % len(scales)]
        if scale != 1.0:
            with __import__("torch").no_grad():
                for param in mlp.parameters():
                    param.mul_(scale)

    curiosity = CuriositySignal(forward_model, window_k=config.PROGRESS_WINDOW_LIVE)
    threshold = AdaptiveThreshold(window_size=50, percentile=75.0)
    action_proj = ActionProjection()
    exploration_gen = ExplorationGenerator(model_name=config.OLLAMA_MODEL)

    return ws, encoder, curiosity, threshold, exploration_gen, action_proj


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
