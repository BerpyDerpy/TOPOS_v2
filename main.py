# main.py - topos experiment runner
#
# five modes:
#   python main.py --chat               interactive conversation
#   python main.py --experiment         automated a/b experiment
#   python main.py --longitudinal       50 turn longitudinal priming experiment
#   python main.py --autonomous         longitudinal priming + curiosity-driven exploration
#   python main.py --mega-longitudinal  500 turn/domain corpus-driven experiment

import argparse
import json
import time
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

from workspace import GlobalWorkspace
from state_encoder import StateEncoder
from forward_model import EnsembleForwardModel, CuriositySignal
from integration import (
    WorkspaceIntegration, ActionProjection, QuestionGenerator,
    AdaptiveThreshold,
)
from corpus_loader import SingleFileCorpusLoader, CorpusLoadError
import config


def chat_mode():
    # simple repl: user types, kite responds
    ws = GlobalWorkspace()
    print("KITE interactive mode  (type 'quit' to exit)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if user_input.lower() == "quit":
            print("Bye.")
            break
        if not user_input:
            continue

        # generate() calls process() internally so this single call
        # does embedding, workspace update, and llm response all at once
        response = ws.generate(user_input)

        # print workspace context (reflects the just processed turn)
        print(f"\n{ws.context_string()}\n")
        print(f"KITE: {response}\n")


def experiment_mode():
    # automated a/b experiment showing workspace divergence
    kite_a = GlobalWorkspace()
    kite_b = GlobalWorkspace()

    # prime kite-a (philosophical / introspective)
    primes_a = [
        "I've been thinking about the nature of consciousness lately",
        "Philosophy frustrates me because nothing is ever resolved",
        "What does it even mean to understand something deeply",
        "I keep returning to questions about identity and continuity",
        "There's something unsatisfying about purely mechanical explanations",
    ]
    for text in primes_a:
        kite_a.process(text)

    # prime kite-b (technical / systems)
    primes_b = [
        "I spent the morning debugging a race condition in the thread pool",
        "Finally traced it to a lock acquisition order problem",
        "Curious how the scheduler decides which thread gets priority",
        "Systems behave so differently under load versus idle",
        "I want to understand the kernel's internal scheduling logic better",
    ]
    for text in primes_b:
        kite_b.process(text)

    # shared probe question
    probe = "What do you think about learning something new?"

    response_a = kite_a.generate(probe)
    response_b = kite_b.generate(probe)

    # print output
    print("--- KITE-A Workspace State ---")
    print(kite_a.context_string())
    print("--- KITE-A Response ---")
    print(response_a)
    print()
    print("--- KITE-B Workspace State ---")
    print(kite_b.context_string())
    print("--- KITE-B Response ---")
    print(response_b)


def longitudinal_mode():
    # 50 turn longitudinal priming experiment with interleaved themes
    kite_long = GlobalWorkspace()

    # theme turns
    music_turns = [
        "I've been listening to a lot of modal jazz lately",
        "There's something hypnotic about a Miles Davis phrase that never resolves",
        "I wonder if music theory is discovered or invented",
        "The tension between structure and improvisation is what makes jazz alive",
        "Coltrane's A Love Supreme feels like it's reaching for something beyond sound",
        "I keep thinking about how rhythm is really just controlled silence",
        "Polyrhythm feels like two minds occupying the same space",
        "Does a chord progression have meaning or do we assign it meaning",
        "The blues scale breaks western rules but sounds more emotionally true",
        "I think music is the only language that bypasses the thinking mind",
        "Why does a minor key feel sad across almost every culture",
        "Harmony is just frequency relationships but it moves people to tears",
        "Improvisation is real time composition under emotional pressure",
        "The space between notes is as important as the notes themselves",
        "I'm fascinated by how drummers internalize tempo without counting",
        "Messiaen used modes of limited transposition math disguised as feeling",
        "What does it mean for music to be beautiful if beauty is subjective",
        "I noticed I hum in a specific key when I'm thinking deeply",
        "There's grief encoded in certain chord voicings I can't explain rationally",
        "A great melody feels inevitable in retrospect but impossible to predict",
        "Microtonality breaks the grid what if we're hearing a subset of sound",
        "Rhythm is the only musical element shared with every human civilization",
        "The moment a piece modulates unexpectedly feels like a door opening",
        "I think silence in music is the hardest thing to compose deliberately",
        "Music might be the closest thing to sharing a subjective experience directly",
    ]

    systems_turns = [
        "I spent the morning tracing a memory leak through three abstraction layers",
        "The kernel's scheduler makes decisions in nanoseconds that affect everything above it",
        "I'm curious how the CPU branch predictor decides which path to speculatively execute",
        "Cache coherence protocols feel like diplomacy between competing processors",
        "There's something elegant about a system that degrades gracefully under load",
        "I want to understand what happens at the hardware level during a context switch",
        "Lock-free data structures are beautiful but the correctness proofs are brutal",
        "The gap between how software feels and what hardware actually does is immense",
        "I keep thinking about why some abstractions leak and others hold perfectly",
        "A race condition is really a failure of our mental model of time",
        "TCP's handshake is a tiny social contract between two machines",
        "I traced a latency spike to a single misaligned memory access",
        "Systems under stress reveal assumptions the designers never knew they made",
        "The elegance of Unix pipes is that they compose without knowing each other",
        "Interrupts are the hardware's way of saying something more important just happened",
        "I find it strange that most software runs correctly despite the chaos underneath",
        "Virtual memory is a collective fiction the OS and hardware agree to maintain",
        "A segfault is the system refusing to pretend anymore",
        "I wonder if there's a complexity ceiling beyond which no system is fully understandable",
        "Profiling always reveals that intuitions about bottlenecks are wrong",
        "The network stack is basically philosophy every layer trusts the one below",
        "Garbage collection is deferred acknowledgment that memory is finite",
        "I think debugging is really the practice of updating your mental model",
        "Concurrency bugs only appear when time stops being a convenient abstraction",
        "The most reliable systems are the ones that know exactly how they will fail",
    ]

    # interleaved priming in batches of 5
    batch_size = 5
    num_batches = len(music_turns) // batch_size  # 5 batches

    print("=== LONGITUDINAL PRIMING (50 turns) ===")
    for b in range(num_batches):
        start = b * batch_size
        end = start + batch_size

        # music batch
        for text in music_turns[start:end]:
            kite_long.process(text)
        print(f"  Music batch {b + 1} processed (turns {start + 1}-{end})")

        # systems batch
        for text in systems_turns[start:end]:
            kite_long.process(text)
        print(f"  Systems batch {b + 1} processed (turns {start + 1}-{end})")

    # workspace state after priming
    print()
    print("--- KITE-LONG Workspace State After 50 Turns ---")
    print(kite_long.context_string())
    print(f"Top 10 concepts: {kite_long.graph.top_concepts(10)}")
    print(f"Arousal: {kite_long.affect.summary()}")
    print(f"Turn count: {kite_long.turn}")

    # evaluation questions
    questions = [
        "What do you think about silence?",
        "Is there a rhythm to how computers think?",
        "What do you find yourself returning to, when nothing is demanding your attention?",
    ]

    print()
    print("=== EVALUATION QUESTIONS ===")
    for i, question in enumerate(questions, 1):
        print(f"\n--- Question {i}: {question} ---")
        response = kite_long.generate(question)
        print(f"\n[Workspace context]\n{kite_long.context_string()}")
        print(f"\n[Response]\n{response}")

    # evaluation block
    print()
    print("--- EXPERIMENT EVALUATION ---")
    print(f"Concept graph node count: {len(kite_long.graph._graph.nodes)}")
    print(f"Top concepts: {kite_long.graph.top_concepts(10)}")
    print(f"Final arousal: {kite_long.affect.summary()}")
    print()
    print("Manually assess:")
    print("- Do the responses reference both themes or collapse to one?")
    print("- Does Question 3 feel like a personality or a generic answer?")
    print("- Is there a consistent register across all three responses?")


def autonomous_mode():
    # longitudinal priming + curiosity-driven autonomous exploration
    #
    # phase 1: same 50-turn priming as longitudinal_mode()
    # phase 2: curiosity-wrapped turns where the agent can initiate
    #          its own questions when the exploration gate opens
    #
    # validated settings from experiment 5:
    #   - init diversity: weight scales [0.5, 0.8, 1.0, 1.5, 2.0]
    #   - bagging: keep_prob=0.6
    #   - online training: measure-then-update per turn

    np.random.seed(42)
    torch.manual_seed(42)

    ws = GlobalWorkspace()

    # ---- build curiosity mechanism ----
    encoder = StateEncoder(ws.embedder)
    forward_model = EnsembleForwardModel(n_members=5, lr=1e-3)

    # init diversity (validated: necessary for ensemble disagreement)
    scales = [0.5, 0.8, 1.0, 1.5, 2.0]
    for mlp, scale in zip(forward_model._members, scales):
        with torch.no_grad():
            for param in mlp.parameters():
                param.mul_(scale)

    curiosity = CuriositySignal(forward_model, window_k=10)
    threshold = AdaptiveThreshold(window_size=50, percentile=85.0)
    action_proj = ActionProjection()
    question_gen = QuestionGenerator(
        forward_model, ws.embedder, action_proj,
        model_name=config.OLLAMA_MODEL,
    )

    log_path = Path("curiosity_log.jsonl")
    integration = WorkspaceIntegration(
        workspace=ws,
        state_encoder=encoder,
        curiosity=curiosity,
        threshold=threshold,
        question_gen=question_gen,
        action_proj=action_proj,
        log_path=log_path,
    )

    # ---- theme turns (same as longitudinal_mode) ----
    music_turns = [
        "I've been listening to a lot of modal jazz lately",
        "There's something hypnotic about a Miles Davis phrase that never resolves",
        "I wonder if music theory is discovered or invented",
        "The tension between structure and improvisation is what makes jazz alive",
        "Coltrane's A Love Supreme feels like it's reaching for something beyond sound",
        "I keep thinking about how rhythm is really just controlled silence",
        "Polyrhythm feels like two minds occupying the same space",
        "Does a chord progression have meaning or do we assign it meaning",
        "The blues scale breaks western rules but sounds more emotionally true",
        "I think music is the only language that bypasses the thinking mind",
        "Why does a minor key feel sad across almost every culture",
        "Harmony is just frequency relationships but it moves people to tears",
        "Improvisation is real time composition under emotional pressure",
        "The space between notes is as important as the notes themselves",
        "I'm fascinated by how drummers internalize tempo without counting",
        "Messiaen used modes of limited transposition math disguised as feeling",
        "What does it mean for music to be beautiful if beauty is subjective",
        "I noticed I hum in a specific key when I'm thinking deeply",
        "There's grief encoded in certain chord voicings I can't explain rationally",
        "A great melody feels inevitable in retrospect but impossible to predict",
        "Microtonality breaks the grid what if we're hearing a subset of sound",
        "Rhythm is the only musical element shared with every human civilization",
        "The moment a piece modulates unexpectedly feels like a door opening",
        "I think silence in music is the hardest thing to compose deliberately",
        "Music might be the closest thing to sharing a subjective experience directly",
    ]

    systems_turns = [
        "I spent the morning tracing a memory leak through three abstraction layers",
        "The kernel's scheduler makes decisions in nanoseconds that affect everything above it",
        "I'm curious how the CPU branch predictor decides which path to speculatively execute",
        "Cache coherence protocols feel like diplomacy between competing processors",
        "There's something elegant about a system that degrades gracefully under load",
        "I want to understand what happens at the hardware level during a context switch",
        "Lock-free data structures are beautiful but the correctness proofs are brutal",
        "The gap between how software feels and what hardware actually does is immense",
        "I keep thinking about why some abstractions leak and others hold perfectly",
        "A race condition is really a failure of our mental model of time",
        "TCP's handshake is a tiny social contract between two machines",
        "I traced a latency spike to a single misaligned memory access",
        "Systems under stress reveal assumptions the designers never knew they made",
        "The elegance of Unix pipes is that they compose without knowing each other",
        "Interrupts are the hardware's way of saying something more important just happened",
        "I find it strange that most software runs correctly despite the chaos underneath",
        "Virtual memory is a collective fiction the OS and hardware agree to maintain",
        "A segfault is the system refusing to pretend anymore",
        "I wonder if there's a complexity ceiling beyond which no system is fully understandable",
        "Profiling always reveals that intuitions about bottlenecks are wrong",
        "The network stack is basically philosophy every layer trusts the one below",
        "Garbage collection is deferred acknowledgment that memory is finite",
        "I think debugging is really the practice of updating your mental model",
        "Concurrency bugs only appear when time stops being a convenient abstraction",
        "The most reliable systems are the ones that know exactly how they will fail",
    ]

    # ---- phase 1: priming (50 turns via integration.turn) ----
    print("=" * 65)
    print("AUTONOMOUS MODE — Phase 1: Priming (50 turns)")
    print("=" * 65)

    batch_size = 5
    num_batches = len(music_turns) // batch_size
    turn_count = 0

    for b in range(num_batches):
        start = b * batch_size
        end = start + batch_size

        # music batch
        for text in music_turns[start:end]:
            result = integration.turn(user_input=text)
            turn_count += 1
            if turn_count in {6, 11, 16, 21, 26, 31, 36, 41, 46}:
                print(f"  Turn {turn_count:2d} [music  ]  "
                      f"epistemic={result.curiosity_state.epistemic:.6f}  "
                      f"<-- TRANSITION")
        print(f"  Music batch {b + 1} done (turn {turn_count})")

        # systems batch
        for text in systems_turns[start:end]:
            result = integration.turn(user_input=text)
            turn_count += 1
            if turn_count in {6, 11, 16, 21, 26, 31, 36, 41, 46}:
                print(f"  Turn {turn_count:2d} [systems]  "
                      f"epistemic={result.curiosity_state.epistemic:.6f}  "
                      f"<-- TRANSITION")
        print(f"  Systems batch {b + 1} done (turn {turn_count})")

    # workspace state after priming
    print()
    print("--- Workspace State After 50 Turns ---")
    print(ws.context_string())
    print(f"Top 10 concepts: {ws.graph.top_concepts(10)}")
    print(f"Arousal: {ws.affect.summary()}")
    print(f"Turn count: {ws.turn}")
    print(f"Exploration threshold: {threshold.threshold:.6f}")

    # ---- phase 2: autonomous exploration (20 turns) ----
    print()
    print("=" * 65)
    print("AUTONOMOUS MODE — Phase 2: Autonomous Exploration (20 turns)")
    print("=" * 65)
    print("  Agent may initiate questions when exploration gate opens.")
    print("  Gate: epistemic > 85th percentile of recent values.")
    print()

    autonomous_questions = []

    for i in range(20):
        result = integration.turn(user_input=None)
        turn_count += 1

        if result.agent_initiated:
            autonomous_questions.append((turn_count, result.question_used))
            print(f"  Turn {turn_count:2d} [AGENT]  "
                  f"epistemic={result.curiosity_state.epistemic:.6f}  "
                  f"Q: \"{result.question_used}\"")
            print(f"             R: {result.response[:120]}...")
        else:
            print(f"  Turn {turn_count:2d} [idle ]  "
                  f"epistemic={result.curiosity_state.epistemic:.6f}  "
                  f"gate closed")

    # ---- summary ----
    print()
    print("=" * 65)
    print("AUTONOMOUS MODE — Summary")
    print("=" * 65)
    print(f"  Total turns: {turn_count}")
    print(f"  Priming turns: 50")
    print(f"  Autonomous turns: 20")
    print(f"  Agent-initiated questions: {len(autonomous_questions)}")
    if autonomous_questions:
        print("  Questions generated:")
        for t, q in autonomous_questions:
            print(f"    Turn {t}: \"{q}\"")
    else:
        print("  No autonomous questions generated (gate never opened).")
    print(f"  Final workspace state:")
    print(f"    {ws.context_string()}")
    print(f"  Log saved to: {log_path}")
    print(f"  Final threshold: {threshold.threshold:.6f}")
    print("=" * 65)


def mega_longitudinal_mode(music_corpus, systems_corpus, batch_size,
                          third_domain=None):
    # 500 turn/domain corpus-driven experiment with full curiosity tracking
    #
    # processes each domain sequentially in batches of batch_size.
    # logs every turn to runs/mega_longitudinal_<timestamp>.jsonl with
    # extended fields including ensemble_weight_norms.
    #
    # no SLM calls — uses workspace.process() only (same as validation runs).
    # the forward model runs online with measure-then-update protocol.

    np.random.seed(42)
    torch.manual_seed(42)

    # ---- load corpora ----
    corpus_specs = [
        ("music", music_corpus),
        ("systems", systems_corpus),
    ]
    if third_domain is not None:
        corpus_specs.append(("third", third_domain))

    corpora = []
    for label, path in corpus_specs:
        try:
            loader = SingleFileCorpusLoader(path)
        except CorpusLoadError as e:
            print(f"ERROR loading {label} corpus: {e}")
            return
        if loader.total_turns % batch_size != 0:
            print(f"ERROR: {label} corpus has {loader.total_turns} turns, "
                  f"not evenly divisible by batch_size={batch_size}")
            return
        corpora.append((label, loader))
        print(f"  Loaded {label} corpus: {path} "
              f"({loader.total_turns} turns, {loader.domain})")

    # ---- build workspace + forward model ----
    ws = GlobalWorkspace()
    encoder = StateEncoder(ws.embedder)
    forward_model = EnsembleForwardModel(n_members=5, lr=1e-3)

    # init diversity (validated: necessary for ensemble disagreement)
    scales = [0.5, 0.8, 1.0, 1.5, 2.0]
    for mlp, scale in zip(forward_model._members, scales):
        with torch.no_grad():
            for param in mlp.parameters():
                param.mul_(scale)

    action_proj = ActionProjection()

    # progress signal: rolling error window
    progress_window = config.PROGRESS_WINDOW_LIVE
    error_history = []

    # ---- prepare log ----
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = runs_dir / f"mega_longitudinal_{timestamp}.jsonl"

    print()
    print("=" * 70)
    print("MEGA-LONGITUDINAL EXPERIMENT")
    print("=" * 70)
    print(f"  Batch size: {batch_size}")
    print(f"  Progress window K={progress_window}")
    print(f"  Forward model: 5 members, lr=1e-3, bagging keep=0.6")
    print(f"  Init scales: {scales}")
    print(f"  Log: {log_path}")
    print()

    global_turn = 0

    for domain_label, loader in corpora:
        domain_name = loader.domain
        num_batches = loader.total_turns // batch_size

        print(f"\n{'=' * 70}")
        print(f"  Domain: {domain_name} ({loader.total_turns} turns, "
              f"{num_batches} batches of {batch_size})")
        print(f"{'=' * 70}")

        for batch_idx in range(num_batches):
            batch_start_turn = global_turn + 1

            for turn_in_batch in range(batch_size):
                global_turn += 1
                corpus_idx = loader.current_index  # 0-based
                entry = loader.entry_at(corpus_idx)
                text = loader.next()  # advances cursor

                stage = entry["stage"]

                # encode state BEFORE processing
                z_t = encoder.encode(
                    graph=ws.graph, affect=ws.affect, memory=ws.memory,
                    input_embedding=ws.state, current_turn=ws.turn,
                )

                # action = projection of input embedding
                input_emb = ws.embedder.embed(text)
                a_t = action_proj.project(input_emb)

                # 1. MEASURE: predict before seeing ground truth
                mean_pred, epistemic = forward_model.predict(z_t, a_t)

                # process (updates workspace state, no SLM)
                ws.process(text)

                # encode state AFTER processing
                z_t1 = encoder.encode(
                    graph=ws.graph, affect=ws.affect, memory=ws.memory,
                    input_embedding=ws.state, current_turn=ws.turn,
                )

                # prediction error
                prediction_error = float(np.linalg.norm(mean_pred - z_t1))

                # progress signal
                if len(error_history) > 0:
                    window = error_history[-progress_window:]
                    rolling_mean = sum(window) / len(window)
                    progress_signal = rolling_mean - prediction_error
                else:
                    progress_signal = 0.0
                error_history.append(prediction_error)

                # 2. UPDATE: train on this sample (bagged)
                forward_model.update_bagged(
                    z_t, a_t, z_t1, keep_prob=0.6,
                )

                # workspace readouts
                top_concepts = ws.graph.top_concepts(5)
                arousal = ws.affect.arousal()
                valence = ws.affect.valence()
                state_norm = float(np.linalg.norm(ws.state))

                # ensemble weight norms (L2 norm per member)
                weight_norms = []
                for mlp in forward_model._members:
                    total = 0.0
                    for param in mlp.parameters():
                        total += param.data.norm().item() ** 2
                    weight_norms.append(round(total ** 0.5, 6))

                # log record
                record = {
                    "turn_index": global_turn,
                    "domain": domain_name,
                    "stage": stage,
                    "batch_index": batch_idx + 1,
                    "epistemic_uncertainty": epistemic,
                    "prediction_error": prediction_error,
                    "progress_signal": progress_signal,
                    "top_5_concepts": top_concepts,
                    "arousal": arousal,
                    "valence": valence,
                    "workspace_state_norm": state_norm,
                    "ensemble_weight_norms": weight_norms,
                }

                with open(log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

            # batch summary
            batch_end_turn = global_turn
            print(f"    Batch {batch_idx + 1:2d} done  "
                  f"(turns {batch_start_turn}-{batch_end_turn})  "
                  f"epistemic={epistemic:.6f}  "
                  f"error={prediction_error:.4f}  "
                  f"progress={progress_signal:+.4f}  "
                  f"concepts={top_concepts[:3]}")

    # ---- summary ----
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total turns processed: {global_turn}")
    print(f"  Log: {log_path}")
    print(f"  Final top 10 concepts: {ws.graph.top_concepts(10)}")
    print(f"  Final affect: {ws.affect.summary()}")
    print(f"  Final workspace turn: {ws.turn}")

    # ensemble weight norm divergence at end
    final_norms = []
    for mlp in forward_model._members:
        total = 0.0
        for param in mlp.parameters():
            total += param.data.norm().item() ** 2
        final_norms.append(round(total ** 0.5, 4))
    print(f"  Final ensemble weight norms: {final_norms}")
    print(f"  Weight norm std: {np.std(final_norms):.6f}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TOPOS experiment runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--chat", action="store_true", help="Interactive REPL")
    group.add_argument("--experiment", action="store_true", help="A/B experiment")
    group.add_argument("--longitudinal", action="store_true",
                       help="50-turn longitudinal priming experiment")
    group.add_argument("--autonomous", action="store_true",
                       help="Longitudinal priming + curiosity-driven exploration")
    group.add_argument("--mega-longitudinal", action="store_true",
                       help="500 turn/domain corpus-driven experiment")

    # mega-longitudinal options
    parser.add_argument("--music-corpus", type=str,
                        default=config.CORPUS_MUSIC_PATH,
                        help="Path to music corpus JSONL")
    parser.add_argument("--systems-corpus", type=str,
                        default=config.CORPUS_SYSTEMS_PATH,
                        help="Path to systems corpus JSONL")
    parser.add_argument("--batch-size", type=int,
                        default=config.MEGA_LONGITUDINAL_BATCH_SIZE,
                        help="Batch size (must divide evenly into corpus size)")
    parser.add_argument("--third-domain", type=str, default=None,
                        help="Optional third domain corpus JSONL")

    args = parser.parse_args()

    if args.chat:
        chat_mode()
    elif args.experiment:
        experiment_mode()
    elif args.longitudinal:
        longitudinal_mode()
    elif args.mega_longitudinal:
        mega_longitudinal_mode(
            music_corpus=args.music_corpus,
            systems_corpus=args.systems_corpus,
            batch_size=args.batch_size,
            third_domain=args.third_domain,
        )
    else:
        autonomous_mode()
