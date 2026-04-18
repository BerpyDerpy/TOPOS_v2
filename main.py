# main.py - topos experiment runner
#
# three modes:
#   python main.py --chat          interactive conversation
#   python main.py --experiment    automated a/b experiment
#   python main.py --longitudinal  50 turn longitudinal priming experiment

#todo: add mega longitude mode with 100+ turn priming and test with new nlp stuff
import argparse
from workspace import GlobalWorkspace


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TOPOS experiment runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--chat", action="store_true", help="Interactive REPL")
    group.add_argument("--experiment", action="store_true", help="A/B experiment")
    group.add_argument("--longitudinal", action="store_true",
                       help="50-turn longitudinal priming experiment")
    args = parser.parse_args()

    if args.chat:
        chat_mode()
    elif args.experiment:
        experiment_mode()
    else:
        longitudinal_mode()
