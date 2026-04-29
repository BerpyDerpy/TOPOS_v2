# workspace.py - the global workspace for topos
#
# central coordination module. owns the embedder, episodic memory,
# affective state, and concept graph. has a process method
# that does one turn of the cognitive loop.

import numpy as np
import ollama

import config
from config import (
    WORKSPACE_ALPHA, SENTIMENT_SCALE, NER_BOOST_THRESHOLD,
    MEMORY_RECALL_N, TOP_CONCEPTS_N,
)
from embedder import Embedder
from memory import EpisodicMemory
from affect import AffectiveState
from graph import ConceptGraph

from nlp_extractor import extract_concepts

# leftover from before spacy, not used anymore but whatever
STOPWORDS = {
    "about", "between", "feels", "think", "thinking", "there's",
    "something", "every", "really", "would", "could", "should",
    "their", "which", "where", "while", "these", "those", "often",
    "still", "might", "other", "after", "before", "through", "across",
    "under", "being", "doing", "having", "makes", "means",
    "never", "always", "maybe", "almost",
    "since", "until", "along", "above", "below",
    "things", "closer", "certain", "anyone",
    "don't", "can't", "what's", "that's", "here's", "let's",
    "doesn't", "we're", "they're", "won't",
}

# manual sentiment lexicon
# about 35 positive, 35 negative. crude but at least it gives
# nonzero results on domain text where textblob just returns 0

POSITIVE_WORDS = {
    # general
    "beautiful", "elegant", "alive", "fascinating", "hypnotic",
    "love", "great", "best", "perfect", "true",
    "joy", "graceful", "harmonious", "sublime", "powerful",
    "satisfying", "inspired", "brilliant", "remarkable", "wonderful",
    # music + systems domain
    "closest", "resonates", "inevitable", "compelling", "deeply",
    "creative", "comforting", "intriguing", "reliable", "smooth",
    "effortless", "ingenious", "intuitive", "clarity", "warmth",
}

NEGATIVE_WORDS = {
    # general
    "frustrates", "unsatisfying", "brutal", "chaos", "grief",
    "failure", "wrong", "broke", "leak", "bug",
    "crash", "sad", "refuse", "refusing", "strange",
    "worst", "painful", "fear", "terrible", "problem",
    # music + systems domain
    "hardest", "breaks", "pressure", "finite", "broken",
    "frustrating", "difficult", "impossible", "struggling", "confused",
    "fragile", "unpredictable", "latency", "spike", "misaligned",
}


def _lexicon_sentiment(text):
    # returns sentiment between -1 and 1 using simple word counting
    words = text.lower().split()
    if not words:
        return 0.0
    pos = sum(1 for w in words if w.strip(".,;:!?'\"") in POSITIVE_WORDS)
    neg = sum(1 for w in words if w.strip(".,;:!?'\"") in NEGATIVE_WORDS)
    return max(-1.0, min(1.0, (pos - neg) / len(words) * SENTIMENT_SCALE))


class GlobalWorkspace:
    # the main brain. holds all the submodules and runs the cognitive loop.

    def __init__(self):
        self.embedder = Embedder()
        self.memory = EpisodicMemory(self.embedder)
        self.affect = AffectiveState()
        self.graph = ConceptGraph()

        self.state = np.zeros(config.WORKSPACE_DIM, dtype=float)
        self.turn = 0

        # cache last surprise so context_string can use it
        self._last_surprise = 0.0
        # cache last input for memory recall
        self._last_input = ""

    def process(self, user_input):
        # runs one full cognitive turn, returns the context string
        #
        # steps:
        # 1. embed input
        # 2. compute surprise = 1 - similarity(embedding, current state)
        # 3. ema update state toward new embedding
        # 4. extract concepts via nlp
        # 5. update concept graph with concepts + surprise
        # 6. update affect with surprise and sentiment
        # 7. write to memory
        # 8. increment turn
        # 9. return context string

        # 1. embed
        embedding = self.embedder.embed(user_input)

        # 2. surprise
        surprise = 1.0 - self.embedder.similarity(embedding, self.state)
        self._last_surprise = surprise
        self._last_input = user_input

        # 3. ema update
        self.state = (1 - WORKSPACE_ALPHA) * self.state + WORKSPACE_ALPHA * embedding

        # 4. extract concepts via nlp pipeline
        # pass existing graph nodes so adj extraction can anchor to them
        existing = self.graph.node_names()
        weighted_concepts = extract_concepts(user_input, existing_concepts=existing)

        # 4.5 decay all graph weights before adding this turn's contributions
        # makes top-n reflect recent salience, not unbounded history
        self.graph.decay()

        # 5. update the graph
        # batch the strings for co-occurrence edges first
        # then re-boost ner concepts individually
        if weighted_concepts:
            concept_strings = [c.text for c in weighted_concepts]
            self.graph.update(concept_strings, surprise)

            # bonus boost for high weight ner entities
            for wc in weighted_concepts:
                if wc.weight > NER_BOOST_THRESHOLD:
                    self.graph.update([wc.text], surprise * (wc.weight - 1.0))

        # 6. affect update with lexicon sentiment
        sentiment = _lexicon_sentiment(user_input)
        self.affect.update(surprise, sentiment=sentiment)

        # 7. write to memory
        self.memory.write(user_input, {"turn": self.turn})

        # 8. turn counter
        self.turn += 1

        # 9. context string
        return self.context_string()

    def context_string(self):
        # builds a first person context block
        # uses tendency framing so the slm cant smooth it into generic prose

        # recall recent memories relevant to last input
        if self._last_input:
            recalls = self.memory.recall(self._last_input, n=MEMORY_RECALL_N)
            recall_lines = "; ".join(
                f'"{r["text"]}" (turn {r["turn"]})'
                for r in recalls
            )
        else:
            recall_lines = "none yet"

        # top concepts as first person tendencies
        top = self.graph.top_concepts(TOP_CONCEPTS_N)
        if top:
            concepts_line = (
                f"You find yourself drawn to: {', '.join(top)}. "
                "These are not topics, they are how you think."
            )
        else:
            concepts_line = "No strong conceptual tendencies yet."

        # R6: pass continuous affect values directly instead of bucketed
        # mood strings. the slm interprets the numbers; we don't prescribe
        # what arousal=0.7 or valence=-0.3 "feels like" in prose.
        v = self.affect.valence()
        a = self.affect.arousal()

        return (
            "[Workspace state]\n"
            f"Arousal: {a:.2f}  Valence: {v:+.2f}\n"
            f"Surprise this turn: {self._last_surprise:.2f}\n"
            f"{concepts_line}\n"
            f"You keep returning to: {recall_lines}"
        )

    def generate(self, user_input):
        # process the input, build a prompt, get llm response
        context = self.process(user_input)

        system = (
            "You are KITE. Your internal state below defines who you are "
            "right now. Respond from inside that state, your specific "
            "associations, not general ones. Do not deflect to the user. "
            "Do not narrate your state. Speak from it."
        )
        full_prompt = f"{context}\n\nUser: {user_input}\nKITE:"

        try:
            response = ollama.chat(
                model=config.OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": full_prompt},
                ],
            )
            return response["message"]["content"]
        except Exception as e:
            return f"[KITE error: could not reach Ollama {e}]"
