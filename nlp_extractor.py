# nlp_extractor.py - spacy based concept extraction
#
# three stage pipeline:
#   1. ner - named entities with label based boosting
#   2. pos - noun/propn tokens only (lemmatised, deduplicated)
#   3. chunks - noun chunk heads for multi word phrases
#
# all three streams merge, lemmatise, deduplicate, and return weighted concepts

import spacy
from dataclasses import dataclass

from config import (
    NER_BOOSTED_WEIGHT, NER_DEFAULT_WEIGHT,
    NOUN_CHUNK_WEIGHT, POS_TOKEN_WEIGHT,
    MIN_CONCEPT_CHARS,
)

# load once at module level - this is expensive so dont reload every call
# python -m spacy download en_core_web_sm
_NLP = None

def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


# ner labels we care about more - they get a score boost
_BOOSTED_ENTITY_LABELS = {
    "ORG",      # organisations, frameworks, companies
    "PRODUCT",  # named products/tools
    "GPE",      # geopolitical stuff
    "EVENT",    # conferences etc
    "LAW",      # standards and specs
    "WORK_OF_ART",  # titles for music/culture turns
    "LANGUAGE", # programming languages
}

# pos tags we actually want
_ACCEPTED_POS = {"NOUN", "PROPN"}

# dependency relations to skip even for accepted pos
# these are usually grammar filler not real concepts
_EXCLUDED_DEPS = {"det", "poss", "case", "mark", "aux", "auxpass", "cc", "punct"}


@dataclass
class WeightedConcept:
    text: str          # lemmatised, lowercased concept string
    weight: float      # 1.0 baseline, higher for boosted entities
    source: str        # "ner" | "pos" | "chunk"


def extract_concepts(text, min_chars=MIN_CONCEPT_CHARS):
    # returns a deduplicated, weighted list of concepts from the text
    # sorted by weight descending, duplicates collapsed (highest weight wins)
    nlp = _get_nlp()
    doc = nlp(text)

    seen = {}

    def _add(raw, weight, source):
        key = raw.lower().strip()
        if len(key) < min_chars:
            return
        if key in seen:
            # keep the highest weight, prefer ner source
            if weight > seen[key].weight:
                seen[key] = WeightedConcept(key, weight, source)
        else:
            seen[key] = WeightedConcept(key, weight, source)

    # stage 1: named entities
    for ent in doc.ents:
        boost = NER_BOOSTED_WEIGHT if ent.label_ in _BOOSTED_ENTITY_LABELS else NER_DEFAULT_WEIGHT
        # use entity root lemma for normalisation
        _add(ent.root.lemma_, boost, "ner")

    # stage 2: pos filtering on individual tokens
    for token in doc:
        if token.pos_ not in _ACCEPTED_POS:
            continue
        if token.dep_ in _EXCLUDED_DEPS:
            continue
        if token.is_stop or token.is_punct or token.is_space:
            continue
        _add(token.lemma_, POS_TOKEN_WEIGHT, "pos")

    # stage 3: noun chunk heads (multi word concepts)
    for chunk in doc.noun_chunks:
        # head token lemma is the most important word
        head = chunk.root
        if head.is_stop:
            continue
        # full chunk text normalised, useful for compound terms
        chunk_text = " ".join(t.lemma_ for t in chunk
                              if not t.is_stop and not t.is_punct)
        if chunk_text:
            _add(chunk_text, NOUN_CHUNK_WEIGHT, "chunk")  # slight boost for multi word

    # sort by weight descending
    return sorted(seen.values(), key=lambda c: c.weight, reverse=True)


def top_concept_strings(text, n=10):
    # convenience wrapper, just the concept strings ranked by weight
    concepts = extract_concepts(text)
    return [c.text for c in concepts[:n]]