# nlp_extractor.py - spacy based concept extraction
#
# four stage pipeline:
#   1. ner - named entities with label based boosting
#   2. pos - noun/propn tokens only (lemmatised, deduplicated)
#   3. chunks - noun chunk heads for multi word phrases
#   4. adj - qualifying adjectives anchored to extracted or existing concepts
#
# all four streams merge, lemmatise, deduplicate, and return weighted concepts

import spacy
from dataclasses import dataclass

from config import (
    NER_BOOSTED_WEIGHT, NER_DEFAULT_WEIGHT,
    NOUN_CHUNK_WEIGHT, POS_TOKEN_WEIGHT,
    MIN_CONCEPT_CHARS, ACCEPTED_ADJ_DEPS,
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

# pos tags we actually want for stage 2
_ACCEPTED_POS = {"NOUN", "PROPN"}

# dependency relations to skip even for accepted pos
# these are usually grammar filler not real concepts
_EXCLUDED_DEPS = {"det", "poss", "case", "mark", "aux", "auxpass", "cc", "punct"}


@dataclass
class WeightedConcept:
    text: str          # lemmatised, lowercased concept string
    weight: float      # 1.0 baseline, higher for boosted entities
    source: str        # "ner" | "pos" | "chunk" | "adj"


def _get_modified_noun(token):
    """Find the lemma of the noun an adjective modifies, or None.

    Handles four dep relations:
      amod  - direct modifier: "inharmonic partials" -> "partial"
      acomp - predicate adj:   "music is immaterial" -> "music"
      attr  - predicate adj:   "pitch is perceptual" -> "pitch"
      conj  - conjoined adj:   "physical and immaterial" -> traces up to noun
    """
    if token.dep_ == "amod":
        # direct adjectival modifier: head is the noun
        return token.head.lemma_.lower().strip()
    elif token.dep_ in ("acomp", "attr"):
        # predicate adjective: find nsubj of the governing verb
        verb = token.head
        for child in verb.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                return child.lemma_.lower().strip()
    elif token.dep_ == "conj":
        # conjoined with another adjective: trace up to find the noun
        head = token.head
        # walk up through chained conjunctions
        while head.pos_ == "ADJ" and head.dep_ == "conj":
            head = head.head
        if head.dep_ == "amod":
            return head.head.lemma_.lower().strip()
        elif head.dep_ in ("acomp", "attr"):
            verb = head.head
            for child in verb.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    return child.lemma_.lower().strip()
    return None


def extract_concepts(text, min_chars=MIN_CONCEPT_CHARS, existing_concepts=None):
    # returns a deduplicated, weighted list of concepts from the text
    # sorted by weight descending, duplicates collapsed (highest weight wins)
    #
    # existing_concepts: optional set of concept strings already in the graph,
    # used to anchor adjective extraction to established concepts
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

    # stage 4: qualifying adjectives anchored to known concepts
    # only admit ADJ tokens whose dep is in ACCEPTED_ADJ_DEPS and whose
    # modified noun is either already extracted in this turn or already
    # established in the concept graph from previous turns
    anchors = set(seen.keys())
    if existing_concepts:
        anchors |= set(existing_concepts)

    for token in doc:
        if token.pos_ != "ADJ":
            continue
        if token.dep_ not in ACCEPTED_ADJ_DEPS:
            continue
        if token.is_stop or token.is_punct or token.is_space:
            continue
        lemma = token.lemma_.lower().strip()
        if len(lemma) < min_chars:
            continue
        # check that the noun being modified is a known concept
        modified_noun = _get_modified_noun(token)
        if modified_noun and modified_noun in anchors:
            _add(lemma, POS_TOKEN_WEIGHT, "adj")

    # sort by weight descending
    return sorted(seen.values(), key=lambda c: c.weight, reverse=True)


def top_concept_strings(text, n=10):
    # convenience wrapper, just the concept strings ranked by weight
    concepts = extract_concepts(text)
    return [c.text for c in concepts[:n]]