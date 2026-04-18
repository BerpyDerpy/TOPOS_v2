# config.py - all the settings and stuff for topos
#
# basically every number that matters in the system is here
# if you change something for an experiment write it down
# in experiments.md first or you'll forget

# model stuff
EMBED_MODEL       = "all-MiniLM-L6-v2"
OLLAMA_MODEL      = "qwen2.5:7b"

# how big things are
WORKSPACE_DIM     = 384       # has to match whatever the embed model spits out
AFFECT_DIM        = 64

# workspace ema
WORKSPACE_ALPHA   = 0.4       # blend rate: state = (1-a)*state + a*embedding

# affect stuff
AFFECT_ALPHA      = 0.3       # how much new stuff matters
AFFECT_DECAY      = 0.95      # how fast it fades back to zero
AROUSAL_HIGH      = 0.6       # when we call it "high"
AROUSAL_MODERATE  = 0.3       # when we call it "moderate"
VALENCE_POS       = 0.2       # positive vibes threshold
VALENCE_NEG       = -0.2      # negative vibes threshold

# sentiment
SENTIMENT_SCALE   = 8.0       # magic number for lexicon normalisation

# graph
MAX_GRAPH_NODES   = 100       # chop the weakest node if we go over this
NER_BOOST_THRESHOLD = 1.5     # ner concepts above this get extra graph juice

# nlp extractor weights
NER_BOOSTED_WEIGHT  = 1.8     # good ner labels get this
NER_DEFAULT_WEIGHT  = 1.2     # other ner entities get this
NOUN_CHUNK_WEIGHT   = 1.1     # noun chunks
POS_TOKEN_WEIGHT    = 1.0     # basic nouns
MIN_CONCEPT_CHARS   = 3       # anything shorter is probably junk

# memory
MEMORY_COLLECTION = "episodic"
MEMORY_RECALL_N   = 2         # how many memories to pull back
TOP_CONCEPTS_N    = 5         # how many top concepts for the context string

# not used yet but keeping it around
SURPRISE_THRESHOLD = 0.35     # maybe for gating memory writes later idk
