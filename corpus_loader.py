# corpus_loader.py - staged corpus loader for topos validation runs
#
# loads per-stage jsonl files (e.g. corpus/music_stage_01.jsonl through
# corpus/music_stage_10.jsonl), validates each file's structure, and
# exposes a simple iterator over the full sequence with global turn indices.
#
# strict validation — fails loud on any irregularity:
#   - all 10 stage files must be present, contiguous, no gaps
#   - each file must have exactly 50 turns with turn_index 1-50
#   - stage field in each line must match the filename stage number
#   - domain field must be consistent across all files
#   - required fields: turn_index, stage, domain, text

from __future__ import annotations

import json
import glob
import re
from pathlib import Path
from typing import Iterator, Tuple


# each stage has exactly this many turns
TURNS_PER_STAGE = 50
# expect exactly this many stages
EXPECTED_STAGES = 10


class CorpusLoadError(Exception):
    """Raised when corpus files are missing, malformed, or inconsistent."""
    pass


class CorpusLoader:
    """Load and validate a staged corpus, iterate as (global_turn, stage, text).

    Usage:
        loader = CorpusLoader("corpus/music_stage_*.jsonl")
        # or: loader = CorpusLoader.from_directory("corpus", "music")
        for global_turn, stage, text in loader:
            print(global_turn, stage, text[:60])
    """

    _REQUIRED_FIELDS = {"turn_index", "stage", "domain", "text"}

    def __init__(self, pattern: str):
        """Discover and validate stage files matching a glob pattern.

        pattern: a glob like 'corpus/music_stage_*.jsonl'
        """
        self._entries: list[dict] = []
        self._stage_files: list[Path] = []

        # discover files
        matched = sorted(glob.glob(pattern))
        if not matched:
            raise CorpusLoadError(
                f"No files matched pattern: {pattern}"
            )

        self._stage_files = [Path(p) for p in matched]
        self._validate_stage_coverage()
        self._load_and_validate()

    @classmethod
    def from_directory(cls, directory: str, domain: str) -> "CorpusLoader":
        """Convenience: construct from a directory and domain name.

        Equivalent to CorpusLoader("directory/domain_stage_*.jsonl")
        """
        d = Path(directory)
        if not d.is_dir():
            raise CorpusLoadError(f"Not a directory: {directory}")
        pattern = str(d / f"{domain}_stage_*.jsonl")
        return cls(pattern)

    # -------
    # iteration
    # -------

    def __iter__(self) -> Iterator[Tuple[int, int, str]]:
        """Yield (global_turn_index, stage, text) for each turn."""
        for entry in self._entries:
            yield entry["global_turn"], entry["stage"], entry["text"]

    def __len__(self) -> int:
        return len(self._entries)

    # -------
    # validation internals
    # -------

    def _extract_stage_number(self, path: Path) -> int:
        """Extract the stage number from a filename like music_stage_07.jsonl."""
        match = re.search(r"_stage_(\d+)\.jsonl$", path.name)
        if not match:
            raise CorpusLoadError(
                f"Cannot parse stage number from filename: {path.name}  "
                f"Expected format: <domain>_stage_<NN>.jsonl"
            )
        return int(match.group(1))

    def _validate_stage_coverage(self):
        """Check that we have exactly stages 1-10, no gaps, no duplicates."""
        stage_numbers = []
        for path in self._stage_files:
            stage_numbers.append(self._extract_stage_number(path))

        stage_numbers_sorted = sorted(stage_numbers)
        expected = list(range(1, EXPECTED_STAGES + 1))

        if stage_numbers_sorted != expected:
            missing = set(expected) - set(stage_numbers_sorted)
            extra = set(stage_numbers_sorted) - set(expected)
            parts = []
            if missing:
                parts.append(f"missing stages: {sorted(missing)}")
            if extra:
                parts.append(f"unexpected stages: {sorted(extra)}")
            if len(stage_numbers_sorted) != len(set(stage_numbers_sorted)):
                parts.append("duplicate stage numbers detected")
            raise CorpusLoadError(
                f"Stage coverage check failed. {'; '.join(parts)}. "
                f"Found stages: {stage_numbers_sorted}, "
                f"expected: {expected}"
            )

        # re-sort file list by stage number (glob sort may differ
        # from numeric sort for two-digit numbers, but sorted() on
        # zero-padded names like _01, _02 ... _10 works fine.
        # Belt-and-suspenders: sort by extracted stage number.)
        file_by_stage = {
            self._extract_stage_number(p): p for p in self._stage_files
        }
        self._stage_files = [file_by_stage[s] for s in expected]

    def _load_and_validate(self):
        """Load all files, validate every line, build the entry list."""
        observed_domain = None

        for file_idx, path in enumerate(self._stage_files):
            expected_stage = file_idx + 1

            lines = path.read_text().strip().splitlines()
            if not lines:
                raise CorpusLoadError(
                    f"Stage file is empty: {path}"
                )

            if len(lines) != TURNS_PER_STAGE:
                raise CorpusLoadError(
                    f"{path.name}: expected {TURNS_PER_STAGE} turns, "
                    f"got {len(lines)}"
                )

            seen_turn_indices = set()

            for line_num, raw_line in enumerate(lines, 1):
                # parse json
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as e:
                    raise CorpusLoadError(
                        f"{path.name} line {line_num}: invalid JSON: {e}"
                    )

                # required fields
                missing = self._REQUIRED_FIELDS - set(record.keys())
                if missing:
                    raise CorpusLoadError(
                        f"{path.name} line {line_num}: "
                        f"missing required fields: {missing}"
                    )

                turn_index = record["turn_index"]
                stage = record["stage"]
                domain = record["domain"]
                text = record["text"]

                # type checks
                if not isinstance(turn_index, int):
                    raise CorpusLoadError(
                        f"{path.name} line {line_num}: "
                        f"turn_index must be int, got {type(turn_index).__name__}"
                    )
                if not isinstance(stage, int):
                    raise CorpusLoadError(
                        f"{path.name} line {line_num}: "
                        f"stage must be int, got {type(stage).__name__}"
                    )
                if not isinstance(text, str) or not text.strip():
                    raise CorpusLoadError(
                        f"{path.name} line {line_num}: "
                        f"text must be a non-empty string"
                    )

                # stage must match filename
                if stage != expected_stage:
                    raise CorpusLoadError(
                        f"{path.name} line {line_num}: "
                        f"stage field is {stage}, expected {expected_stage} "
                        f"(from filename)"
                    )

                # turn_index must be 1-50
                if turn_index < 1 or turn_index > TURNS_PER_STAGE:
                    raise CorpusLoadError(
                        f"{path.name} line {line_num}: "
                        f"turn_index {turn_index} out of range [1, {TURNS_PER_STAGE}]"
                    )

                # no duplicate turn indices within a stage
                if turn_index in seen_turn_indices:
                    raise CorpusLoadError(
                        f"{path.name} line {line_num}: "
                        f"duplicate turn_index {turn_index}"
                    )
                seen_turn_indices.add(turn_index)

                # domain consistency
                if observed_domain is None:
                    observed_domain = domain
                elif domain != observed_domain:
                    raise CorpusLoadError(
                        f"{path.name} line {line_num}: "
                        f"domain '{domain}' does not match "
                        f"expected domain '{observed_domain}'"
                    )

                # compute global turn index: (stage-1)*50 + turn_index
                global_turn = (expected_stage - 1) * TURNS_PER_STAGE + turn_index

                self._entries.append({
                    "global_turn": global_turn,
                    "stage": expected_stage,
                    "turn_index": turn_index,
                    "domain": domain,
                    "text": text,
                })

            # verify we got all turn indices 1-50
            expected_indices = set(range(1, TURNS_PER_STAGE + 1))
            if seen_turn_indices != expected_indices:
                missing_idx = expected_indices - seen_turn_indices
                raise CorpusLoadError(
                    f"{path.name}: missing turn_index values: "
                    f"{sorted(missing_idx)}"
                )

        # sort by global turn index (should already be in order if
        # turn_index runs 1-50 per file, but be explicit)
        self._entries.sort(key=lambda e: e["global_turn"])

    @property
    def domain(self) -> str:
        """The domain string from the corpus (e.g. 'music')."""
        if not self._entries:
            return "unknown"
        return self._entries[0]["domain"]

    @property
    def total_turns(self) -> int:
        return len(self._entries)

    @property
    def total_stages(self) -> int:
        return EXPECTED_STAGES


# =====================
# standalone test
# =====================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python corpus_loader.py <glob_pattern>")
        print("  e.g. python corpus_loader.py 'corpus/music_stage_*.jsonl'")
        sys.exit(1)

    pattern = sys.argv[1]
    print(f"Loading corpus: {pattern}")

    try:
        loader = CorpusLoader(pattern)
    except CorpusLoadError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"  Domain: {loader.domain}")
    print(f"  Total turns: {loader.total_turns}")
    print(f"  Total stages: {loader.total_stages}")
    print()

    # print first and last 3 turns
    entries = list(loader)
    for label, items in [("First 3:", entries[:3]), ("Last 3:", entries[-3:])]:
        print(f"  {label}")
        for gt, stage, text in items:
            print(f"    global_turn={gt:3d}  stage={stage:2d}  "
                  f"text={text[:70]}...")
    print()
    print("Validation passed.")
