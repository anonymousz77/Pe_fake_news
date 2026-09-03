"""Contract tests for the dataset register (data/sources.yaml).

Offline by construction: nothing here touches the network or requires a single
byte of raw data to be on disk.

Two guards carry the weight here.

``test_register_holds_exactly_the_expected_datasets`` asserts the exact name
SET. An earlier version of this file asserted only ``len(entries) == 12``,
which passed happily while four unrequested datasets sat in the register and
four requested ones were missing. A count is not an identity check.

``test_include_never_overlaps_forbidden`` is the leak guard. It exists because
the failure it catches is silent: widen one glob, and held-out evidence quietly
joins the training corpus with no error anywhere. It is checked in both
directions, and the guard's own logic is unit-tested below so a guard that has
stopped guarding cannot pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.paths import RAW, SOURCES_YAML, dataset, dataset_names, raw_dir
from scripts.manifest import (
    BUDGET_GB,
    METHODS,
    REQUIRED_FIELDS,
    audit,
    entry_leak_problems,
    normalise,
    patterns_overlap,
)

#: The correct twelve, and the only twelve.
EXPECTED_NAMES = frozenset(
    {
        "mocheg",
        "factify2",
        "averitec",
        "averimatec",
        "verite",
        "fakeddit",
        "welfake",
        "liar",
        "isot",
        "fakenewsnet",
        "visualnews",
        "newsclippings",
    }
)

DISABLED_BY_DESIGN = {"visualnews", "newsclippings"}
AVERITEC_FORBIDDEN = [
    "data_store/knowledge_store/test/*",          # superseded 15 Nov 2024
    "data_store/knowledge_store/test_updated/*",  # live held-out shard
]


@pytest.fixture(scope="module")
def document() -> dict:
    with SOURCES_YAML.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def entries(document) -> list[dict]:
    return document["datasets"]


# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------


def test_register_exists_and_parses(document):
    assert isinstance(document, dict), f"{SOURCES_YAML} is not a YAML mapping"
    assert "datasets" in document, "register has no top-level 'datasets:' key"
    assert isinstance(document["datasets"], list)


def test_register_holds_exactly_the_expected_datasets(entries):
    """The exact name set — not the count.

    Counting alone cannot tell a correct register from one where the right
    number of wrong datasets have been swapped in.
    """
    actual = {e["name"] for e in entries}
    missing = sorted(EXPECTED_NAMES - actual)
    unexpected = sorted(actual - EXPECTED_NAMES)
    assert not missing and not unexpected, (
        f"register identity mismatch\n"
        f"  missing (requested, absent): {missing}\n"
        f"  unexpected (present, not requested): {unexpected}"
    )


def test_register_has_no_duplicate_entries(entries):
    """Set equality above would pass even if a name appeared twice."""
    names = [e["name"] for e in entries]
    assert len(names) == len(EXPECTED_NAMES), (
        f"expected {len(EXPECTED_NAMES)} entries, got {len(names)}: {names}"
    )
    assert len(set(names)) == len(names), f"duplicate dataset name in {names}"


# --------------------------------------------------------------------------
# shape
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_every_entry_declares_required_field(entries, field):
    """Required means the KEY is present; null is a legitimate value."""
    missing = [e.get("name", "<unnamed>") for e in entries if field not in e]
    assert not missing, f"entries missing '{field}': {missing}"


def test_forbidden_patterns_is_a_list_even_when_empty(entries):
    """Required field, not merely a conventional one. `null` is not `[]`."""
    for entry in entries:
        value = entry["forbidden_patterns"]
        assert isinstance(value, list), (
            f"{entry['name']}: forbidden_patterns must be a list "
            f"(use [] for 'nothing held back'), got {value!r}"
        )


def test_method_is_recognised(entries):
    for entry in entries:
        assert entry["method"] in METHODS, (
            f"{entry['name']}: method {entry['method']!r} not in {sorted(METHODS)}"
        )


def test_repo_type_only_set_for_hf(entries):
    for entry in entries:
        if entry["method"] != "hf":
            assert entry["repo_type"] is None, (
                f"{entry['name']}: repo_type is only meaningful for method 'hf'"
            )


def test_nothing_is_marked_redistributable(entries):
    """False is the conservative default; true needs a verified licence.

    If this test ever fails, the fix is not to change the test — it is to
    confirm the licence by hand and record it in docs/provenance.md.
    """
    for entry in entries:
        assert entry["redistributable"] is False, (
            f"{entry['name']}: redistributable was set true; confirm the "
            "licence in docs/provenance.md before relying on this"
        )


def test_pattern_lists_are_strings(entries):
    for entry in entries:
        for key in ("include_patterns", "forbidden_patterns"):
            for pattern in entry.get(key) or []:
                assert isinstance(pattern, str), (
                    f"{entry['name']}.{key} contains a non-string: {pattern!r} "
                    "(an unquoted leading '*' parses as a YAML alias)"
                )


# --------------------------------------------------------------------------
# expected counts
# --------------------------------------------------------------------------


def test_expected_counts_are_positive_integers(entries):
    for entry in entries:
        expected = entry["expected"]
        if expected is None:
            continue
        assert isinstance(expected, dict), (
            f"{entry['name']}: expected must be a mapping or null"
        )
        for key, value in expected.items():
            assert isinstance(key, str), (
                f"{entry['name']}: expected key {key!r} is {type(key).__name__}, "
                "not str — a bare YAML `true:` key parses as a boolean, so "
                "class labels that spell out true/false/yes/no must be quoted"
            )
            assert isinstance(value, int) and not isinstance(value, bool), (
                f"{entry['name']}.expected[{key!r}] must be an int, got {value!r}"
            )
            assert value > 0, f"{entry['name']}.expected[{key!r}] must be positive"


def test_verite_class_labels_survived_yaml_parsing(entries):
    """Regression: `true:` unquoted would arrive as the boolean True."""
    verite = next(e for e in entries if e["name"] == "verite")
    assert set(verite["expected"]) == {"true", "ooc", "miscaptioned"}


def test_welfake_expected_describes_the_actual_download(entries):
    """`expected` must describe the file we have, not the source's headline.

    The register once carried csv_rows: 78098, which is the source's
    pre-filtering count. The Zenodo distribution already ships the filtered
    subset, so that figure described no file on disk and belongs in notes.
    """
    welfake = next(e for e in entries if e["name"] == "welfake")
    expected = welfake["expected"]
    assert "csv_rows" not in expected, (
        "csv_rows is a property of the source, not of the downloaded file")
    assert expected["rows"] == expected["real"] + expected["fake"], (
        "the class counts must reconcile to the row count")


def test_verite_keeps_the_row_that_differs_from_the_paper(entries):
    """VERITE ships 1,001 rows; the published out-of-context figure is 324.

    Investigated and found genuine: contiguous index, no duplicates, and three
    shipped files agree. The register records what we HAVE. Never drop data to
    match a published number.
    """
    verite = next(e for e in entries if e["name"] == "verite")
    expected = verite["expected"]
    assert expected["ooc"] == 325
    assert sum(expected.values()) == 1001


def test_averimatec_holds_back_its_test_archive(entries):
    averimatec = next(e for e in entries if e["name"] == "averimatec")
    assert averimatec["forbidden_patterns"] == ["test_data.zip"]
    assert averimatec.get("include_patterns"), (
        "an entry that forbids something must declare what it includes, or the "
        "implicit '*' reaches the forbidden path anyway")


# --------------------------------------------------------------------------
# the leak guard
# --------------------------------------------------------------------------


def test_include_never_overlaps_forbidden(entries):
    """No include glob may reach anything a forbidden glob protects.

    Checked in both directions: it is just as wrong for a forbidden pattern to
    swallow an included one as the reverse.
    """
    collisions = []
    for entry in entries:
        collisions.extend(entry_leak_problems(entry))
    assert not collisions, "held-out data reachable via include_patterns:\n" + "\n".join(
        collisions
    )


def test_averitec_holds_back_the_test_knowledge_store(entries):
    """The specific leak this project was bitten by, pinned as a regression."""
    averitec = next(e for e in entries if e["name"] == "averitec")
    assert averitec["forbidden_patterns"] == AVERITEC_FORBIDDEN
    assert averitec["enabled"] is True


def test_absent_includes_count_as_an_implicit_catch_all():
    """An entry with no include_patterns takes everything, forbidden included."""
    problems = entry_leak_problems(
        {
            "name": "synthetic",
            "forbidden_patterns": ["data_store/knowledge_store/test/*"],
        }
    )
    assert problems, "an absent include list must count as '*' against forbidden globs"
    assert "implicit '*'" in problems[0]


def test_absent_includes_are_fine_when_nothing_is_forbidden():
    """The common case: take everything, nothing held back."""
    assert not entry_leak_problems({"name": "synthetic", "forbidden_patterns": []})


def test_entries_without_includes_forbid_nothing(entries):
    """The register's own instance of the rule above."""
    for entry in entries:
        if not entry.get("include_patterns"):
            assert entry["forbidden_patterns"] == [], (
                f"{entry['name']}: declares forbidden_patterns but no "
                "include_patterns, so it would download them anyway"
            )


# ---- the guard's own unit tests: a broken guard must not pass silently ----


@pytest.mark.parametrize(
    "include, forbidden",
    [
        # identical
        ("data_store/knowledge_store/test/*", "data_store/knowledge_store/test/*"),
        # glob swallows the forbidden subtree (fnmatch '*' spans '/')
        ("data_store/knowledge_store/*", "data_store/knowledge_store/test/*"),
        # literal directory prefix, no wildcard on the include side at all
        ("data_store/knowledge_store/", "data_store/knowledge_store/test/*"),
        ("data_store/knowledge_store", "data_store/knowledge_store/test/*"),
        # bare catch-all
        ("*", "data_store/knowledge_store/test/*"),
        # reversed argument order — the relation must be symmetric
        ("data_store/knowledge_store/test/*", "data_store/knowledge_store/*"),
        # separator and './' normalisation must not defeat it
        (r"data_store\knowledge_store\*", "data_store/knowledge_store/test/*"),
        ("./data_store/knowledge_store/*", "data_store/knowledge_store/test/*"),
    ],
)
def test_patterns_overlap_detects_collisions(include, forbidden):
    assert patterns_overlap(include, forbidden), (
        f"{include!r} vs {forbidden!r} should be flagged as overlapping"
    )


@pytest.mark.parametrize(
    "include, forbidden",
    [
        # sibling split directories — the real averitec configuration
        ("data_store/knowledge_store/train/*", "data_store/knowledge_store/test/*"),
        ("data_store/knowledge_store/dev/*", "data_store/knowledge_store/test/*"),
        ("data/train.json", "data_store/knowledge_store/test/*"),
        # 'test' as a prefix of a longer sibling name, not a parent directory
        ("data_store/knowledge_store/testimony/*", "data_store/knowledge_store/test/*"),
        # unrelated files
        ("multimodal_train.tsv", "multimodal_test_public.tsv"),
        ("train.csv", "test_unlabeled.csv"),
    ],
)
def test_patterns_overlap_allows_disjoint_patterns(include, forbidden):
    assert not patterns_overlap(include, forbidden), (
        f"{include!r} vs {forbidden!r} are disjoint but were flagged"
    )


def test_normalise_is_idempotent():
    for pattern in ("./a/b/*", r"a\b\*", "a/b/*"):
        assert normalise(normalise(pattern)) == normalise(pattern) == "a/b/*"


# --------------------------------------------------------------------------
# budget
# --------------------------------------------------------------------------


def test_enabled_datasets_fit_the_budget(entries):
    enabled = [e for e in entries if e["enabled"]]
    total = sum(float(e["est_gb"]) for e in enabled)
    assert total <= BUDGET_GB, (
        f"enabled datasets estimate {total:.2f} GB, over the "
        f"{BUDGET_GB:.0f} GB budget: "
        + ", ".join(f"{e['name']}={e['est_gb']}" for e in enabled)
    )


def test_declared_budget_matches_enforced_budget(document):
    assert float(document["budget_gb"]) == BUDGET_GB, (
        "data/sources.yaml budget_gb and scripts/manifest.py BUDGET_GB have "
        "drifted; raising the budget must be a single deliberate change"
    )


def test_est_gb_is_a_positive_number(entries):
    for entry in entries:
        value = entry["est_gb"]
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        assert value > 0, f"{entry['name']}: est_gb must be positive"


def test_deferred_multimodal_pair_stays_disabled(entries):
    """visualnews + newsclippings are 68 GB and are deferred by design.

    verite is the only enabled out-of-context corpus while they are off, which
    is why enabling them is a budget decision rather than a flag flip.
    """
    by_name = {e["name"]: e for e in entries}
    for name in DISABLED_BY_DESIGN:
        assert name in by_name, f"{name} missing from the register"
        assert by_name[name]["enabled"] is False, (
            f"{name} was enabled; that needs a budget change, not a flag flip"
        )


def test_verite_is_enabled(entries):
    """Load-bearing: the only enabled out-of-context corpus."""
    verite = next(e for e in entries if e["name"] == "verite")
    assert verite["enabled"] is True, (
        "verite carries out-of-context coverage on its own while "
        "visualnews/newsclippings are deferred"
    )


# --------------------------------------------------------------------------
# register <-> paths integration
# --------------------------------------------------------------------------


def test_dataset_names_match_the_file(entries):
    assert list(dataset_names()) == [e["name"] for e in entries]


def test_raw_dir_resolves_under_raw():
    for name in dataset_names():
        assert raw_dir(name) == RAW / name


def test_raw_dir_rejects_unknown_datasets():
    with pytest.raises(KeyError) as excinfo:
        raw_dir("not_a_real_dataset")
    assert "not_a_real_dataset" in str(excinfo.value)


def test_raw_dir_rejects_the_pre_rename_visual_news_spelling():
    """visual_news was renamed to visualnews; the old spelling must not work."""
    with pytest.raises(KeyError):
        raw_dir("visual_news")


def test_register_lookup_is_cached_not_import_time():
    """The register is loaded on first use and reused, never re-read per call."""
    first = dataset("averitec")
    second = dataset("averitec")
    assert first is second, "register entry re-parsed; the lru_cache is not holding"


def test_audit_reports_no_problems():
    """The script's own audit agrees with these tests."""
    problems = audit()
    assert not problems, "manifest.py audit found:\n" + "\n".join(problems)
