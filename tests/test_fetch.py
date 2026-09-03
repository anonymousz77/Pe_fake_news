"""Offline tests for the acquisition layer.

Nothing here touches the network, and nothing requires a byte of real data.
Every gate that protects the disk or the held-out splits is exercised against
synthetic entries, because a guard that has only ever been observed passing on
the real register is a guard nobody has actually tested.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.paths import dataset  # noqa: E402
from scripts import fetch, fetchlib  # noqa: E402
from scripts.fetchlib import (  # noqa: E402
    BIBLE_RESERVE_GB,
    GB,
    PROJECTED_CAP_GB,
    WORKING_RESERVE_GB,
    CredentialError,
    DiskBudgetError,
    DiskProjection,
    FetchError,
    ForbiddenPathError,
    HydrationReport,
    allow_patterns_for,
    backoff_delays,
    check_free_space,
    check_projection,
    check_selection_size,
    completion_status,
    download_resumable,
    find_forbidden,
    guard_or_abort,
    select_included,
    should_retry,
    stratified_indices,
    write_marker,
)


def entry(**overrides):
    """A minimal synthetic register entry."""
    base = {
        "name": "synthetic",
        "method": "git",
        "url": "example.com/repo",
        "include_patterns": ["data/*"],
        "forbidden_patterns": [],
        "est_gb": 1.0,
        "enabled": True,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# the disk model
# --------------------------------------------------------------------------


def test_projection_sums_reserves():
    p = DiskProjection(raw_now_gb=10.0, remaining_gb=20.0, pending=("a", "b"))
    assert p.total_gb == 10.0 + 20.0 + BIBLE_RESERVE_GB + WORKING_RESERVE_GB
    assert p.ok is True
    assert p.over_by_gb == 0.0


def test_projection_knows_when_it_is_over():
    p = DiskProjection(raw_now_gb=100.0, remaining_gb=50.0, pending=("big",))
    assert p.ok is False
    assert p.over_by_gb == pytest.approx(p.total_gb - PROJECTED_CAP_GB)


def test_real_register_fits_under_the_cap(monkeypatch):
    """The register's own arithmetic fits, before any data is on disk.

    Deliberately pinned to an empty disk rather than the live tree: walking
    ~100k real files made this test take two minutes, and the live figure is
    an operational check that `fetch.py --dry-run` and `manifest.py audit`
    already print.
    """
    monkeypatch.setattr(fetchlib, "dir_bytes", lambda _root: 0)
    projection = fetchlib.project_disk(fetchlib.enabled_names())
    assert projection.ok, projection.render()


def test_projection_does_not_double_count_data_already_on_disk(monkeypatch):
    """est_gb already spent must not be charged again as 'remaining'."""
    monkeypatch.setattr(fetchlib, "est_gb", lambda _name: 10.0)
    monkeypatch.setattr(fetchlib, "raw_dir", lambda name: Path("raw") / name)
    # 8 GB of this dataset's 10 GB estimate is already downloaded
    monkeypatch.setattr(fetchlib, "dir_bytes", lambda _root: int(8 * GB))
    assert fetchlib.remaining_gb("solo") == pytest.approx(2.0)

    projection = fetchlib.project_disk(["solo"])
    # raw_now 8 + remaining 2 + reserves 50 = 60, not 8 + 10 + 50 = 68
    assert projection.total_gb == pytest.approx(60.0)


def test_remaining_never_goes_negative(monkeypatch):
    monkeypatch.setattr(fetchlib, "est_gb", lambda _name: 1.0)
    monkeypatch.setattr(fetchlib, "raw_dir", lambda name: Path("raw") / name)
    monkeypatch.setattr(fetchlib, "dir_bytes", lambda _root: int(5 * GB))
    assert fetchlib.remaining_gb("over") == 0.0


def test_unregistered_name_charges_its_full_estimate(monkeypatch):
    """raw_dir refuses unknown datasets, so nothing can be on disk for one."""
    monkeypatch.setattr(fetchlib, "est_gb", lambda _name: 3.0)
    assert fetchlib.remaining_gb("not_a_real_dataset") == 3.0


def test_check_projection_refuses_and_names_what_to_cut(monkeypatch):
    sizes = {"huge": 70.0, "medium": 20.0, "small": 1.0}
    monkeypatch.setattr(fetchlib, "dir_bytes", lambda _root: 0)
    monkeypatch.setattr(fetchlib, "est_gb", lambda name: sizes[name])

    with pytest.raises(DiskBudgetError) as excinfo:
        check_projection(["huge", "medium", "small"])

    message = str(excinfo.value)
    assert "exceeds the 140 GB cap" in message
    # Advice stops at the first cut that gets back under the cap: dropping
    # the 70 GB dataset is sufficient, so it must not also demand the others.
    assert "cut huge" in message
    assert "cut medium" not in message
    assert "71.00 GB" in message, "it should show the total the cut achieves"


def test_cut_advice_lists_several_when_one_is_not_enough(monkeypatch):
    sizes = {"a": 40.0, "b": 30.0, "c": 25.0}
    monkeypatch.setattr(fetchlib, "dir_bytes", lambda _root: 0)
    monkeypatch.setattr(fetchlib, "est_gb", lambda name: sizes[name])

    with pytest.raises(DiskBudgetError) as excinfo:
        check_projection(["a", "b", "c"])

    message = str(excinfo.value)
    # 40+30+25+50 = 145; cutting only the largest leaves 105 -- under. So one
    # cut suffices here too, and it must be the biggest one named first.
    assert "cut a" in message
    assert message.index("cut a") < len(message)


def test_cut_advice_admits_when_no_cut_is_enough(monkeypatch):
    """The reserves alone can exceed the cap; say so rather than bluffing."""
    monkeypatch.setattr(fetchlib, "dir_bytes", lambda _root: int(200 * GB))
    monkeypatch.setattr(fetchlib, "est_gb", lambda _name: 1.0)

    with pytest.raises(DiskBudgetError) as excinfo:
        check_projection(["a"])
    assert "still over even with every pending dataset cut" in str(excinfo.value)


def test_check_projection_passes_when_it_fits(monkeypatch):
    monkeypatch.setattr(fetchlib, "dir_bytes", lambda _root: 0)
    monkeypatch.setattr(fetchlib, "est_gb", lambda _name: 1.0)
    assert check_projection(["a", "b"]).ok


@pytest.mark.parametrize(
    "free, expected_ok",
    [
        (15.1, True),   # comfortably above 1.5 x 10
        (15.0, True),   # exactly at the boundary
        (14.9, False),  # a hair under
        (0.0, False),
    ],
)
def test_free_space_gate_at_the_boundary(monkeypatch, free, expected_ok):
    monkeypatch.setattr(fetchlib, "est_gb", lambda _name: 10.0)
    monkeypatch.setattr(fetchlib, "free_gb", lambda *_a, **_k: free)
    if expected_ok:
        assert check_free_space("synthetic") == free
    else:
        with pytest.raises(DiskBudgetError, match="not enough free space"):
            check_free_space("synthetic")


def test_selection_size_refuses_what_cannot_fit(monkeypatch):
    monkeypatch.setattr(fetchlib, "est_gb", lambda _name: 50.0)
    with pytest.raises(DiskBudgetError, match="cannot finish"):
        check_selection_size("synthetic", int(40 * GB), available_gb=10.0)


def test_selection_size_refuses_a_wrong_scope(monkeypatch):
    """A selection far past the estimate is a bad fetch scope, not a stale number."""
    monkeypatch.setattr(fetchlib, "est_gb", lambda _name: 12.0)
    with pytest.raises(DiskBudgetError, match="wrong fetch scope"):
        check_selection_size("synthetic", int(400 * GB), available_gb=1000.0)


def test_selection_size_warns_but_allows_a_mild_overshoot(monkeypatch):
    monkeypatch.setattr(fetchlib, "est_gb", lambda _name: 10.0)
    selected, warning = check_selection_size("synthetic", int(12 * GB), available_gb=500.0)
    assert selected == pytest.approx(12.0)
    assert warning and "update est_gb" in warning


# --------------------------------------------------------------------------
# the forbidden-path guard
# --------------------------------------------------------------------------


def test_guard_aborts_on_a_forbidden_path():
    e = entry(forbidden_patterns=["data_store/knowledge_store/test/*"])
    paths = ["data/train.json", "data_store/knowledge_store/test/claim_1.json"]
    with pytest.raises(ForbiddenPathError) as excinfo:
        guard_or_abort(e, paths)
    message = str(excinfo.value)
    assert "ABORTING THE WHOLE DATASET" in message
    assert "claim_1.json" in message
    assert "data_store/knowledge_store/test/*" in message


def test_guard_allows_sibling_split_directories():
    e = entry(forbidden_patterns=["data_store/knowledge_store/test/*"])
    guard_or_abort(e, ["data_store/knowledge_store/dev/claim_1.json"])


def test_guard_catches_the_superseded_and_the_live_averitec_shards():
    """Both must be caught — the old one and the one that replaced it."""
    e = dataset("averitec")
    for shard in ("test", "test_updated"):
        with pytest.raises(ForbiddenPathError):
            guard_or_abort(e, [f"data_store/knowledge_store/{shard}/claim_0.json"])


def test_guard_is_a_no_op_when_nothing_is_forbidden():
    guard_or_abort(entry(forbidden_patterns=[]), ["anything/at/all.bin"])


def test_find_forbidden_reports_every_pair():
    e = entry(forbidden_patterns=["a/*", "b/*"])
    hits = find_forbidden(e, ["a/1", "b/2", "c/3"])
    assert sorted(hits) == [("a/1", "a/*"), ("b/2", "b/*")]


def test_select_included_filters_to_the_include_patterns():
    e = entry(include_patterns=["data/*"])
    assert select_included(e, ["data/a", "other/b"]) == ["data/a"]


def test_select_included_takes_everything_when_no_patterns():
    e = entry(include_patterns=None)
    assert select_included(e, ["data/a", "other/b"]) == ["data/a", "other/b"]


# --------------------------------------------------------------------------
# the unfiltered-fetch stop
# --------------------------------------------------------------------------


def test_unfiltered_fetch_refused_when_the_entry_forbids_something():
    e = entry(include_patterns=[], forbidden_patterns=["secret/*"], method="hf")
    with pytest.raises(DiskBudgetError, match="refusing an unfiltered"):
        allow_patterns_for(e)


def test_unfiltered_fetch_allowed_when_nothing_is_forbidden():
    """averimatec's shape: no includes, no prohibitions, a whole small repo."""
    e = entry(include_patterns=None, forbidden_patterns=[], method="hf")
    assert allow_patterns_for(e) == []


def test_averitec_resolves_to_a_non_empty_allow_list():
    patterns = allow_patterns_for(dataset("averitec"))
    assert patterns, "averitec must never resolve to an unfiltered fetch"
    # The dev knowledge store is a single ZIP, not a directory. A dev/* glob
    # matched nothing and failed silently on the first real run.
    assert "data_store/knowledge_store/dev_knowledge_store.zip" in patterns
    assert not any(p.endswith("knowledge_store/dev/*") for p in patterns)


def test_averimatec_is_now_filtered_because_it_forbids_its_test_split():
    """It used to take the whole repo, which pulled the held-out test archive.

    The general "no forbidden globs means an unfiltered fetch is fine" rule is
    still covered by test_unfiltered_fetch_allowed_when_nothing_is_forbidden;
    averimatec is simply no longer an instance of it.
    """
    entry = dataset("averimatec")
    patterns = allow_patterns_for(entry)
    assert patterns, "an entry that forbids something must declare its includes"
    assert entry["forbidden_patterns"] == ["test_data.zip"]
    assert not any("test_data" in p for p in patterns)


# --------------------------------------------------------------------------
# .done markers
# --------------------------------------------------------------------------


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    directory = tmp_path / "_fetch_state"
    monkeypatch.setattr(fetchlib, "FETCH_STATE", directory)
    return directory


def test_marker_round_trip_skips_an_intact_tree(tmp_path, state_dir):
    root = tmp_path / "raw" / "synthetic"
    root.mkdir(parents=True)
    (root / "a.txt").write_text("alpha", encoding="utf-8")

    assert completion_status("synthetic", root) == (False, "no .done marker")
    marker = write_marker("synthetic", root, note="test")
    assert marker["files"] == 1

    skip, reason = completion_status("synthetic", root)
    assert skip is True
    assert "complete" in reason


def test_marker_stats_cannot_overwrite_measured_values(tmp_path, state_dir):
    """A handler reporting its own counts must not overwrite the real ones."""
    root = tmp_path / "raw" / "synthetic"
    root.mkdir(parents=True)
    for name in ("a.txt", "b.txt", "c.txt"):
        (root / name).write_text(name, encoding="utf-8")

    marker = write_marker("synthetic", root, files=1, bytes=999, extracted=2)
    assert marker["files"] == 3, "measured file count must win over handler stats"
    assert marker["bytes"] == sum(len(n) for n in ("a.txt", "b.txt", "c.txt"))
    assert marker["stats"] == {"files": 1, "bytes": 999, "extracted": 2}


def test_marker_detects_a_changed_tree(tmp_path, state_dir):
    root = tmp_path / "raw" / "synthetic"
    root.mkdir(parents=True)
    (root / "a.txt").write_text("alpha", encoding="utf-8")
    write_marker("synthetic", root)

    (root / "a.txt").write_text("tampered", encoding="utf-8")
    skip, reason = completion_status("synthetic", root)
    assert skip is False
    assert "MISMATCH" in reason


def test_marker_detects_a_renamed_file(tmp_path, state_dir):
    """Content-only hashing would miss this; the tree hash includes names."""
    root = tmp_path / "raw" / "synthetic"
    root.mkdir(parents=True)
    (root / "a.txt").write_text("alpha", encoding="utf-8")
    write_marker("synthetic", root)

    (root / "a.txt").rename(root / "b.txt")
    skip, _ = completion_status("synthetic", root)
    assert skip is False


def test_marker_notices_a_vanished_tree(tmp_path, state_dir):
    root = tmp_path / "raw" / "synthetic"
    root.mkdir(parents=True)
    (root / "a.txt").write_text("alpha", encoding="utf-8")
    write_marker("synthetic", root)

    (root / "a.txt").unlink()
    root.rmdir()
    skip, reason = completion_status("synthetic", root)
    assert skip is False
    assert "missing" in reason


# --------------------------------------------------------------------------
# backoff policy
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", [404, 410, 401, 403, 400, 451])
def test_terminal_statuses_are_never_retried(status):
    assert should_retry(status) is False


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 408])
def test_transient_statuses_are_retried(status):
    assert should_retry(status) is True


def test_transport_errors_are_retried():
    assert should_retry(None) is True


def test_backoff_ceiling_grows_and_is_capped():
    import random

    delays = backoff_delays(8, base=0.5, cap=10.0, rng=random.Random(0))
    assert len(delays) == 8
    assert all(0 <= d <= 10.0 for d in delays)
    # full jitter samples below the ceiling, so compare ceilings not samples
    assert min(0.5 * 2**7, 10.0) == 10.0


class _Response:
    def __init__(self, status, body=b""):
        self.status_code = status
        self._body = body
        self.content = body

    def iter_content(self, _size):
        yield self._body


class _Session:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.calls = 0

    def get(self, url, **kwargs):
        self.calls += 1
        status = self.statuses[min(self.calls - 1, len(self.statuses) - 1)]
        return _Response(status, b"payload")


def test_download_does_not_retry_a_404(tmp_path):
    session = _Session([404])
    with pytest.raises(FetchError, match="not retryable"):
        download_resumable(session, "https://example.com/x", tmp_path / "x.bin",
                           retries=5, sleeper=lambda _s: None)
    assert session.calls == 1, "a 404 must cost exactly one request"


def test_download_retries_a_503_then_succeeds(tmp_path):
    session = _Session([503, 503, 200])
    dest = tmp_path / "x.bin"
    size, digest = download_resumable(session, "https://example.com/x", dest,
                                      retries=5, sleeper=lambda _s: None)
    assert session.calls == 3
    assert dest.read_bytes() == b"payload"
    assert size == len(b"payload") and len(digest) == 64


def test_download_gives_up_after_the_retry_budget(tmp_path):
    session = _Session([503])
    with pytest.raises(FetchError, match="giving up after 4 attempts"):
        download_resumable(session, "https://example.com/x", tmp_path / "x.bin",
                           retries=3, sleeper=lambda _s: None)
    assert session.calls == 4


def test_download_leaves_no_file_behind_on_failure(tmp_path):
    dest = tmp_path / "x.bin"
    with pytest.raises(FetchError):
        download_resumable(_Session([404]), "https://example.com/x", dest,
                           retries=1, sleeper=lambda _s: None)
    assert not dest.exists(), "a failed download must not look like a finished one"


# --------------------------------------------------------------------------
# stratified sampling
# --------------------------------------------------------------------------


LABELS = ["a"] * 600 + ["b"] * 300 + ["c"] * 100


def test_sample_is_deterministic_for_a_seed():
    assert stratified_indices(LABELS, 100, 42) == stratified_indices(LABELS, 100, 42)


def test_sample_changes_with_the_seed():
    assert stratified_indices(LABELS, 100, 42) != stratified_indices(LABELS, 100, 7)


def test_sample_preserves_class_proportions():
    from collections import Counter

    chosen = Counter(LABELS[i] for i in stratified_indices(LABELS, 100, 42))
    assert chosen == {"a": 60, "b": 30, "c": 10}


def test_sample_size_is_exact_even_with_awkward_remainders():
    labels = ["a"] * 7 + ["b"] * 5 + ["c"] * 3
    assert len(stratified_indices(labels, 10, 1)) == 10


def test_sample_returns_everything_when_asked_for_more_than_exists():
    assert stratified_indices(LABELS, 99_999, 1) == list(range(len(LABELS)))


def test_sample_of_zero_is_empty():
    assert stratified_indices(LABELS, 0, 1) == []


def test_sample_ignores_input_ordering():
    """Row order must not change which rows are chosen for a given seed."""
    labels = ["a", "b"] * 50
    forward = {labels[i] for i in stratified_indices(labels, 10, 5)}
    assert forward == {"a", "b"}


# --------------------------------------------------------------------------
# hydration reporting
# --------------------------------------------------------------------------


def test_report_refuses_to_serialise_without_a_per_class_breakdown():
    report = HydrationReport(dataset="synthetic")
    with pytest.raises(FetchError, match="per-class breakdown"):
        report.to_dict()


def test_report_write_refuses_before_touching_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(fetchlib, "REPORTS", tmp_path)
    report = HydrationReport(dataset="synthetic")
    with pytest.raises(FetchError):
        report.write()
    assert list(tmp_path.iterdir()) == [], "no partial report may be left behind"


def test_report_computes_overall_and_per_class_rates():
    report = HydrationReport(dataset="synthetic", label_field="label")
    for _ in range(8):
        report.record("keep", ok=True)
    for _ in range(2):
        report.record("keep", ok=False, reason="http_404")
    for _ in range(9):
        report.record("lossy", ok=False, reason="http_404")
    report.record("lossy", ok=True)

    assert report.recovery_rate == pytest.approx(0.45)
    assert report.recovery_rate_per_class == {"keep": 0.8, "lossy": 0.1}
    assert report.failure_reasons == {"http_404": 11}


def test_report_surfaces_a_class_specific_collapse():
    """The whole reason the per-class field is mandatory."""
    report = HydrationReport(dataset="synthetic", label_field="label")
    for _ in range(90):
        report.record("common", ok=True)
    for _ in range(10):
        report.record("rare", ok=False, reason="http_404")

    assert report.recovery_rate == pytest.approx(0.9)  # looks fine
    assert report.recovery_rate_per_class["rare"] == 0.0  # is not fine
    assert "<-- LOW" in report.render()


def test_report_round_trips_through_json(tmp_path, monkeypatch):
    monkeypatch.setattr(fetchlib, "REPORTS", tmp_path)
    report = HydrationReport(dataset="synthetic", label_field="label", seed=7)
    report.record("a", ok=True)
    report.record("b", ok=False, reason="timeout")

    payload = json.loads(report.write().read_text(encoding="utf-8"))
    assert payload["recovery_rate_per_class"] == {"a": 1.0, "b": 0.0}
    assert payload["seed"] == 7
    assert payload["failure_reasons"] == {"timeout": 1}


# --------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------


def test_kaggle_credentials_error_names_every_option(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch.Path, "home", staticmethod(lambda: tmp_path))
    for var in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(CredentialError) as excinfo:
        fetch.kaggle_token_source()

    message = str(excinfo.value)
    for expected in ("kaggle auth login", "KAGGLE_API_TOKEN",
                     "~/.kaggle/access_token", "kaggle.json"):
        assert expected in message


def test_kaggle_credentials_found_via_access_token_file(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    (tmp_path / ".kaggle").mkdir()
    (tmp_path / ".kaggle" / "access_token").write_text("tok", encoding="utf-8")
    assert fetch.kaggle_token_source() == "~/.kaggle/access_token"


def test_kaggle_credentials_found_via_env(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setenv("KAGGLE_API_TOKEN", "tok")
    assert "KAGGLE_API_TOKEN" in fetch.kaggle_token_source()


# --------------------------------------------------------------------------
# licence gate
# --------------------------------------------------------------------------


def test_averitec_requires_explicit_licence_acceptance(capsys, monkeypatch):
    monkeypatch.setattr(fetchlib, "FETCH_LOG", Path("nul") if sys.platform == "win32"
                        else Path("/dev/null"))
    with pytest.raises(fetchlib.LicenceNotAcceptedError, match="--accept-licence"):
        fetch.require_licence(dataset("averitec"), fetch.Context(accept_licence=False))
    assert "CC-BY-NC-4.0" in capsys.readouterr().out


def test_unlicensed_datasets_pass_through():
    fetch.require_licence(dataset("liar"), fetch.Context(accept_licence=False))


# --------------------------------------------------------------------------
# handler routing
# --------------------------------------------------------------------------


def test_every_enabled_dataset_has_a_handler():
    for name in fetchlib.enabled_names():
        assert callable(fetch.handler_for(dataset(name))), name


def test_dataset_handlers_win_over_method_handlers():
    assert fetch.handler_for(dataset("fakeddit")) is fetch.fetch_fakeddit
    assert fetch.handler_for(dataset("factify2")) is fetch.fetch_factify2
    assert fetch.handler_for(dataset("liar")) is fetch.fetch_direct


# --------------------------------------------------------------------------
# failure isolation
# --------------------------------------------------------------------------


def test_one_dataset_failure_does_not_abandon_the_others(monkeypatch, capsys):
    """A third-party client exception must not end the whole run.

    gdown, huggingface_hub and kaggle raise their own exception types. An
    earlier version caught only FetchError, so a gdown.DownloadError on
    factify2 killed the run and the seven datasets after it never ran.
    """
    calls: list[str] = []

    def fake_fetch_one(name, ctx, pending):
        calls.append(name)
        if name == "factify2":
            raise RuntimeError("simulated third-party client failure")
        return "fetched"

    monkeypatch.setattr(fetch, "fetch_one", fake_fetch_one)
    monkeypatch.setattr(fetch, "print_plan", lambda _names: None)
    monkeypatch.setattr(fetch, "check_projection", lambda _pending: None)
    monkeypatch.setattr(fetch, "enabled_names",
                        lambda: ["mocheg", "factify2", "averitec", "liar"])

    exit_code = fetch.main(["--all", "--yes"])

    assert calls == ["mocheg", "factify2", "averitec", "liar"], (
        "every dataset must be attempted despite the failure in the middle"
    )
    assert exit_code == 1, "a failure must still be reported in the exit code"
    assert "FAILED (RuntimeError)" in capsys.readouterr().out


def test_git_bookkeeping_is_excluded_from_markers(tmp_path, state_dir):
    """A sparse clone leaves a .git dir; hashing it would break idempotence.

    FETCH_HEAD and friends change on every fetch without the data changing, so
    a marker that covered them would report MISMATCH forever and re-download
    the dataset each run.
    """
    root = tmp_path / "raw" / "synthetic"
    (root / ".git").mkdir(parents=True)
    (root / "data.tsv").write_text("real data", encoding="utf-8")
    (root / ".git" / "FETCH_HEAD").write_text("abc123", encoding="utf-8")

    marker = write_marker("synthetic", root)
    assert marker["files"] == 1, ".git contents must not count as data"

    (root / ".git" / "FETCH_HEAD").write_text("def456 changed", encoding="utf-8")
    skip, _ = completion_status("synthetic", root)
    assert skip is True, "a changed .git must not invalidate a complete fetch"


# --------------------------------------------------------------------------
# post-fetch estimate reconciliation
# --------------------------------------------------------------------------


def test_a_fetch_far_under_its_estimate_is_flagged(monkeypatch):
    """The averitec failure: include_patterns matched nothing, and nothing
    complained because 12 MB against a 12 GB estimate trips no disk guard."""
    monkeypatch.setattr(fetchlib, "est_gb", lambda _name: 12.0)
    actual, warning = fetchlib.reconcile_estimate("synthetic", int(0.012 * GB))
    assert warning is not None
    assert "include_patterns match less than intended" in warning
    assert "0.1%" in warning


def test_a_fetch_over_its_estimate_asks_for_an_update(monkeypatch):
    monkeypatch.setattr(fetchlib, "est_gb", lambda _name: 1.0)
    _actual, warning = fetchlib.reconcile_estimate("synthetic", int(3 * GB))
    assert warning and "update est_gb" in warning


def test_a_fetch_close_to_its_estimate_is_quiet(monkeypatch):
    monkeypatch.setattr(fetchlib, "est_gb", lambda _name: 10.0)
    _actual, warning = fetchlib.reconcile_estimate("synthetic", int(8 * GB))
    assert warning is None


def test_hf_cache_is_excluded_from_the_manifest(tmp_path, state_dir):
    """huggingface_hub writes .cache beside a local_dir snapshot."""
    root = tmp_path / "raw" / "synthetic"
    (root / ".cache" / "huggingface").mkdir(parents=True)
    (root / "data.json").write_text("real", encoding="utf-8")
    (root / ".cache" / "huggingface" / "x.metadata").write_text("m", encoding="utf-8")
    assert write_marker("synthetic", root)["files"] == 1


def test_changing_include_patterns_invalidates_a_marker(tmp_path, state_dir, monkeypatch):
    """A corrected glob must re-fetch, not silently keep the stale data."""
    root = tmp_path / "raw" / "averitec"
    root.mkdir(parents=True)
    (root / "data.json").write_text("stale", encoding="utf-8")
    write_marker("averitec", root)
    assert completion_status("averitec", root)[0] is True

    widened = {**dataset("averitec"), "include_patterns": ["data/*", "something/else/*"]}
    monkeypatch.setattr(fetchlib, "dataset", lambda _n: widened)
    skip, reason = completion_status("averitec", root)
    assert skip is False
    assert "FETCH SCOPE CHANGED" in reason


def test_a_marker_without_a_scope_is_not_trusted(tmp_path, state_dir):
    """Markers predating scope tracking must re-verify, not be waved through."""
    import json as _json

    root = tmp_path / "raw" / "averitec"
    root.mkdir(parents=True)
    (root / "data.json").write_text("legacy", encoding="utf-8")
    write_marker("averitec", root)

    path = fetchlib.marker_path("averitec")
    marker = _json.loads(path.read_text(encoding="utf-8"))
    del marker["scope"]  # simulate a marker written by the previous version
    path.write_text(_json.dumps(marker), encoding="utf-8")

    skip, reason = completion_status("averitec", root)
    assert skip is False
    assert "before scope tracking existed" in reason




# --------------------------------------------------------------------------
# hydration input parsing
# --------------------------------------------------------------------------


def test_tab_separated_csv_is_detected(tmp_path):
    """Factify2 ships tab-separated data in files named .csv.

    Trusting the extension parsed the entire header as a single column, so
    every label lookup failed and hydration refused to run.
    """
    from scripts import hydrate

    path = tmp_path / "train.csv"
    path.write_text(
        "\tclaim\tclaim_image\tCategory\n0\thi\thttp://x/i.jpg\tSupport_Text\n",
        encoding="utf-8",
    )
    assert hydrate.sniff_delimiter(path) == "\t"
    header, rows = hydrate.read_rows(path)
    assert "Category" in header and "claim_image" in header
    assert rows[0]["Category"] == "Support_Text"


def test_comma_separated_csv_still_works(tmp_path):
    from scripts import hydrate

    path = tmp_path / "x.csv"
    path.write_text("id,label,image_url\n1,fake,http://x/i.jpg\n", encoding="utf-8")
    assert hydrate.sniff_delimiter(path) == ","
    _header, rows = hydrate.read_rows(path)
    assert rows[0]["label"] == "fake"


def test_oversized_csv_fields_are_readable(tmp_path):
    """MOCHEG stores whole scraped articles in one cell, past the 128 KB default."""
    from scripts import hydrate

    path = tmp_path / "big.csv"
    path.write_text("id,label,text\n1,x," + "a" * 200_000 + "\n", encoding="utf-8")
    _header, rows = hydrate.read_rows(path)
    assert len(rows[0]["text"]) == 200_000


def test_blank_header_columns_are_dropped(tmp_path):
    """Factify2's first column has no name; it must not become a label candidate."""
    from scripts import hydrate

    path = tmp_path / "t.csv"
    path.write_text("\tclaim\tCategory\n0\thi\tRefute\n", encoding="utf-8")
    header, _rows = hydrate.read_rows(path)
    assert "" not in header


# --------------------------------------------------------------------------
# retry policy, throttling, ledgers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", [403, 406, 429, 500, 503])
def test_unblock_policy_retries_blocked_statuses(status):
    assert fetchlib.UNBLOCK_POLICY.should_retry(status) is True


@pytest.mark.parametrize("status", [404, 410])
def test_no_policy_ever_retries_a_deleted_resource(status):
    assert fetchlib.DEFAULT_POLICY.should_retry(status) is False
    assert fetchlib.UNBLOCK_POLICY.should_retry(status) is False


def test_default_policy_still_treats_403_as_final():
    """The ordinary pass must not change behaviour."""
    assert fetchlib.DEFAULT_POLICY.should_retry(403) is False
    assert fetchlib.should_retry(403) is False


def test_unblock_headers_set_referer_to_the_urls_own_origin():
    headers = fetchlib.unblock_headers("https://www.snopes.com/tachyon/a/b.jpg?x=1")
    assert headers["Referer"] == "https://www.snopes.com/"
    assert "Mozilla/5.0" in headers["User-Agent"]
    assert headers["Accept"].startswith("image/avif,image/webp,image/apng")


def test_throttle_spaces_requests_to_the_same_host():
    now = [0.0]
    slept = []

    def clock():
        return now[0]

    def sleeper(seconds):
        slept.append(seconds)
        now[0] += seconds

    throttle = fetchlib.HostThrottle(1.0, clock=clock, sleeper=sleeper)
    throttle.wait("https://a.example/1.jpg")
    throttle.wait("https://a.example/2.jpg")
    assert slept == [1.0], "the second same-host request must wait a full delay"


def test_throttle_does_not_penalise_other_hosts():
    now = [0.0]
    slept = []
    throttle = fetchlib.HostThrottle(
        1.0, clock=lambda: now[0], sleeper=lambda s: slept.append(s)
    )
    throttle.wait("https://a.example/1.jpg")
    throttle.wait("https://b.example/1.jpg")
    assert slept == [], "a different host must not be delayed by the first"


def test_throttle_is_a_no_op_when_delay_is_zero():
    throttle = fetchlib.HostThrottle(0.0, sleeper=lambda s: pytest.fail("slept"))
    assert throttle.wait("https://a.example/x") == 0.0


def test_wayback_raw_url_requests_the_unmodified_original():
    url = fetchlib.wayback_raw_url("20220103120000", "https://x.example/a.jpg")
    assert url == "https://web.archive.org/web/20220103120000id_/https://x.example/a.jpg"
    assert "id_/" in url, "without id_ the archive returns a rewritten HTML page"


def test_failure_ledger_marks_only_deleted_resources_terminal(tmp_path):
    ledger = fetchlib.FailureLedger(tmp_path / "f.jsonl")
    ledger.write([
        {"key": "gone", "url": "u", "label": "a", "reason": "http_404"},
        {"key": "blocked", "url": "u", "label": "a", "reason": "http_403"},
        {"key": "flaky", "url": "u", "label": "a", "reason": "readtimeout"},
    ])
    assert ledger.terminal_keys() == {"gone"}


def test_provenance_ledger_keeps_origin_and_wayback_apart(tmp_path):
    ledger = fetchlib.ProvenanceLedger(tmp_path / "p.jsonl")
    ledger.append(key="a", label="Refute", url="u1", source=fetchlib.ORIGIN)
    ledger.append(key="b", label="Refute", url="u2", source=fetchlib.WAYBACK,
                  wayback_timestamp="20220103120000")
    assert ledger.counts_by_source() == {"origin": 1, "wayback": 1}
    records = ledger.read()
    assert records[1]["wayback_timestamp"] == "20220103120000"
    assert all("fetched_at" in r for r in records)


def test_report_records_recovery_by_source():
    report = HydrationReport(dataset="d", label_field="l")
    report.record("a", ok=True)
    report.per_source = {"origin": 1, "wayback": 3}
    assert report.to_dict()["recovered_by_source"] == {"origin": 1, "wayback": 3}


# --------------------------------------------------------------------------
# domain / label confound diagnostic
# --------------------------------------------------------------------------


def test_nmi_is_one_when_domain_determines_label():
    from scripts import confound

    pairs = [("a.com", "x")] * 50 + [("b.com", "y")] * 50
    info = confound.mutual_information(pairs)
    assert info["nmi_sqrt"] == pytest.approx(1.0, abs=1e-9)


def test_nmi_is_zero_when_domain_is_independent_of_label():
    from scripts import confound

    pairs = []
    for domain in ("a.com", "b.com"):
        pairs += [(domain, "x")] * 50 + [(domain, "y")] * 50
    info = confound.mutual_information(pairs)
    assert info["nmi_sqrt"] == pytest.approx(0.0, abs=1e-9)


def test_domain_classifier_beats_baseline_when_domain_leaks_the_label():
    from scripts import confound

    pairs = [("snopes.com", "Refute")] * 300 + [("cnn.com", "Support")] * 300
    assert confound.baseline_accuracy(pairs) == pytest.approx(0.5)
    assert confound.heldout_accuracy(pairs) == pytest.approx(1.0)


def test_heldout_is_lower_than_resubstitution_for_singleton_domains():
    from scripts import confound

    # every domain appears once, so resubstitution is perfect and meaningless
    pairs = [(f"d{i}.com", "x" if i % 2 else "y") for i in range(200)]
    resub = confound.resubstitution_accuracy(pairs)
    held = confound.heldout_accuracy(pairs)
    assert resub == pytest.approx(1.0)
    assert held < resub, "held-out must expose what resubstitution flatters"


def test_domain_of_strips_www_and_lowercases():
    from scripts import confound

    assert confound.domain_of("https://WWW.Snopes.com/a/b.jpg?x=1") == "snopes.com"
    assert confound.domain_of("not a url") == "unknown"


def test_majority_map_breaks_ties_deterministically():
    from scripts import confound

    pairs = [("d.com", "b"), ("d.com", "a")]
    table, _fallback = confound.majority_map(pairs)
    assert table["d.com"] == "a", "ties must resolve by label name, not dict order"


def test_entropy_of_a_constant_is_zero():
    from scripts import confound

    assert confound.entropy([10]) == 0.0
    assert confound.entropy([]) == 0.0


def test_circuit_breaker_writes_off_a_host_that_keeps_refusing():
    """snopes refused 5,814/5,814; retrying each with a full budget is ~29,000
    pointless requests at a host behind a blanket block."""
    breaker = fetchlib.HostCircuitBreaker(threshold=3)
    url = "https://blocked.example/a.jpg"
    for _ in range(3):
        assert breaker.is_open(url) is False
        breaker.record(url, ok=False)
    assert breaker.is_open(url) is True
    assert breaker.blocked_hosts() == {"blocked.example": 3}


def test_circuit_breaker_leaves_other_hosts_alone():
    breaker = fetchlib.HostCircuitBreaker(threshold=2)
    for _ in range(2):
        breaker.record("https://bad.example/x", ok=False)
    assert breaker.is_open("https://bad.example/x") is True
    assert breaker.is_open("https://good.example/x") is False


def test_a_success_closes_the_breaker():
    """A merely flaky host must not be written off permanently."""
    breaker = fetchlib.HostCircuitBreaker(threshold=2)
    breaker.record("https://flaky.example/x", ok=False)
    breaker.record("https://flaky.example/x", ok=True)
    breaker.record("https://flaky.example/x", ok=False)
    assert breaker.is_open("https://flaky.example/x") is False


# --------------------------------------------------------------------------
# manifest robustness
# --------------------------------------------------------------------------


def test_scan_survives_one_unreadable_file(tmp_path, monkeypatch):
    """One transient errno 22 must not abandon a 142,000-file manifest.

    That is not hypothetical: it happened to fakeddit, and every one of its
    142,434 media files went unrecorded because a single JPEG was briefly
    unopenable.
    """
    from scripts import manifest as m

    root = tmp_path / "raw" / "synthetic"
    root.mkdir(parents=True)
    good = root / "fine.bin"
    bad = root / "locked.bin"
    good.write_bytes(b"ok")
    bad.write_bytes(b"nope")

    monkeypatch.setattr(m, "_targets", lambda _sel: ["synthetic"])
    monkeypatch.setattr(m, "raw_dir", lambda _n: root)
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)

    real = m.sha256

    def flaky(path, *a, **k):
        if path.name == "locked.bin":
            raise OSError(22, "Invalid argument")
        return real(path, *a, **k)

    monkeypatch.setattr(m, "sha256", flaky)

    errors = []
    found = m.scan(["synthetic"], errors=errors)
    assert list(found) == ["raw/synthetic/fine.bin"]
    assert len(errors) == 1 and "locked.bin" in errors[0][0]


def test_scan_still_raises_when_no_error_sink_is_given(tmp_path, monkeypatch):
    """Default behaviour is unchanged: a caller that wants the exception gets it."""
    from scripts import manifest as m

    root = tmp_path / "raw" / "synthetic"
    root.mkdir(parents=True)
    (root / "locked.bin").write_bytes(b"x")
    monkeypatch.setattr(m, "_targets", lambda _sel: ["synthetic"])
    monkeypatch.setattr(m, "raw_dir", lambda _n: root)
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(m, "sha256",
                        lambda *a, **k: (_ for _ in ()).throw(OSError(22, "nope")))
    with pytest.raises(OSError):
        m.scan(["synthetic"])


def test_sha256_retries_a_transient_error(tmp_path, monkeypatch):
    from scripts import manifest as m

    target = tmp_path / "f.bin"
    target.write_bytes(b"payload")
    calls = {"n": 0}
    real_open = Path.open

    def flaky_open(self, *a, **k):
        if self == target:
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError(22, "Invalid argument")
        return real_open(self, *a, **k)

    monkeypatch.setattr(Path, "open", flaky_open)
    digest = m.sha256(target, retries=2, pause=0)
    assert len(digest) == 64
    assert calls["n"] == 2, "it should succeed on the retry"


def test_item_record_defaults_to_key():
    """One-image-per-row corpora need no explicit record id."""
    from scripts.hydrate import Item

    assert Item(key="k", url="u", label="l").record_id == "k"
    assert Item(key="row_claim", url="u", label="l", record="row").record_id == "row"


def test_complete_cases_needs_every_image_on_the_row():
    """A Factify2 row has two images; one missing discards the whole row."""
    from scripts import confound
    from scripts.hydrate import Item

    items = [
        Item(key="r1_a", url="http://x/1.jpg", label="Refute", record="r1"),
        Item(key="r1_b", url="http://x/2.jpg", label="Refute", record="r1"),
        Item(key="r2_a", url="http://x/3.jpg", label="Support", record="r2"),
        Item(key="r2_b", url="http://x/4.jpg", label="Support", record="r2"),
    ]
    # r1 complete, r2 half-missing
    have = {"r1_a", "r1_b", "r2_a"}
    confound_present = lambda _n, it, _s: it.key in have
    import scripts.confound as c
    original = c._present
    c._present = confound_present
    try:
        result = c.complete_cases("factify2", items, "all", set())
    finally:
        c._present = original

    assert result["records_total"] == 2
    assert result["records_complete"] == 1
    assert result["per_class"]["Refute"]["records_complete"] == 1
    assert result["per_class"]["Support"]["records_lost"] == 1
    assert result["images_per_record"] == 2.0


def test_excluding_a_domain_removes_its_contribution():
    """Excluding a pure domain must lower the measured association."""
    from scripts import confound

    pairs = [("neutral.com", "a")] * 100 + [("neutral.com", "b")] * 100
    pure = [("pure.com", "a")] * 100
    with_pure = confound.mutual_information(pairs + pure)["nmi_sqrt"]
    without = confound.mutual_information(pairs)["nmi_sqrt"]
    assert with_pure > without
    assert without == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# source-disjoint split generator
# --------------------------------------------------------------------------


def _toy():
    """A corpus with one near-uniform hub domain and two pure leaky domains."""
    domains, labels = {}, {}
    for i in range(200):                       # hub: spread across classes
        domains[f"h{i}"] = {"hub.com"}
        labels[f"h{i}"] = "a" if i % 2 else "b"
    for i in range(40):                        # pure -> always "a"
        domains[f"p{i}"] = {"pure_a.com"}
        labels[f"p{i}"] = "a"
    for i in range(40):                        # pure -> always "b"
        domains[f"q{i}"] = {"pure_b.com"}
        labels[f"q{i}"] = "b"
    return domains, labels


def test_purity_identifies_the_leaky_domains():
    from scripts import splits

    domains, labels = _toy()
    purity = splits.domain_purity(domains, labels)
    assert purity["pure_a.com"]["purity"] == 1.0
    assert purity["pure_a.com"]["dominant_class"] == "a"
    assert purity["hub.com"]["purity"] == pytest.approx(0.5)
    assert purity["hub.com"]["records"] == 200


def test_predictive_mode_constrains_only_pure_domains():
    from scripts import splits

    domains, labels = _toy()
    purity = splits.domain_purity(domains, labels)
    chosen = splits.constrained_domains(purity, "predictive", 0.90)
    assert chosen == {"pure_a.com", "pure_b.com"}
    assert splits.constrained_domains(purity, "none", 0.90) == set()
    assert splits.constrained_domains(purity, "strict", 0.90) == set(purity)


def test_strict_mode_makes_one_giant_component():
    """The hub joins nearly everything, which is what dooms strict mode."""
    from scripts import splits

    domains, labels = _toy()
    purity = splits.domain_purity(domains, labels)
    strict = splits.components(domains, splits.constrained_domains(purity, "strict", 0.9))
    predictive = splits.components(
        domains, splits.constrained_domains(purity, "predictive", 0.9))
    assert len(strict[0]) == 200, "the hub forces its records together"
    assert len(predictive[0]) == 40, "only the pure domains bind records"
    assert len(predictive) > len(strict)


def test_strict_mode_refuses_rather_than_writing_a_useless_split(monkeypatch):
    from scripts import splits

    domains, labels = _toy()
    monkeypatch.setattr(splits, "complete_records", lambda _n: (domains, labels))
    with pytest.raises(splits.SplitError) as excinfo:
        splits.build("factify2", mode="strict")
    message = str(excinfo.value)
    assert "IMPOSSIBLE" in message
    assert "--disjoint predictive" in message


def test_predictive_split_hits_roughly_the_requested_ratios(monkeypatch):
    from scripts import splits

    domains, labels = _toy()
    monkeypatch.setattr(splits, "complete_records", lambda _n: (domains, labels))
    result = splits.build("factify2", mode="predictive")
    sizes = result["split_sizes"]
    assert sum(sizes.values()) == 280
    assert sizes["train"] > sizes["val"] and sizes["train"] > sizes["test"]
    assert 0.5 < sizes["train"] / 280 < 0.9


def test_a_pure_domain_never_straddles_two_splits(monkeypatch):
    from scripts import splits

    domains, labels = _toy()
    monkeypatch.setattr(splits, "complete_records", lambda _n: (domains, labels))
    result = splits.build("factify2", mode="predictive")
    placement = result["_placement"]
    for domain in ("pure_a.com", "pure_b.com"):
        splits_used = {placement[r] for r, d in domains.items() if domain in d}
        assert len(splits_used) == 1, f"{domain} straddles {splits_used}"
    assert result["leakage"]["violations"] == 0


def test_leakage_assertion_fails_loudly_on_a_planted_violation():
    from scripts import splits

    domains, labels = _toy()
    purity = splits.domain_purity(domains, labels)
    placement = {r: "train" for r in domains}
    placement["p0"] = "test"          # split a 100%-pure domain across the boundary
    violations = splits.leakage(placement, domains, purity, 0.90)
    assert any(v["domain"] == "pure_a.com" for v in violations)
    with pytest.raises(splits.SplitError, match="LEAKAGE"):
        splits.assert_no_leakage(violations, "predictive", 0.90)


def test_none_mode_records_leakage_instead_of_aborting(monkeypatch):
    """The random baseline leaks by design; that is the measurement."""
    from scripts import splits

    domains, labels = _toy()
    monkeypatch.setattr(splits, "complete_records", lambda _n: (domains, labels))
    result = splits.build("factify2", mode="none")
    assert result["leakage"]["enforced"] is False
    assert result["leakage"]["violations"] > 0, "a random split should leak here"


def test_domain_only_accuracy_collapses_when_domains_are_disjoint(monkeypatch):
    from scripts import splits

    domains, labels = _toy()
    monkeypatch.setattr(splits, "complete_records", lambda _n: (domains, labels))
    predictive = splits.build("factify2", mode="predictive")
    leaky = splits.build("factify2", mode="none")
    assert (predictive["domain_only"]["domain_only_test_accuracy"]
            <= leaky["domain_only"]["domain_only_test_accuracy"]), (
        "constraining the leaky domains must not make hostname guessing easier")


def test_split_is_deterministic_for_a_seed(monkeypatch):
    from scripts import splits

    domains, labels = _toy()
    monkeypatch.setattr(splits, "complete_records", lambda _n: (domains, labels))
    a = splits.build("factify2", seed=7)["_placement"]
    b = splits.build("factify2", seed=7)["_placement"]
    assert a == b


def test_report_carries_every_field_the_rule_requires(monkeypatch):
    from scripts import splits

    domains, labels = _toy()
    monkeypatch.setattr(splits, "complete_records", lambda _n: (domains, labels))
    result = splits.build("factify2", mode="predictive")
    for field in ("mode", "purity_threshold", "seed", "split_sizes",
                  "per_class_by_split", "constrained_domains",
                  "component_sizes", "leakage", "domain_only"):
        assert field in result, f"split report is missing {field}"
    entry = result["constrained_domains"][0]
    for field in ("domain", "purity", "records", "dominant_class"):
        assert field in entry


def test_records_missing_an_image_are_excluded(monkeypatch):
    """A Factify2 row needs both images to be usable multimodally."""
    from scripts import splits
    from scripts.hydrate import Item

    items = [
        Item(key="r1_a", url="http://x/1.jpg", label="a", record="r1"),
        Item(key="r1_b", url="http://x/2.jpg", label="a", record="r1"),
        Item(key="r2_a", url="http://y/3.jpg", label="b", record="r2"),
        Item(key="r2_b", url="http://y/4.jpg", label="b", record="r2"),
    ]
    monkeypatch.setitem(splits.RESOLVERS, "factify2", lambda: (items, "label", {}))
    have = {"r1_a", "r1_b", "r2_a"}   # r2 is half-missing
    monkeypatch.setattr(splits, "existing_destination",
                        lambda _n, it: object() if it.key in have else None)
    domains, labels = splits.complete_records("factify2")
    assert set(domains) == {"r1"}
    assert labels == {"r1": "a"}


def test_a_class_nearly_absent_from_a_split_is_reported(monkeypatch):
    """Factify2 hits this unavoidably; it must never pass silently.

    All Refute records sit in one indivisible component, so whichever split
    misses out has essentially none -- and a class you cannot evaluate is worth
    saying out loud rather than leaving in a results table.
    """
    from scripts import splits

    domains, labels = {}, {}
    for i in range(300):
        domains[f"h{i}"] = {"hub.com"}
        labels[f"h{i}"] = "common"
    for i in range(60):                     # one indivisible pure block
        domains[f"r{i}"] = {"pure.com"}
        labels[f"r{i}"] = "rare"

    monkeypatch.setattr(splits, "complete_records", lambda _n: (domains, labels))
    result = splits.build("factify2", mode="predictive")

    thin = {(t["split"], t["class"]) for t in result["thin_classes"]}
    assert any(cls == "rare" for _s, cls in thin), (
        "a split with almost none of a class must be flagged")
    assert "UNUSABLE CLASS/SPLIT COMBINATIONS" in splits.render(result)


def test_a_balanced_split_reports_no_thin_classes(monkeypatch):
    from scripts import splits

    domains = {f"r{i}": {f"d{i}.com"} for i in range(600)}
    labels = {f"r{i}": ("a" if i % 2 else "b") for i in range(600)}
    monkeypatch.setattr(splits, "complete_records", lambda _n: (domains, labels))
    result = splits.build("factify2", mode="none")
    assert result["thin_classes"] == []


def test_class_aware_assignment_keeps_the_rare_class_out_of_train_only(monkeypatch):
    """The bug this replaced put every Refute component in train."""
    from scripts import splits

    domains, labels = {}, {}
    for i in range(300):
        domains[f"h{i}"] = {"hub.com"}
        labels[f"h{i}"] = "common"
    for block, n in (("A", 60), ("B", 30)):   # two indivisible rare blocks
        for i in range(n):
            domains[f"{block}{i}"] = {f"pure_{block}.com"}
            labels[f"{block}{i}"] = "rare"

    monkeypatch.setattr(splits, "complete_records", lambda _n: (domains, labels))
    result = splits.build("factify2", mode="predictive")
    per_split = result["per_class_by_split"]
    with_rare = [s for s in ("train", "val", "test") if per_split[s].get("rare")]
    assert "test" in with_rare, "the test set must receive a rare-class block"


def test_generated_bytecode_is_not_data(tmp_path, state_dir):
    """pytest once imported a vendored test suite out of data/raw and left a
    .pyc behind, which then read as a change to immutable raw data."""
    root = tmp_path / "raw" / "synthetic"
    (root / "_aux" / "test" / "__pycache__").mkdir(parents=True)
    (root / "data.csv").write_text("real", encoding="utf-8")
    (root / "_aux" / "test" / "__pycache__" / "t.cpython-311.pyc").write_bytes(
        b"fake bytecode")
    assert write_marker("synthetic", root)["files"] == 1, (
        "generated bytecode must not count as data")


# --------------------------------------------------------------------------
# verify.py: date recovery
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("2023-03-17", "2023-03-17"),
        ("17 March 2023", "2023-03-17"),
        ("25-8-2020", "2020-08-25"),      # AVeriTeC: day-first, unpadded
        ("5-11-2021", "2021-11-05"),
        ("1551641244.0", "2019-03-03"),   # Fakeddit: unix timestamp
    ],
)
def test_explicit_dates_are_recovered(value, expected):
    from scripts import verify

    assert verify.iso_date(value) == expected


@pytest.mark.parametrize("value", ["", "n/a", "none", "NaN", "-", None,
                                   "2023-02-30", "32-1-2020", "not a date"])
def test_impossible_or_absent_dates_return_none(value):
    from scripts import verify

    assert verify.iso_date(value) is None


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://abc.net.au/news/2020-08-21/story/12578866", "2020-08-21"),
        ("https://x.example/2020/08/21/story", "2020-08-21"),
        ("https://x.example/story", None),
        ("https://x.example/2020-13-45/story", None),   # impossible date
        (None, None),
    ],
)
def test_url_derived_dates(url, expected):
    from scripts import verify

    assert verify.date_from_url(url) == expected


def test_explicit_and_url_dates_are_never_merged():
    """Only an explicit field is authoritative; a URL date is a filing habit."""
    from scripts import verify

    d = verify.Dates("claim_date")
    d.add_explicit("2020-01-01")
    d.add_url("https://x.example/2021/05/06/a")
    d.add_url("https://x.example/no-date")
    out = d.to_dict()
    assert out["with_explicit_date"] == 1
    assert out["with_url_derived_date"] == 1
    assert out["items"] == 3
    # the two are accounted separately and only their SUM is 'any'
    assert out["any_coverage"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["range"] == {"min": "2020-01-01", "max": "2021-05-06"}

    lopsided = verify.Dates("d")
    for _ in range(9):
        lopsided.add_explicit("2020-01-01")
    lopsided.add_url("https://x.example/no-date")
    got = lopsided.to_dict()
    assert got["explicit_coverage"] == pytest.approx(0.9)
    assert got["url_derived_coverage"] == 0.0, (
        "a URL with no date must never borrow the explicit field's coverage")


def test_dates_with_nothing_recovered_report_zero_not_crash():
    from scripts import verify

    out = verify.Dates(None).to_dict()
    assert out["explicit_coverage"] == 0.0 and out["range"] is None


# --------------------------------------------------------------------------
# verify.py: modality and expected-vs-measured
# --------------------------------------------------------------------------


def test_modality_fractions_partition_the_records():
    from scripts import verify

    m = verify.Modality()
    m.add(has_text=True, has_image=True)
    m.add(has_text=True, has_image=False)
    m.add(has_text=False, has_image=True)
    m.add(has_text=False, has_image=False)
    out = m.to_dict()
    assert out["records"] == 4
    assert (out["both"], out["text_only"], out["image_only"], out["neither"]) == (1, 1, 1, 1)
    assert sum(out["fractions"].values()) == pytest.approx(1.0)


def test_mismatch_is_reported_when_a_count_differs(monkeypatch):
    from scripts import verify

    monkeypatch.setattr(verify, "dataset",
                        lambda _n: {"expected": {"rows": 1000, "ooc": 324}})
    monkeypatch.setitem(verify.EXPECTED_LABEL_ALIASES, "synthetic",
                        {"ooc": "out-of-context"})
    out = verify.compare_expected("synthetic", rows=1001,
                                  labels={"out-of-context": 325})
    fields = {m["field"]: m for m in out}
    assert fields["rows"]["difference"] == 1
    assert fields["ooc"]["measured"] == 325
    assert "+1" in fields["ooc"]["detail"]


def test_no_mismatch_when_everything_agrees(monkeypatch):
    from scripts import verify

    monkeypatch.setattr(verify, "dataset",
                        lambda _n: {"expected": {"rows": 10, "real": 4, "fake": 6}})
    assert verify.compare_expected("x", rows=10, labels={"real": 4, "fake": 6}) == []


def test_a_missing_measurement_is_flagged_not_skipped(monkeypatch):
    """An expected field with nothing to compare against must still surface."""
    from scripts import verify

    monkeypatch.setattr(verify, "dataset", lambda _n: {"expected": {"ghost": 5}})
    out = verify.compare_expected("x", rows=1, labels={})
    assert out and out[0]["measured"] is None


# --------------------------------------------------------------------------
# prepush_check.py
# --------------------------------------------------------------------------


def _stage(tmp_path, name, text):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return name


def test_guard_refuses_an_image(tmp_path):
    from scripts import prepush_check as g

    _stage(tmp_path, "docs/figure.png", "not really a png")
    found = g.check(tmp_path, ["docs/figure.png"])
    assert "docs/figure.png" in found
    assert any("data/media file" in r for r in found["docs/figure.png"])


def test_guard_refuses_anything_under_data_raw(tmp_path):
    from scripts import prepush_check as g

    _stage(tmp_path, "data/raw/liar/train.tsv", "a\tb\n")
    found = g.check(tmp_path, ["data/raw/liar/train.tsv"])
    assert any("data directory" in r for r in found["data/raw/liar/train.tsv"])


def test_guard_refuses_a_split_csv_carrying_labels(tmp_path):
    from scripts import prepush_check as g

    _stage(tmp_path, "factify2_train.csv", "record_id,label,domains\nr1,Refute,x.com\n")
    found = g.check(tmp_path, ["factify2_train.csv"])
    reasons = " ".join(found["factify2_train.csv"])
    assert "label" in reasons and "factify2" in reasons


def test_guard_allows_a_split_csv_of_ids_and_split(tmp_path):
    from scripts import prepush_check as g

    _stage(tmp_path, "factify2_train.csv", "record_id,split\nr1,train\n")
    assert g.check(tmp_path, ["factify2_train.csv"]) == {}


def test_guard_allows_an_aggregate_json_mapping(tmp_path):
    from scripts import prepush_check as g

    _stage(tmp_path, "factify2_split_report.json",
           '{"mode": "predictive", "domain_only": {"acc": 0.222}}')
    assert g.check(tmp_path, ["factify2_split_report.json"]) == {}


def test_guard_refuses_a_json_list_of_labelled_records(tmp_path):
    from scripts import prepush_check as g

    _stage(tmp_path, "factify2_rows.json", '[{"record_id": "r1", "label": "Refute"}]')
    found = g.check(tmp_path, ["factify2_rows.json"])
    assert any("JSON LIST" in r for r in found["factify2_rows.json"])


def test_guard_refuses_a_per_record_jsonl_with_labels(tmp_path):
    from scripts import prepush_check as g

    _stage(tmp_path, "hydration_factify2_failures.jsonl",
           '{"key": "k", "label": "Refute", "url": "http://x/1.jpg"}\n')
    found = g.check(tmp_path, ["hydration_factify2_failures.jsonl"])
    reasons = " ".join(found["hydration_factify2_failures.jsonl"])
    assert "label" in reasons


def test_guard_refuses_a_file_containing_a_gate_credential(tmp_path):
    """Removing the passwords from source must not be quietly undoable."""
    from scripts import prepush_check as g

    secret = "factify2" + "taskaaai" + "@22"
    _stage(tmp_path, "scripts/fetch.py", f'PASSWORDS = {{"a": "{secret}"}}\n')
    found = g.check(tmp_path, ["scripts/fetch.py"])
    assert any("gate credential" in r for r in found["scripts/fetch.py"])


def test_guard_passes_ordinary_source_and_docs(tmp_path):
    from scripts import prepush_check as g

    _stage(tmp_path, "scripts/fetch.py", "def fetch():\n    return 1\n")
    _stage(tmp_path, "docs/data_card.md", "# Data card\n\nlabel distribution\n")
    assert g.check(tmp_path, ["scripts/fetch.py", "docs/data_card.md"]) == {}


def test_every_register_dataset_counts_as_gated():
    """The rule keys off redistributable, not off a hand-maintained list."""
    from scripts import prepush_check as g
    from configs.paths import dataset_names

    assert g.gated_datasets() == set(dataset_names())


def test_split_files_carry_ids_and_split_only(tmp_path, monkeypatch):
    """Labels are the gated annotation and must never reach a split file."""
    import csv as _csv
    from scripts import splits

    domains = {f"r{i}": {f"d{i%3}.com"} for i in range(30)}
    labels = {f"r{i}": ("a" if i % 2 else "b") for i in range(30)}
    monkeypatch.setattr(splits, "complete_records", lambda _n: (domains, labels))
    monkeypatch.setattr(splits, "SPLITS", tmp_path)

    result = splits.build("factify2", mode="none")
    _report, written = splits.write_outputs(result)

    for path in written:
        with path.open(encoding="utf-8", newline="") as fh:
            header = next(_csv.reader(fh))
        assert header == ["record_id", "split"], f"{path.name} header is {header}"


def test_guard_allows_an_empty_gitkeep_under_a_data_directory(tmp_path):
    """The repo tracks these deliberately to hold the directory shape.

    A guard that refuses them cries wolf on every push, and a guard that cries
    wolf is one people learn to bypass.
    """
    from scripts import prepush_check as g

    for rel in ("data/raw/.gitkeep", "data/bible/.gitkeep",
                "data/interim/.gitkeep"):
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")
    assert g.check(tmp_path, ["data/raw/.gitkeep", "data/bible/.gitkeep",
                              "data/interim/.gitkeep"]) == {}


def test_guard_still_refuses_a_gitkeep_with_content_in_it(tmp_path):
    """The exemption is for emptiness, not for the filename."""
    from scripts import prepush_check as g

    target = tmp_path / "data" / "raw" / ".gitkeep"
    target.parent.mkdir(parents=True)
    target.write_text("record_id,label" + chr(10) + "r1,Refute" + chr(10),
                      encoding="utf-8")
    assert g.check(tmp_path, ["data/raw/.gitkeep"]) != {}
