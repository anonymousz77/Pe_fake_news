"""Image recovery: politeness, routes, provenance, and the index it trusts.

Offline: every HTTP response here is synthetic, and the clock is fake.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import hydrate, images, recover  # noqa: E402
from scripts.fetchlib import ProvenanceLedger  # noqa: E402
from scripts.hydrate import Item  # noqa: E402
from scripts.recover import Req, Resp  # noqa: E402


def _png(colour=(10, 20, 30), size=(8, 8)) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


class FakeClock:
    def __init__(self):
        self.t = 1000.0
        self.slept = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += max(0.0, s)


# --------------------------------------------------------------------------
# the single worker
# --------------------------------------------------------------------------


def test_the_host_delay_cannot_be_lowered_for_real_traffic():
    with pytest.raises(ValueError):
        recover.Worker(session=None, delay=1.0)


def test_requests_to_one_host_start_at_least_five_seconds_apart():
    clock = FakeClock()
    starts = []

    def transport(req):
        starts.append((recover.host_of(req.url), clock()))
        clock.t += 0.4  # the request itself takes time
        return Resp(req.url, 404, {}, b"", None, 0.4)

    worker = recover.Worker(None, delay=5.0, clock=clock, sleeper=clock.sleep,
                            transport=transport)

    def three(host):
        for i in range(3):
            yield Req(f"https://{host}/{i}")

    worker.start(three("a.example"))
    worker.start(three("b.example"))
    worker.run()
    for host in ("a.example", "b.example"):
        times = [t for h, t in starts if h == host]
        assert len(times) == 3
        assert all(b - a >= 5.0 - 1e-9 for a, b in zip(times, times[1:]))


def test_hosts_are_interleaved_rather_than_waited_out():
    """While a's five seconds run down, the worker talks to b."""
    clock = FakeClock()
    order = []

    def transport(req):
        order.append(recover.host_of(req.url))
        return Resp(req.url, 404, {}, b"", None, 0.0)

    worker = recover.Worker(None, delay=5.0, clock=clock, sleeper=clock.sleep,
                            transport=transport)
    worker.start(r for r in [Req("https://a.example/1"), Req("https://a.example/2")])
    worker.start(r for r in [Req("https://b.example/1")])
    worker.run()
    assert order[:2] == ["a.example", "b.example"]


def test_only_one_request_is_ever_in_flight():
    in_flight = {"n": 0, "max": 0}
    clock = FakeClock()

    def transport(req):
        in_flight["n"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["n"])
        in_flight["n"] -= 1
        return Resp(req.url, 404, {}, b"", None, 0.0)

    worker = recover.Worker(None, delay=5.0, clock=clock, sleeper=clock.sleep,
                            transport=transport)
    for h in "abcde":
        worker.start((Req(f"https://{h}.example/{i}") for i in range(3)))
    worker.run()
    assert in_flight["max"] == 1 and worker.requests == 15


def test_the_transfer_budget_stops_the_run():
    clock = FakeClock()
    worker = recover.Worker(None, delay=5.0, clock=clock, sleeper=clock.sleep,
                            transport=lambda r: Resp(r.url, 200, {}, b"x" * 100, None, 0.0),
                            budget_bytes=150)
    worker.start((Req(f"https://a.example/{i}") for i in range(5)))
    with pytest.raises(recover.BudgetExceeded):
        worker.run()


# --------------------------------------------------------------------------
# small pure pieces
# --------------------------------------------------------------------------


@pytest.mark.parametrize("message,expected", [
    ("HTTPSConnectionPool: Failed to resolve 'gone.example' ([Errno 11001] getaddrinfo failed)", "dns_failure"),
    ("Read timed out. (read timeout=40)", "timeout"),
    ("SSLError: certificate verify failed", "ssl_error"),
    ("Connection reset by peer", "connection_error"),
])
def test_transport_failures_are_named_so_dns_is_distinguishable(message, expected):
    assert recover.classify_exception(RuntimeError(message)) == expected


def test_cdx_json_is_parsed_by_its_header_row():
    body = json.dumps([["timestamp", "original", "statuscode", "mimetype", "digest"],
                       ["20200101000000", "http://x.example/a.jpg", "200", "image/jpeg", "D1"]]).encode()
    rows = recover.parse_cdx(body)
    assert rows == [{"timestamp": "20200101000000", "original": "http://x.example/a.jpg",
                     "statuscode": "200", "mimetype": "image/jpeg", "digest": "D1"}]
    assert recover.parse_cdx(b"") == [] and recover.parse_cdx(b"[]") == []


def test_snapshots_are_this_path_only_exact_query_first_one_per_content():
    url = "https://www.snopes.com/tachyon/2018/02/a.jpg?resize=865%2C452"
    rows = [
        {"timestamp": "20190101000000", "original": "https://www.snopes.com/tachyon/2018/02/a.jpg?w=10", "mimetype": "image/jpeg", "digest": "B"},
        {"timestamp": "20220101000000", "original": "http://snopes.com/tachyon/2018/02/a.jpg?resize=865%2C452", "mimetype": "image/jpeg", "digest": "A"},
        {"timestamp": "20220102000000", "original": "https://www.snopes.com/tachyon/2018/02/a.jpg?resize=865%2C452", "mimetype": "image/jpeg", "digest": "A"},
        {"timestamp": "20220101000000", "original": "https://www.snopes.com/tachyon/2018/02/ab.jpg", "mimetype": "image/jpeg", "digest": "C"},
        {"timestamp": "20220101000000", "original": "https://www.snopes.com/tachyon/2018/02/a.jpg", "mimetype": "text/html", "digest": "H"},
    ]
    chosen = recover.choose_snapshots(url, rows, prefer_year=2022)
    assert [r["digest"] for r in chosen] == ["A", "B"]


def test_the_snopes_cdn_path_maps_to_the_media_host_and_nothing_else_does():
    assert recover.snopes_equivalent(
        "https://www.snopes.com/tachyon/2016/10/x.jpg?resize=1") == "https://media.snopes.com/2016/10/x.jpg"
    assert recover.snopes_equivalent("https://media.snopes.com/2016/10/x.jpg") is None
    assert recover.snopes_equivalent("https://i.imgur.com/abc.jpg") is None


def test_snopes_directories_are_batched_and_others_are_not():
    d = recover.CdxCache.directory
    assert d("https://www.snopes.com/tachyon/2016/10/x.jpg") == "www.snopes.com/tachyon/2016/10/"
    assert d("https://i.imgur.com/abc.jpg") is None
    assert d("https://www.snopes.com/x.jpg") is None


def test_mediaproxy_wrapped_urls_yield_the_url_they_wrap():
    item = Item("images/img_0.jpg",
                "https://mediaproxy.snopes.com/width/600/https://media.snopes.com/2022/12/a.jpg", "true")
    assert recover.declared_alternates("verite", item) == [
        ("https://media.snopes.com/2022/12/a.jpg", "mediaproxy_inner_url")]
    assert recover.declared_alternates("fakeddit", Item("k", "https://i.redd.it/x.jpg", "0")) == []


def test_cdx_queries_do_not_double_encode_an_encoded_url():
    q = recover.cdx_url("https://images.thequint.com/thequint%2F2018-09%2Fa.jpg", prefix=False)
    assert "thequint%2F2018-09" in q and "%252F" not in q


# --------------------------------------------------------------------------
# the pipeline, end to end, against a fake network and a temp tree
# --------------------------------------------------------------------------


@pytest.fixture
def tree(tmp_path, monkeypatch):
    raw = tmp_path / "data" / "raw"

    def fake_raw_dir(name):
        return raw / name

    monkeypatch.setattr(hydrate, "raw_dir", fake_raw_dir)
    monkeypatch.setattr(recover, "raw_dir", fake_raw_dir)
    monkeypatch.setattr(images, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(recover, "PROJECT_ROOT", tmp_path)
    ledgers = {}

    def fake_ledger(name):
        return ledgers.setdefault(name, ProvenanceLedger(raw / name / "media" / "provenance.jsonl"))

    monkeypatch.setattr(recover, "provenance_ledger", fake_ledger)
    return tmp_path


def _run(units, transport, routes=recover.NETWORK_ROUTES):
    clock = FakeClock()
    log = recover.Log()
    cache = recover.CdxCache()
    tally = recover.Tally()
    index = {n: {} for n in recover.DATASETS}
    done = {n: {} for n in recover.DATASETS}
    worker = recover.Worker(None, delay=5.0, clock=clock, sleeper=clock.sleep,
                            transport=transport)
    for unit in units:
        worker.start(recover.pipeline(unit, routes, done, log, cache, tally, index, clock))
    worker.run()
    return tally, index, log


def test_origin_refusal_then_archive_success_lands_in_the_archive_directory(tree):
    image = _png()
    item = Item("k1", "https://blocked.example/a.jpg", "Refute")
    cdx = json.dumps([["timestamp", "original", "statuscode", "mimetype", "digest"],
                      ["20220101000000", "https://blocked.example/a.jpg", "200", "image/jpeg", "D"]]).encode()

    def transport(req):
        if req.url.startswith("https://blocked.example"):
            return Resp(req.url, 403, {}, b"", None, 0.1)
        if "/cdx/" in req.url:
            return Resp(req.url, 200, {}, cdx, None, 0.1)
        return Resp(req.url, 200, {}, image, None, 0.1)

    tally, index, log = _run([recover.Unit(item.url, [recover.Member("factify2", item)])], transport)
    written = tree / "data" / "raw" / "factify2" / "media" / "images_wayback" / "k1.jpg"
    assert written.read_bytes() == image
    assert tally.by_route[("factify2", "wayback")] == 1
    rows = log.ledgers["factify2"].read()
    attempts = [r for r in rows if r["event"] == "attempt"]
    assert {"url", "host", "http_status", "outcome", "attempt_reason", "pass"} <= set(attempts[0])
    assert attempts[0]["http_status"] == 403 and attempts[0]["host"] == "blocked.example"
    results = {r["route"]: r for r in rows if r["event"] == "route_result"}
    assert results["origin"]["outcome"] == "failed" and results["origin"]["attempt_reason"] == "http_403"
    assert results["wayback"]["outcome"] == "ok" and "alternate" not in results


def test_a_placeholder_is_a_failure_even_with_status_200(tree, monkeypatch):
    placeholder = _png((255, 255, 255), (1, 1))
    import hashlib
    from scripts import placeholders
    digest = hashlib.sha256(placeholder).hexdigest()
    monkeypatch.setattr(placeholders, "missing_hashes", lambda: frozenset({digest}))
    monkeypatch.setattr(placeholders, "describe", lambda d: "blank: test")
    item = Item("k2", "https://i.example/a.jpg", "4")

    def transport(req):
        if "/cdx/" in req.url:
            return Resp(req.url, 200, {}, b"[]", None, 0.1)
        return Resp(req.url, 200, {}, placeholder, None, 0.1)

    tally, _index, log = _run([recover.Unit(item.url, [recover.Member("fakeddit", item)])], transport)
    assert not tally.by_route
    results = {r["route"]: r for r in log.ledgers["fakeddit"].read() if r["event"] == "route_result"}
    assert results["origin"]["attempt_reason"].startswith("placeholder_image")
    assert results["wayback"]["attempt_reason"] == "no_snapshot"
    assert results["alternate"]["attempt_reason"] == "no_alternate_in_metadata"


def test_an_unreadable_index_is_never_recorded_as_no_snapshot(tree):
    item = Item("k3", "https://gone.example/a.jpg", "0")

    def transport(req):
        if "/cdx/" in req.url:
            return Resp(req.url, 503, {}, b"", None, 0.1)
        return Resp(req.url, 404, {}, b"", None, 0.1)

    _t, _i, log = _run([recover.Unit(item.url, [recover.Member("fakeddit", item)])],
                       transport, routes=("origin", "wayback"))
    results = {r["route"]: r for r in log.ledgers["fakeddit"].read() if r["event"] == "route_result"}
    assert results["wayback"]["attempt_reason"].startswith("cdx_unavailable")


def test_a_shared_url_is_fetched_once_and_fills_every_item(tree):
    image = _png((1, 1, 1))
    a, b = Item("a", "https://x.example/same.jpg", "0"), Item("b", "https://x.example/same.jpg", "1")
    calls = []

    def transport(req):
        calls.append(req.url)
        return Resp(req.url, 200, {}, image, None, 0.1)

    tally, _i, _l = _run([recover.Unit(a.url, [recover.Member("fakeddit", a),
                                               recover.Member("fakeddit", b)])], transport)
    assert calls == ["https://x.example/same.jpg"]
    assert tally.by_route[("fakeddit", "origin")] == 2


def test_recovery_never_overwrites_a_file_already_in_raw(tree):
    item = Item("k4", "https://x.example/a.jpg", "0")
    occupied = hydrate.destination("fakeddit", item, hydrate.ORIGIN)
    occupied.parent.mkdir(parents=True)
    occupied.write_bytes(b"the placeholder that is already there")
    target = recover.free_destination("fakeddit", item, recover.ORIGIN_RETRY)
    assert target != occupied and target.parent.name == "images_retry"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"first retry")
    again = recover.free_destination("fakeddit", item, recover.ORIGIN_RETRY)
    assert again != target and again.name == "k4.v2.jpg"


def test_redirect_hops_are_throttled_and_logged(tree):
    image = _png((3, 3, 3))
    item = Item("k5", "https://short.example/a", "0")
    seen = []

    def transport(req):
        seen.append(req.url)
        if req.url == "https://short.example/a":
            return Resp(req.url, 302, {"location": "https://cdn.example/a.jpg"}, b"", None, 0.0)
        return Resp(req.url, 200, {}, image, None, 0.0)

    tally, _i, log = _run([recover.Unit(item.url, [recover.Member("fakeddit", item)])], transport)
    assert seen == ["https://short.example/a", "https://cdn.example/a.jpg"]
    outcomes = [r["outcome"] for r in log.ledgers["fakeddit"].read() if r["event"] == "attempt"]
    assert outcomes == ["redirect", "ok"]


# --------------------------------------------------------------------------
# the image index
# --------------------------------------------------------------------------


def test_a_truncated_jpeg_verifies_but_does_not_load():
    """Why the index decodes fully: verify() alone passes a cut-off JPEG."""
    from PIL import Image
    buf = io.BytesIO()
    Image.effect_noise((64, 64), 50).convert("RGB").save(buf, format="JPEG", quality=95)
    cut = buf.getvalue()[: len(buf.getvalue()) // 2]
    row = images.examine_bytes(cut)
    assert row["loads"] is False
    assert images.status(row) == images.UNDECODABLE


def test_html_is_undecodable_and_a_registered_hash_is_not_usable(monkeypatch):
    from scripts import placeholders
    html = images.examine_bytes(b"<!DOCTYPE html><html>error</html>")
    assert images.status(html) == images.UNDECODABLE and "head" in html
    good = images.examine_bytes(_png())
    assert images.status(good) == images.USABLE
    monkeypatch.setattr(placeholders, "registry",
                        lambda: {good["sha256"]: {"status": "furniture"}})
    assert images.status(good) == "furniture"


def test_a_flat_image_is_marked_flat():
    assert images.examine_bytes(_png((9, 9, 9)))["flat"] is True


def test_a_wayback_replay_url_is_rewritten_to_ask_for_original_bytes():
    assert recover.raw_snapshot(
        "https://web.archive.org/web/20220404045630/https://x.example/a.jpg"
    ) == "https://web.archive.org/web/20220404045630id_/https://x.example/a.jpg"
    assert recover.raw_snapshot(
        "https://web.archive.org/web/20220404045630im_/https://x.example/a.jpg"
    ) == "https://web.archive.org/web/20220404045630id_/https://x.example/a.jpg"
    assert recover.raw_snapshot("https://x.example/a.jpg") == "https://x.example/a.jpg"


def test_directory_listings_survive_a_restart(tmp_path):
    cache = recover.CdxCache(tmp_path)
    rows = [{"timestamp": "20220101000000", "original": "https://www.snopes.com/tachyon/2016/10/a.jpg"}]
    cache.save("www.snopes.com/tachyon/2016/10/", rows)
    again = recover.CdxCache(tmp_path)
    assert again.state["www.snopes.com/tachyon/2016/10/"] == "done"
    assert again.rows["www.snopes.com/tachyon/2016/10/"] == rows


@pytest.mark.parametrize("reason,status,expected", [
    ("http_403", 403, "blocked_403"), ("http_404", 404, "gone_404"), ("http_410", 410, "gone_410"),
    ("dns_failure", None, "dns_failure"), ("timeout", None, "unreachable_timeout"),
    ("placeholder_image[placeholder]+redirect_to_removed", 200, "removed_by_host"),
    ("placeholder_image[furniture]", 200, "host_serves_placeholder"),
    ("not_an_image", 200, "host_serves_non_image"), ("http_503", 503, "server_error_5xx"),
    (None, None, "not_attempted"),
])
def test_terminal_reasons_name_what_the_host_did(reason, status, expected):
    assert recover.origin_category(reason, status) == expected


def test_an_unreadable_archive_index_is_never_called_no_snapshot():
    assert recover.archive_category("cdx_unavailable:http_503") == "archive_index_unavailable"
    assert recover.archive_category("no_snapshot") == "no_snapshot"
    assert recover.archive_category("snapshot_placeholder_image[placeholder]") == "snapshots_are_placeholders"


def test_a_prior_success_whose_file_was_later_rejected_moves_on_to_the_next_route(tree):
    """Bytes registered or fingerprint-rejected after the fetch: try the routes
    after the one that produced them, never that route again (same bytes)."""
    item = Item("images/img_1.jpg",
                "https://mediaproxy.snopes.com/width/600/https://media.snopes.com/a.jpg", "out-of-context")
    m = recover.Member("verite", item)
    rel = "data/raw/verite/data/VERITE/images_wayback/img_1.jpg"
    index = {n: {} for n in recover.DATASETS}
    index["verite"][rel] = {"path": rel, "loads": True, "sha256": "f" * 64}
    done = {n: {} for n in recover.DATASETS}
    done["verite"][item.key] = {
        "origin": {"outcome": "failed", "attempt_reason": "http_404"},
        "wayback": {"outcome": "ok", "path": "data/VERITE/images_wayback/img_1.jpg"}}
    calls = []

    def transport(req):
        calls.append(req.url)
        return Resp(req.url, 404, {}, b"", None, 0.1)

    def drive():
        clock = FakeClock()
        worker = recover.Worker(None, delay=5.0, clock=clock, sleeper=clock.sleep, transport=transport)
        worker.start(recover.pipeline(recover.Unit(item.url, [m]), recover.NETWORK_ROUTES, done,
                                      recover.Log(), recover.CdxCache(), recover.Tally(), index, clock))
        worker.run()

    drive()                       # file still usable: the earlier success stands
    assert calls == []
    images.fingerprint_rejections.cache_clear()
    try:
        rejection = {("verite", "f" * 64): {"dataset": "verite", "sha256": "f" * 64}}
        original = images.fingerprint_rejections
        images.fingerprint_rejections = lambda: rejection
        drive()                   # rejected: the alternate route runs, wayback does not
    finally:
        images.fingerprint_rejections = original
    assert calls and all("media.snopes.com/a.jpg" in u for u in calls)
    assert not any("mediaproxy" in u for u in calls)


def test_a_fingerprint_rejection_names_bytes_within_one_dataset(monkeypatch):
    """Keyed by (dataset, sha256): any path in that dataset with those bytes,
    never the same bytes in another corpus, never different bytes."""
    monkeypatch.setattr(images, "fingerprint_rejections",
                        lambda: {("verite", "a" * 64): {"dataset": "verite", "sha256": "a" * 64}})
    here = "data/raw/verite/data/VERITE/images_wayback/img_9.jpg"
    assert images.status({"path": here, "loads": True, "sha256": "a" * 64}) == images.WRONG_IMAGE
    assert images.status({"path": here, "loads": True, "sha256": "b" * 64}) == images.USABLE
    other = "data/raw/fakeddit/media/images/abc.jpg"
    assert images.status({"path": other, "loads": True, "sha256": "a" * 64}) == images.USABLE


def test_an_unreachable_archive_index_is_retried_but_a_real_answer_is_not():
    assert not recover._route_settled(None)
    assert not recover._route_settled({"outcome": "failed", "attempt_reason": "cdx_unavailable:http_503"})
    assert recover._route_settled({"outcome": "failed", "attempt_reason": "no_snapshot"})
    assert recover._route_settled({"outcome": "failed", "attempt_reason": "http_404"})
    assert recover._route_settled({"outcome": "ok", "path": "x.jpg"})


def test_an_index_refusal_is_terminal_but_an_outage_is_not():
    assert recover.archive_category("cdx_unavailable:cdx_http_404") == "archive_refuses_host_404"
    assert recover.archive_category("cdx_unavailable:http_503") == "archive_index_unavailable"
    assert recover._route_settled({"outcome": "failed", "attempt_reason": "cdx_unavailable:cdx_http_404"})
    assert not recover._route_settled({"outcome": "failed", "attempt_reason": "cdx_unavailable:timeout"})
