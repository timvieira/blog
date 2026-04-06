"""Tests for Atom feed generation."""

import xml.etree.ElementTree as ET
from datetime import datetime

import build

NS = {"atom": "http://www.w3.org/2005/Atom"}


def make_post(title="Test Post", slug="test-post", date=None, external_url=""):
    if date is None:
        date = datetime(2025, 1, 15)
    return {
        "title": title,
        "date": date,
        "url": f"{slug}/",
        "external_url": external_url,
    }


def parse_feed(tmp_path, monkeypatch, posts, **kwargs):
    monkeypatch.setattr(build, "OUTPUT_DIR", tmp_path)
    build.build_feed(posts, **kwargs)
    xml_text = (tmp_path / "atom.xml").read_text()
    return ET.fromstring(xml_text)


def test_feed_valid_xml(tmp_path, monkeypatch):
    root = parse_feed(tmp_path, monkeypatch, [make_post()])
    assert root.tag == f"{{{NS['atom']}}}feed"
    entries = root.findall("atom:entry", NS)
    assert len(entries) == 1


def test_external_url_uses_external_link(tmp_path, monkeypatch):
    post = make_post(
        title="External Post",
        slug="external-post",
        external_url="https://example.com/article",
    )
    root = parse_feed(tmp_path, monkeypatch, [post])
    entry = root.find("atom:entry", NS)
    link = entry.find("atom:link", NS).get("href")
    entry_id = entry.find("atom:id", NS).text

    assert link == "https://example.com/article"
    assert entry_id == f"{build.FEED_URL}/external-post/"


def test_regular_post_uses_canonical_url(tmp_path, monkeypatch):
    root = parse_feed(tmp_path, monkeypatch, [make_post()])
    entry = root.find("atom:entry", NS)
    link = entry.find("atom:link", NS).get("href")
    assert link == f"{build.FEED_URL}/test-post/"


def test_all_entry_links_are_valid_urls(tmp_path, monkeypatch):
    posts = [
        make_post(title="Regular", slug="regular", date=datetime(2025, 2, 1)),
        make_post(
            title="External",
            slug="ext",
            date=datetime(2025, 1, 1),
            external_url="https://example.com/ext",
        ),
    ]
    root = parse_feed(tmp_path, monkeypatch, posts)
    for entry in root.findall("atom:entry", NS):
        href = entry.find("atom:link", NS).get("href")
        assert href.startswith("https://"), f"Bad URL: {href}"


def test_xml_special_chars_in_title(tmp_path, monkeypatch):
    post = make_post(title="Foo & Bar <Baz>")
    root = parse_feed(tmp_path, monkeypatch, [post])
    entry = root.find("atom:entry", NS)
    title = entry.find("atom:title", NS).text
    assert title == "Foo & Bar <Baz>"
