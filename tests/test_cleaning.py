from phishing_benchmark.data.cleaning import filter_by_length, normalize_text, strip_html


def test_strip_html_removes_tags():
    html_text = "<html><body><h1>Alert</h1><script>bad()</script>Click <a href='http://test'>here</a></body></html>"
    cleaned, was_html = strip_html(html_text)
    assert "<" not in cleaned
    assert "bad" not in cleaned
    assert was_html is True


def test_normalize_text_placeholders():
    text = "Contact admin@example.com or visit https://example.com"
    normalized = normalize_text(text)
    assert "<email>" in normalized
    assert "<url>" in normalized


def test_filter_by_length():
    items = ["short", "", "long" * 100]
    mask = list(filter_by_length(items, min_length=3, max_length=50))
    assert mask == [True, False, False]
