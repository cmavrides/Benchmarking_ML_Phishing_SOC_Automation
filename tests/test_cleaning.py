from common_utils.text_cleaning import normalize, strip_html


def test_strip_html_removes_scripts_and_decodes_entities():
    raw = "<html><body><script>alert('x')</script>Hello &amp; welcome!</body></html>"
    cleaned, was_html = strip_html(raw)
    assert cleaned == "Hello & welcome!"
    assert was_html


def test_normalize_masks_urls_emails_numbers():
    text = "Contact me at Alice@example.com after visiting https://example.com at 5pm"
    normalized = normalize(text)
    assert "<email>" in normalized
    assert normalized.count("<url>") == 1
    assert "<num>" in normalized
