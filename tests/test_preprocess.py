import sys
import os
import unicodedata
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocess import clean_text, tokenize_and_normalize, preprocess_for_features

def test_unicode_normalization():
    """Test that homoglyph attacks are normalized away."""
    # Latin 'o' vs Greek 'ο'
    homoglyph_text = "Verify yοur accοunt" # contains greek omicrons
    normalized = clean_text(homoglyph_text)
    
    # After normalization, it should match the standard ASCII 'o'
    assert "Verify your account" in normalized
    assert any(ord(c) == 111 for c in normalized) # 111 is ASCII 'o'

def test_url_replacement():
    """Test that URLs are replaced with the URL token."""
    text = "Check this out: http://malicious-site.biz/login"
    cleaned = clean_text(text)
    assert "URL" in cleaned
    assert "http" not in cleaned

def test_lemmatization():
    """Test that words are reduced to their root forms."""
    text = "verifying accounts suspended"
    tokens = tokenize_and_normalize(text)
    assert "verify" in tokens
    assert "account" in tokens
    assert "suspend" in tokens

def test_stop_word_preservation():
    """Test that critical phishing words are NOT removed as stop words."""
    text = "verify your account now"
    tokens = tokenize_and_normalize(text)
    assert "verify" in tokens
    assert "now" in tokens
    assert "account" in tokens

def test_preprocess_for_features():
    """Test the full pipeline for features."""
    text = "URGENT: Please verify your portal! http://link.biz"
    processed = preprocess_for_features(text)
    # lowercase, lemmatized, URLs removed
    assert "urgent" in processed
    assert "verify" in processed
    assert "portal" in processed
    assert "url" in processed
