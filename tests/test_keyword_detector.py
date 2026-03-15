import sys
import os
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from keyword_detector import scan_text

def test_urgency_detection():
    text = "ACT NOW! Final notice for your account."
    result = scan_text(text)
    assert "urgency" in result.categories
    assert any("act now" in kw.lower() for kw in result.found_keywords)

def test_credential_harvesting_detection():
    text = "Please verify your password at this link."
    result = scan_text(text)
    assert "credential_harvesting" in result.categories
    assert any("verify your password" in kw.lower() for kw in result.found_keywords)

def test_suspicious_url_tld():
    text = "Login at http://secure-update.biz/login"
    result = scan_text(text)
    assert "suspicious_links" in result.categories
    assert any(".biz" in sig["url"] for sig in result.url_signals)

def test_aviation_sector_detection():
    text = "Access your crew portal to view flight schedules."
    result = scan_text(text)
    assert "aviation_sector" in result.categories
    assert any("crew portal" in kw.lower() for kw in result.found_keywords)

def test_risk_score_calculation():
    # Multiple categories should yield higher risk
    low_risk_text = "Hello, how are you?"
    high_risk_text = "URGENT: Verify your crew portal account password now at http://scam.biz"
    
    low_result = scan_text(low_risk_text)
    high_result = scan_text(high_risk_text)
    
    assert high_result.risk_score > low_result.risk_score
    assert high_result.risk_score > 0.7
