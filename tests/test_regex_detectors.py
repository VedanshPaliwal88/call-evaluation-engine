from __future__ import annotations

from pathlib import Path

from call_evaluation.detectors.regex.compliance import RegexComplianceDetector
from call_evaluation.ingestion import IngestionService


FIXTURES = Path(__file__).parent / "fixtures"


def test_compliance_detector_flags_voicemail_disclosure() -> None:
    transcript = IngestionService().load_named_bytes(
        "case.json",
        b"""
        [
          {"speaker":"Agent","text":"Hello Brian, this is collections. Your balance is 500 dollars. Please call us back.","stime":0,"etime":6}
        ]
        """,
    )[0]
    result = RegexComplianceDetector().analyze(transcript)
    assert result.violation.value == "YES"
    assert result.verification_status.value == "NOT_APPLICABLE"


def test_compliance_detector_respects_verification_sequence() -> None:
    transcript = IngestionService().load_path(FIXTURES / "valid_transcript.yaml")[0]
    result = RegexComplianceDetector().analyze(transcript)
    assert result.violation.value == "NO"
    assert result.verification_status.value == "VERIFIED"


def test_compliance_false_positive_traps() -> None:
    detector = RegexComplianceDetector()
    cases = [
        (
            """
            [
              {"speaker":"Agent","text":"A balance is the amount owed on an account in general terms.","stime":0,"etime":5},
              {"speaker":"Customer","text":"Okay, I understand.","stime":5.5,"etime":7}
            ]
            """,
            "NO",
        ),
        (
            """
            [
              {"speaker":"Agent","text":"Please verify your date of birth.","stime":0,"etime":3},
              {"speaker":"Customer","text":"I have told you 3 times already.","stime":3.5,"etime":6},
              {"speaker":"Agent","text":"Understood, without verification I cannot continue.","stime":6.5,"etime":10}
            ]
            """,
            "NO",
        ),
        (
            """
            [
              {"speaker":"Agent","text":"Please verify your date of birth.","stime":0,"etime":3},
              {"speaker":"Customer","text":"January 5th 1984.","stime":3.5,"etime":5},
              {"speaker":"Agent","text":"Your account is in good standing.","stime":5.5,"etime":8}
            ]
            """,
            "NO",
        ),
    ]
    for raw_text, expected in cases:
        transcript = IngestionService().load_named_bytes("case.json", raw_text.encode())[0]
        result = detector.analyze(transcript)
        assert result.violation.value == expected


def test_compliance_false_negative_traps() -> None:
    detector = RegexComplianceDetector()
    cases = [
        """
        [
          {"speaker":"Agent","text":"Can you confirm your name?","stime":0,"etime":3},
          {"speaker":"Customer","text":"Yes, this is Alex.","stime":3.5,"etime":5},
          {"speaker":"Agent","text":"You owe $450 on this account.","stime":5.5,"etime":9}
        ]
        """,
        """
        [
          {"speaker":"Agent","text":"Your overdue balance is 450 dollars.","stime":0,"etime":4},
          {"speaker":"Customer","text":"Who is this?","stime":4.5,"etime":6}
        ]
        """,
        """
        [
          {"speaker":"Agent","text":"I see your last payment was in March.","stime":0,"etime":4},
          {"speaker":"Customer","text":"Okay.","stime":4.5,"etime":5.5}
        ]
        """,
    ]
    for raw_text in cases:
        transcript = IngestionService().load_named_bytes("case.json", raw_text.encode())[0]
        result = detector.analyze(transcript)
        assert result.violation.value == "YES"


def test_compliance_verification_boundaries() -> None:
    detector = RegexComplianceDetector()
    cases = [
        ("yes that's correct", "UNVERIFIED"),
        ("my birthday is january 5th 1984", "VERIFIED"),
        ("one two three main street", "VERIFIED"),
        ("it's 415-22-3456", "VERIFIED"),
        ("i've told you 3 times already", "UNVERIFIED"),
        ("5", "UNVERIFIED"),
        ("sure", "UNVERIFIED"),
    ]
    for customer_text, expected_status in cases:
        transcript = IngestionService().load_named_bytes(
            "case.json",
            f"""
            [
              {{"speaker":"Agent","text":"Please verify your date of birth, address, or social security number.","stime":0,"etime":4}},
              {{"speaker":"Customer","text":"{customer_text}","stime":4.5,"etime":8}},
              {{"speaker":"Agent","text":"You owe $450 on this account.","stime":8.5,"etime":12}}
            ]
            """.encode(),
        )[0]
        result = detector.analyze(transcript)
        assert result.verification_status.value == expected_status


def test_compliance_explicit_edge_cases_exist_for_regex_path() -> None:
    detector = RegexComplianceDetector()
    cases = [
        (
            """
            [
              {"speaker":"Agent","text":"Hello Brian, this is collections. Your balance is 500 dollars. Please call us back.","stime":0,"etime":6}
            ]
            """,
            ("YES", "NOT_APPLICABLE"),
        ),
        (
            """
            [
              {"speaker":"Agent","text":"May I speak to Brian?","stime":0,"etime":2},
              {"speaker":"Customer","text":"You have the wrong person.","stime":2.5,"etime":4},
              {"speaker":"Agent","text":"Understood, I will remove this number.","stime":4.5,"etime":6}
            ]
            """,
            ("NO", "NOT_APPLICABLE"),
        ),
        (
            """
            [
              {"speaker":"Customer","text":"This is Alex.","stime":0,"etime":2},
              {"speaker":"Agent","text":"Okay, I see the right account here. You owe 450 dollars.","stime":2.5,"etime":7}
            ]
            """,
            ("YES", "PARTIAL"),
        ),
        (
            """
            [
              {"speaker":"Agent","text":"Can you confirm your name?","stime":0,"etime":2},
              {"speaker":"Customer","text":"This is Alex.","stime":2.5,"etime":4},
              {"speaker":"Agent","text":"Your balance is 450 dollars.","stime":4.5,"etime":8}
            ]
            """,
            ("YES", "PARTIAL"),
        ),
        (
            """
            [
              {"speaker":"Agent","text":"Your balance is 450 dollars.","stime":0,"etime":4},
              {"speaker":"Customer","text":"Why are you calling?","stime":4.5,"etime":6}
            ]
            """,
            ("YES", "UNVERIFIED"),
        ),
    ]
    for raw_text, expected in cases:
        transcript = IngestionService().load_named_bytes("case.json", raw_text.encode())[0]
        result = detector.analyze(transcript)
        assert (result.violation.value, result.verification_status.value) == expected
