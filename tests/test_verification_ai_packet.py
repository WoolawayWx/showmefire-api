import json

from ai.verification_summary import build_verification_ai_packet, write_verification_ai_packet


def test_verification_ai_packet_contains_today_and_recent_context(tmp_path):
    report = {
        "date": "2026-08-29",
        "generated_at": "2026-08-30T03:00:00Z",
        "record_count": 12,
        "stations_count": 4,
        "metrics": {
            "Temperature (C)": {"mae": 1.2, "bias": -0.4, "count": 12},
        },
        "comparison_rows": [{"ignored": True}],
    }
    packet = build_verification_ai_packet(
        report,
        [{"date": "2026-08-28", "metrics": {"Temperature (C)": {"mae": 2.0}}}],
        [{"station": "ABC", "forecast": {"temperature_c": 30}}],
    )

    assert packet["schema_version"] == "verification-ai-packet.v1"
    assert packet["today"]["metrics"]["Temperature (C)"]["bias"] == -0.4
    assert packet["recent_history"][0]["date"] == "2026-08-28"
    assert packet["representative_comparisons"][0]["station"] == "ABC"
    assert "output" in packet["instructions"]

    output = write_verification_ai_packet(
        report, [], [], tmp_path / "verification_ai_packet.json"
    )
    assert json.loads(output.read_text())["today"]["date"] == "2026-08-29"
