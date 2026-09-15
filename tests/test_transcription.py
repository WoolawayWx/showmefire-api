from services.transcription import classify_transcript


def test_classifies_brush_fire_as_wildland_signal():
    result = classify_transcript("Engine responding to a brush fire in the woods with smoke showing")
    assert result["classification"] == "wildland_fire"
    assert "brush fire" in result["evidence"]


def test_rejects_structure_fire_as_wildland_signal():
    result = classify_transcript("Structure fire at a house, medical requested")
    assert result["classification"] == "other"
    assert result["confidence"] < 0.65


def test_empty_transcript_is_rejected():
    result = classify_transcript("")
    assert result["classification"] == "other"
    assert result["confidence"] == 0
