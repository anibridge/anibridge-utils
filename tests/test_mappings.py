"""Tests for AniBridge mapping schema helpers."""

import pytest

from anibridge.utils.mappings import (
    AnibridgeDescriptorMapping,
    AnibridgeMapping,
    AnibridgeMappingRange,
    descriptor_key,
    format_mapping_range,
    is_valid_source_range,
    is_valid_target_range,
    parse_mapping_descriptor,
    ratio_to_weight,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0", True),
        ("3-7", True),
        ("10-", True),
        (" 42 ", True),
        ("", False),
        ("-1", False),
        ("1--2", False),
        ("1-2-3", False),
        ("abc", False),
        ("1|2", False),
    ],
)
def test_is_valid_source_range(value: str, expected: bool) -> None:
    """Source range validation should accept only source-schema forms."""
    assert is_valid_source_range(value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0", True),
        ("2-4", True),
        ("2-", True),
        ("2,4-6", True),
        ("2-4|2", True),
        ("2,4-6|-3", True),
        (" 2-4|2 ", True),
        ("", False),
        ("2|", False),
        ("2||3", False),
        ("2,", False),
        (",2", False),
        ("2,4|x", False),
        ("2|3,4", False),
    ],
)
def test_is_valid_target_range(value: str, expected: bool) -> None:
    """Target range validation should include optional trailing ratio only."""
    assert is_valid_target_range(value) is expected


def test_mapping_range_constructor_validates_bounds() -> None:
    """Range invariants should reject negative starts and descending ranges."""
    with pytest.raises(ValueError, match="start must be >= 0"):
        AnibridgeMappingRange(start=-1, end=1)

    with pytest.raises(ValueError, match="end must be >= start"):
        AnibridgeMappingRange(start=4, end=3)


def test_mapping_range_contains_for_closed_and_open_ranges() -> None:
    """Containment checks should work for bounded and open-ended ranges."""
    closed = AnibridgeMappingRange(start=3, end=5)
    assert closed.contains(2) is False
    assert closed.contains(3) is True
    assert closed.contains(5) is True
    assert closed.contains(6) is False

    open_ended = AnibridgeMappingRange(start=7, end=None)
    assert open_ended.contains(6) is False
    assert open_ended.contains(7) is True
    assert open_ended.contains(9999) is True


def test_mapping_range_length_for_closed_and_open_ranges() -> None:
    """Length should be inclusive for bounded ranges and None for open-ended."""
    assert AnibridgeMappingRange(start=8, end=8).length == 1
    assert AnibridgeMappingRange(start=8, end=12).length == 5
    assert AnibridgeMappingRange(start=8, end=None).length is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("4", AnibridgeMappingRange(start=4, end=4)),
        ("4-6", AnibridgeMappingRange(start=4, end=6)),
        ("4-", AnibridgeMappingRange(start=4, end=None)),
        (" 9-10 ", AnibridgeMappingRange(start=9, end=10)),
    ],
)
def test_mapping_range_parse_valid(value: str, expected: AnibridgeMappingRange) -> None:
    """Range parser should normalize all valid source-range forms."""
    assert AnibridgeMappingRange.parse(value) == expected


@pytest.mark.parametrize("value", ["", "-1", "1--2", "abc", "1|2"])
def test_mapping_range_parse_invalid(value: str) -> None:
    """Range parser should reject non-source-schema forms."""
    with pytest.raises(ValueError, match="Invalid source range"):
        AnibridgeMappingRange.parse(value)


def test_format_mapping_range_serializes_single_closed_and_open() -> None:
    """Formatting should produce canonical source/target range strings."""
    assert format_mapping_range(AnibridgeMappingRange(start=5, end=5)) == "5"
    assert format_mapping_range(AnibridgeMappingRange(start=5, end=8)) == "5-8"
    assert format_mapping_range(AnibridgeMappingRange(start=5, end=None)) == "5-"


def test_mapping_parse_and_serialize_round_trip() -> None:
    """Mapping parser should preserve source and target serialization."""
    mapping = AnibridgeMapping.parse("1-3", "10-12,20")

    assert mapping.source_range == AnibridgeMappingRange(start=1, end=3)
    assert mapping.target_ranges == (
        AnibridgeMappingRange(start=10, end=12),
        AnibridgeMappingRange(start=20, end=20),
    )
    assert mapping.target_ratio is None
    assert mapping.source_key == "1-3"
    assert mapping.target_value == "10-12,20"
    assert mapping.as_pair() == ("1-3", "10-12,20")


def test_mapping_parse_includes_optional_ratio() -> None:
    """Mapping parser should parse trailing target ratio and expose weights."""
    mapping = AnibridgeMapping.parse("1-4", "20-21|2")

    assert mapping.target_ratio == 2
    assert mapping.target_value == "20-21|2"
    assert mapping.source_weight == 0.5
    assert mapping.target_weight == 0.5


def test_mapping_parse_rejects_invalid_target_ranges() -> None:
    """Mapping parser should validate target syntax before parsing."""
    with pytest.raises(ValueError, match="Invalid target range"):
        AnibridgeMapping.parse("1-3", "10-12|")


def test_mapping_parse_rejects_invalid_source_range() -> None:
    """Mapping parser should surface source-range parse errors."""
    with pytest.raises(ValueError, match="Invalid source range"):
        AnibridgeMapping.parse("1|2", "10-12")


def test_mapping_constructor_requires_at_least_one_target_range() -> None:
    """Direct construction should reject mappings without targets."""
    with pytest.raises(ValueError, match="At least one target range is required"):
        AnibridgeMapping(
            source_range=AnibridgeMappingRange(start=1, end=1),
            target_ranges=(),
        )


def test_mapping_ratio_validation_positive_ratio_valid() -> None:
    """Positive ratio should require target length == source length / ratio."""
    mapping = AnibridgeMapping.parse("1-4", "10-11|2")

    assert mapping.target_ratio == 2


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("1-3", "10-11|2"),
        ("1-4", "10-12|2"),
    ],
)
def test_mapping_ratio_validation_positive_ratio_invalid(
    source: str, target: str
) -> None:
    """Positive ratio combinations with non-matching lengths should fail."""
    with pytest.raises(ValueError, match="Mapping ratio does not align"):
        AnibridgeMapping.parse(source, target)


def test_mapping_ratio_validation_negative_ratio_valid() -> None:
    """Negative ratio should require target length == source length * abs(ratio)."""
    mapping = AnibridgeMapping.parse("1-2", "20-25|-3")

    assert mapping.target_ratio == -3


def test_mapping_ratio_validation_negative_ratio_invalid() -> None:
    """Negative ratio combinations with mismatched lengths should fail."""
    with pytest.raises(ValueError, match="Mapping ratio does not align"):
        AnibridgeMapping.parse("1-2", "20-23|-3")


def test_mapping_ratio_validation_zero_ratio_skips_length_check() -> None:
    """Zero ratio should bypass ratio-length validation logic."""
    mapping = AnibridgeMapping.parse("1-3", "10-20|0")

    assert mapping.target_ratio == 0
    assert mapping.source_weight == 0.0
    assert mapping.target_weight == 0.0


def test_mapping_ratio_validation_skips_open_ended_source_or_target() -> None:
    """Ratio-length validation should be skipped for open-ended ranges."""
    source_open = AnibridgeMapping.parse("1-", "10-12|2")
    target_open = AnibridgeMapping.parse("1-4", "10-|2")

    assert source_open.target_ratio == 2
    assert target_open.target_ratio == 2


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (None, 1.0),
        (1, 1.0),
        (0, 0.0),
        (2, 0.5),
        (-4, 4.0),
    ],
)
def test_ratio_to_weight(ratio: int | None, expected: float) -> None:
    """Ratio-to-weight conversion should match mapping weighting semantics."""
    assert ratio_to_weight(ratio) == expected


def test_parse_mapping_descriptor_valid_with_and_without_scope() -> None:
    """Descriptor parser should support provider:id and provider:id:scope."""
    assert parse_mapping_descriptor("mal:123") == ("mal", "123", None)
    assert parse_mapping_descriptor(" ani:abc:tv ") == ("ani", "abc", "tv")


@pytest.mark.parametrize(
    "value",
    [
        "",
        "missing-delimiter",
        "provider:",
        ":id",
        "provider:id:scope:extra",
    ],
)
def test_parse_mapping_descriptor_invalid(value: str) -> None:
    """Descriptor parser should reject malformed descriptor strings."""
    with pytest.raises(ValueError, match="Invalid descriptor"):
        parse_mapping_descriptor(value)


def test_descriptor_key_serialization() -> None:
    """Descriptor key serialization should include optional scope only when set."""
    assert descriptor_key(("mal", "123", None)) == "mal:123"
    assert descriptor_key(("mal", "123", "tv")) == "mal:123:tv"


def test_descriptor_mapping_from_strings_parses_descriptors() -> None:
    """Factory should parse source and target descriptor strings."""
    mapping = AnibridgeDescriptorMapping.from_strings("mal:1", "ani:2:tv")

    assert mapping.source == ("mal", "1", None)
    assert mapping.target == ("ani", "2", "tv")
    assert mapping.mappings == []


def test_descriptor_mapping_add_mapping_appends_and_returns_mapping() -> None:
    """Adding a mapping should append parsed mapping and return the same object."""
    descriptor_mapping = AnibridgeDescriptorMapping.from_strings("mal:1", "ani:2")

    created = descriptor_mapping.add_mapping("1-2", "10-11")

    assert descriptor_mapping.mappings == [created]
    assert created.as_pair() == ("1-2", "10-11")


def test_descriptor_mapping_from_schema_map_parses_all_entries() -> None:
    """Schema-map factory should parse and preserve all range mappings."""
    descriptor_mapping = AnibridgeDescriptorMapping.from_schema_map(
        source="mal:100:tv",
        target="ani:200",
        range_map={
            "1-3": "10-12",
            "4": "13|1",
        },
    )

    assert descriptor_mapping.source == ("mal", "100", "tv")
    assert descriptor_mapping.target == ("ani", "200", None)
    assert [entry.as_pair() for entry in descriptor_mapping.mappings] == [
        ("1-3", "10-12"),
        ("4", "13|1"),
    ]


def test_descriptor_mapping_to_schema_map_serializes_entries() -> None:
    """Schema-map serialization should emit source->target mapping pairs."""
    descriptor_mapping = AnibridgeDescriptorMapping.from_strings("mal:1", "ani:2")
    descriptor_mapping.add_mapping("1-3", "10-12")
    descriptor_mapping.add_mapping("4", "13|1")

    assert descriptor_mapping.to_schema_map() == {
        "1-3": "10-12",
        "4": "13|1",
    }


def test_descriptor_mapping_from_schema_map_propagates_mapping_errors() -> None:
    """Schema-map factory should raise when any range entry is invalid."""
    with pytest.raises(ValueError, match="Invalid target range"):
        AnibridgeDescriptorMapping.from_schema_map(
            source="mal:1",
            target="ani:2",
            range_map={"1": "10|"},
        )
