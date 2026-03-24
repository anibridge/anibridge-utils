"""AniBridge mapping schema parsing helpers."""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache

from anibridge.utils.types import MappingDescriptor

RANGE_PATTERN = r"\d+(?:-\d*)?"
RANGE_RE = re.compile(rf"^{RANGE_PATTERN}$")
SOURCE_RANGE_RE = re.compile(rf"^{RANGE_PATTERN}$")
TARGET_RANGE_RE = re.compile(rf"^{RANGE_PATTERN}(?:,{RANGE_PATTERN})*(?:\|-?\d+)?$")
DESCRIPTOR_RE = re.compile(
    r"^(?P<provider>\w+):(?P<entry_id>[^:]+)(?::(?P<scope>[^:]+))?$"
)

__all__ = [
    "AnibridgeDescriptorMapping",
    "AnibridgeMapping",
    "AnibridgeMappingRange",
    "descriptor_key",
    "format_mapping_range",
    "is_valid_source_range",
    "is_valid_target_range",
    "ratio_to_weight",
    "parse_mapping_descriptor",
]


@dataclass(frozen=True, slots=True)
class AnibridgeMapping:
    """One source range mapped to one or more target ranges."""

    source_range: AnibridgeMappingRange
    target_ranges: tuple[AnibridgeMappingRange, ...]
    target_ratio: int | None = None

    def __post_init__(self) -> None:
        """Validate mappings."""
        if not self.target_ranges:
            raise ValueError("At least one target range is required.")

        ratio = self.target_ratio
        if ratio is None or ratio == 0:
            return

        source_length = self.source_range.length
        if source_length is None:
            return

        target_lengths = [target_range.length for target_range in self.target_ranges]
        if any(length is None for length in target_lengths):
            return

        total_target_length = sum(length for length in target_lengths if length is not None)

        if ratio > 0:
            expected_target_length = Fraction(source_length, ratio)
        else:
            expected_target_length = Fraction(source_length * abs(ratio), 1)

        if expected_target_length.denominator != 1 or (
            total_target_length != expected_target_length.numerator
        ):
            raise ValueError(
                "Mapping ratio does not align with source/target lengths "
                f"(source_length={source_length}, "
                f"target_length={total_target_length}, ratio={ratio})."
            )

    @classmethod
    def parse(cls, source_range: str, target_ranges: str) -> AnibridgeMapping:
        """Build a mapping from schema string values.

        Args:
            source_range (str): Source range schema text.
            target_ranges (str): Target ranges with optional trailing ratio.

        Returns:
            AnibridgeMapping: Parsed mapping.

        Raises:
            ValueError: If either range string is invalid.
        """
        raw = target_ranges.strip()
        if not bool(TARGET_RANGE_RE.match(raw)):
            raise ValueError(f"Invalid target range '{target_ranges}'")

        if "|" in raw:
            ranges_part, ratio_part = raw.rsplit("|", 1)
            ratio = int(ratio_part)
        else:
            ranges_part = raw
            ratio = None

        ranges = tuple(
            AnibridgeMappingRange.parse(part) for part in ranges_part.split(",")
        )

        return cls(
            source_range=AnibridgeMappingRange.parse(source_range),
            target_ranges=ranges,
            target_ratio=ratio,
        )

    @property
    def source_key(self) -> str:
        """Return serialized source range key.

        Returns:
            str: Serialized source range.
        """
        return format_mapping_range(self.source_range)

    @property
    def target_value(self) -> str:
        """Return serialized target range value.

        Returns:
            str: Serialized target ranges and optional trailing ratio.
        """
        base = ",".join(
            format_mapping_range(segment) for segment in self.target_ranges
        )
        if self.target_ratio is None:
            return base
        return f"{base}|{self.target_ratio}"

    @property
    def source_weight(self) -> float:
        """Return the weight of one source index in this mapping."""
        if self.target_ratio is None:
            return 1.0
        return ratio_to_weight(self.target_ratio)

    @property
    def target_weight(self) -> float:
        """Return the weight of one target index in this mapping."""
        return ratio_to_weight(self.target_ratio)

    def as_pair(self) -> tuple[str, str]:
        """Return `(source_key, target_value)` suitable for schema maps.

        Returns:
            tuple[str, str]: Serialized (source, target) pair.
        """
        return self.source_key, self.target_value


@dataclass(frozen=True, slots=True)
class AnibridgeMappingRange:
    """Normalized representation of one range segment."""

    start: int
    end: int | None

    def __post_init__(self) -> None:
        """Validate normalized range invariants."""
        if self.start < 0:
            raise ValueError("start must be >= 0")
        if self.end is not None and self.end < self.start:
            raise ValueError("end must be >= start")

    def contains(self, value: int) -> bool:
        """Whether an index value falls within [start, end].

        Args:
            value (int): Value to test.

        Returns:
            bool: True if value falls within [start, end].
        """
        if value < self.start:
            return False
        return not (self.end is not None and value > self.end)

    @property
    def length(self) -> int | None:
        """Return inclusive range length when bounded.

        Returns:
            int | None: Inclusive length, or None for open-ended ranges.
        """
        if self.end is None:
            return None
        return self.end - self.start + 1

    @classmethod
    def parse(cls, value: str) -> AnibridgeMappingRange:
        """Parse a range segment from schema text.

        Args:
            value (str): Source-compatible range segment (no ratio supported).

        Returns:
            AnibridgeMappingRange: Parsed normalized range segment.

        Raises:
            ValueError: If value does not match source range schema.
        """
        raw = value.strip()
        if not bool(RANGE_RE.match(raw)):
            raise ValueError(f"Invalid source range '{raw}'")

        if "-" in raw:
            start_str, end_str = raw.split("-", 1)
            start = int(start_str)
            end = int(end_str) if end_str else None
        else:
            start = int(raw)
            end = start

        return AnibridgeMappingRange(start=start, end=end)


@dataclass(slots=True)
class AnibridgeDescriptorMapping:
    """Collection of mappings between a source and target descriptor."""

    source: MappingDescriptor
    target: MappingDescriptor
    mappings: list[AnibridgeMapping] = field(default_factory=list)

    @classmethod
    def from_strings(cls, source: str, target: str) -> AnibridgeDescriptorMapping:
        """Create an empty descriptor mapping from schema descriptor strings.

        Args:
            source (str): Source descriptor text.
            target (str): Target descriptor text.

        Returns:
            AnibridgeDescriptorMapping: Empty mapping with parsed descriptors.

        Raises:
            ValueError: If either descriptor string is invalid.
        """
        return cls(
            source=parse_mapping_descriptor(source),
            target=parse_mapping_descriptor(target),
        )

    @classmethod
    def from_schema_map(
        cls,
        source: str,
        target: str,
        range_map: Mapping[str, str],
    ) -> AnibridgeDescriptorMapping:
        """Parse one schema range map into an AnibridgeDescriptorMapping.

        Args:
            source (str): Source descriptor text.
            target (str): Target descriptor text.
            range_map (Mapping[str, str]): Mapping of source-range to target-ranges.

        Returns:
            AnibridgeDescriptorMapping: Parsed descriptor mapping.

        Raises:
            ValueError: If descriptors or any range entry is invalid.
        """
        mapping = cls.from_strings(source=source, target=target)
        for source_range, target_ranges in range_map.items():
            mapping.add_mapping(source_range=source_range, target_ranges=target_ranges)
        return mapping

    def add_mapping(self, source_range: str, target_ranges: str) -> AnibridgeMapping:
        """Parse and append one mapping.

        Args:
            source_range (str): Source range schema text.
            target_ranges (str): Target ranges schema text.

        Returns:
            AnibridgeMapping: Parsed mapping that was appended.

        Raises:
            ValueError: If either range string is invalid.
        """
        mapping = AnibridgeMapping.parse(
            source_range=source_range,
            target_ranges=target_ranges,
        )
        self.mappings.append(mapping)
        return mapping

    def to_schema_map(self) -> dict[str, str]:
        """Serialize mappings to `{source_range: target_ranges}` schema format.

        Returns:
            dict[str, str]: Serialized mapping entries.
        """
        return {mapping.source_key: mapping.target_value for mapping in self.mappings}

def descriptor_key(descriptor: MappingDescriptor) -> str:
    """Serialize a descriptor tuple into `provider:id[:scope]` format."""
    provider, entry_id, scope = descriptor
    return f"{provider}:{entry_id}:{scope}" if scope else f"{provider}:{entry_id}"


def parse_mapping_descriptor(value: str) -> MappingDescriptor:
    """Parse descriptor text into a `(provider, entry_id, scope)` tuple."""
    raw = value.strip()
    match = DESCRIPTOR_RE.match(raw)
    if match is None:
        raise ValueError(
            f"Invalid descriptor '{value}'. "
            "Expected 'provider:id' or 'provider:id:scope'."
        )
    return (
        match.group("provider"),
        match.group("entry_id"),
        match.group("scope"),
    )

def is_valid_source_range(value: str) -> bool:
    """Check if a string is a valid source range.

    Args:
        value (str): String to validate.

    Returns:
        bool: True if value matches source range schema, False otherwise.
    """
    return bool(SOURCE_RANGE_RE.match(value.strip()))

def is_valid_target_range(value: str) -> bool:
    """Check if a string is a valid target range.

    Args:
        value (str): String to validate.

    Returns:
        bool: True if value matches target range schema, False otherwise.
    """
    return bool(TARGET_RANGE_RE.match(value.strip()))


def format_mapping_range(range_segment: AnibridgeMappingRange) -> str:
    """Serialize one AnibridgeMappingRange into schema text.

    Args:
        range_segment (AnibridgeMappingRange): Range segment to serialize.

    Returns:
        str: Serialized range segment.
    """
    if range_segment.end is None:
        return f"{range_segment.start}-"
    if range_segment.start == range_segment.end:
        return str(range_segment.start)
    return f"{range_segment.start}-{range_segment.end}"


def ratio_to_weight(ratio: int | None) -> float:
    """Convert ratio into weight.

    Args:
        ratio (int | None): Mapping ratio modifier.

    Returns:
        float: Weight expressed as source-per-target contribution.
    """
    if ratio is None or ratio == 1:
        return 1.0
    if ratio == 0:
        return 0.0
    if ratio > 0:
        return 1.0 / ratio
    return float(abs(ratio))
