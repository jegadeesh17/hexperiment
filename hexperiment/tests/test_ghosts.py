"""Tests of the ghost catalogue (Requirements 4.3, 4.5)."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hexperiment import canary
from hexperiment.checks import row_overlap, target_proxy, temporal
from hexperiment.errors import HexInputError
from hexperiment.ghosts import CATALOG, Ghost, find_ghost, to_dict
from hexperiment.verdict import FINDING_ORDER

# The ghosts of the product description.
PRODUCT_GHOSTS = {
    "row_overlap": "The Doppelganger",
    "target_proxy": "The Possession",
    "canary": "The Peeking Spirit",
    "temporal": "The Prophet",
    "determinism": "The Shapeshifter",
}


def test_the_catalogue_has_one_entry_for_every_check_in_finding_order():
    assert sorted(entry.check for entry in CATALOG) == sorted(FINDING_ORDER)
    assert len({entry.check for entry in CATALOG}) == len(CATALOG)


def test_the_five_ghost_names_equal_the_product_table():
    named = {entry.check: entry.ghost for entry in CATALOG if entry.ghost}
    assert named == PRODUCT_GHOSTS


def test_ghost_names_equal_the_constants_of_the_core_modules():
    assert find_ghost("row_overlap").ghost == row_overlap.GHOST
    assert find_ghost("target_proxy").ghost == target_proxy.GHOST
    assert find_ghost("temporal").ghost == temporal.GHOST
    assert find_ghost("canary").ghost == canary.PEEKING_SPIRIT
    assert find_ghost("determinism").ghost == canary.SHAPESHIFTER


def test_the_doppelgangers_fix_removes_the_overlap_after_the_split():
    entry = find_ghost("The Doppelganger")
    assert entry.how_to_fix == (
        "After `hexperiment.split`, drop every test row that also appears in the "
        "training rows. Do not deduplicate the whole frame before the split; the "
        "canary run would then end `UNVERIFIED`."
    )


def test_the_doppelgangers_catalogue_fix_agrees_with_the_finding_fix_hint():
    # The finding says it in one clause; the catalogue adds the reason.
    finding_fix = row_overlap.FIX
    assert finding_fix == (
        "after the split, drop every test row that also appears in the training rows"
    )
    assert finding_fix.split(", ", 1)[1] in find_ghost("row_overlap").how_to_fix


def test_every_entry_has_all_three_texts():
    for entry in CATALOG:
        assert entry.meaning.strip()
        assert entry.why_it_matters.strip()
        assert entry.how_to_fix.strip()


@pytest.mark.parametrize(
    "name",
    ["The Peeking Spirit", "peeking spirit", "peeking-spirit", "canary", "CANARY"],
)
def test_the_peeking_spirit_is_found_under_its_variants(name):
    assert find_ghost(name).check == "canary"


def test_a_check_without_a_ghost_is_found_by_its_check_name():
    entry = find_ghost("test_set_uses")
    assert entry.ghost is None
    assert entry.check == "test_set_uses"


def test_to_dict_has_the_five_fields():
    data = to_dict(find_ghost("the prophet"))
    assert set(data) == {"check", "ghost", "meaning", "why_it_matters", "how_to_fix"}
    assert data["ghost"] == "The Prophet"
    assert data["check"] == "temporal"


@pytest.mark.parametrize("name", ["", "   ", "the", "the ghost", "banshee", "tem"])
def test_an_unknown_name_raises_and_lists_the_known_names(name):
    with pytest.raises(HexInputError) as caught:
        find_ghost(name)
    message = str(caught.value)
    for entry in CATALOG:
        assert entry.check in message
        if entry.ghost:
            assert entry.ghost in message


def test_a_name_that_is_not_a_string_is_unknown():
    with pytest.raises(HexInputError):
        find_ghost(None)  # type: ignore[arg-type]


def test_a_ghost_is_a_frozen_dataclass():
    entry = CATALOG[0]
    assert isinstance(entry, Ghost)
    with pytest.raises(AttributeError):
        entry.check = "other"  # type: ignore[misc]


SEPARATORS = st.sampled_from([" ", "_", "-", "  ", " - "])


@st.composite
def name_variants(draw):
    entry = draw(st.sampled_from(CATALOG))
    base = draw(st.sampled_from([entry.check] + ([entry.ghost] if entry.ghost else [])))
    words = base.replace("_", " ").split()
    if words and words[0].lower() == "the" and draw(st.booleans()):
        words = words[1:]
    separator = draw(SEPARATORS)
    joined = separator.join(words)
    if draw(st.booleans()):
        joined = joined.upper()
    elif draw(st.booleans()):
        joined = joined.lower()
    padding = draw(st.sampled_from(["", " ", "\t"]))
    return entry, f"{padding}{joined}{padding}"


@given(name_variants())
def test_find_ghost_returns_the_entry_for_any_variant_of_its_names(case):
    entry, name = case
    assert find_ghost(name) is entry
