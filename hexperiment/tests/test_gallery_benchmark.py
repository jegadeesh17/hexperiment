import copy
import json
import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hexperiment.errors import HexInputError
from hexperiment.gallery.benchmark import (
    BENCHMARK_FORMAT,
    benchmark_section,
    parse_benchmark,
    percent,
    run_count,
)
from hexperiment.ghosts import CATALOG

SOURCE = "results.json"
GHOSTS = frozenset(entry.ghost for entry in CATALOG if entry.ghost is not None)


def mechanism(leak_type="L1.1", name="Pre-processing", **changes):
    value = {
        "leak_type": leak_type,
        "leak_type_name": name,
        "mechanism": f"A leak of type {leak_type}.",
        "designed_to_catch": ["The Peeking Spirit"],
        "scripts": 12,
        "rejected": 9,
        "by_ghost": {"The Peeking Spirit": 9, "split_order": 2},
        "statuses": {"REJECTED": 9, "APPROVED": 3},
    }
    value.update(changes)
    return value


def valid():
    return {
        "format": BENCHMARK_FORMAT,
        "created_at": "2026-11-11T19:50:51Z",
        "complete": True,
        "environment": {
            "hexperiment": "0.1.0",
            "python": "3.13.1",
            "numpy": "2.0",
            "pandas": "2.2",
            "scikit-learn": "1.6.0",
            "os": "Windows",
            "cpu_count": 8,
            "run_seconds": 41.5,
        },
        "source": "Kapoor and Narayanan, doi:10.1016/j.patter.2023.100804",
        "mechanisms": [
            mechanism("L1.1", "No test set"),
            mechanism("L1.2", "Pre-processing on training and test set"),
            mechanism(
                "L1.2",
                "Pre-processing on training and test set",
                designed_to_catch=[],
                rejected=0,
                by_ghost={},
            ),
        ],
        "clean": {"scripts": 12, "false_alarms": 1, "by_ghost": {"x": 1}},
        "not_measured": [
            {
                "leak_type": "L3.3",
                "leak_type_name": "Sampling bias in test distribution",
                "reason": "It depends on the population.",
            }
        ],
    }


def parse(data):
    text = data if isinstance(data, str) else json.dumps(data)
    return parse_benchmark(text, source=SOURCE)


def section(data=None):
    return benchmark_section(parse(data or valid()), GHOSTS).html


# the table


def test_a_valid_file_parses_into_the_records():
    benchmark = parse(valid())
    assert benchmark.day == "2026-11-11"
    assert benchmark.created_at == "2026-11-11T19:50:51Z"
    assert benchmark.scikit_learn_version == "1.6.0"
    assert len(benchmark.mechanisms) == 3
    assert benchmark.mechanisms[0].by_ghost == (
        ("The Peeking Spirit", 9),
        ("split_order", 2),
    )
    assert (benchmark.clean_scripts, benchmark.false_alarms) == (12, 1)
    assert run_count(benchmark) == 36 + 12


def test_the_section_has_the_structure_and_the_notes():
    html = section()
    assert '<section aria-labelledby="benchmark">' in html
    assert '<h2 id="benchmark">Benchmark</h2>' in html
    assert "Leaky scripts rejected, by leakage type and mechanism" in html
    assert 'aria-label="Benchmark results"' in html
    assert html.count('scope="col"') == 5
    assert (
        "Leakage types follow Kapoor and Narayanan, doi:10.1016/j.patter.2023.100804."
        in html
    )
    assert (
        "Measured with hexperiment 0.1.0, Python 3.13.1 and scikit-learn 1.6.0 "
        "on 2026-11-11, in 48 runs." in html
    )
    assert "<h3>Not measured</h3>" in html
    assert (
        "L3.3 Sampling bias in test distribution: It depends on the population." in html
    )


def test_types_are_grouped_with_one_rowgroup_cell():
    html = section()
    assert html.count('scope="rowgroup"') == 2
    assert (
        '<th scope="rowgroup" rowspan="2">L1.2 Pre-processing on training and '
        "test set</th>" in html
    )
    assert '<th scope="rowgroup" rowspan="1">L1.1 No test set</th>' in html
    assert html.count("<tbody>") == 3


def test_a_file_that_interleaves_two_types_is_grouped_by_first_appearance():
    data = valid()
    a, b, c = data["mechanisms"]
    data["mechanisms"] = [a, c, mechanism("L1.1", "No test set", mechanism="second")]
    html = section(data)
    assert html.count('scope="rowgroup"') == 2
    assert '<th scope="rowgroup" rowspan="2">L1.1 No test set</th>' in html
    assert html.index("second") < html.index("L1.2 Pre-processing")


def test_designed_ghosts_and_the_absence_of_a_check():
    html = section()
    assert '<em class="ghost">The Peeking Spirit</em>' in html
    assert "No check is designed for it" in html


def test_rejected_n_of_m_with_the_rate():
    html = section()
    assert "rejected 9 of 12 (75.0%)" in html
    assert "rejected 0 of 12 (0.0%)" in html


def test_by_ghost_in_count_order_with_ghosts_and_checks_told_apart():
    data = valid()
    data["mechanisms"][0]["by_ghost"] = {"b": 1, "The Possession": 4, "a": 1}
    data["mechanisms"][0]["rejected"] = 9
    html = section(data)
    ordered = [
        '<em class="ghost">The Possession</em> (4)',
        "a (1)",
        "b (1)",
    ]
    positions = [html.index(text) for text in ordered]
    assert positions == sorted(positions)
    assert "none" in html  # the mechanism with an empty by_ghost
    assert '<em class="ghost">split_order</em>' not in html
    assert "split_order (2)" in html


def test_the_clean_row_in_the_singular_and_the_plural():
    one = section()
    assert "1 false alarm in 12 scripts (8.3%)" in one
    data = valid()
    data["clean"] = {"scripts": 1, "false_alarms": 0}
    zero = section(data)
    assert "0 false alarms in 1 script (0.0%)" in zero
    data["clean"] = {"scripts": 4, "false_alarms": 2}
    assert "2 false alarms in 4 scripts (50.0%)" in section(data)
    assert '<th scope="row" colspan="3">Clean twins</th>' in one


def test_no_not_measured_list_when_it_is_empty():
    data = valid()
    data["not_measured"] = []
    assert "Not measured" not in section(data)


def test_percent():
    assert percent(1, 3) == "33.3%"
    assert percent(0, 5) == "0.0%"
    assert percent(5, 5) == "100.0%"


def test_hostile_text_is_escaped_and_unknown_names_are_not_ghosts():
    hostile = "<script>alert(1)</script>"
    data = valid()
    data["source"] = hostile
    data["mechanisms"][0]["mechanism"] = hostile
    data["mechanisms"][0]["designed_to_catch"] = [hostile]
    data["not_measured"][0]["reason"] = hostile
    html = section(data)
    assert "<script>" not in html
    assert html.count("&lt;script&gt;alert(1)&lt;/script&gt;") >= 4
    assert f'<em class="ghost">{hostile}' not in html


# parsing: the error table, one test per row, one mutation of a valid file


def drop(*path):
    def mutate(data):
        target = data
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]

    return mutate


def put(*path_and_value):
    *path, value = path_and_value

    def mutate(data):
        target = data
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


ROWS = [
    (1, "not json", "the file is not valid JSON"),
    (2, "[1]", "the top level must be a JSON object, not a list"),
    (3, drop("format"), "format is missing"),
    (
        4,
        put("format", "hexperiment-benchmark/2"),
        "format is 'hexperiment-benchmark/2'",
    ),
    (5, put("complete", "yes"), "complete is missing or is not true or false"),
    (6, put("complete", False), "complete is false"),
    (7, drop("source"), "source is missing"),
    (
        8,
        put("created_at", "2026-11-11T19:50:51+02:00"),
        "created_at must be an ISO 8601 time in UTC",
    ),
    (
        9,
        put("environment", "scikit-learn", ""),
        "environment.scikit-learn is missing or is not a non-empty string",
    ),
    (10, put("source", ""), "source must be a non-empty string"),
    (11, put("mechanisms", []), "mechanisms must be a non-empty list"),
    (12, drop("mechanisms", 1, "by_ghost"), "mechanisms[1].by_ghost is missing"),
    (
        13,
        put("mechanisms", 0, "mechanism", ""),
        "mechanisms[0].mechanism must be a non-empty string",
    ),
    (
        14,
        put("mechanisms", 0, "designed_to_catch", ["a", 3]),
        "mechanisms[0].designed_to_catch must be a list of non-empty strings",
    ),
    (
        15,
        put("mechanisms", 2, "scripts", 0),
        "mechanisms[2].scripts must be an integer of at least 1",
    ),
    (
        16,
        put("mechanisms", 1, "rejected", 13),
        "mechanisms[1].rejected must be an integer from 0 to 12 "
        "(the value of scripts), not 13",
    ),
    (
        17,
        put("mechanisms", 0, "by_ghost", "split_order", 10),
        "mechanisms[0].by_ghost.split_order must be an integer from 1 to 9",
    ),
    (18, drop("clean", "false_alarms"), "clean.false_alarms is missing"),
    (19, put("clean", "scripts", 0), "clean.scripts must be an integer of at least 1"),
    (
        20,
        put("clean", "false_alarms", 13),
        "clean.false_alarms must be an integer from 0 to 12",
    ),
    (
        21,
        put("not_measured", 0, "reason", ""),
        "not_measured[0].reason is missing or is not a non-empty string",
    ),
]


def apply(mutation):
    if isinstance(mutation, str):
        return mutation
    data = valid()
    mutation(data)
    return data


@pytest.mark.parametrize(
    ("row", "mutation", "expected"), ROWS, ids=[r[0] for r in ROWS]
)
def test_each_row_of_the_error_table(row, mutation, expected):
    with pytest.raises(HexInputError) as caught:
        parse(apply(mutation))
    error = caught.value
    assert error.problem.startswith(f"{SOURCE}: ")
    assert expected in error.problem
    assert error.fix


def test_rows_are_all_21():
    assert [r[0] for r in ROWS] == list(range(1, 22))


def test_the_complete_false_message_names_only():
    data = valid()
    data["complete"] = False
    with pytest.raises(HexInputError, match="--only"):
        parse(data)


def test_complete_false_is_reported_before_a_bad_created_at():
    data = valid()
    data["complete"] = False
    data["created_at"] = "yesterday"
    with pytest.raises(HexInputError, match="complete is false"):
        parse(data)


def test_a_missing_key_is_reported_before_a_wrong_value_in_the_same_object():
    data = valid()
    del data["clean"]
    data["created_at"] = "yesterday"
    with pytest.raises(HexInputError, match="clean is missing"):
        parse(data)


@pytest.mark.parametrize("value", [True, 3.0, "3"])
def test_only_a_json_integer_is_an_integer(value):
    data = valid()
    data["mechanisms"][0]["scripts"] = value
    with pytest.raises(HexInputError, match=r"mechanisms\[0\]\.scripts"):
        parse(data)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_constants_are_refused(constant):
    text = json.dumps(valid()).replace(
        '"run_seconds": 41.5', f'"run_seconds": {constant}'
    )
    assert constant in text
    with pytest.raises(HexInputError, match="not valid JSON"):
        parse(text)


def test_a_byte_order_mark_is_accepted():
    assert parse("\ufeff" + json.dumps(valid())).day == "2026-11-11"


def test_extra_keys_are_ignored():
    data = valid()
    data["clean"]["by_ghost"] = {"anything": "at all"}
    data["extra"] = [1, 2]
    data["mechanisms"][0]["statuses"] = "ignored"
    assert parse(data).clean_scripts == 12


def test_created_at_with_a_zero_offset_is_accepted_and_naive_is_not():
    data = valid()
    data["created_at"] = "2026-11-11T23:50:51+00:00"
    assert parse(data).day == "2026-11-11"
    data["created_at"] = "2026-11-11T23:50:51"
    with pytest.raises(HexInputError, match="created_at"):
        parse(data)


def test_by_ghost_may_add_up_to_more_than_rejected():
    data = valid()
    data["mechanisms"][0]["by_ghost"] = {"a": 9, "b": 9}
    assert parse(data).mechanisms[0].rejected == 9


def test_the_input_is_not_changed_by_parsing():
    data = valid()
    before = copy.deepcopy(data)
    parse(data)
    assert data == before


@settings(max_examples=50, deadline=None)
@given(
    st.lists(
        st.integers(min_value=1, max_value=12).flatmap(
            lambda scripts: st.tuples(
                st.just(scripts),
                st.integers(min_value=0, max_value=scripts),
                st.sampled_from(["L1.1", "L1.2", "L2.1"]),
            ).flatmap(
                lambda t: st.tuples(
                    st.just(t),
                    st.dictionaries(
                        st.sampled_from(["The Possession", "canary", "a"]),
                        st.integers(min_value=1, max_value=max(t[1], 1)),
                        max_size=3 if t[1] else 0,
                    ),
                )
            )
        ),
        min_size=1,
        max_size=5,
    ),
    st.integers(min_value=1, max_value=20),
    st.data(),
)
def test_random_valid_counts_parse_and_render_one_row_per_mechanism(items, clean, data):
    file = valid()
    file["mechanisms"] = [
        mechanism(leak_type, scripts=scripts, rejected=rejected, by_ghost=by_ghost)
        for (scripts, rejected, leak_type), by_ghost in items
    ]
    file["clean"] = {"scripts": clean, "false_alarms": data.draw(st.integers(0, clean))}
    benchmark = parse(file)
    html = benchmark_section(benchmark, GHOSTS).html
    assert len(benchmark.mechanisms) == len(items)
    assert len(re.findall(r"rejected \d+ of \d+ \(", html)) == len(items)
    assert html.count("<tr>") == len(items) + 2  # header row and clean row
