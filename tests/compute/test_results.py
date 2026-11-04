import pytest
import qcelemental as qcel

from crystalatte.compute.results import (
    IncompleteResultsError,
    Row,
    Status,
    assemble,
    multimer_fields,
    select_active,
)

HK = qcel.constants.hartree2kJmol


def _mixed():
    return [
        Row("dimer-a", 2, 1, 0.5, -0.001, "complete"),
        Row("dimer-b", 2, 1, 0.5, None, "failed", "SCF did not converge"),
        Row("dimer-c", 2, 1, 0.5, None, "pending"),
    ]


def test_lattice_energy_sums_weighted_kjmol():
    r = assemble(
        [Row("dimer-a", 2, 2, 1.0, -0.001, "complete"), Row("trimer-b", 3, 3, 1.0, 0.0001, "complete")],
        partial=False,
    )
    assert r.complete
    assert r.lattice_energy == pytest.approx((-0.001 + 0.0001) * HK)
    assert r.by_order[2] == pytest.approx(-0.001 * HK)
    assert r.by_order[3] == pytest.approx(0.0001 * HK)


def test_weight_applied():
    r = assemble([Row("dimer-a", 2, 2, 0.5, -0.002, "complete")], partial=False)
    assert r.lattice_energy == pytest.approx(-0.001 * HK)


def test_collect_refuses_incomplete():
    with pytest.raises(IncompleteResultsError) as e:
        assemble(_mixed(), partial=False)
    assert e.value.missing == {2: {"failed": ["dimer-b"], "pending": ["dimer-c"]}}
    assert "SCF did not converge" in str(e.value)
    assert "dimer-c" in str(e.value)


def test_message_truncates():
    rows = [Row(f"d{i}", 2, 1, 1.0, None, "pending") for i in range(25)]
    rows.append(Row("f", 2, 1, 1.0, None, "failed", "x" * 500))
    with pytest.raises(IncompleteResultsError) as e:
        assemble(rows, partial=False)
    msg = str(e.value)
    assert "... and 5 more" in msg
    assert "x" * 200 in msg and "x" * 201 not in msg


def test_partial_exposes_rows_not_total():
    r = assemble(_mixed(), partial=True)
    assert len(r.table) == 3 and not r.complete
    assert r.missing == {2: {"failed": ["dimer-b"], "pending": ["dimer-c"]}}
    with pytest.raises(IncompleteResultsError):
        r.lattice_energy
    with pytest.raises(IncompleteResultsError):
        r.by_order


def test_to_dataframe():
    pd = pytest.importorskip("pandas")
    df = assemble(_mixed(), partial=True).to_dataframe()
    assert isinstance(df, pd.DataFrame) and len(df) == 3


def test_select_active_ignores_stale_and_marks_unknown_pending():
    known = {
        "dimer-a": Row("dimer-a", 2, 1, 1.0, -0.001, "complete"),
        "dimer-old": Row("dimer-old", 2, 1, 1.0, -0.002, "complete"),
    }
    rows = select_active({2: ["dimer-a", "dimer-new"]}, known)
    assert [(x.id, x.status) for x in rows] == [("dimer-a", "complete"), ("dimer-new", "pending")]
    assert rows[1].N == 2 and rows[1].energy is None


def test_status_str():
    s = Status({2: {"complete": 40, "pending": 2, "failed": 1}, 3: {"complete": 5, "pending": 0, "failed": 0}})
    assert str(s).splitlines() == [
        "dimers: 40 complete, 2 pending, 1 failed",
        "trimers: 5 complete, 0 pending, 0 failed",
    ]


def test_table_and_dataframe_expose_multimer_metadata():
    pd = pytest.importorskip("pandas")
    row = Row("dimer-a", 2, 1, 0.5, -0.001, "complete", geometric_mean=4.0, min_separation=3.0, formulas=("A", "B"))
    r = assemble([row], partial=False)
    assert r.table[0]["geometric_mean"] == 4.0
    assert r.table[0]["min_separation"] == 3.0
    assert r.table[0]["formulas"] == ("A", "B")
    assert r.to_dataframe().loc[0, "min_separation"] == 3.0


def test_multimer_fields_are_none_for_entries_that_predate_them():
    assert multimer_fields({"multiplicity": 1, "weight": 0.5}) == {
        "geometric_mean": None,
        "min_separation": None,
        "formulas": None,
    }


def test_multimer_fields_read_a_stored_entry():
    stored = {"geometric_mean": 4.0, "min_separation": 3.0, "formulas": ["A", "B"]}
    assert multimer_fields(stored) == {"geometric_mean": 4.0, "min_separation": 3.0, "formulas": ("A", "B")}
