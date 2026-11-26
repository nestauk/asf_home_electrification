"""Tests for the Dwelling class and its weather-resampling helpers.

Covers TABULA archetype YAML loading and heat-transfer (`h_tr`) calculation,
per-orientation PVGIS weather fetching, hourly-to-half-hourly UTC-to-London
resampling, and validation errors for archetype inputs.

Usage:
    uv run pytest -m dwelling
"""

import io

import pandas as pd
import pytest
import yaml

from asf_home_electrification.core_model import dwelling as dwelling_module
from asf_home_electrification.core_model.dwelling import ORIENTATIONS, Dwelling, _hourly_to_half_hourly

pytestmark = pytest.mark.dwelling

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_pvgis_df() -> pd.DataFrame:
    """Mock hourly PVGIS response with two consecutive UTC hours."""
    return pd.DataFrame(
        {
            "time": ["20230701:0010", "20230701:0110"],
            "T2m": [15.0, 16.5],
            "G(i)": [0.0, 120.0],
        }
    )


@pytest.fixture
def sample_archetype_data() -> dict:
    """Minimal valid TABULA archetype dictionary."""
    return {
        "description": "Test archetype",
        "a_ref": 100.0,
        "f_nu": 0.9,
        "n_air_infiltr": 0.5,
        "g_gl": 0.65,
        "delta_u_tb": 0.1,
        "elements": [
            {"u": 1.5, "area": 80.0, "b_tr": 1.0},  # 120.0
            {"u": 0.5, "area": 50.0, "b_tr": 0.8},  # 20.0
        ],  # total area = 130.0 -> h_bridging = 13.0 -> h_tr = 153.0
        "window_area": {"south": 10.0, "north": 5.0},
    }


@pytest.fixture
def mock_external_io(monkeypatch, sample_archetype_data, sample_pvgis_df):
    """Mocks fsspec.open and pvgis.get_hourly_pvgis_data for Dwelling tests."""
    yaml_str = yaml.safe_dump(sample_archetype_data, sort_keys=False)
    pvgis_calls = []

    def mock_fsspec_open(path, mode="r"):
        return io.StringIO(yaml_str)

    def mock_get_pvgis(*args, **kwargs):
        pvgis_calls.append(kwargs)
        return sample_pvgis_df.copy()

    monkeypatch.setattr(dwelling_module.fsspec, "open", mock_fsspec_open)
    monkeypatch.setattr(dwelling_module.pvgis, "get_hourly_pvgis_data", mock_get_pvgis)

    return pvgis_calls


# ============================================================================
# Initialisation
# ============================================================================


def test_dwelling_init_parses_params_and_calculates_h_tr(mock_external_io):
    dwelling = Dwelling(lat=51.5, lon=-0.1, archetype_name="TEST.ARCHETYPE")

    assert dwelling.lat == 51.5
    assert dwelling.lon == -0.1
    assert dwelling.archetype_name == "TEST.ARCHETYPE"
    assert dwelling.archetype_description == "Test archetype"

    params = dwelling.archetype_params
    assert params["a_ref"] == 100.0
    assert params["f_nu"] == 0.9
    assert params["n_air_infiltr"] == 0.5
    assert params["g_gl"] == 0.65
    assert params["window_area"] == {"south": 10.0, "north": 5.0}
    # h_elements = (1.5 * 80 * 1.0) + (0.5 * 50 * 0.8) = 140.0
    # h_bridging = 0.1 * (80 + 50) = 13.0
    assert params["h_tr"] == pytest.approx(153.0)


def test_dwelling_fetches_weather_for_each_orientation(mock_external_io):
    dwelling = Dwelling(lat=51.5, lon=-0.1, archetype_name="TEST.ARCHETYPE")

    assert len(mock_external_io) == 2

    called_orientations = {(call["aspect"], call["angle"]) for call in mock_external_io}
    assert called_orientations == {
        (ORIENTATIONS["south"]["aspect"], ORIENTATIONS["south"]["angle"]),
        (ORIENTATIONS["north"]["aspect"], ORIENTATIONS["north"]["angle"]),
    }

    assert isinstance(dwelling.outside_temperature, pd.Series)
    assert len(dwelling.outside_temperature) == 4
    assert list(dwelling.outside_temperature) == [15.0, 15.0, 16.5, 16.5]

    assert isinstance(dwelling.solar_irradiance, pd.DataFrame)
    assert set(dwelling.solar_irradiance.columns) == {"south", "north"}
    assert len(dwelling.solar_irradiance) == 4
    assert list(dwelling.solar_irradiance["south"]) == [0.0, 0.0, 120.0, 120.0]
    assert list(dwelling.solar_irradiance["north"]) == [0.0, 0.0, 120.0, 120.0]
    pd.testing.assert_index_equal(dwelling.outside_temperature.index, dwelling.solar_irradiance.index)


def test_dwelling_reads_expected_s3_path(monkeypatch, sample_archetype_data, sample_pvgis_df):
    captured_paths = []

    def mock_open(path, mode="r"):
        captured_paths.append((path, mode))
        return io.StringIO(yaml.safe_dump(sample_archetype_data))

    monkeypatch.setattr(dwelling_module.fsspec, "open", mock_open)
    monkeypatch.setattr(
        dwelling_module.pvgis,
        "get_hourly_pvgis_data",
        lambda *args, **kwargs: sample_pvgis_df.copy(),
    )

    Dwelling(lat=51.5, lon=-0.1, archetype_name="GB.ENG.TEST.001")

    assert captured_paths == [
        ("s3://asf-home-electrification/inputs/tabula/archetype_heat_parameters/GB.ENG.TEST.001.yaml", "r")
    ]


# ============================================================================
# Checking _hourly_to_half_hourly functionality
# ============================================================================


def test_hourly_to_half_hourly_expands_and_converts_timezone(sample_pvgis_df):
    result = _hourly_to_half_hourly(sample_pvgis_df.copy())

    assert len(result) == 4
    assert str(result["time"].dt.tz) == "Europe/London"

    expected_times = pd.DatetimeIndex(
        [
            "2023-07-01 01:00:00+01:00",
            "2023-07-01 01:30:00+01:00",
            "2023-07-01 02:00:00+01:00",
            "2023-07-01 02:30:00+01:00",
        ],
        tz="Europe/London",
        name="time",
    )
    pd.testing.assert_index_equal(pd.DatetimeIndex(result["time"]), expected_times)
    assert list(result["T2m"]) == [15.0, 15.0, 16.5, 16.5]
    assert list(result["G(i)"]) == [0.0, 0.0, 120.0, 120.0]


# ============================================================================
# Error handling
# ============================================================================


def test_only_requests_orientations_in_window_area(monkeypatch, sample_archetype_data, sample_pvgis_df):
    sample_archetype_data["window_area"] = {"south": 8.0}
    pvgis_calls = []

    monkeypatch.setattr(
        dwelling_module.fsspec,
        "open",
        lambda *args, **kwargs: io.StringIO(yaml.safe_dump(sample_archetype_data)),
    )
    monkeypatch.setattr(
        dwelling_module.pvgis,
        "get_hourly_pvgis_data",
        lambda *args, **kwargs: pvgis_calls.append(kwargs) or sample_pvgis_df.copy(),
    )

    Dwelling(lat=51.5, lon=-0.3, archetype_name="GB.ENG.SFH.TEST")

    assert len(pvgis_calls) == 1
    assert pvgis_calls[0]["aspect"] == 0


def test_dwelling_raises_on_unknown_window_orientation(monkeypatch, sample_archetype_data):
    sample_archetype_data["window_area"] = {"southeast": 12.0}
    monkeypatch.setattr(
        dwelling_module.fsspec,
        "open",
        lambda *args, **kwargs: io.StringIO(yaml.safe_dump(sample_archetype_data)),
    )

    with pytest.raises(ValueError, match="Unknown window orientation 'southeast'"):
        Dwelling(lat=51.5, lon=-0.1, archetype_name="TEST.ARCHETYPE")


def test_dwelling_raises_on_empty_window_area(monkeypatch, sample_archetype_data):
    sample_archetype_data["window_area"] = {}
    monkeypatch.setattr(
        dwelling_module.fsspec,
        "open",
        lambda *args, **kwargs: io.StringIO(yaml.safe_dump(sample_archetype_data)),
    )

    with pytest.raises(ValueError, match="No window orientations"):
        Dwelling(lat=51.5, lon=-0.1, archetype_name="TEST.ARCHETYPE")


@pytest.mark.parametrize(
    "missing_key",
    ["a_ref", "f_nu", "n_air_infiltr", "window_area", "g_gl", "elements", "delta_u_tb", "description"],
)
def test_dwelling_raises_key_error_on_missing_yaml_field(monkeypatch, sample_archetype_data, missing_key):
    del sample_archetype_data[missing_key]
    monkeypatch.setattr(
        dwelling_module.fsspec,
        "open",
        lambda *args, **kwargs: io.StringIO(yaml.safe_dump(sample_archetype_data)),
    )

    with pytest.raises(KeyError):
        Dwelling(lat=51.5, lon=-0.1, archetype_name="TEST.ARCHETYPE")
