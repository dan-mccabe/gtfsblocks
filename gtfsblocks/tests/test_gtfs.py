import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from gtfsblocks.gtfs import (
    _load_table,
    _haversine_np,
    _manhattan_np,
    add_deadhead,
    get_shape,
    Feed,
)


# Test for _load_table
def test_load_table(tmp_path):
    # Create a temporary CSV file
    csv_file = tmp_path / "test.csv"
    data = {"col1": [1, 2], "col2": [3, 4], "col3": [5, 6]}
    pd.DataFrame(data).to_csv(csv_file, index=False)

    # Test loading required columns
    df = _load_table(csv_file, required_cols=["col1", "col2"])
    assert list(df.columns) == ["col1", "col2"]

    # Test missing required columns
    with pytest.raises(ValueError):
        _load_table(csv_file, required_cols=["missing_col"])

    # Test optional columns
    # Make sure optional columns are included if they exist
    df = _load_table(csv_file, required_cols=["col1"], optional_cols=["col3"])
    assert list(df.columns) == ["col1", "col3"]

    # Ignore optional columns that don't exist
    df = _load_table(csv_file, required_cols=["col1"], optional_cols=["col4"])
    assert list(df.columns) == ["col1"]


# Test for _haversine_np
def test_haversine_np():
    lon1, lat1, lon2, lat2 = np.array([0]), np.array([0]), np.array([1]), np.array([1])
    result = _haversine_np(lon1, lat1, lon2, lat2)
    assert result[0] > 0  # Ensure the result is positive


# Test for _manhattan_np
def test_manhattan_np():
    lon1, lat1, lon2, lat2 = np.array([0]), np.array([0]), np.array([1]), np.array([1])
    result = _manhattan_np(lon1, lat1, lon2, lat2)
    assert result[0] > 0  # Ensure the result is positive

    hav_result = _haversine_np(lon1, lat1, lon2, lat2)
    assert (
        result[0] > hav_result[0]
    )  # Manhattan distance should be greater than haversine


# Test for add_deadhead
def test_add_deadhead():
    trips_df = pd.DataFrame(
        {
            "block_id": [1, 1],
            "trip_idx": [1, 2],
            "start_lat": [0, 1],
            "start_lon": [0, 1],
            "end_lat": [1, 2],
            "end_lon": [1, 2],
        }
    )
    result = add_deadhead(trips_df)
    assert "dh_dist" in result.columns
    assert result["dh_dist"].iloc[0] == 0  # Deadhead is zero in this example

    trips_df = pd.DataFrame(
        {
            "block_id": [1, 1],
            "trip_idx": [1, 2],
            "start_lat": [0, 1],
            "start_lon": [0, 1],
            "end_lat": [1, 2],
            "end_lon": [2, 2],
        }
    )
    result = add_deadhead(trips_df)
    assert "dh_dist" in result.columns
    assert result["dh_dist"].iloc[0] > 0  # Deadhead is nonzero in this example


# Test for get_shape
def test_get_shape():
    shapes_df = pd.DataFrame(
        {
            "shape_id": ["shape1", "shape1", "shape2"],
            "shape_pt_sequence": [1, 2, 1],
            "shape_pt_lon": [0, 1, 2],
            "shape_pt_lat": [0, 1, 2],
        }
    )
    lon, lat = get_shape(shapes_df, "shape1")
    assert list(lon) == [0, 1]
    assert list(lat) == [0, 1]


# Test for Feed.from_dir
def test_feed_from_dir_default():
    # Load test feed
    # feed = Feed.from_dir(Path(__file__).parent / "data" / "nantucket")
    feed = Feed.from_dir("../../data/nantucket")

    # Check that the correct columns are loaded by default
    default_cols = {
        "agency": ["agency_id", "agency_name", "agency_url", "agency_timezone"],
        "trips": ["trip_id", "route_id", "service_id", "block_id", "shape_id"],
        "routes": ["route_short_name", "route_type", "route_desc", "agency_id"],
        "calendar": [
            "service_id",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
            "start_date",
            "end_date",
        ],
        "calendar_dates": ["service_id", "date", "exception_type"],
        "shapes": ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"],
        "stops": ["stop_id", "stop_lat", "stop_lon"],
        "stop_times": ["trip_id", "stop_sequence", "arrival_time", "stop_id"],
    }
    print(feed.stop_times.columns.tolist())
    print(default_cols["stop_times"])

    assert set(feed.agency.columns.tolist()) == set(default_cols["agency"])
    assert set(feed.trips.columns.tolist()) == set(default_cols["trips"])
    assert set(feed.routes.columns.tolist()) == set(default_cols["routes"])
    assert set(feed.calendar.columns.tolist()) == set(default_cols["calendar"])
    assert set(feed.calendar_dates.columns.tolist()) == set(
        default_cols["calendar_dates"]
    )
    assert set(feed.shapes.columns.tolist()) == set(default_cols["shapes"])
    assert set(feed.stops.columns.tolist()) == set(default_cols["stops"])
    assert set(feed.stop_times.columns.tolist()) == set(default_cols["stop_times"])

    # TODO: check that only bus trips are included


def test_feed_from_dir_optional_cols():
    # Load test feed
    feed = Feed.from_dir(
        Path(__file__).parent / "data" / "nantucket",
        columns={
            "agency": ["agency_id", "agency_name"],
            "routes": ["route_long_name"],
            "stop_times": ["arrival_time", "departure_time"],
        },
    )

    # Check that the correct columns are loaded by default
    expect_cols = {
        "agency": ["agency_id", "agency_name", "agency_url", "agency_timezone"],
        "trips": ["trip_id", "route_id", "service_id", "block_id", "shape_id"],
        "routes": [
            "route_short_name",
            "route_long_name",
            "route_type",
            "agency_id",
            "route_desc",
        ],
        "calendar": [
            "service_id",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
            "start_date",
            "end_date",
        ],
        "calendar_dates": ["service_id", "date", "exception_type"],
        "shapes": ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"],
        "stops": ["stop_id", "stop_lat", "stop_lon"],
        "stop_times": [
            "trip_id",
            "stop_sequence",
            "arrival_time",
            "departure_time",
            "stop_id",
        ],
    }

    assert set(feed.agency.columns.tolist()) == set(expect_cols["agency"])
    assert set(feed.trips.columns.tolist()) == set(expect_cols["trips"])
    assert set(feed.routes.columns.tolist()) == set(expect_cols["routes"])
    assert set(feed.calendar.columns.tolist()) == set(expect_cols["calendar"])
    assert set(feed.calendar_dates.columns.tolist()) == set(
        expect_cols["calendar_dates"]
    )
    assert set(feed.shapes.columns.tolist()) == set(expect_cols["shapes"])
    assert set(feed.stops.columns.tolist()) == set(expect_cols["stops"])
    assert set(feed.stop_times.columns.tolist()) == set(expect_cols["stop_times"])


def test_feed_dtypes():
    # Load test feed
    feed = Feed.from_dir(
        Path(__file__).parent / "data" / "nantucket",
        columns={"stop_times": ["departure_time"]},
    )

    # Check that the correct dtypes are loaded by default
    assert feed.agency.dtypes["agency_id"] == "object"
    assert feed.agency.dtypes["agency_name"] == "object"
    assert feed.agency.dtypes["agency_url"] == "object"
    assert feed.agency.dtypes["agency_timezone"] == "object"

    assert feed.trips.dtypes["trip_id"] == "object"
    assert feed.trips.dtypes["route_id"] == "object"
    assert feed.trips.dtypes["service_id"] == "object"
    assert feed.trips.dtypes["block_id"] == "object"
    assert feed.trips.dtypes["shape_id"] == "object"

    assert feed.routes.dtypes["route_short_name"] == "object"
    assert feed.routes.dtypes["route_type"] == "int64"
    assert feed.routes.dtypes["agency_id"] == "object"
    assert feed.routes.dtypes["route_desc"] == "object"

    assert feed.calendar.dtypes["service_id"] == "object"
    assert feed.calendar.dtypes["monday"] == "int64"
    assert feed.calendar.dtypes["tuesday"] == "int64"
    assert feed.calendar.dtypes["wednesday"] == "int64"
    assert feed.calendar.dtypes["thursday"] == "int64"
    assert feed.calendar.dtypes["friday"] == "int64"
    assert feed.calendar.dtypes["saturday"] == "int64"
    assert feed.calendar.dtypes["sunday"] == "int64"
    assert str(feed.calendar.dtypes["start_date"]) == "datetime64[ns]"
    assert str(feed.calendar.dtypes["end_date"]) == "datetime64[ns]"

    assert feed.calendar_dates.dtypes["service_id"] == "object"
    assert str(feed.calendar_dates.dtypes["date"]) == "datetime64[ns]"
    assert feed.calendar_dates.dtypes["exception_type"] == "int64"

    assert feed.shapes.dtypes["shape_id"] == "object"
    assert feed.shapes.dtypes["shape_pt_lat"] == "float64"
    assert feed.shapes.dtypes["shape_pt_lon"] == "float64"
    assert feed.shapes.dtypes["shape_pt_sequence"] == "int64"

    assert feed.stops.dtypes["stop_id"] == "object"
    assert feed.stops.dtypes["stop_lat"] == "float64"
    assert feed.stops.dtypes["stop_lon"] == "float64"

    assert feed.stop_times.dtypes["trip_id"] == "object"
    assert feed.stop_times.dtypes["stop_id"] == "object"
    assert feed.stop_times.dtypes["stop_sequence"] == "int64"
    assert str(feed.stop_times.dtypes["arrival_time"]) == "timedelta64[ns]"
    assert str(feed.stop_times.dtypes["departure_time"]) == "timedelta64[ns]"


# # Test for Feed.get_service_ids_from_date
# def test_get_service_ids_from_date(tmp_path):
#     # Create temporary GTFS files
#     (tmp_path / "calendar.txt").write_text("service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n1,1,1,1,1,1,0,0,20230101,20231231")
#     (tmp_path / "calendar_dates.txt").write_text("service_id,date,exception_type\n1,20230101,1")
#     feed = Feed.from_dir(tmp_path)

#     # Test service IDs for a specific date
#     service_ids = feed.get_service_ids_from_date("2023-01-01")
#     assert "1" in service_ids

# # Test for Feed.add_trip_data
# def test_add_trip_data(tmp_path):
#     # Create temporary GTFS files
#     (tmp_path / "trips.txt").write_text("trip_id,route_id,service_id,block_id,shape_id\n1,1,1,1,1")
#     (tmp_path / "shapes.txt").write_text("shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence\n1,0,0,1\n1,1,1,2")
#     (tmp_path / "stop_times.txt").write_text("trip_id,stop_sequence,arrival_time,stop_id\n1,1,08:00:00,1\n1,2,09:00:00,2")
#     feed = Feed.from_dir(tmp_path)

#     # Test adding trip data
#     trips_df = feed.trips
#     result = feed.add_trip_data(trips_df, "2023-01-01")
#     assert "start_time" in result.columns
#     assert "end_time" in result.columns
#     assert "start_lat" in result.columns
#     assert "end_lat" in result.columns

# # Test for Feed.get_trips_from_date
# def test_get_trips_from_date(tmp_path):
#     # Create temporary GTFS files
#     (tmp_path / "calendar.txt").write_text("service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n1,1,1,1,1,1,0,0,20230101,20231231")
#     (tmp_path / "trips.txt").write_text("trip_id,route_id,service_id,block_id,shape_id\n1,1,1,1,1")
#     feed = Feed.from_dir(tmp_path)

#     # Test getting trips for a specific date
#     trips = feed.get_trips_from_date("2023-01-01")
#     assert not trips.empty
#     assert "trip_id" in trips.columns
