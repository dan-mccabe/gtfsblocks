# GTFS Blocks
`gtfsblocks` is a Python package that pieces together GTFS feed data to assemble individual vehicle blocks and compile relevant data from various tables. The code was originally developed to analyze transit bus electrification, but the functionality can be helpful to other applications as well. The package was predominantly built off of the GTFS processing code from [`ebusopt`](https://github.com/dan-mccabe/ebusopt).

Some core functions include:
- Reading static GTFS tables into Pandas `DataFrame` objects and performing a bit of basic validation that necessary columns are populated while dropping what isn't needed.
- Parsing the `calendar.txt` and `calendar_dates.txt` files to identify the active `service_id` values on each day of service.
- Merging together trip-level data from different GTFS tables for easy manipulation and analysis. For example:
    - Adding trip start and end times from `stop_times.txt`
    - Adding trip start and end locations from `shapes.txt`
        - Esimating deadhead distances between consecutive trips based on these coordinates
    - Adding trip distances calculated from the lat/lon coordinates in `shapes.txt`
 
See [this gist](https://gist.github.com/dan-mccabe/2c0b0a4d58ab7f3f3068f7102b121672) for an overview of core functionality as well as the example usage below. Documentation is a work in progress.

## Installation
`gtfsblocks` is installable via `pip`:

`pip install gtfsblocks`

## Example Usage
It's easy to read in a GTFS feed with `gtfsblocks`. Just supply the path to the directory where unzipped GTFS files are housed:

```python
from gtfsblocks import Feed
gtfs = Feed.from_dir('/path/to/your/data')
```

This will load all relevant files into memory as Pandas DataFrames. Future releases may take advantage of [`partridge`](https://github.com/remix/partridge) for better memory management.

From here, you can access predictably named tables like `gtfs.trips_df` or `gtfs.stop_times_df`, or call various methods on `Feed` to perform some transformations and aggregations for you.

### Getting active trips on a particular day
```python
# Get a Pandas Series of the number of trips per day in the scope of these files
trips_per_day = gtfs.get_n_trips_per_day()

# Filter down trips.txt to just those happening on a particular day
test_date = '2/25/25'
day_trips = gtfs.get_trips_from_date(test_date)
```

### Only include blocks serving a specific set of routes
```python
from gtfsblocks import filter_blocks_by_route
routes = ['D Line', 'E Line']
route_trips = filter_blocks_by_route(
    trips=day_trips,
    routes=routes,
    route_method=route_method,
    route_column='route_short_name'
)
```

### Add data from other GTFS tables to trips DataFrame
```python
# Add all trip data columns (e.g. locations and distances)
route_trips = gtfs.add_trip_data(route_trips, test_date)
```

### Estimate deadhead distance between trips
```python
from gtfsblocks import add_deadhead
trips_with_dh = add_deadhead(route_trips)
```

### Plot the trips on an interactive Plotly map
```python
from gtfsblocks import plot_trips_and_terminals
fig = plot_trips_and_terminals(
    trips_df=route_trips,
    shapes_df=gtfs.shapes_df
)
```
