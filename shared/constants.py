from enum import Enum
from typing import Dict

GAEVLE_LONGITUDE = 17.1412
GAEVLE_LATITUDE = 60.6748


class FeedType(Enum):
    TRIP_UPDATES = "TripUpdates"
    VEHICLE_POSITIONS = "VehiclePositions"
    SERVICE_ALERTS = "ServiceAlerts"


class OperatorsWithRT(Enum):
    DALATRAFIK = "dt"
    JLT = "jlt"
    KALMAR_LANSTRAFIK = "klt"
    KRONOBERGS_LANSTRAFIK = "krono"
    LANSTRAFIKEN_OREBRO = "orebro"
    SKANETRAFIKEN = "skane"
    SL = "sl"
    UL = "ul"
    VL = "vastmanland"
    VARMLANDSTRAFIK = "varm"
    X_TRAFIK = "xt"
    OSTGOTATRAFIKEN = "otraf"


class StaticDataTypes(Enum):
    AGENCY = "agency"
    ATTRIBUTIONS = "attributions"
    CALENDAR = "calendar"
    CALENDAR_DATES = "calendar_dates"
    FEED_INFO = "feed_info"
    ROUTES = "routes"
    SHAPES = "shapes"
    STOP_TIMES = "stop_times"
    STOPS = "stops"
    TRANSFERS = "transfers"
    TRIPS = "trips"


route_types: Dict[int, str] = {
    100: "Railway Service",
    101: "High Speed Rail Service",
    102: "Long Distance Rail Service",
    103: "Inter Regional Rail Service",
    105: "Sleeper Rail Service",
    106: "Regional Rail Service",
    401: "Metro Service",
    700: "Bus Service",
    714: "Rail Replacement Bus Service",
    900: "Tram Service",
    1000: "Water Transport Service",
    1501: "Communal Taxi Service"
}
