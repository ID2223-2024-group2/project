from enum import Enum


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