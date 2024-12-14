from koda.koda_constants import OperatorsWithRT
import koda.koda_pipeline as kp
import koda.koda_transform as kt

if __name__ == "__main__":
    # Example usage
    DATE = "2023-02-06"
    OPERATOR = OperatorsWithRT.X_TRAFIK
    df = kp.get_trip_updates_for_day(DATE, OPERATOR)
    print(f"1. Read {df.shape} from realtime data")

    columns_to_keep = [
        'trip_id', 'start_date', 'timestamp',
        'vehicle_id', 'stop_sequence', 'stop_id', 'arrival_delay',
        'arrival_time', 'departure_delay', 'departure_time'
    ]
    df = df[columns_to_keep]
    print(f"2. Reduced to {df.shape}")
    df = kt.keep_only_latest_stop_updates(df)
    print(f"3. Reduced to {df.shape}")

    average_delays_per_trip = df.groupby('trip_id')[['arrival_delay', 'departure_delay']].mean().reset_index()

    print(average_delays_per_trip)
