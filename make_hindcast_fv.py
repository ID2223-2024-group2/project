import hopsworks

DELAYS_VERSION = 10
MONITOR_VERSION = 1
FV_VERSION = 1


if __name__ == "__main__":
    api_key = open(".hw_key").read()
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store(name='tsedmid2223_featurestore')
    delays_fg = fs.get_feature_group(name='delays', version=DELAYS_VERSION)
    monitor_fg = fs.get_feature_group(name='delays_predictions', version=MONITOR_VERSION)
    selected_features = delays_fg.select_all().join(monitor_fg.select_all(),
        left_on=['arrival_time_bin', 'route_type'],
        right_on=['date', 'transport_type'],
        join_type='inner'
    )
    labels = ['mean_delay_change_seconds', 'max_delay_change_seconds', 'min_delay_change_seconds',
                             'var_delay_change_seconds',
                             'mean_arrival_delay_seconds', 'max_arrival_delay_seconds', 'min_arrival_delay_seconds',
                             'var_arrival_delay',
                             'mean_departure_delay_seconds', 'max_departure_delay_seconds',
                             'min_departure_delay_seconds', 'var_departure_delay',
                             'mean_on_time_percent', 'mean_final_stop_delay_seconds']
    feature_view = fs.get_or_create_feature_view(
        name='monitor_fv',
        description="monitoring predictions",
        version=FV_VERSION,
        labels=labels,
        query=selected_features,
    )
