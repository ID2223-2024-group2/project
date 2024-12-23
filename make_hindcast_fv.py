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
    feature_view = fs.get_or_create_feature_view(
        name='monitor_fv',
        description="monitoring predictions",
        version=FV_VERSION,
        query=selected_features,
    )
