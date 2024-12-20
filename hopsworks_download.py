import training_helpers
import hopsworks
import os


if __name__ == "__main__":
    api_key = open(".hw_key").read().strip()
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store("tsedmid2223_featurestore")
    fv = fs.get_feature_view("delays_fv", training_helpers.LATEST_FV)
    if os.getenv("REMAKE", "") == "true":
        print("Creating training data")
        fv.create_training_data(description="All datapoints given as training data")
    else:
        x_all, y_all = fv.get_training_data(training_dataset_version=training_helpers.LATEST_TD)
        print("Downloaded dataset of size", x_all.shape[0])
        training_helpers.save_dataset(x_all, y_all)
