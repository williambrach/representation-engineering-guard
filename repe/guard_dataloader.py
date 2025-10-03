import pandas as pd


def get_affinity_dict(row : pd.Series) -> dict:
    s1 = row['affinity_score_cls_0']
    s2 = row['affinity_score_cls_1']
    s3 = row['affinity_score_cls_2']
    output = {}
    layers = s1.keys()
    for layer in layers:
        output[layer] = [s1[layer], s2[layer], s3[layer]]
    return output

def load_data_for_repre_guard(number_of_samples: int, root_path: str = "data/guard/") -> tuple:
    directions_df = pd.read_parquet(f"{root_path}/{number_of_samples}_directions.parquet")
    inference_test = pd.read_parquet(f"{root_path}/{number_of_samples}_inference_test.parquet")
    inference_train = pd.read_parquet(f"{root_path}/{number_of_samples}_inference_train.parquet")
    for cls in [0, 1, 2]:
        inference_train[f"hidden_states_cls_{cls}"] = inference_train[f"hidden_states_cls_{cls}"].apply(lambda x: {(layer + 1) * -1: state for layer, state in enumerate(x)})
        inference_test[f"hidden_states_cls_{cls}"] = inference_test[f"hidden_states_cls_{cls}"].apply(lambda x: {(layer + 1) * -1: state for layer, state in enumerate(x)})

        inference_train[f"affinity_score_cls_{cls}"] = inference_train[f"affinity_score_cls_{cls}"].apply(lambda x: {(layer + 1) * -1: value for layer, value in enumerate(x)})
        inference_test[f"affinity_score_cls_{cls}"] = inference_test[f"affinity_score_cls_{cls}"].apply(lambda x: {(layer + 1) * -1: value for layer, value in enumerate(x)})
    inference_train['affinity_scores'] = inference_train[['affinity_score_cls_0', 'affinity_score_cls_1', 'affinity_score_cls_2']].apply(lambda row: get_affinity_dict(row), axis=1)
    inference_test['affinity_scores'] = inference_test[['affinity_score_cls_0', 'affinity_score_cls_1', 'affinity_score_cls_2']].apply(lambda row: get_affinity_dict(row), axis=1)
    return directions_df, inference_test, inference_train
