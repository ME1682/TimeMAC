import numpy as np
import pandas as pd
from collections.abc import Callable
from tsfeatures.tsfeatures import _get_feats
from ml_model import MLPTrainer, XGBoost
from datasetsforecast.hierarchical import HierarchicalData
from TimeMAC.utils.experiment_handler import ExperimentDataset
from tsfeatures import (
    acf_features,
    arch_stat,
    crossing_points,
    entropy,
    flat_spots,
    heterogeneity,
    holt_parameters,
    hurst,
    hw_parameters,
    lumpiness,
    nonlinearity,
    pacf_features,
    series_length,
    stability,
    stl_features,
    unitroot_kpss,
    unitroot_pp,
)

TSFEATURES: dict[str, Callable] = {
    "acf_features": acf_features,
    "arch_stat": arch_stat,
    "crossing_points": crossing_points,
    "entropy": entropy,
    "flat_spots": flat_spots,
    "heterogeneity": heterogeneity,
    "holt_parameters": holt_parameters,
    "lumpiness": lumpiness,
    "nonlinearity": nonlinearity,
    "pacf_features": pacf_features,
    "stl_features": stl_features,
    "stability": stability,
    "hw_parameters": hw_parameters,
    "unitroot_kpss": unitroot_kpss,
    "unitroot_pp": unitroot_pp,
    "series_length": series_length,
    "hurst": hurst,
}

def tsfeatures_tool(ctx: ExperimentDataset, features: list[str]):
    callable_features = []
    for feature in features:
        callable_features.append(TSFEATURES[feature])
    features_dfs = []
    for uid in ctx.df["unique_id"].unique():
        features_df_uid = _get_feats(
            index=uid,
            ts=ctx.df,
            features=callable_features,
            freq=ctx.seasonality,
        )
        features_dfs.append(features_df_uid)
        break
    features_df = pd.concat(features_dfs) if features_dfs else pd.DataFrame()
    features_df = features_df.rename_axis("unique_id")  # type: ignore
    # self.features_df = features_df
    return features_df

if __name__ == '__main__':
    dataset_names = ['TourismLarge', 'Traffic', 'Labour', 'Wiki2']
    models = ['TiRex', 'TSRNN', 'TSNHITS', 'TSPatchTST', 'TSTimeXer', 'TSFEDformer', 'TSiTransformer', 'TSTimeMixer',
              'ADIDA', 'AutoARIMA', 'AutoETS', 'DynamicOptimizedTheta', 'CrostonClassic']
    model_num = len(models)
    features = ["arch_stat", "crossing_points", "entropy", "flat_spots", "lumpiness", "nonlinearity",
                "stability", "unitroot_kpss", "unitroot_pp", "series_length", "hurst"]
    # MASE
    performance = [
        # Your performance data
        # our model is trained by MASE results of 'TourismLarge', 'Traffic', 'Labour', 'Wiki2' dataset.
                   ]
    x = np.array([])
    y = np.array([])
    model_feature = np.eye(model_num)
    dataset_feature = np.array([])
    # data_process
    for dataset_name in dataset_names:
        Y_df, S_df, tags = HierarchicalData.load('./data', f'{dataset_name}')
        Y_df['ds'] = pd.to_datetime(Y_df['ds'])
        h = 10
        df_data = ExperimentDataset(df=Y_df, freq=None, h=h, seasonality=None)
        df_data.freq = 'M'
        df_data.seasonality = 1
        df_stl_features = tsfeatures_tool(df_data, features)
        if dataset_feature.size == 0:
            dataset_feature = df_stl_features.to_numpy()[0]
        else:
            dataset_feature = np.vstack([dataset_feature, df_stl_features.to_numpy()[0]])
    for i in range(model_num):
        for j in range(len(dataset_names)):
            cnt = np.hstack([dataset_feature[j], model_feature[i]])
            if x.size == 0:
                x = cnt
            else:
                x = np.vstack([x, cnt])
    y = np.array([performance]).T
    feature_dim = model_num + len(features)

    # define ML model
    trainer = XGBoost()
    trainer.train(x, y)
    trainer.save_model()


