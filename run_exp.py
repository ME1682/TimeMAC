import sys

import numpy as np
import pandas as pd
from pydantic_ai.providers.openai import OpenAIProvider
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pydantic_ai.models.openai import OpenAIChatModel
from TimeMAC.agent import TimeCopilot
from ml_model import MLPTrainer, XGBoost
from datasetsforecast.hierarchical import HierarchicalData
from train_performance_predictor import tsfeatures_tool
from TimeMAC.utils.experiment_handler import ExperimentDataset
import m5_dataset
import pickle
import math
'''
# only test
provider = OpenAIProvider(
    base_url="https://router.huggingface.co/v1",
    api_key='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'
)
llm_model = OpenAIChatModel(
    "HuggingFaceTB/SmolLM3-3B:hf-inference",
    provider=provider)
'''
# If users do not have LLM API, they can use "Without_LLM" type. But this type cannot output the analysis.
provider = OpenAIProvider(
    base_url="Your LLM Website",
    api_key='Your API KEY'
)
llm_model = OpenAIChatModel(
    "Your LLM's Name",
    provider=provider)

tc = TimeCopilot(
    llm=llm_model,
    retries=10)

if __name__ == '__main__':
    dataset_name = "Wiki2"
    default_models = ['TiRex', 'TSRNN', 'TSNHITS', 'TSPatchTST', 'TSTimeXer', 'TSFEDformer', 'TSiTransformer', 'TSTimeMixer',
                      'ADIDA', 'AutoARIMA', 'AutoETS', 'DynamicOptimizedTheta', 'CrostonClassic']
    models = ['TiRex', 'TSRNN', 'TSNHITS', 'TSPatchTST', 'TSTimeXer', 'TSFEDformer', 'TSiTransformer', 'TSTimeMixer',
              'ADIDA', 'AutoARIMA', 'AutoETS', 'DynamicOptimizedTheta', 'CrostonClassic']
    # model_choice: {node_name: model_name} {str, str}
    model_choice = {}
    # If users do not have LLM API, they can use Without LLM type
    choice_type = "LLM" # Default: LLM, Without_LLM
    # result: {node_name: result} {str, AgentRunResult[ForecastAgentOutput]}
    result = {}
    # fcst_df: {str, np.Series}
    fcst_df = {}
    # fcst_df_np: [] -> numpy
    fcst_df_np = np.array([])
    # data_y_df: [fcst] only hierarchy 0 ->numpy
    data_y_df = np.array([])
    cnt_df1 = np.array([])
    cnt_df2 = np.array([])
    future_fcst = {}
    future_fcst_np = np.array([])
    future_test_np = np.array([])
    # Model choice part
    if dataset_name == "amazon":
        hierarchy_node = [['NewYork', 'Alabama', 'NewJersey', 'Pennsylvania', 'Kentucky', 'Mississippi', 'Tennessee',
                           'Alaska', 'California', 'Hawaii', 'Oregon', 'Illinois', 'Indiana', 'Ohio', 'Connecticut',
                           'Maine', 'RhodeIsland', 'Vermont'],
                          ['Mid-Alantic', 'SouthCentral', 'Pacific', 'EastNorthCentral', 'NewEngland'],
                          ['total']]
        total_node_number = 24
        hierarchy_0_node_number = 18

        hierarchy_level = 3

        cross_validation_h = 90
        cross_number = 1
        future_step = cross_validation_h

        df_read = pd.read_csv(f"{dataset_name}/all.csv")
        df_read = df_read.drop('item', axis=1)
        df_read = df_read.drop('country', axis=1)
        df_read = df_read.drop('number', axis=1)
        df_read["ds"] = pd.to_datetime(df_read["ds"])
        df_read2 = df_read.query(f'ds > "2009-04-29"').copy()
        '''
        df_read2 = pd.read_csv(f"{dataset_name}/test.csv")
        df_read2 = df_read2.drop('item', axis=1)
        df_read2 = df_read2.drop('country', axis=1)
        df_read2 = df_read2.drop('number', axis=1)
        df_read2["ds"] = pd.to_datetime(df_read2["ds"])
        '''
        hierarchy_name = ["state", "region"]
        if choice_type == 'LLM':
            # choose model in dataset level
            if models != default_models:
                print('Models are not default models. Please retrain dataset model select part or ban it.')
                sys.exit()
            features = ["arch_stat", "crossing_points", "entropy", "flat_spots", "lumpiness", "nonlinearity",
                        "stability", "unitroot_kpss", "unitroot_pp", "series_length", "hurst"]
            x = np.array([])
            model_num = len(models)
            model_feature = np.eye(model_num)
            df_train = df_read.rename(columns={hierarchy_name[0]: 'unique_id'})

            for i in range(hierarchy_level - 2):
                df_train = df_train.drop(hierarchy_name[i + 1], axis=1)
            df_data = ExperimentDataset(df=df_train, freq=None, h=None, seasonality=None)
            df_data.freq = 'M'
            df_data.seasonality = 1
            df_stl_features = tsfeatures_tool(df_data, features)
            dataset_feature = df_stl_features.to_numpy()[0]
            for i in range(model_num):
                    cnt = np.hstack([dataset_feature, model_feature[i]])
                    if x.size == 0:
                        x = cnt
                    else:
                        x = np.vstack([x, cnt])
            performance_predictor = XGBoost()
            performance_predictor.load_model()
            performance = performance_predictor.predict(x)
            print(models)
            print(performance)
            models = tc.choose_model_in_dataset_level(models, performance, ST_num=3, NN_num=3, FM_num=1)
            print('For this dataset, we use the following models:')
            print(models)

        # train
        for model in models:
            if model not in ['ADIDA', 'AutoARIMA', 'AutoETS', 'DynamicOptimizedTheta', 'SeasonalNaive',
                             'CrostonClassic', 'HistoricAverage', 'Chronos', 'Sundial', 'TiRex']:
                df_train = df_read.rename(columns={hierarchy_name[0]: 'unique_id'})
                # 除了最下层和最上层都要舍去
                for i in range(hierarchy_level-2):
                    df_train = df_train.drop(hierarchy_name[i+1], axis=1)
                cnt = tc.train_model(df=df_train,
                                     freq='D',
                                     model=model,
                                     dataset_name=dataset_name,
                                     h=future_step,)
        load_path = "True"

        for hierarchy in range(hierarchy_level):
            if hierarchy == 0:
                df_train = df_read.rename(columns={hierarchy_name[0]: 'unique_id'})
                df_test = df_read2.rename(columns={hierarchy_name[0]: 'unique_id'})

                for i in range(hierarchy_level-2):
                    df_train = df_train.drop(hierarchy_name[i+1], axis=1)
                    df_test = df_test.drop(hierarchy_name[i+1], axis=1)
                for cnt in hierarchy_node[hierarchy]:
                    print("Hierarchy 0: "+cnt)
                    df = df_train.loc[df_train['unique_id'] == cnt]
                    df2 = df_test.loc[df_test['unique_id'] == cnt]

                    result[cnt], fcst_df[cnt], future_fcst[cnt] = tc.forecast(df=df, df_test=df2, choice_type=choice_type,
                                                            freq='D', hierarchy=hierarchy, models=models,
                                                            future_step=future_step, cross_validation_h=cross_validation_h,
                                                            cross_number=cross_number,load_path=load_path)
                    cnt_series = fcst_df[cnt]
                    if cnt_series is not None:
                        if fcst_df_np.size == 0:
                            fcst_df_np = cnt_series.to_numpy()
                        else:
                            fcst_df_np = np.vstack([fcst_df_np, cnt_series.to_numpy()])
                    cnt_series = df.iloc[-100:,]
                    if cnt_series is not None:
                        if data_y_df.size == 0:
                            data_y_df = np.array(cnt_series.to_numpy()[:, cnt_series.columns.get_loc('y')].tolist())
                        else:
                            data_y_df = np.vstack([data_y_df, np.array(cnt_series.to_numpy()[:, cnt_series.columns.get_loc('y')].tolist())])
                    cnt_series = future_fcst[cnt]
                    if cnt_series is not None:
                        if future_fcst_np.size == 0:
                            future_fcst_np = cnt_series[result[cnt].output.selected_model].to_numpy()
                        else:
                            future_fcst_np = np.vstack([future_fcst_np, cnt_series[result[cnt].output.selected_model].to_numpy()])

                    # output the fcst of node_0
                    #np.save('fcst/fcst_df_cnt.npy', fcst_df_np)
                    if result[cnt] is not None:
                        print("selected_model: "+result[cnt].output.selected_model)
                        print("model_details: "+result[cnt].output.model_details)
                        print("model_comparison: "+result[cnt].output.model_comparison)
                        #print("reason_for_selection: "+result[cnt].output.reason_for_selection)
                        #print("user_query_response: "+result[cnt].output.user_query_response)
                        model_choice[cnt] = result[cnt].output.selected_model

            elif hierarchy == hierarchy_level - 1:
                df_train = df_read
                df_test = df_read2

                for i in range(1, hierarchy_level-1):
                    df_train = df_train.drop(hierarchy_name[i], axis=1)
                    df_test = df_test.drop(hierarchy_name[i], axis=1)
                for cnt in hierarchy_node[hierarchy]:
                    print("Hierarchy 2, " + cnt)
                    df = df_train.set_index(["ds", hierarchy_name[0]])
                    df = df.groupby(level="ds").sum()
                    df = df.assign(unique_id="total")
                    df = df.sort_values(by='ds')
                    df.to_csv(f"{dataset_name}/cnt.csv")
                    df = pd.read_csv(f"{dataset_name}/cnt.csv")
                    df["ds"] = pd.to_datetime(df["ds"])
                    df = df.sort_values(by='ds')

                    df2 = df_test.set_index(["ds", hierarchy_name[0]])
                    df2 = df2.groupby(level="ds").sum()
                    df2 = df2.assign(unique_id="total")
                    df2 = df2.sort_values(by='ds')
                    df2.to_csv(f"{dataset_name}/cnt.csv")
                    df2 = pd.read_csv(f"{dataset_name}/cnt.csv")
                    df2["ds"] = pd.to_datetime(df2["ds"])
                    df2 = df2.sort_values(by='ds')
                    #df.to_csv("amazon/cnt.csv")
                    result[cnt], fcst_df[cnt], future_fcst[cnt] = tc.forecast(df=df, df_test=df2, choice_type=choice_type,
                                                            freq='D', hierarchy=hierarchy, models=models,
                                                            future_step = future_step, cross_validation_h=cross_validation_h,
                                                            cross_number=cross_number,load_path=load_path)
                    cnt_series = fcst_df[cnt]
                    if cnt_series is not None:
                        if fcst_df_np.size == 0:
                            fcst_df_np = cnt_series.to_numpy()
                        else:
                            fcst_df_np = np.vstack([fcst_df_np, cnt_series.to_numpy()])
                    cnt_series = future_fcst[cnt]
                    if cnt_series is not None:
                        if future_fcst_np.size == 0:
                            future_fcst_np = cnt_series[result[cnt].output.selected_model].to_numpy()
                        else:
                            future_fcst_np = np.vstack([future_fcst_np, cnt_series[result[cnt].output.selected_model].to_numpy()])

                    if result[cnt] is not None:
                        print("selected_model: "+result[cnt].output.selected_model)
                        print("model_details: "+result[cnt].output.model_details)
                        print("model_comparison: "+result[cnt].output.model_comparison)
                        #print("reason_for_selection: "+result[cnt].output.reason_for_selection)
                        #print("user_query_response: "+result[cnt].output.user_query_response)
                        model_choice[cnt] = result[cnt].output.selected_model

            else:
                df_train = df_read.rename(columns={'region': 'unique_id'})
                df_test = df_read2.rename(columns={'region': 'unique_id'})

                for i in range(1, hierarchy_level-1):

                    if i != hierarchy:
                        df_train = df_train.drop(hierarchy_name[i], axis=1)
                        df_test = df_test.drop(hierarchy_name[i], axis=1)
                for cnt in hierarchy_node[hierarchy]:
                    print("Hierarchy middle: " + cnt)
                    df = df_train.loc[df_train['unique_id'] == cnt] # 抽出该层需要的节点的数据

                    df = df.set_index(["ds", hierarchy_name[0]])
                    df = df.groupby(level="ds").sum()
                    df.to_csv(f"{dataset_name}/cnt.csv")
                    df = pd.read_csv(f"{dataset_name}/cnt.csv")
                    df["ds"] = pd.to_datetime(df["ds"])
                    df = df.sort_values(by='ds')

                    df2 = df_test.loc[df_test['unique_id'] == cnt] # 抽出该层需要的节点的数据

                    df2 = df2.set_index(["ds", hierarchy_name[0]])
                    df2 = df2.groupby(level="ds").sum()
                    df2.to_csv(f"{dataset_name}/cnt.csv")
                    df2 = pd.read_csv(f"{dataset_name}/cnt.csv")
                    df2["ds"] = pd.to_datetime(df2["ds"])
                    df2 = df2.sort_values(by='ds')
                    #df.to_csv("amazon/cnt.csv")
                    result[cnt], fcst_df[cnt], future_fcst[cnt] = tc.forecast(df=df, df_test=df2, choice_type=choice_type,
                                                            freq='D', hierarchy=hierarchy, models=models,
                                                            future_step = future_step, cross_validation_h=cross_validation_h,
                                                            cross_number=cross_number,load_path=load_path)
                    cnt_series = fcst_df[cnt]
                    if cnt_series is not None:
                        if fcst_df_np.size == 0:
                            fcst_df_np = cnt_series.to_numpy()
                        else:
                            fcst_df_np = np.vstack([fcst_df_np, cnt_series.to_numpy()])
                    cnt_series = future_fcst[cnt]
                    if cnt_series is not None:
                        if future_fcst_np.size == 0:
                            future_fcst_np = cnt_series[result[cnt].output.selected_model].to_numpy()
                        else:
                            future_fcst_np = np.vstack([future_fcst_np, cnt_series[result[cnt].output.selected_model].to_numpy()])
                    if result[cnt] is not None:
                        print("selected_model: "+result[cnt].output.selected_model)
                        print("model_details: "+result[cnt].output.model_details)
                        print("model_comparison: "+result[cnt].output.model_comparison)
                        #print("reason_for_selection: "+result[cnt].output.reason_for_selection)
                        #print("user_query_response: "+result[cnt].output.user_query_response)
                        model_choice[cnt] = result[cnt].output.selected_model
        np.save(f'fcst_{dataset_name}/fcst_df_np.npy', fcst_df_np)
        np.save(f'fcst_{dataset_name}/data_y_df.npy', data_y_df)
        np.save(f'fcst_{dataset_name}/future_fcst_np.npy', future_fcst_np)
        print(model_choice)

        # Train Forecasting time series model for hierarchy
        # transform numpy to train One MLP
        fcst_df_np = np.load(f'fcst_{dataset_name}/fcst_df_np.npy', allow_pickle=True)
        data_y_df = np.load(f'fcst_{dataset_name}/data_y_df.npy', allow_pickle=True)
        future_fcst_np = np.load(f'fcst_{dataset_name}/future_fcst_np.npy', allow_pickle=True)
        fcst_df_np = fcst_df_np.T
        data_y_df = data_y_df.T
        future_fcst_np = future_fcst_np.T
        if len(fcst_df_np) > 15:
            # define ML model
            trainer = MLPTrainer(input_dim=total_node_number, output_dim=hierarchy_0_node_number, dropout_rate=0.0)
            # prepare data, please make sure input_dim and output_dim both > 1
            trainer.prepare_data(fcst_df_np[:-10], data_y_df[:-10], test_size=0.2, batch_size=32)
            # train
            train_losses, val_losses = trainer.train(epochs=20000, learning_rate=0.001)
            # Test
            predictions = trainer.predict(fcst_df_np[-10:])
            mae = mean_absolute_error(predictions, data_y_df[-10:])
            if mae < val_losses[-1]:
                predictions = trainer.predict(future_fcst_np)
            else:
                predictions = future_fcst_np[:, :hierarchy_0_node_number]
        else:
            predictions = future_fcst_np[:, :hierarchy_0_node_number]

        # ground truth
        df_test = pd.read_csv(f"{dataset_name}/test.csv")
        df_test = df_test.drop('item', axis=1)
        df_test = df_test.drop('country', axis=1)
        df_test = df_test.drop('number', axis=1)
        hierarchy = 0
        df = df_test.rename(columns={hierarchy_name[0]: 'unique_id'})

        for i in range(hierarchy_level - 2):
            df = df.drop(hierarchy_name[i + 1], axis=1)
        for cnt in hierarchy_node[hierarchy]:
            df1 = df.loc[df['unique_id'] == cnt]
            if future_test_np.size == 0:
                future_test_np = np.array(df1.to_numpy()[:, df1.columns.get_loc('y')].tolist())
            else:
                future_test_np = np.vstack(
                    [future_test_np, np.array(df1.to_numpy()[:, df1.columns.get_loc('y')].tolist())])
        future_test_np = future_test_np.T
        np.save(f'fcst_{dataset_name}/predictions.npy', predictions)
        np.save(f'fcst_{dataset_name}/future_test_np.npy', future_test_np)
        # Pre-fcst Evaluation
        mae1 = mean_absolute_error(future_fcst_np[:, :hierarchy_0_node_number], future_test_np)
        print(f"Pre-fcst Evaluation MAE: {mae1:.4f}")
        # Final Evaluation
        mae2 = mean_absolute_error(predictions, future_test_np)
        print(f"Final Evaluation MAE: {mae2:.4f}")
        mse1 = mean_squared_error(future_fcst_np[:, :hierarchy_0_node_number], future_test_np)
        print(f"MSE1: {mse1:.4f}")
        mse2 = mean_squared_error(predictions, future_test_np)
        print(f"MSE2: {mse2:.4f}")

    else:
        Y_df, S_df, tags = HierarchicalData.load('./data', f'{dataset_name}')
        Y_df['ds'] = pd.to_datetime(Y_df['ds'])
        cross_number = 1
        if dataset_name == "TourismLarge":
            cross_validation_h = 20
            future_step = 20
        elif dataset_name == "Traffic":
            cross_validation_h = 60
            future_step = 60
        elif dataset_name == "TourismSmall": # need to set start_padding_enabled=True
            cross_validation_h = 4
            future_step = 4
        elif dataset_name == "Labour":
            cross_validation_h = 90
            future_step = 90
        elif dataset_name == "Wiki2":
            cross_validation_h = 60
            future_step = 60
        # split train/test sets
        Y_test_df = Y_df.groupby('unique_id').tail(future_step)
        Y_train_df = Y_df.drop(Y_test_df.index)
        # get node names
        hierarchy_node = [list(S_df.axes[1].values),
                          list(set(list(S_df.axes[0].values)) - set(list(S_df.axes[1].values)))]
        total_node_number = len(list(S_df.axes[0].values))
        hierarchy_0_node_number = len(list(S_df.axes[1].values))
        print("hierarchy_0_node_number: " + str(hierarchy_0_node_number))
        print("total_node_number: " + str(total_node_number))
        freq = None

        if choice_type == 'LLM':
            # choose model in dataset level
            if models != default_models:
                print('Models are not default models. Please retrain dataset model select part or ban it.')
                sys.exit()
            features = ["arch_stat", "crossing_points", "entropy", "flat_spots", "lumpiness", "nonlinearity",
                        "stability", "unitroot_kpss", "unitroot_pp", "series_length", "hurst"]
            x = np.array([])
            model_num = len(models)
            model_feature = np.eye(model_num)
            df_data = ExperimentDataset(df=Y_train_df, freq=None, h=None, seasonality=None)
            df_data.freq = 'M'
            df_data.seasonality = 1
            df_stl_features = tsfeatures_tool(df_data, features)
            dataset_feature = df_stl_features.to_numpy()[0]
            for i in range(model_num):
                    cnt = np.hstack([dataset_feature, model_feature[i]])
                    if x.size == 0:
                        x = cnt
                    else:
                        x = np.vstack([x, cnt])
            performance_predictor = XGBoost()
            performance_predictor.load_model()
            performance = performance_predictor.predict(x)
            models = tc.choose_model_in_dataset_level(models, performance, ST_num=3, NN_num=3, FM_num=1)
            print('For this dataset, we use the following models:')
            print(models)

        # train
        for model in models:
            if model not in ['ADIDA', 'AutoARIMA', 'AutoETS', 'DynamicOptimizedTheta', 'SeasonalNaive',
                             'CrostonClassic', 'HistoricAverage', 'Chronos', 'Sundial', 'TiRex']:
                cnt = tc.train_model(df=Y_train_df,
                                     freq=freq,
                                     model=model,
                                     dataset_name=dataset_name,
                                     h=future_step,)
        load_path = "True"
        # predict
        number = 0
        for hierarchy in range(2):
            for cnt in hierarchy_node[hierarchy]:
                number = number + 1
                print("node_number: " + str(number))
                df = Y_df.loc[Y_df['unique_id'] == cnt]
                df_train = Y_train_df.loc[Y_train_df['unique_id'] == cnt]
                df_test = Y_test_df.loc[Y_test_df['unique_id'] == cnt]
                result[cnt], fcst_df[cnt], future_fcst[cnt] = tc.forecast(df=df, df_test=df_test,
                                                                          choice_type=choice_type,
                                                                          freq=freq,
                                                                          hierarchy=hierarchy,
                                                                          models=models,
                                                                          future_step=future_step,
                                                                          cross_validation_h=cross_validation_h,
                                                                          cross_number=cross_number,
                                                                          dataset_name=dataset_name,
                                                                          load_path=load_path)
                if hierarchy == 0:
                    cnt_series = df_train.iloc[-cross_validation_h:, ]
                    if cnt_series is not None:
                        if data_y_df.size == 0:
                            data_y_df = np.array(cnt_series.to_numpy()[:, cnt_series.columns.get_loc('y')].tolist())
                        else:
                            data_y_df = np.vstack(
                                [data_y_df, np.array(cnt_series.to_numpy()[:, cnt_series.columns.get_loc('y')].tolist())])
                cnt_series = fcst_df[cnt]
                if cnt_series is not None:
                    if fcst_df_np.size == 0:
                        fcst_df_np = cnt_series[result[cnt].output.selected_model].to_numpy()
                    else:
                        fcst_df_np = np.vstack(
                            [fcst_df_np, cnt_series[result[cnt].output.selected_model].to_numpy()])
                cnt_series = future_fcst[cnt]
                if cnt_series is not None:
                    if future_fcst_np.size == 0:
                        future_fcst_np = cnt_series[result[cnt].output.selected_model].to_numpy()
                    else:
                        future_fcst_np = np.vstack(
                            [future_fcst_np, cnt_series[result[cnt].output.selected_model].to_numpy()])
                if result[cnt] is not None:
                    print("selected_model: " + result[cnt].output.selected_model)
                    print("model_details: " + result[cnt].output.model_details)
                    print("model_comparison: " + result[cnt].output.model_comparison)
                    # print("reason_for_selection: "+result[cnt].output.reason_for_selection)
                    # print("user_query_response: "+result[cnt].output.user_query_response)
                    model_choice[cnt] = result[cnt].output.selected_model

        np.save(f'fcst_{dataset_name}/fcst_df_np.npy', fcst_df_np)
        np.save(f'fcst_{dataset_name}/data_y_df.npy', data_y_df)
        np.save(f'fcst_{dataset_name}/future_fcst_np.npy', future_fcst_np)
        print(model_choice)

        # Train Forecasting time series model for hierarchy
        fcst_df_np = np.load(f'fcst_{dataset_name}/fcst_df_np.npy', allow_pickle=True)
        data_y_df = np.load(f'fcst_{dataset_name}/data_y_df.npy', allow_pickle=True)
        future_fcst_np = np.load(f'fcst_{dataset_name}/future_fcst_np.npy', allow_pickle=True)
        fcst_df_np = fcst_df_np.T
        data_y_df = data_y_df.T
        future_fcst_np = future_fcst_np.T
        if len(fcst_df_np) > 15:
            # define ML model
            trainer = MLPTrainer(input_dim=total_node_number, output_dim=hierarchy_0_node_number, dropout_rate=0.0)
            # prepare data, please make sure input_dim and output_dim both > 1
            trainer.prepare_data(fcst_df_np[:-10], data_y_df[:-10], test_size=0.2, batch_size=32)
            # train
            train_losses, val_losses = trainer.train(epochs=20000, learning_rate=0.001)
            # Test
            predictions = trainer.predict(fcst_df_np[-10:])
            mae = mean_absolute_error(predictions, data_y_df[-10:])
            if mae < val_losses[-1]:
                predictions = trainer.predict(future_fcst_np)
            else:
                predictions = future_fcst_np[:, :hierarchy_0_node_number]
        else:
            predictions = future_fcst_np[:, :hierarchy_0_node_number]

        # ground truth
        for cnt in hierarchy_node[0]:
            df1 = Y_test_df.loc[Y_test_df['unique_id'] == cnt]
            if future_test_np.size == 0:
                future_test_np = np.array(df1.to_numpy()[:, df1.columns.get_loc('y')].tolist())
            else:
                future_test_np = np.vstack(
                    [future_test_np, np.array(df1.to_numpy()[:, df1.columns.get_loc('y')].tolist())])
        future_test_np = future_test_np.T
        np.save(f'fcst_{dataset_name}/predictions.npy', predictions)
        np.save(f'fcst_{dataset_name}/future_test_np.npy', future_test_np)

        # Pre-fcst Evaluation
        mae1 = mean_absolute_error(future_fcst_np[:, :hierarchy_0_node_number], future_test_np)
        print(f"Pre-fcst Evaluation MAE: {mae1:.4f}")
        # Final Evaluation
        mae2 = mean_absolute_error(predictions, future_test_np)
        print(f"Final Evaluation MAE: {mae2:.4f}")
        mse1 = mean_squared_error(future_fcst_np[:, :hierarchy_0_node_number], future_test_np)
        print(f"MSE1: {mse1:.4f}")
        mse2 = mean_squared_error(predictions, future_test_np)
        print(f"MSE2: {mse2:.4f}")
