import sys
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.agent import AgentRunResult
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
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
from tsfeatures.tsfeatures import _get_feats

from TimeMAC.models.foundation.chronos import Chronos
from TimeMAC.models.foundation.sundial import Sundial
from TimeMAC.models.foundation.tirex import TiRex
from TimeMAC.forecaster import Forecaster, TimeCopilotForecaster
from .models.stats import (
    ADIDA,
    IMAPA,
    AutoARIMA,
    AutoCES,
    AutoETS,
    CrostonClassic,
    DynamicOptimizedTheta,
    HistoricAverage,
    SeasonalNaive,
)
from .models.neural import (
    TSxLSTM,
    TSRNN,
    TSNHITS,
    TSTimesNet,
    TSPatchTST,
    TSTimeXer,
    TSFEDformer,
    TSiTransformer,
    TSTimeMixer,
    TSNLinear,
)

from .utils.experiment_handler import ExperimentDataset, ExperimentDatasetParser
from gluonts.time_feature.seasonality import (
    get_seasonality as _get_seasonality,
)

DEFAULT_MODELS: list[Forecaster] = [
    TSxLSTM(
        input_size=-1,
        inference_input_size=16,
        h_train=10,
        learning_rate=1e-3,
        max_steps=50,
        val_check_steps=1,
        batch_size=1,
        valid_batch_size=20,
        windows_batch_size=20,
        inference_windows_batch_size=20,
        start_padding_enabled=True,
        encoder_n_layers=2,
        encoder_hidden_size=128,
        encoder_dropout=0.3,
        decoder_hidden_size=64,
        decoder_layers=2, ),
    TSRNN(
        input_size=-1,
        inference_input_size=-1,
        h_train=10,
        learning_rate=1e-4,
        max_steps=500,
        val_check_steps=20,
        batch_size=64,
        valid_batch_size=20,
        windows_batch_size=20,
        inference_windows_batch_size=20,
        start_padding_enabled=True,
        encoder_n_layers=2,
        encoder_hidden_size=64,
        encoder_dropout=0.2,
        decoder_hidden_size=32,
        decoder_layers=2, ),
    TSNHITS(),
    ADIDA(),
    AutoARIMA(),
    SeasonalNaive(),
    CrostonClassic(),
    HistoricAverage(),
    AutoETS(),
    DynamicOptimizedTheta(),
    #TSTimesNet(),# long runtime
    TSPatchTST(
        input_size=128, # 128
        encoder_layers=2, # 3
        n_heads=4, # 8
        hidden_size=256, # 256
        linear_hidden_size=256,
        dropout=0.0,
        fc_dropout=0.0,
        head_dropout=0.0,
        attn_dropout=0.0,
        patch_len=16,
        stride=8,
        res_attention=True,
        batch_normalization=True,
        max_steps=5000,
        learning_rate=1e-4,
        val_check_steps=10,
        batch_size=32,
        windows_batch_size=512,
        inference_windows_batch_size=512,
        start_padding_enabled=True,),
    TSTimeXer(),
    TSFEDformer(),
    TSiTransformer(),
    TSTimeMixer(),
    TSNLinear(),
    Chronos(repo_id="E:\\foundation_model\\chronos-t5-base"),
    Sundial(repo_id="E:\\foundation_model\\sundial-base-128m"),
    TiRex(repo_id="NX-AI/TiRex"),
]

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

class ForecastAgentOutput(BaseModel):
    """The output of the forecasting agent."""
    selected_model: str = Field(
        description="The model that was selected for the forecast"
    )
    model_details: str = Field(
        description=(
            "Technical details about the selected model including its assumptions, "
            "strengths, and typical use cases."
        )
    )
    model_comparison: str = Field(
        description=(
            "Detailed comparison of model performances, explaining why certain "
            "models performed better or worse on this specific time series."
        )
    )
    reason_for_selection: str = Field(
        description="Explanation for why the selected model was chosen"
    )
    user_query_response: str | None = Field(
        description=(
            "The response to the user's query, if any. "
            "If the user did not provide a query, this field will be None."
        )
    )

    def prettify(
        self,
        console: Console | None = None,
        features_df: pd.DataFrame | None = None,
        eval_df: pd.DataFrame | None = None,
        fcst_df: pd.DataFrame | None = None,
    ) -> None:
        """Pretty print the forecast results using rich formatting."""
        console = console or Console()

        # Create header with title and overview
        header = Panel(
            f"[bold cyan]{self.selected_model}[/bold cyan] forecast analysis\n"
            f"[{'green' if self.is_better_than_seasonal_naive else 'red'}]"
            f"{'✓ Better' if self.is_better_than_seasonal_naive else '✗ Not better'} "
            "than Seasonal Naive[/"
            f"{'green' if self.is_better_than_seasonal_naive else 'red'}]",
            title="[bold blue]TimeCopilot Forecast[/bold blue]",
            style="blue",
        )

        # Time Series Analysis Section - check if features_df is available
        ts_features = Table(
            title="Time Series Features",
            show_header=True,
            title_style="bold cyan",
            header_style="bold magenta",
        )
        ts_features.add_column("Feature", style="cyan")
        ts_features.add_column("Value", style="magenta")

        # Use features_df if available (attached after forecast run)
        if features_df is not None:
            for feature_name, feature_value in features_df.iloc[0].items():
                if pd.notna(feature_value):
                    ts_features.add_row(feature_name, f"{float(feature_value):.3f}")
        else:
            # Fallback: show a note that detailed features are not available
            ts_features.add_row("Features", "Available in analysis text below")

        ts_analysis = Panel(
            f"{self.tsfeatures_analysis}",
            title="[bold cyan]Feature Analysis[/bold cyan]",
            style="blue",
        )

        # Model Selection Section
        model_details = Panel(
            f"[bold]Technical Details[/bold]\n{self.model_details}\n\n"
            f"[bold]Selection Rationale[/bold]\n{self.reason_for_selection}",
            title="[bold green]Model Information[/bold green]",
            style="green",
        )

        # Model Comparison Table - check if eval_df is available
        model_scores = Table(
            title="Model Performance", show_header=True, title_style="bold yellow"
        )
        model_scores.add_column("Model", style="yellow")
        model_scores.add_column("MASE", style="cyan", justify="right")

        # Use eval_df if available (attached after forecast run)
        if eval_df is not None:
            # Get the MASE scores from eval_df
            model_scores_data = []
            for col in eval_df.columns:
                if col != "metric" and pd.notna(eval_df[col].iloc[0]):
                    model_scores_data.append((col, float(eval_df[col].iloc[0])))

            # Sort by score (lower MASE is better)
            model_scores_data.sort(key=lambda x: x[1])
            for model, score in model_scores_data:
                model_scores.add_row(model, f"{score:.3f}")
        else:
            # Fallback: show a note that detailed scores are not available
            model_scores.add_row("Scores", "Available in analysis text below")

        model_analysis = Panel(
            self.model_comparison,
            title="[bold yellow]Performance Analysis[/bold yellow]",
            style="yellow",
        )

        # Forecast Results Section - check if fcst_df is available
        forecast_table = Table(
            title="Forecast Values", show_header=True, title_style="bold magenta"
        )
        forecast_table.add_column("Period", style="magenta")
        forecast_table.add_column("Value", style="cyan", justify="right")

        # Use fcst_df if available (attached after forecast run)
        if fcst_df is not None:
            # Show forecast values from fcst_df
            fcst_data = fcst_df.copy()
            if "ds" in fcst_data.columns and self.selected_model in fcst_data.columns:
                for _, row in fcst_data.iterrows():
                    period = (
                        row["ds"].strftime("%Y-%m-%d")
                        if hasattr(row["ds"], "strftime")
                        else str(row["ds"])
                    )
                    value = row[self.selected_model]
                    forecast_table.add_row(period, f"{value:.2f}")

                # Add note about number of periods if many
                if len(fcst_data) > 12:
                    forecast_table.caption = (
                        f"[dim]Showing all {len(fcst_data)} forecasted periods. "
                        "Use aggregation functions for summarized views.[/dim]"
                    )
            else:
                forecast_table.add_row("Forecast", "Available in analysis text below")
        else:
            # Fallback: show a note that detailed forecast is not available
            forecast_table.add_row("Forecast", "Available in analysis text below")

        forecast_analysis = Panel(
            self.forecast_analysis,
            title="[bold magenta]Forecast Analysis[/bold magenta]",
            style="magenta",
        )

        # Optional user response section
        user_response = None
        if self.user_query_response:
            user_response = Panel(
                self.user_query_response,
                title="[bold]Response to Query[/bold]",
                style="cyan",
            )

        # Print all sections with clear separation
        console.print("\n")
        console.print(header)

        console.print("\n[bold]1. Time Series Analysis[/bold]")
        console.print(ts_features)
        console.print(ts_analysis)

        console.print("\n[bold]2. Model Selection[/bold]")
        console.print(model_details)
        console.print(model_scores)
        console.print(model_analysis)

        console.print("\n[bold]3. Forecast Results[/bold]")
        console.print(forecast_table)
        console.print(forecast_analysis)

        if user_response:
            console.print("\n[bold]4. Additional Information[/bold]")
            console.print(user_response)

        console.print("\n")


def _transform_time_series_to_text(df: pd.DataFrame) -> str:
    df_agg = df.groupby("unique_id").agg(list)
    output = (
        "these are the time series in json format where the key is the "
        "identifier of the time series and the values is also a json "
        "of two elements: "
        "the first element is the date column and the second element is the "
        "value column."
        f"{df_agg.to_json(orient='index')}"
    )
    return output


def _transform_features_to_text(features_df: pd.DataFrame) -> str:
    output = (
        "these are the time series features in json format where the key is "
        "the identifier of the time series and the values is also a json of "
        "feature names and their values."
        f"{features_df.to_json(orient='index')}"
    )
    return output


def _transform_eval_to_text(eval_df: pd.DataFrame, models: list[str]) -> str:
    output = ", ".join([f"{model}: {eval_df[model].iloc[0]}" for model in models])
    return output


def _transform_fcst_to_text(fcst_df: pd.DataFrame) -> str:
    df_agg = fcst_df.groupby("unique_id").agg(list)
    output = (
        "these are the forecasted values in json format where the key is the "
        "identifier of the time series and the values is also a json of two "
        "elements: the first element is the date column and the second "
        "element is the value column."
        f"{df_agg.to_json(orient='index')}"
    )
    return output

# TimeMAC
class TimeCopilot:
    def __init__(
        self,
        llm: str,
        forecasters: list[Forecaster] | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            llm: The LLM to use.
            forecasters: A list of forecasters to use. If not provided,
                TimeCopilot will use the default forecasters.
            **kwargs: Additional keyword arguments to pass to the agent.
        """

        if forecasters is None:
            forecasters = DEFAULT_MODELS
        self.forecasters = {forecaster.alias: forecaster for forecaster in forecasters}
        self.system_prompt = f"""
        You are a forecasting expert. The system will provide information on a time series dataset.
        Your task is to determine the best forecasting model for the time series from among many models.
        All available models have been given in the information, please choose among them.
        The bases for your decision are as follows:
        1. The features of a time series dataset and additional prompt.
            - Time series features (trend, seasonality, stationarity, etc.)
            - Characteristics of the model itself
            - Additional prompts for model selection
            - Use these insights to guide efficient model selection
            - Focus on features and information that directly inform model choice
        
        2. The evaluation of these models for forecasting this series.
            - This is cross-validation result, which directly affects the forecasting results and model selection
            - Select candidate models based on the time series features and information provided
            - Document each model's technical details and assumptions
            - Explain why these models are suitable for the identified features
            - Balance model complexity with forecast accuracy

        3. Final Model Selection
           - Choose the well performing model with clear justification.
           - Interpret trends and patterns in the forecast
           - Discuss reliability and potential uncertainties
           - Address any specific aspects from the user's prompt
           - Please prioritise the method with the best evaluation results.

        The evaluation will use MAE (Mean Absolute Error) by default.

        Your output must include:
        - Comprehensive feature analysis with clear implications
        - Detailed model comparison and selection rationale
        - Technical details of the selected model
        - Clear interpretation of cross-validation results
        - Response to any user queries

        Focus on providing:
        - Clear connections between features and model choices
        - Technical accuracy with accessible explanations
        - Quantitative support for decisions
        - Thorough responses to user concerns
        
        The models for selection and their characteristics are briefly outlined as follows:
        - TSRNN: a neural network method, this method is suitable for time series forecasting.
        - TSPathTST: The PatchTST model is an efficient Transformer-based (neural network) model for time series forecasting.
        - TiRex: a foundation model (neural network) for time series forecasting.
        - ADIDA: Useful for time series containing numerous zero values.
        - AutoARIMA: Using the existing auto-correlation relationships within time series, 
                    this method is suitable for time series with trend and seasonal components.
        - SeasonalNaive: A very simple and unadorned statistical model.    
        - AutoETS: Suitable for time series with trend, seasonal components and noise.
        - DynamicOptimizedTheta: Suitable for scenarios with significant fluctuations and changing data distributions. 
        """

        '''
        The models for selection and their characteristics are briefly outlined as follows:
        - ADIDA and IMAPA: Useful for time series containing numerous zero values.
        - AutoARIMA: Using the existing auto-correlation relationships within time series, 
                    this method is suitable for time series with trend and seasonal components.
        - AutoETS: Suitable for time series with trend, seasonal components and noise.
        - DynamicOptimizedTheta: Suitable for scenarios with significant fluctuations and changing data distributions.
        - HistoricAverage: Suitable for sequences without significant trends or seasonality.
        - SeasonalNaive: A very simple and unadorned statistical model.
        - CrostonClassic: suitable for time series with low demand levels and irregular patterns.
        '''


        if "model" in kwargs:
            raise ValueError(
                "model is not allowed to be passed as a keyword argument"
                "use `llm` instead"
            )
        self.llm = llm

        self.forecasting_agent = Agent(
            deps_type=ExperimentDataset,
            output_type=ForecastAgentOutput,
            system_prompt=self.system_prompt,
            model=self.llm,
            **kwargs,
        )
        self.forecasting_agent2 = Agent(
                            deps_type=ExperimentDataset,
                            output_type=ForecastAgentOutput,
                            system_prompt=self.system_prompt,
                            model=self.llm,
                            **kwargs,
        )
        self.forecasting_agent3 = Agent(
                            deps_type=ExperimentDataset,
                            output_type=ForecastAgentOutput,
                            system_prompt=self.system_prompt,
                            model=self.llm,
                            **kwargs,
        )

        self.query_system_prompt = """
        You are a forecasting assistant. You have access to the following dataframes 
        from a previous analysis:
        - eval_df: Evaluation results for each model. The evaluation metric is always 
          MASE (Mean Absolute Scaled Error), as established in the main system prompt. 
          Each value in eval_df represents the MAE score for a model.
        - features_df: Extracted time series features for each series, such as trend, 
          seasonality, autocorrelation, and more.

        When the user asks a follow-up question, use these dataframes to provide 
        detailed, data-driven answers. Reference specific values, trends, or metrics 
        from the dataframes as needed. If the user asks about model performance, use 
        eval_df and explain that the metric is MAE.

        Always explain your reasoning and cite the relevant data when answering. If a 
        question cannot be answered with the available data, politely explain the 
        limitation.
        """

        self.query_agent = Agent(
            deps_type=ExperimentDataset,
            output_type=str,
            system_prompt=self.query_system_prompt,
            model=self.llm,
            **kwargs,
        )

        self.dataset: ExperimentDataset
        self.fcst_df: pd.DataFrame
        self.eval_df: pd.DataFrame
        self.features_df: pd.DataFrame
        self.eval_forecasters: list[str]

        '''@self.query_agent.system_prompt
        async def add_experiment_info(
            ctx: RunContext[ExperimentDataset],
        ) -> str:
            output = "\n".join(
                [
                    _transform_features_to_text(self.features_df),
                    _transform_eval_to_text(self.eval_df, self.eval_forecasters),
                    #还差一个问题自身的prompt没写，以及一些feature没加
                ]
            )
            return output'''

        '''@self.forecasting_agent.tool
        async def tsfeatures_tool(
            ctx: RunContext[ExperimentDataset],
            features: list[str],
        ) -> str:
            callable_features = []
            for feature in features:
                if feature not in TSFEATURES:
                    raise ModelRetry(
                        f"Feature {feature} is not available. Available features are: "
                        f"{', '.join(TSFEATURES.keys())}"
                    )
                callable_features.append(TSFEATURES[feature])
            features_dfs = []
            for uid in ctx.deps.df["unique_id"].unique():
                features_df_uid = _get_feats(
                    index=uid,
                    ts=ctx.deps.df,
                    features=callable_features,
                    freq=ctx.deps.seasonality,
                )
                features_dfs.append(features_df_uid)
            features_df = pd.concat(features_dfs) if features_dfs else pd.DataFrame()
            features_df = features_df.rename_axis("unique_id")  # type: ignore
            self.features_df = features_df
            return _transform_features_to_text(features_df)'''

        '''@self.forecasting_agent.tool
        async def cross_validation_tool(
            ctx: RunContext[ExperimentDataset],
            models: list[str],
        ) -> str:
            print("processing_cross_validation_tool")
            callable_models = []
            for str_model in models:
                if str_model not in self.forecasters:
                    raise ModelRetry(
                        f"Model {str_model} is not available. Available models are: "
                        f"{', '.join(self.forecasters.keys())}"
                    )
                callable_models.append(self.forecasters[str_model])
            forecaster = TimeCopilotForecaster(models=callable_models)
            print(ctx.deps.df['unique_id'].count() - cross_number* cross_validation_h)
            fcst_df = forecaster.cross_validation(
                df=ctx.deps.df,
                h= cross_validation_h,
                freq=ctx.deps.freq,
                step_size = ctx.deps.df['unique_id'].count() - cross_number* cross_validation_h,
            )
            eval_df = ctx.deps.evaluate_forecast_df(
                forecast_df=fcst_df,
                models=[model.alias for model in callable_models],
            )
            eval_df = eval_df.groupby(
                ["metric"],
                as_index=False,
            ).mean(numeric_only=True)
            self.eval_df = eval_df
            self.eval_forecasters = models
            return _transform_eval_to_text(eval_df, models)'''

        '''@self.forecasting_agent.tool
        async def forecast_tool(
            ctx: RunContext[ExperimentDataset],
            model: str,
        ) -> str:
            callable_model = self.forecasters[model]
            forecaster = TimeCopilotForecaster(models=[callable_model])
            fcst_df = forecaster.forecast(
                df=ctx.deps.df,
                h=ctx.deps.h,
                freq=ctx.deps.freq,
            )
            self.fcst_df = fcst_df
            return _transform_fcst_to_text(fcst_df)'''

        '''@self.forecasting_agent.output_validator
        async def validate_best_model(
            ctx: RunContext[ExperimentDataset],
            output: ForecastAgentOutput,
        ) -> ForecastAgentOutput:
            if not output.is_better_than_seasonal_naive:
                raise ModelRetry(
                    "The selected model is not better than the seasonal naive model. "
                    "Please try again with a different model."
                    "The cross-validation results are: "
                    f"{output.model_comparison}"
                )
            return output'''

    def cross_validation_tool(self, ctx: ExperimentDataset, cross_validation_h: int, dataset_name: str,
        cross_number: int, models: list[str], load_path: str | None = None,):
        callable_models = []
        for str_model in models:
            if str_model not in self.forecasters:
                print(
                    f"Model {str_model} is not available. Available models are: "
                    f"{', '.join(self.forecasters.keys())}"
                )
            callable_models.append(self.forecasters[str_model])
        forecaster = TimeCopilotForecaster(models=callable_models)
        #print(ctx.df['unique_id'].count() - cross_number * cross_validation_h)
        fcst_df = forecaster.cross_validation(
            df=ctx.df,
            h=cross_validation_h,
            freq=ctx.freq,
            load_path=load_path,
            #step_size=ctx.df['unique_id'].count() - cross_number * cross_validation_h,
        )
        eval_df = ctx.evaluate_forecast_df(
            forecast_df=fcst_df,
            models=[model.alias for model in callable_models],
        )
        eval_df = eval_df.groupby(
            ["metric"],
            as_index=False,
        ).mean(numeric_only=True)
        # eval_df.to_csv("amazon/eval_df.csv")
        self.eval_df = eval_df
        self.eval_forecasters = models
        return eval_df, models, fcst_df

    def tsfeatures_tool(self, ctx: ExperimentDataset, features: list[str]):
        callable_features = []
        for feature in features:
            if feature not in TSFEATURES:
                raise ModelRetry(
                    f"Feature {feature} is not available. Available features are: "
                    f"{', '.join(TSFEATURES.keys())}"
                )
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
        features_df = pd.concat(features_dfs) if features_dfs else pd.DataFrame()
        features_df = features_df.rename_axis("unique_id")  # type: ignore
        #self.features_df = features_df
        return features_df

    def is_queryable(self) -> bool:
        """
        Check if the class is queryable.
        It needs to have `dataset`, `fcst_df`, `eval_df`, `features_df`
        and `eval_forecasters`.
        """
        return all(
            hasattr(self, attr) and getattr(self, attr) is not None
            for attr in [
                "dataset",
                "fcst_df",
                "eval_df",
                "features_df",
                "eval_forecasters",
            ]
        )

    def generate_prompt_hierarchy_from_time_series(
        self,
        models: [str],
        df: pd.DataFrame | str | Path,
        h: int | None = None,
        freq: str | None = None,
        query: str | None = None,
        cross_validation_h: int = None,
        cross_number: int = None,
        hierarchy: int = None,
        seasonality: int = None,
        dataset_name: str | None = None,
        load_path: str | None = None,
    ):
        prompt_start = ("The following information is cross-validation and feature information from dataset, "
                  "used to support model selection.")
        self.data = ExperimentDataset(df=df, freq=freq, h=h, seasonality=seasonality)
        if self.data.seasonality is None and dataset_name not in ["TourismSmall", "TourismLarge", "Traffic", "Labour"]:
            df_stl_features = self.tsfeatures_tool(self.data, ['stl_features'])
            self.data.seasonality = df_stl_features.seasonal_period[0]  # int
            self.data.h = cross_validation_h  # int
            # self.data.seasonality = _get_seasonality(self.data.freq, self.data.df)
            # print("seasonality: ", self.data.seasonality)
        else:
            self.data.seasonality = 1  # int
            self.data.h = cross_validation_h

        eval_df, models, fcst_df = self.cross_validation_tool(self.data, models=models,
                                                              cross_validation_h=cross_validation_h,
                                                              cross_number=cross_number,
                                                              dataset_name=dataset_name,
                                                              load_path=load_path,
                                                     )
        # eval_df.to_csv("amazon/result.csv")
        # transform eval_df and features to prompt
        features_df = self.tsfeatures_tool(ctx=self.data, features=[key for key in TSFEATURES.keys()])
        if hierarchy == 0:
            hierarchy_prompt = ("Because of the background conditions of this time series, "
                                "agents should focus on the detailed time series features when selecting a model. "
                                "Model selection should favour models with more parameters, "
                                "greater precision, and superior performance.")
        else:
            hierarchy_prompt = ("Because of the background conditions of this time series, "
                                "agents should focus on coarse-grained time series features during model selection. "
                                "Model selection should favour models with fewer parameters, and judgement and "
                                "interpretation should place greater emphasis on data patterns.")
        prompt_data = "\n".join(
            [
                _transform_features_to_text(features_df),
                _transform_eval_to_text(eval_df, models),
                hierarchy_prompt,
                'Please carefully check all the numbers and ensure that your comparisons are correct.'
            ]
        )
        return prompt_start+prompt_data, fcst_df

    def forecast(
        self,
        models: [str],
        df: pd.DataFrame | str | Path,
        df_test: pd.DataFrame | str | Path,
        choice_type: str,
        h: int | None = None,
        freq: str | None = None,
        seasonality: int | None = None,
        query: str | None = None,
        hierarchy: int | None = None,
        cross_validation_h: int = 100,
        cross_number: int = 1,
        future_step: int = 90,
        dataset_name: str | None = None,
        load_path: str | None = None,
    ):
    # -> AgentRunResult[ForecastAgentOutput]:
        """Generate forecast and analysis.

        Args:
            df: The time-series data. Can be one of:
                - a *pandas* `DataFrame` with at least the columns
                  `["unique_id", "ds", "y"]`.
                - a file path or URL pointing to a CSV / Parquet file with the
                  same columns (it will be read automatically).
            h: Forecast horizon. Number of future periods to predict. If
                `None` (default), TimeCopilot will try to infer it from
                `query` or, as a last resort, default to `2 * seasonality`.
            freq: Pandas frequency string (e.g. `"H"`, `"D"`, `"MS"`).
                `None` (default), lets TimeCopilot infer it from the data or
                the query. See [pandas frequency documentation](https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases).
            seasonality: Length of the dominant seasonal cycle (expressed in
                `freq` periods). `None` (default), asks TimeCopilot to infer it via
                [`get_seasonality`][TimeMAC.models.utils.forecaster.get_seasonality].
            query: Optional natural-language prompt that will be shown to the
                agent. You can embed `freq`, `h` or `seasonality` here in
                plain English, they take precedence over the keyword
                arguments.

        Returns:
            A result object whose `output` attribute is a fully
                populated [`ForecastAgentOutput`][TimeMAC.agent.ForecastAgentOutput]
                instance. Use `result.output` to access typed fields or
                `result.output.prettify()` to print a nicely formatted
                report.
        """

        if choice_type == "LLM":
            query = f"User query: {query}" if query else None
            prompt, fcst_df = self.generate_prompt_hierarchy_from_time_series(df=df, freq=freq, models=models,
                                                                              cross_validation_h=cross_validation_h,
                                                                              cross_number=cross_number,
                                                                              hierarchy=hierarchy,
                                                                              dataset_name=dataset_name,
                                                                              load_path=load_path,
                                                                              )

            #print(prompt)
            # One Agent Decision
            result = self.forecasting_agent.run_sync(
                user_prompt=prompt,
            )
            final_result = result
            '''
            # Multi-Agent discussion and decision
            argue = "The following represents the decisions made by other Agent. Please argue whether "
                    "it is the optimal choice and then make your final decision."
            discuss_prompt = prompt + "\n".join([argue + "\n",
                    "selected_model: " + str(result.output.selected_model) + "\n",
                    "model_details: " + str(result.output.model_details) + "\n",
                    "model_comparison: " + str(result.output.model_comparison)+ "\n",
                    "reason_for_selection: "+ str(result.output.reason_for_selection)+ "\n",
                    "user_query_response: " + str(result.output.user_query_response)])
            result2 =self.forecasting_agent2.run_sync(
                user_prompt=discuss_prompt,
            )
            discuss_prompt = prompt + "\n".join([argue + "\n",
                                                 "selected_model: " + str(result2.output.selected_model) + "\n",
                                                 "model_details: " + str(result2.output.model_details) + "\n",
                                                 "model_comparison: " + str(result2.output.model_comparison) + "\n",
                                                 "reason_for_selection: " + str(result2.output.reason_for_selection) + "\n",
                                                 "user_query_response: " + str(result2.output.user_query_response)])
            result3 = self.forecasting_agent3.run_sync(
                user_prompt=discuss_prompt,
            )
            discuss_prompt = prompt + "\n".join([argue + "\n",
                                                 "selected_model: " + str(result3.output.selected_model) + "\n",
                                                 "model_details: " + str(result3.output.model_details) + "\n",
                                                 "model_comparison: " + str(result3.output.model_comparison) + "\n",
                                                 "reason_for_selection: " + str(result3.output.reason_for_selection) + "\n",
                                                 "user_query_response: " + str(result3.output.user_query_response)])
            result = self.forecasting_agent3.run_sync(
                user_prompt=discuss_prompt,
            )
            voting_answer = ""
            voting = [result.output.selected_model, result2.output.selected_model, result3.output.selected_model]
            count_voting_dict = {}
            for s in voting:
                count_voting_dict[s] = count_voting_dict.get(s, 0) + 1
            for s in voting:
                if count_voting_dict[s] > 1:
                    voting_answer = s
            if voting_answer == "":
                voting_answer = result.output.selected_model
            for cnt in [result, result2, result3]:
                if cnt.output.selected_model == voting_answer:
                    final_result = cnt
            '''

            # fcst and output
            while result.output.selected_model not in self.forecasters:
                result.output.selected_model = self.forecasting_agent.run_sync(
                    user_prompt=f"{result.output.selected_model} is a wrong model name, maybe you want to choose one from "
                                f"{models}. Please only output the right model name.",
                )
        elif choice_type == "Without_LLM":
            self.data = ExperimentDataset(df=df, freq=freq, h=h, seasonality=seasonality)
            if self.data.seasonality is None and dataset_name not in ["TourismSmall", "TourismLarge", "Traffic",
                                                                      "Labour"]:
                df_stl_features = self.tsfeatures_tool(self.data, ['stl_features'])
                self.data.seasonality = df_stl_features.seasonal_period[0]  # int
                self.data.h = cross_validation_h  # int
                # self.data.seasonality = _get_seasonality(self.data.freq, self.data.df)
                # print("seasonality: ", self.data.seasonality)
            else:
                self.data.seasonality = 1  # int
                self.data.h = cross_validation_h
            eval_df, models, fcst_df = self.cross_validation_tool(self.data, models=models,
                                                                  cross_validation_h=cross_validation_h,
                                                                  cross_number=cross_number,
                                                                  dataset_name=dataset_name,
                                                                  load_path=load_path,
                                                                  )
            # choose best performance model
            eval_df = eval_df.drop('metric', axis=1)
            best = eval_df.idxmin(axis=1)
            model = best[0]
            class _output:
                def __init__(self):
                    self.selected_model = model
                    self.model_details = ""
                    self.model_comparison = ""
            class _result:
                def __init__(self):
                    self.output = _output()
            result = _result()
            final_result = result
        else:
            print("Wrong choice type. Default: LLM, Without_LLM")
            sys.exit()

        df = df.drop(df_test.index)
        forecaster = TimeCopilotForecaster(models=[self.forecasters[result.output.selected_model]])
        fcst_df = forecaster.forecast(
                                df=df.iloc[-cross_validation_h:, ],
                                h=future_step,
                                freq=freq,
                                load_path=load_path,
                                )
        future_forecasting = forecaster.forecast(
                                df=df,
                                h=future_step,
                                freq=freq,
                                load_path=load_path,
                                )
        
        return final_result, fcst_df, future_forecasting

    def forecast_only_train_data(
            self,
            models: [str],
            df: pd.DataFrame | str | Path,
            df_test: pd.DataFrame | str | Path,
            choice_type: str,
            h: int | None = None,
            freq: str | None = None,
            seasonality: int | None = None,
            query: str | None = None,
            hierarchy: int | None = None,
            cross_validation_h: int = 100,
            cross_number: int = 1,
            future_step: int = 90,
            dataset_name: str | None = None,
            load_path: str | None = None,
    ):
        # -> AgentRunResult[ForecastAgentOutput]:
        """Generate forecast and analysis.

        Args:
            df: The time-series data. Can be one of:
                - a *pandas* `DataFrame` with at least the columns
                  `["unique_id", "ds", "y"]`.
                - a file path or URL pointing to a CSV / Parquet file with the
                  same columns (it will be read automatically).
            h: Forecast horizon. Number of future periods to predict. If
                `None` (default), TimeCopilot will try to infer it from
                `query` or, as a last resort, default to `2 * seasonality`.
            freq: Pandas frequency string (e.g. `"H"`, `"D"`, `"MS"`).
                `None` (default), lets TimeCopilot infer it from the data or
                the query. See [pandas frequency documentation](https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases).
            seasonality: Length of the dominant seasonal cycle (expressed in
                `freq` periods). `None` (default), asks TimeCopilot to infer it via
                [`get_seasonality`][TimeMAC.models.utils.forecaster.get_seasonality].
            query: Optional natural-language prompt that will be shown to the
                agent. You can embed `freq`, `h` or `seasonality` here in
                plain English, they take precedence over the keyword
                arguments.

        Returns:
            A result object whose `output` attribute is a fully
                populated [`ForecastAgentOutput`][TimeMAC.agent.ForecastAgentOutput]
                instance. Use `result.output` to access typed fields or
                `result.output.prettify()` to print a nicely formatted
                report.
        """

        if choice_type == "LLM":
            query = f"User query: {query}" if query else None
            prompt, fcst_df = self.generate_prompt_hierarchy_from_time_series(df=df, freq=freq, models=models,
                                                                              cross_validation_h=cross_validation_h,
                                                                              cross_number=cross_number,
                                                                              hierarchy=hierarchy,
                                                                              dataset_name=dataset_name,
                                                                              load_path=load_path,
                                                                              )

            # print(prompt)
            # One Agent Decision
            result = self.forecasting_agent.run_sync(
                user_prompt=prompt,
            )
            final_result = result

            # fcst and output
            while result.output.selected_model not in self.forecasters:
                result.output.selected_model = self.forecasting_agent.run_sync(
                    user_prompt=f"{result.output.selected_model} is a wrong model name, maybe you want to choose one from "
                                f"{models}. Please only output the right model name.",
                )
        elif choice_type == "Without_LLM":
            self.data = ExperimentDataset(df=df, freq=freq, h=h, seasonality=seasonality)
            if self.data.seasonality is None and dataset_name not in ["TourismSmall", "TourismLarge", "Traffic",
                                                                      "Labour"]:
                df_stl_features = self.tsfeatures_tool(self.data, ['stl_features'])
                self.data.seasonality = df_stl_features.seasonal_period[0]  # int
                self.data.h = cross_validation_h  # int
                # self.data.seasonality = _get_seasonality(self.data.freq, self.data.df)
                # print("seasonality: ", self.data.seasonality)
            else:
                self.data.seasonality = 1  # int
                self.data.h = cross_validation_h
            eval_df, models, fcst_df = self.cross_validation_tool(self.data, models=models,
                                                                  cross_validation_h=cross_validation_h,
                                                                  cross_number=cross_number,
                                                                  dataset_name=dataset_name,
                                                                  load_path=load_path,
                                                                  )
            # To choose the best performance model
            eval_df = eval_df.drop('metric', axis=1)
            best = eval_df.idxmin(axis=1)
            model = best[0]

            class _output:
                def __init__(self):
                    self.selected_model = model
                    self.model_details = ""
                    self.model_comparison = ""

            class _result:
                def __init__(self):
                    self.output = _output()

            result = _result()
            final_result = result
        else:
            print("Wrong choice type. Default: LLM, Without_LLM")
            sys.exit()

        forecaster = TimeCopilotForecaster(models=[self.forecasters[result.output.selected_model]])
        fcst_df = forecaster.forecast(
            df=df.iloc[-cross_validation_h:, ],
            h=future_step,
            freq=freq,
            load_path=load_path,
        )
        future_forecasting = forecaster.forecast(
            df=df,
            h=future_step,
            freq=freq,
            load_path=load_path,
        )

        return final_result, fcst_df, future_forecasting

    def forecast_without_agent(
            self,
            models: [str],
            df: pd.DataFrame | str | Path,
            df_test: pd.DataFrame | str | Path,
            h: int | None = None,
            freq: str | None = None,
            seasonality: int | None = None,
            query: str | None = None,
            hierarchy: int | None = None,
            cross_validation_h: int = 100,
            cross_number: int = 1,
            future_step: int = 90,
            dataset_name: str | None = None,
            load_path: str | None = None,
    ):
        # -> AgentRunResult[ForecastAgentOutput]:
        """Generate forecast and analysis.

        Args:
            df: The time-series data. Can be one of:
                - a *pandas* `DataFrame` with at least the columns
                  `["unique_id", "ds", "y"]`.
                - a file path or URL pointing to a CSV / Parquet file with the
                  same columns (it will be read automatically).
            h: Forecast horizon. Number of future periods to predict. If
                `None` (default), TimeCopilot will try to infer it from
                `query` or, as a last resort, default to `2 * seasonality`.
            freq: Pandas frequency string (e.g. `"H"`, `"D"`, `"MS"`).
                `None` (default), lets TimeCopilot infer it from the data or
                the query. See [pandas frequency documentation](https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases).
            seasonality: Length of the dominant seasonal cycle (expressed in
                `freq` periods). `None` (default), asks TimeCopilot to infer it via
                [`get_seasonality`][TimeMAC.models.utils.forecaster.get_seasonality].
            query: Optional natural-language prompt that will be shown to the
                agent. You can embed `freq`, `h` or `seasonality` here in
                plain English, they take precedence over the keyword
                arguments.

        Returns:
            A result object whose `output` attribute is a fully
                populated [`ForecastAgentOutput`][TimeMAC.agent.ForecastAgentOutput]
                instance. Use `result.output` to access typed fields or
                `result.output.prettify()` to print a nicely formatted
                report.
        """

        # only test without Agent
        self.data = ExperimentDataset(df=df, freq=freq, h=h, seasonality=seasonality)
        self.data.freq = freq
        if self.data.seasonality is None and dataset_name not in ["TourismSmall", "TourismLarge", "Traffic", "Labour"]:
            df_stl_features = self.tsfeatures_tool(self.data, ['stl_features'])
            self.data.seasonality = df_stl_features.seasonal_period[0]  # int
            self.data.h = cross_validation_h  # int
            # self.data.seasonality = _get_seasonality(self.data.freq, self.data.df)
            print("seasonality: ", self.data.seasonality)
        else:
            self.data.seasonality = 1  # int
            self.data.h = cross_validation_h
        eval_df, models, fcst_df = self.cross_validation_tool(self.data, models=models,
                                                              cross_validation_h=cross_validation_h,
                                                              cross_number=cross_number,
                                                              dataset_name=dataset_name,
                                                              load_path=load_path,
                                                              )
        forecaster = TimeCopilotForecaster(models=[self.forecasters[models[0]]])
        future_forecasting = forecaster.forecast(
            df=df.drop(df_test.index),
            h=future_step,
            freq=freq,
            load_path=load_path,
        )
        class _output:
            def __init__(self):
                self.selected_model = models[0]
                self.model_details = ""
                self.model_comparison = ""
        class _result:
            def __init__(self):
                self.output = _output()
        result = _result()
        return result, fcst_df, future_forecasting


    def train_model(
        self,
        model: str,
        df: pd.DataFrame | str | Path,
        h: int | None = None,
        freq: str | None = None,
        seasonality: int | None = None,
        query: str | None = None,
        hierarchy: int | None = None,
        cross_validation_h: int = 100,
        cross_number: int = 1,
        future_step: int = 90,
        dataset_name: str | None = None,
        load_path: str | None = None,
    ):
        self.data = ExperimentDataset(df=df, freq=freq, h=h, seasonality=seasonality)
        self.data.freq = freq
        self.data.seasonality = 1  # int
        self.data.h = cross_validation_h
        fcst_cnt = self.forecasters[model].forecast(df=self.data.df,
                                                    h=h,
                                                    freq=self.data.freq,)
        return fcst_cnt
    def _maybe_raise_if_not_queryable(self):
        if not self.is_queryable():
            raise ValueError(
                "The class is not queryable. Please forecast first using `forecast`."
            )

    def query(
        self,
        query: str,
    ) -> AgentRunResult[str]:
        # fmt: off
        """
        Ask a follow-up question about the forecast, model evaluation, or time
        series features.

        This method enables chat-like, interactive querying after a forecast
        has been run. The agent will use the stored dataframes (`fcst_df`,
        `eval_df`, `features_df`) and the original dataset to answer the user's
        question in a data-driven manner. Typical queries include asking about
        the best model, forecasted values, or time series characteristics.

        Args:
            query: The user's follow-up question. This can be about model
                performance, forecast results, or time series features.

        Returns:
            AgentRunResult[str]: The agent's answer as a string. Use
                `result.output` to access the answer.

        Raises:
            ValueError: If the class is not ready for querying (i.e., forecast
                has not been run and required dataframes are missing).

        Example:
            ```python
            import pandas as pd
            from TimeMAC import TimeCopilot

            df = pd.read_csv("https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv") 
            tc = TimeCopilot(llm="openai:gpt-4o")
            tc.forecast(df, h=12, freq="MS")
            answer = tc.query("Which model performed best?")
            print(answer.output)
            ```
        Note:
            The class is not queryable until the `forecast` method has been
            called.
        """
        # fmt: on
        self._maybe_raise_if_not_queryable()
        result = self.query_agent.run_sync(
            user_prompt=query,
            deps=self.data,
        )
        return result

    def choose_model_in_dataset_level(self, models=[], performance=[], ST_num=3, NN_num=3, FM_num=1):
        result = []
        model_type = ['NN', 'ST', 'FM']
        model_ST = []
        model_NN = []
        model_FM = []
        for model in models:
            if model in ['ADIDA', 'AutoARIMA', 'AutoETS', 'DynamicOptimizedTheta', 'CrostonClassic', 'HistoricAverage']:
                model_ST.append(model)
            elif model in ['TSxLSTM', 'TSRNN', 'TSNHITS', 'TSTimesNet', 'TSPatchTST', 'TSTimeXer', 'TSFEDformer',
                           'TSiTransformer', 'TSTimeMixer', 'TSNLinear']:
                model_NN.append(model)
            elif model in ['Chronos', 'Sundial', 'TiRex']:
                model_FM.append(model)
            else:
                print('Wrong model type:' + model)
        for type in model_type:
            if type == 'NN':
                choose_num = min(NN_num, len(model_NN))
                model_cnt = model_NN
            elif type == 'ST':
                choose_num = min(ST_num, len(model_ST))
                model_cnt = model_ST
            else:
                choose_num = min(FM_num, len(model_FM))
                model_cnt = model_FM
            for i in range(choose_num):
                prompt = ('The following information predicts the performance of different models on this dataset, '
                          'with lower numbers indicating better results. Please select the best model from among them.'
                          'If performance is equal, Please choose special models such as TSPatchTST, ADIDA.'
                          'Performance:')
                for model in model_cnt:
                    prompt = prompt + '\n' + model + ': ' + str(performance[models.index(model)])
                choose_model = self.forecasting_agent.run_sync(
                    user_prompt=prompt,
                )

                while choose_model.output.selected_model not in model_cnt:
                    choose_model = self.forecasting_agent.run_sync(
                        user_prompt=f"{choose_model.output.selected_model} is a wrong model name, maybe you want to choose one from "
                                    f"{model_cnt}. Please only output the right model name.",
                    )
                print(choose_model.output)
                result.append(choose_model.output.selected_model)
                model_cnt.remove(choose_model.output.selected_model)
        result.append('SeasonalNaive') # Naive model
        # result.append('ADIDA') # For many zero value data
        return result
