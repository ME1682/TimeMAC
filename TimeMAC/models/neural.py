import os

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.common._base_model import BaseModel as NeuralForecastModel
from .utils.forecaster import Forecaster
import torch

from neuralforecast.models.lstm import LSTM
from neuralforecast.models.xlstm import xLSTM
from neuralforecast.models.rnn import RNN
from neuralforecast.models.nhits import NHITS
from neuralforecast.models.deepar import DeepAR
from neuralforecast.models.timesnet import TimesNet
from neuralforecast.models.patchtst import PatchTST
from neuralforecast.models.timexer import TimeXer
from neuralforecast.models.fedformer import FEDformer
from neuralforecast.models.itransformer import iTransformer
from neuralforecast.models.timemixer import TimeMixer
from neuralforecast.models.nlinear import NLinear


os.environ["NIXTLA_ID_AS_COL"] = "true"

def run_neuralforecast_model(
    model: NeuralForecastModel,
    h: int,
    df: pd.DataFrame,
    freq: str,
    load_path: str | None = None,
) -> pd.DataFrame:

    if load_path is None:
        nf = NeuralForecast(
        models=[model],
        freq=freq,)
        nf.fit(df=df, val_size=h)
        nf.save(f"{nf.models[0].alias}", overwrite=True)
        fcst_df = nf.predict()
    else:
        nf = NeuralForecast(
        models=[model],
        freq=freq,
        )
        nf = nf.load(f"{nf.models[0].alias}")
        fcst_df = nf.predict(df=df, h=h)
    return fcst_df


class TSLSTM(Forecaster):

    def __init__(
        self,
        alias: str = "TSLSTM",
        input_size: int = -1,
        inference_input_size: int = None,
        h_train: int = 10,
        learning_rate: float = 1e-5,
        max_steps: int = 50,
        val_check_steps: int = 10,
        batch_size: int = 1,
        valid_batch_size: int = 1,
        windows_batch_size: int | None = None,
        inference_windows_batch_size: int = 1,
        start_padding_enabled: bool = False,
        encoder_n_layers: int = 3,
        encoder_hidden_size: int = 128,
        encoder_dropout: float = 0.2,
        decoder_hidden_size: int = 128,
        decoder_layers: int = 2,
    ):
        self.alias = alias
        self.input_size = input_size
        self.inference_input_size = inference_input_size
        self.h_train = h_train
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.val_check_steps = val_check_steps
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.windows_batch_size = windows_batch_size
        self.inference_windows_batch_size = inference_windows_batch_size
        self.start_padding_enabled = start_padding_enabled
        self.encoder_n_layers = encoder_n_layers
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_dropout = encoder_dropout
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        load_path: str | None = None,
    ) -> pd.DataFrame:

        if level is not None or quantiles is not None:
            raise ValueError("Level and quantiles are not supported for AutoTFT yet.")
        inferred_freq = self._maybe_infer_freq(df, freq)
        fcst_df = run_neuralforecast_model(
            model=LSTM(
                    h=h,
                    input_size=self.input_size,
                    learning_rate=self.learning_rate,
                    max_steps=self.max_steps,
                    val_check_steps=self.val_check_steps,
                    batch_size=self.batch_size,
                    valid_batch_size=self.valid_batch_size,
                    windows_batch_size=self.windows_batch_size,
                    inference_windows_batch_size=self.inference_windows_batch_size,
                    start_padding_enabled=self.start_padding_enabled,
                    alias=self.alias,
                    encoder_n_layers=self.encoder_n_layers,
                    encoder_hidden_size=self.encoder_hidden_size,
                    inference_input_size=self.inference_input_size,
                    h_train=self.h_train,
                    encoder_dropout=self.encoder_dropout,
                    decoder_hidden_size=self.decoder_hidden_size,
                    decoder_layers=self.decoder_layers,
            ),
            h=h,
            df=df,
            freq=inferred_freq,
            load_path=load_path,
        )
        return fcst_df

class TSxLSTM(Forecaster):

    def __init__(
        self,
        alias: str = "TSxLSTM",
        input_size: int = -1,
        inference_input_size: int = None,
        h_train: int = 10,
        learning_rate: float = 1e-5,
        max_steps: int = 50,
        val_check_steps: int = 10,
        batch_size: int = 1,
        valid_batch_size: int = 1,
        windows_batch_size: int | None = None,
        inference_windows_batch_size: int = 64,
        start_padding_enabled: bool = False,
        encoder_n_layers: int = 3,
        encoder_hidden_size: int = 128,
        encoder_dropout: float = 0.2,
        decoder_hidden_size: int = 128,
        decoder_layers: int = 2,
    ):
        self.alias = alias
        self.input_size = input_size
        self.inference_input_size = inference_input_size
        self.h_train = h_train
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.val_check_steps = val_check_steps
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.windows_batch_size = windows_batch_size
        self.inference_windows_batch_size = inference_windows_batch_size
        self.start_padding_enabled = start_padding_enabled
        self.encoder_n_layers = encoder_n_layers
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_dropout = encoder_dropout
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        load_path: str | None = None,
    ) -> pd.DataFrame:

        if level is not None or quantiles is not None:
            raise ValueError("Level and quantiles are not supported for AutoTFT yet.")
        inferred_freq = self._maybe_infer_freq(df, freq)
        fcst_df = run_neuralforecast_model(
            model=xLSTM(
                    h=h,
                    input_size=self.input_size,
                    learning_rate=self.learning_rate,
                    max_steps=self.max_steps,
                    val_check_steps=self.val_check_steps,
                    batch_size=self.batch_size,
                    valid_batch_size=self.valid_batch_size,
                    windows_batch_size=self.windows_batch_size,
                    inference_windows_batch_size=self.inference_windows_batch_size,
                    start_padding_enabled=self.start_padding_enabled,
                    alias=self.alias,
                    encoder_n_layers=self.encoder_n_layers,
                    encoder_hidden_size=self.encoder_hidden_size,
                    inference_input_size=self.inference_input_size,
                    h_train=self.h_train,
                    encoder_dropout=self.encoder_dropout,
                    decoder_hidden_size=self.decoder_hidden_size,
                    decoder_layers=self.decoder_layers,
            ),
            h=h,
            df=df,
            freq=inferred_freq,
            load_path=load_path,
        )
        return fcst_df


class TSRNN(Forecaster):

    def __init__(
        self,
        alias: str = "TSRNN",
        input_size: int = -1,
        inference_input_size: int = None,
        h_train: int = 10,
        learning_rate: float = 1e-5,
        max_steps: int = 50,
        val_check_steps: int = 10,
        batch_size: int = 1,
        valid_batch_size: int = 1,
        windows_batch_size: int | None = None,
        inference_windows_batch_size: int = 1,
        start_padding_enabled: bool = False,
        encoder_n_layers: int = 3,
        encoder_hidden_size: int = 128,
        encoder_dropout: float = 0.2,
        decoder_hidden_size: int = 128,
        decoder_layers: int = 2,
        load_path: str | None = None,
    ):
        self.alias = alias
        self.input_size = input_size
        self.inference_input_size = inference_input_size
        self.h_train = h_train
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.val_check_steps = val_check_steps
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.windows_batch_size = windows_batch_size
        self.inference_windows_batch_size = inference_windows_batch_size
        self.start_padding_enabled = start_padding_enabled
        self.encoder_n_layers = encoder_n_layers
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_dropout = encoder_dropout
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers
        self.load_path = load_path

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        load_path: str | None = None,
    ) -> pd.DataFrame:

        if level is not None or quantiles is not None:
            raise ValueError("Level and quantiles are not supported for AutoTFT yet.")
        inferred_freq = self._maybe_infer_freq(df, freq)
        fcst_df = run_neuralforecast_model(
            model=RNN(
                    h=h,
                    input_size=self.input_size,
                    learning_rate=self.learning_rate,
                    max_steps=self.max_steps,
                    val_check_steps=self.val_check_steps,
                    batch_size=self.batch_size,
                    valid_batch_size=self.valid_batch_size,
                    windows_batch_size=self.windows_batch_size,
                    inference_windows_batch_size=self.inference_windows_batch_size,
                    start_padding_enabled=self.start_padding_enabled,
                    alias=self.alias,
                    encoder_n_layers=self.encoder_n_layers,
                    encoder_hidden_size=self.encoder_hidden_size,
                    inference_input_size=self.inference_input_size,
                    h_train=self.h_train,
                    encoder_dropout=self.encoder_dropout,
                    decoder_hidden_size=self.decoder_hidden_size,
                    decoder_layers=self.decoder_layers,
            ),
            h=h,
            df=df,
            freq=inferred_freq,
            load_path=load_path,
        )
        return fcst_df

class TSNHITS(Forecaster):
    def __init__(
        self,
        alias: str = "TSNHITS",
        input_size: int = 16,
        inference_input_size: int = None,
        h_train: int = 10,
        learning_rate: float = 1e-5,
        max_steps: int = 50,
        val_check_steps: int = 10,
        batch_size: int = 1,
        valid_batch_size: int = 1,
        windows_batch_size: int | None = None,
        inference_windows_batch_size: int = 1,
        start_padding_enabled: bool = False,
        encoder_n_layers: int = 3,
        encoder_hidden_size: int = 128,
        encoder_dropout: float = 0.2,
        decoder_hidden_size: int = 128,
        decoder_layers: int = 2,
    ):
        self.alias = alias
        self.input_size = input_size
        self.inference_input_size = inference_input_size
        self.h_train = h_train
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.val_check_steps = val_check_steps
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.windows_batch_size = windows_batch_size
        self.inference_windows_batch_size = inference_windows_batch_size
        self.start_padding_enabled = start_padding_enabled
        self.encoder_n_layers = encoder_n_layers
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_dropout = encoder_dropout
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        load_path: str | None = None,
    ) -> pd.DataFrame:

        if level is not None or quantiles is not None:
            raise ValueError("Level and quantiles are not supported for AutoTFT yet.")
        inferred_freq = self._maybe_infer_freq(df, freq)
        fcst_df = run_neuralforecast_model(
            model=NHITS(
                        h=h,
                        input_size=self.input_size,
                        max_steps=self.max_steps,
                        learning_rate=self.learning_rate,
                        val_check_steps=self.val_check_steps,
                        batch_size=self.batch_size,
                        valid_batch_size=self.valid_batch_size,
                        windows_batch_size=self.windows_batch_size,
                        inference_windows_batch_size=self.inference_windows_batch_size,
                        start_padding_enabled=self.start_padding_enabled,
                        alias=self.alias,
            ),
            h=h,
            df=df,
            freq=inferred_freq,
            load_path=load_path,
        )
        return fcst_df

class TSDeepAR(Forecaster):
    def __init__(
        self,
        alias: str = "TSDeepAR",
        input_size: int = 16,
        inference_input_size: int = None,
        h_train: int = 10,
        learning_rate: float = 1e-5,
        max_steps: int = 50,
        val_check_steps: int = 10,
        batch_size: int = 1,
        valid_batch_size: int = 1,
        windows_batch_size: int | None = None,
        inference_windows_batch_size: int = 1,
        start_padding_enabled: bool = False,
        encoder_n_layers: int = 3,
        encoder_hidden_size: int = 128,
        encoder_dropout: float = 0.2,
        decoder_hidden_size: int = 128,
        decoder_layers: int = 2,
    ):
        self.alias = alias
        self.input_size = input_size
        self.inference_input_size = inference_input_size
        self.h_train = h_train
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.val_check_steps = val_check_steps
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.windows_batch_size = windows_batch_size
        self.inference_windows_batch_size = inference_windows_batch_size
        self.start_padding_enabled = start_padding_enabled
        self.encoder_n_layers = encoder_n_layers
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_dropout = encoder_dropout
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        load_path: str | None = None,
    ) -> pd.DataFrame:

        if level is not None or quantiles is not None:
            raise ValueError("Level and quantiles are not supported for AutoTFT yet.")
        inferred_freq = self._maybe_infer_freq(df, freq)
        fcst_df = run_neuralforecast_model(
            model=DeepAR(
                        h=h,
                        input_size=self.input_size,
                        max_steps=self.max_steps,
                        learning_rate=self.learning_rate,
                        val_check_steps=self.val_check_steps,
                        batch_size=self.batch_size,
                        valid_batch_size=self.valid_batch_size,
                        windows_batch_size=self.windows_batch_size,
                        inference_windows_batch_size=self.inference_windows_batch_size,
                        start_padding_enabled=self.start_padding_enabled,
                        alias=self.alias,
            ),
            h=h,
            df=df,
            freq=inferred_freq,
            load_path=load_path,
        )
        return fcst_df

class TSTimesNet(Forecaster):
    def __init__(
        self,
        alias: str = "TSTimesNet",
        input_size: int = 16,
        inference_input_size: int = None,
        h_train: int = 10,
        learning_rate: float = 1e-5,
        max_steps: int = 50,
        val_check_steps: int = 10,
        batch_size: int = 1,
        valid_batch_size: int = 1,
        windows_batch_size: int | None = None,
        inference_windows_batch_size: int = 1,
        start_padding_enabled: bool = False,
        encoder_n_layers: int = 3,
        encoder_hidden_size: int = 128,
        encoder_dropout: float = 0.2,
        decoder_hidden_size: int = 128,
        decoder_layers: int = 2,
    ):
        self.alias = alias
        self.input_size = input_size
        self.inference_input_size = inference_input_size
        self.h_train = h_train
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.val_check_steps = val_check_steps
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.windows_batch_size = windows_batch_size
        self.inference_windows_batch_size = inference_windows_batch_size
        self.start_padding_enabled = start_padding_enabled
        self.encoder_n_layers = encoder_n_layers
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_dropout = encoder_dropout
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        load_path: str | None = None,
    ) -> pd.DataFrame:

        if level is not None or quantiles is not None:
            raise ValueError("Level and quantiles are not supported for AutoTFT yet.")
        inferred_freq = self._maybe_infer_freq(df, freq)
        fcst_df = run_neuralforecast_model(
            model=TimesNet(
                        h=h,
                        input_size=self.input_size,
                        max_steps=self.max_steps,
                        learning_rate=self.learning_rate,
                        val_check_steps=self.val_check_steps,
                        batch_size=self.batch_size,
                        valid_batch_size=self.valid_batch_size,
                        windows_batch_size=self.windows_batch_size,
                        inference_windows_batch_size=self.inference_windows_batch_size,
                        start_padding_enabled=self.start_padding_enabled,
                        alias=self.alias,
            ),
            h=h,
            df=df,
            freq=inferred_freq,
            load_path=load_path,
        )
        return fcst_df

class TSPatchTST(Forecaster):
    def __init__(
        self,
        alias: str = "TSPatchTST",
        input_size: int = 16,
        encoder_layers: int = 3,
        n_heads: int = 16,
        hidden_size: int = 128,
        linear_hidden_size: int = 256,
        dropout: float = 0.2,
        fc_dropout: float = 0.2,
        head_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        patch_len: int = 16,
        stride: int = 8,
        res_attention: bool = True,
        batch_normalization: bool = False,
        max_steps: int = 500,
        learning_rate: float = 1e-4,
        val_check_steps: int = 10,
        batch_size: int = 32,
        windows_batch_size: int =1024,
        inference_windows_batch_size: int = 1024,
        start_padding_enabled=False,
    ):
        self.alias = alias
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.val_check_steps = val_check_steps
        self.batch_size = batch_size
        self.windows_batch_size = windows_batch_size
        self.inference_windows_batch_size = inference_windows_batch_size
        self.start_padding_enabled = start_padding_enabled
        self.encoder_layers = encoder_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.linear_hidden_size = linear_hidden_size
        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.head_dropout = head_dropout
        self.attn_dropout = attn_dropout
        self.patch_len = patch_len
        self.stride = stride
        self.res_attention = res_attention
        self.batch_normalization = batch_normalization

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        load_path: str | None = None,
    ) -> pd.DataFrame:

        if level is not None or quantiles is not None:
            raise ValueError("Level and quantiles are not supported for AutoTFT yet.")
        inferred_freq = self._maybe_infer_freq(df, freq)
        fcst_df = run_neuralforecast_model(
            model=PatchTST(
                        h=h,
                        input_size=self.input_size,
                        max_steps=self.max_steps,
                        learning_rate=self.learning_rate,
                        val_check_steps=self.val_check_steps,
                        batch_size=self.batch_size,
                        windows_batch_size=self.windows_batch_size,
                        inference_windows_batch_size=self.inference_windows_batch_size,
                        start_padding_enabled=self.start_padding_enabled,
                        alias=self.alias,
                        encoder_layers = self.encoder_layers,
                        n_heads = self.n_heads,
                        hidden_size = self.hidden_size,
                        linear_hidden_size = self.linear_hidden_size,
                        dropout = self.dropout,
                        fc_dropout = self.fc_dropout,
                        head_dropout = self.head_dropout,
                        attn_dropout = self.attn_dropout,
                        patch_len = self.patch_len,
                        stride = self.stride,
                        res_attention = self.res_attention,
                        batch_normalization = self.batch_normalization,
            ),
            h=h,
            df=df,
            freq=inferred_freq,
            load_path=load_path,
        )
        return fcst_df

class TSTimeXer(Forecaster):
    def __init__(
        self,
        alias: str = "TSTimeXer",
        input_size: int = 16,
        n_series: int = 1,
        n_heads: int = 8,
        hidden_size: int = 512,
        dropout: float = 0.2,
        patch_len: int = 16,
        max_steps: int = 500,
        learning_rate: float = 1e-4,
        val_check_steps: int = 10,
        batch_size: int = 32,
        windows_batch_size: int = 32,
        inference_windows_batch_size: int = 32,
        start_padding_enabled=False,
    ):
        self.alias = alias
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.val_check_steps = val_check_steps
        self.batch_size = batch_size
        self.windows_batch_size = windows_batch_size
        self.inference_windows_batch_size = inference_windows_batch_size
        self.start_padding_enabled = start_padding_enabled
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.patch_len = patch_len
        self.n_series = n_series

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        load_path: str | None = None,
    ) -> pd.DataFrame:

        if level is not None or quantiles is not None:
            raise ValueError("Level and quantiles are not supported for AutoTFT yet.")
        inferred_freq = self._maybe_infer_freq(df, freq)
        fcst_df = run_neuralforecast_model(
            model=TimeXer(
                        h=h,
                        input_size=self.input_size,
                        max_steps=self.max_steps,
                        learning_rate=self.learning_rate,
                        val_check_steps=self.val_check_steps,
                        batch_size=self.batch_size,
                        windows_batch_size=self.windows_batch_size,
                        inference_windows_batch_size=self.inference_windows_batch_size,
                        start_padding_enabled=self.start_padding_enabled,
                        alias=self.alias,
                        n_heads = self.n_heads,
                        hidden_size = self.hidden_size,
                        dropout = self.dropout,
                        patch_len = self.patch_len,
                        n_series=self.n_series,
            ),
            h=h,
            df=df,
            freq=inferred_freq,
            load_path=load_path,
        )
        return fcst_df


class TSFEDformer(Forecaster):
    def __init__(
        self,
        alias: str = "TSFEDformer",
        input_size: int = 16,
        hidden_size: int = 512,
        dropout: float = 0.2,
        max_steps: int = 500,
        learning_rate: float = 1e-4,
        val_check_steps: int = 10,
        batch_size: int = 32,
        windows_batch_size: int = 32,
        inference_windows_batch_size: int = 32,
        start_padding_enabled=False,
    ):
        self.alias = alias
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.val_check_steps = val_check_steps
        self.batch_size = batch_size
        self.windows_batch_size = windows_batch_size
        self.inference_windows_batch_size = inference_windows_batch_size
        self.start_padding_enabled = start_padding_enabled
        self.hidden_size = hidden_size
        self.dropout = dropout

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        load_path: str | None = None,
    ) -> pd.DataFrame:

        if level is not None or quantiles is not None:
            raise ValueError("Level and quantiles are not supported for AutoTFT yet.")
        inferred_freq = self._maybe_infer_freq(df, freq)
        fcst_df = run_neuralforecast_model(
            model=FEDformer(
                        h=h,
                        input_size=self.input_size,
                        max_steps=self.max_steps,
                        learning_rate=self.learning_rate,
                        val_check_steps=self.val_check_steps,
                        batch_size=self.batch_size,
                        windows_batch_size=self.windows_batch_size,
                        inference_windows_batch_size=self.inference_windows_batch_size,
                        start_padding_enabled=self.start_padding_enabled,
                        alias=self.alias,
                        hidden_size = self.hidden_size,
                        dropout = self.dropout,
            ),
            h=h,
            df=df,
            freq=inferred_freq,
            load_path=load_path,
        )
        return fcst_df

class TSiTransformer(Forecaster):

    def __init__(
        self,
        alias: str = "TSiTransformer",
        input_size: int = 16,
        n_series: int = 1,
        load_path: str | None = None,
    ):
        self.alias = alias
        self.input_size = input_size
        self.load_path = load_path
        self.n_series = n_series

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        load_path: str | None = None,
    ) -> pd.DataFrame:

        if level is not None or quantiles is not None:
            raise ValueError("Level and quantiles are not supported for AutoTFT yet.")
        inferred_freq = self._maybe_infer_freq(df, freq)
        fcst_df = run_neuralforecast_model(
            model=iTransformer(
                    h=h,
                    input_size=self.input_size,
                    n_series=self.n_series,
                    alias = self.alias,
            ),
            h=h,
            df=df,
            freq=inferred_freq,
            load_path=load_path,
        )
        return fcst_df

class TSTimeMixer(Forecaster):

    def __init__(
        self,
        alias: str = "TSTimeMixer",
        input_size: int = 16,
        n_series: int = 1,
        load_path: str | None = None,
    ):
        self.alias = alias
        self.input_size = input_size
        self.load_path = load_path
        self.n_series = n_series

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        load_path: str | None = None,
    ) -> pd.DataFrame:

        if level is not None or quantiles is not None:
            raise ValueError("Level and quantiles are not supported for AutoTFT yet.")
        inferred_freq = self._maybe_infer_freq(df, freq)
        fcst_df = run_neuralforecast_model(
            model=TimeMixer(
                    h=h,
                    input_size=self.input_size,
                    n_series=self.n_series,
                    alias = self.alias,
            ),
            h=h,
            df=df,
            freq=inferred_freq,
            load_path=load_path,
        )
        return fcst_df

class TSNLinear(Forecaster):

    def __init__(
        self,
        alias: str = "TSNLinear",
        input_size: int = 16,
        load_path: str | None = None,
    ):
        self.alias = alias
        self.input_size = input_size
        self.load_path = load_path

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        load_path: str | None = None,
    ) -> pd.DataFrame:

        if level is not None or quantiles is not None:
            raise ValueError("Level and quantiles are not supported for AutoTFT yet.")
        inferred_freq = self._maybe_infer_freq(df, freq)
        fcst_df = run_neuralforecast_model(
            model=NLinear(
                    h=h,
                    input_size=self.input_size,
                    alias=self.alias,
            ),
            h=h,
            df=df,
            freq=inferred_freq,
            load_path=load_path,
        )
        return fcst_df