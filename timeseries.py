import sys
import os
import numpy as np
import pandas as pd
from loguru import logger


class Timeseries:
    def __init__(
        self,
        data_files: str,
        input_params: str,
        output_params: str,
        data_dir: str = ".",
        norm_type: str = "none",
        time_lag: int = 0,
        future_time: int = 0,
    ):
        self.data_dir = data_dir
        self.input_names = [x.strip() for x in input_params.split(" ")]
        self.output_names = [x.strip() for x in output_params.split(" ")]
        self.file_names = [x.strip() for x in data_files.split(" ")]
        self.load_data_from_files(self.input_names, self.output_names)
        input_padding = np.zeros((time_lag, len(self.input_names)), dtype=np.float32)
        norm_fun = self.normalization(norm_type)
        
        if future_time != 0:
            self.input_data = self.input_data[:-future_time]
        self.output_data = np.array(self.output_data)[future_time:]

        (
            self.train_input, 
            self.test_input, 
            self.train_output, 
            self.test_output) = self.split_data(
                                                self.input_data, 
                                                self.output_data
                            )

        if norm_type != "none":
            logger.info("Normalizing")
            self.norm_fun = self.normalization(norm_type)
            (
                self.train_input, 
                self.train_output, 
                self.test_input, 
                self.test_output
            ) = self.norm_fun(
                                self.train_input, 
                                self.train_output, 
                                self.test_input, 
                                self.test_output,
                                )

        self.train_input = np.append(input_padding, self.train_input, axis=0)
        self.test_input = np.append(input_padding, self.test_input, axis=0)

    def split_data(self, in_data, out_data, split_ratio: float = 0.8, norm_fun=None):
        split_point = int(len(in_data) * split_ratio)
        train_input = in_data[:split_point]
        test_input = in_data[split_point:]
        train_output = out_data[:split_point]
        test_output = out_data[split_point:] 
        return train_input, test_input, train_output, test_output
        
        
        return data[:split_point], data[split_point:]

    def normalization(self, norm_type: str = "minmax"):
        normalization_types = {
            "minmax": self.min_max_normalize,
            "mean_std": self.mean_normalize,
            "none": self.none_normalize,
        }
        logger.info(f"using {norm_type.upper()} Normalization")
        return normalization_types[norm_type]

    def load_data_from_files(self, input_names, output_names):
        logger.info(f"loading data from {self.file_names[0]}")
        self.input_data, self.output_data = self.load_data(
            "/".join([self.data_dir, self.file_names[0]]),
            input_names,
            output_names,
        )
        if len(self.file_names) > 1:
            for name in self.file_names[1:]:
                logger.info(f"loading data from {name}")
                input_data, output_data = self.load_data(
                    "/".join([self.data_dir, name]), self.input_names, self.output_names
                )
                self.input_data = pd.concat((self.input_data, input_data))
                self.output_data = pd.concat((self.output_data, output_data))
        # return self.input_data, self.output_data

    def load_data(self, file_name, in_params, out_params) -> None:
        if not os.path.exists(file_name):
            logger.error(f" File: {file_name} Does Not Exist")
            sys.exit()
        #all_param_names = list(set(self.input_names + self.output_names))
        input_param_names = list(set(self.input_names))
        output_param_names = list(set(self.output_names))
        data = pd.read_csv(file_name, sep=",", skipinitialspace=True)
        data[input_param_names] = data[input_param_names].astype(np.float32)
        output_onehot = pd.get_dummies(data[output_param_names])
        #data[output_param_names] = data[output_param_names].astype('category')
        return data[in_params], output_onehot

    def none_normalize(self, data):
        return data

    def mean_normalize(self, data):
        return (data - data.mean()) / data.std()

    def min_max_normalize(self, train_in, train_out, test_in, test_out):
        max_train = np.max(train_in, axis=0)
        min_train = np.min(train_in, axis=0)
        train_in = (train_in - min_train) / (max_train - min_train)
        test_in  = (test_in - min_train)  / (max_train - min_train)
        max_train = np.max(train_out, axis=0)
        min_train = np.min(train_out, axis=0)
        train_out = (train_out - min_train) / (max_train - min_train)
        test_out  = (test_out - min_train)  / (max_train - min_train)
        
        return train_in, train_out, test_in, test_out
      
