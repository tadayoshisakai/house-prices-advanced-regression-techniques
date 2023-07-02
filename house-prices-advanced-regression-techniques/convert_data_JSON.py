#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 12:29:42 2023

@author: sakaitadayoshi
"""

import json
from convert_data_IF import convert_data_IF

def main():
    print(convert_data_IF().targetVariable)
    print(convert_data_json().targetVariable)
    print(convert_data_json().trainMethod_list)
 
class convert_data_json(convert_data_IF):
    def __init__(self):
        super().__init__()
        self._load_json()
    def _load_json(self):
        with open('../config/convDataFrameConfig.json') as f:
            config = json.load(f)['conf_001']
        self.targetVariable = config["targetVariable"]
        self.num2str_list = config["num2str_list"]
        self.outlierException_list = config["outlierException_list"]
        self.trainMethod_list = config["trainMethod_list"]
        
    
if __name__=="__main__":
    main()