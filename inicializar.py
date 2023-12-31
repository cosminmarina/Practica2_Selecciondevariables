import numpy as np
import pandas as pd
from operator import xor
import json

def inicializar(path:str, size:int=100):
    X1, X2, X3, X5 = (np.random.random(size=(4,size)) > 0.5).astype(np.int8)
    X4 = xor(X2, X3)
    Y = xor(X1, X4)
    df = pd.DataFrame(np.array([X1, X2, X3, X4, X5, Y]).T, columns=["X1", "X2", "X3", "X4", "X5", "Y"])
    df.to_csv(path, index=False)


if __name__ == "__main__":
    file_params_name = './params.json'
    try:
        file_params = open(file_params_name)
    except:
        raise OSError(f'File {file_params_name} not found. Your method need a configuration parameters file')
    else:
        params = json.load(file_params)
        file_params.close()
    path = params["data_file"]
    inicializar(path)

