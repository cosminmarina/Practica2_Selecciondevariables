import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union
import json
from ITMO_FS.filters.univariate import spearman_corr
from ITMO_FS.filters.multivariate import FCBFDiscreteFilter
from ITMO_FS.filters.unsupervised.trace_ratio_laplacian import TraceRatioLaplacian
from ITMO_FS.wrappers import BackwardSelection
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, roc_auc_score, f1_score, recall_score, precision_recall_fscore_support, ConfusionMatrixDisplay

def load_data(params:dict):
    df = pd.read_csv(params["data_file"], header=0)
    col_names = np.array(df.columns[:-1])
    print(col_names)
    y = df.get('Y').to_numpy()
    X = df.drop(columns=['Y']).to_numpy()
    return X, y, col_names

def know_selected(array_all:np.ndarray, array_sel:np.ndarray):
    idx = []
    for i, interest in enumerate(array_sel.T):
        for j, elem in enumerate(array_all.T):
            if np.array_equal(interest, elem):
                idx.append(j)
    return idx

def get_stats(y_test: Union[np.ndarray, list], y_pred: Union[np.ndarray, list], file_save:str):
    """
    | get_stats                                         |
    |---------------------------------------------------|
    | Function that obtain the scores of the classifier.|
    |___________________________________________________|
    | ndarray, ndarray, str ->                          |
    |___________________________________________________|
    | Input:                                            |
    | y_test, y_pred: the real and predicted y values.  |
    | file_save: where to save the results.             |
    |___________________________________________________|
    | Output:                                           |
    | Nothing, we save all in files.                    |
    """
    # RECALL
    rec_gen = recall_score(y_test, y_pred, average='weighted')
    # PRECISION
    prec_gen = precision_score(y_test, y_pred, average='weighted')
    # F1
    f1_gen = f1_score(y_test, y_pred, average='weighted')
    # PER CLASS
    prf = precision_recall_fscore_support(y_test, y_pred, average=None)
    
    gen_stats = np.array([prec_gen, rec_gen, f1_gen])
    df = pd.DataFrame(np.concatenate((prf[:-1], np.array([gen_stats]).T), axis=1), index=['Precision', 'Recall', 'Fscore'])
    df.to_csv(f'./results/stats-{file_save}.csv')

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.savefig(f'./results/confusion_matrix-{file_save}.png')
    plt.close()
    return

def main(params):
    # Load data
    X, y, col_names = load_data(params)
    sp_corr_list = []
    sp_corr_sel_list = []
    fcbf_sel_list = []
    tracer_list = []
    tracer_sel_list = []
    recel_sel_list = []
    pca_var_list = []
    for i in range(params["iter"]):
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params["test_size"])
        X_train_selection, X_train_model, y_train_selection, y_train_model = train_test_split(X_train, y_train, test_size=0.5)
        # Base algorithm
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        get_stats(y_test, y_pred, f'base-knn-{i}')
        # Spearman corr
        ## Selection phase
        sp_corr = spearman_corr(X_train_selection, y_train_selection)
        sp_corr_list.append(sp_corr)
        idx = np.argsort(np.abs(sp_corr))[::-1][:params["n_col_filter"]]
        sp_corr_sel_list.append(col_names[idx])
        ## Model phase
        knn = KNeighborsClassifier()
        knn.fit(X_train_model[:,idx],y_train_model)
        y_pred_sp_corr = knn.predict(X_test[:,idx])
        get_stats(y_test, y_pred_sp_corr, f'sp-corr-{i}')
        # FCBFDiscreteFilter
        ## Selection phase
        fcbf = FCBFDiscreteFilter()
        fcbf.fit(X_train_selection, y_train_selection)
        X_train_model_selected = fcbf.transform(X_train_model)
        idx = know_selected(X_train_model, X_train_model_selected)
        fcbf_sel = ['-', '-', '-', '-', '-']
        fcbf_sel[:len(idx)] = col_names[idx] 
        fcbf_sel_list.append(fcbf_sel)
        ## Model phase
        knn = KNeighborsClassifier()
        knn.fit(X_train_model_selected,y_train_model)
        y_pred_fcbf = knn.predict(X_test[:,idx])
        get_stats(y_test, y_pred_fcbf, f'fcbf-{i}')
        # TraceRatio
        ## Selection phase
        tracer = TraceRatioLaplacian(params["n_col_filter"])
        idx = tracer.run(X_train_selection, y_train_selection)
        tracer_sel_list.append(col_names[idx[0]])
        tracer_list.append(idx[1])
        ## Model phase
        knn = KNeighborsClassifier()
        knn.fit(X_train_model[:,idx[0]],y_train_model)
        y_pred_tracer = knn.predict(X_test[:,idx[0]])
        get_stats(y_test, y_pred_tracer, f'tracer-{i}')
        # PCA
        ## Selection phase
        pca = PCA(n_components=params["n_col_filter"])
        pca.fit(X_train_selection)
        pca_var_list.append(pca.explained_variance_)
        X_train_model_selected = pca.transform(X_train_model)
        X_test_selected = pca.transform(X_test)
        ## Model phase
        knn = KNeighborsClassifier()
        knn.fit(X_train_model_selected, y_train_model)
        y_pred_pca = knn.predict(X_test_selected)
        get_stats(y_test, y_pred_pca, f'pca-{i}')
        # Wrapper BackwardSelection
        ## Selection phase
        knn = KNeighborsClassifier()
        recel = BackwardSelection(knn, len(col_names)-params["n_col_filter"], 'f1_weighted')
        recel.fit(X_train_selection, y_train_selection)
        idx = recel.selected_features
        recel_sel_list.append(col_names[idx])
        ## Model phase
        knn.fit(X_train_model[:,idx],y_train_model)
        y_pred_recel = knn.predict(X_test[:,idx])
        get_stats(y_test, y_pred_recel, f'recel-{i}')
    df_sp_corr_rnk = pd.DataFrame(sp_corr_list, columns=col_names)
    df_sp_corr_sel = pd.DataFrame(sp_corr_sel_list, columns=[f'{i}-best' for i in range(params["n_col_filter"])])
    df_fcbf_sel = pd.DataFrame(fcbf_sel_list, columns=[f'{i}-best' for i in range(len(col_names))])
    df_tracer_rnk = pd.DataFrame(tracer_list, columns=col_names)
    df_tracer_sel = pd.DataFrame(tracer_sel_list, columns=[f'{i}-best' for i in range(params["n_col_filter"])])
    df_recel_sel = pd.DataFrame(recel_sel_list, columns=[f'{i}-best' for i in range(params["n_col_filter"])])
    df_pca_var = pd.DataFrame(pca_var_list, columns=[f'{i}-var' for i in range(params["n_col_filter"])])
    df_sp_corr_rnk.to_csv('./results/ranking-sp-corr.csv')
    df_sp_corr_sel.to_csv('./results/selected-sp-corr.csv')
    df_fcbf_sel.to_csv('./results/selected-fcbf.csv')
    df_tracer_rnk.to_csv('./results/ranking-tracer.csv')
    df_tracer_sel.to_csv('./results/selected-tracer.csv')
    df_recel_sel.to_csv('./results/selected-recel.csv')
    df_pca_var.to_csv('./results/selected-pca.csv')
    return

if __name__  == "__main__":
    file_params_name = './params.json'
    try:
        file_params = open(file_params_name)
    except:
        raise OSError(f'File {file_params_name} not found. Your method need a configuration parameters file')
    else:
        params = json.load(file_params)
        file_params.close()
    main(params)