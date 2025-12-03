import pandas as pd

def import_results(file_path):
    csv = pd.read_csv(file_path, index_col=None, header=0, keep_default_na=False)
    return csv

def process_all_results(data_dir):
    import os

    all_results = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv') and file_name.startswith("scores_model_"):
            file_path = os.path.join(data_dir, file_name)
            df = import_results(file_path)
            all_results.append(df)

    combined_results = pd.concat(all_results, axis=0, ignore_index=True)
    check_NaN_values(combined_results)
    return combined_results

def filter_data(df, column, threshold):
    if type(threshold) in [str, float, int]:
        threshold = float(threshold)
        return df[df[column] > threshold]
    elif type(threshold) is list:
        return df[df[column].isin(threshold)]
    else:
        raise ValueError("Threshold must be a string, float, int, or list.")
    
def filter_data_like_paper(df):
    filtered_df = df[
        ((df['method'] == 'glasso') & (df['alpha'] == 1)) |
        ((df['method'] == 'rope') & (df['alpha'] == 0)) |
        ((df['method'] == 'gelnet') & (df['alpha'] == 0.5))
    ]
    return filtered_df
    
def recalculate_metrics(df, metric_column, overwrite=False):
    # we just cover certain metrics for now
    if metric_column == "MCC":
        tp = df['TP']
        tn = df['TN']
        fp = df['FP']
        fn = df['FN']
        mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).pow(0.5)
        # fill NaN values in mcc due to zero
        mcc = mcc.fillna(0)
        # count division by zero and set to NaN
        #mcc[(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0] = float('nan')
        # only overwrite if old value is NaN
        #df.loc[df[metric_column].isna(), metric_column] = mcc
        # just overwrite all values
        df.loc[:, metric_column] = mcc

    elif metric_column == "F1":
        tp = df['TP']
        fp = df['FP']
        fn = df['FN']
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        # fill NaN values in f1 due to zero
        f1 = f1.fillna(0)
        # only overwrite if old value is NaN
        #df.loc[df[metric_column].isna(), metric_column] = f1
        # just overwrite all values
        df.loc[:, metric_column] = f1
    else:
        raise ValueError(f"Metric {metric_column} not supported for recalculation.")

def check_NaN_values(df):
    nan_summary = df.isna().sum()
    return nan_summary

def postprocess_results(data_dir, metric_columns, threshold=None, overwrite=False):
    combined_results = process_all_results(data_dir)
    
    #if threshold is not None:
    #    combined_results = filter_data(combined_results, metric_column, threshold)
    #else:
    #    combined_results = filter_data_like_paper(combined_results)
    
    for metric_column in metric_columns:
        recalculate_metrics(combined_results, metric_column, overwrite)

    return combined_results

def save_results(df, output_path):
    # separate combined results into the original files
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

# main function to run the postprocessing pipeline
def main():
    print("Starting postprocessing of gelnet study results...")
    data_dirs = ["./data/extended_experiments/", "./data/paper_experiments/"]
    output_paths = ["./data/extended_experiments/extended_experiments.csv", "./data/paper_experiments/paper_experiments.csv"]
    metric_columns = ["F1", "MCC"]
    threshold = []
    overwrite = False
    for data_dir, output_path in zip(data_dirs, output_paths):
        processed_results = postprocess_results(data_dir, metric_columns)
        if check_NaN_values(processed_results).sum() > 0:
            print(f"Warning: There are still {check_NaN_values(processed_results).sum()} NaN values in the processed results.")
            print(f"NaN Summary:\n{check_NaN_values(processed_results)}")
            # print all rows with metric_column MCC is NaN
            print("rows with MCC NaN:")
            nan_rows = processed_results[processed_results["MCC"].isna()]
            # mcc rows which have NaN values in TP, TN, FP, FN
            print(nan_rows[nan_rows[["TP", "TN", "FP", "FN"]].isna().any(axis=1)])
        elif processed_results[processed_results["MCC"].isna()].shape[0] > 0:
            print(f"Warning: There are {processed_results[processed_results['MCC'].isna()].shape[0]} rows with NaN MCC values, but TP, TN, FP, FN are all present.")
            print("These rows are:")
            print(processed_results[processed_results["MCC"].isna()])
        else:
            print("No NaN values found in the processed results.")
            
        save_results(processed_results, output_path)

if __name__ == "__main__":
    main()