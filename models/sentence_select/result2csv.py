import pandas as pd
import os

def compare(tup):

    filename = tup[-1][:-4]
    period = filename.split("_")[-1]
    note = "_".join(filename.split("_")[1:-1])
    return (tup[1], tup[0], note, period, tup[-1])


segment = "window_5"
all_res = []
with open(f'{segment}_results.csv', 'w') as f:
    f.write("task,model,note,period,test_pr,test_roc\n")
    models = os.listdir("results")
    for model in models:
        tasks = os.listdir(os.path.join("results", model))
        for task in tasks:
            results = os.listdir(os.path.join("results", model, task, segment))
            for res in results:
                all_res.append((model, task, segment, res))

    all_res.sort(key=compare)
    for res in all_res:
        task, model, note, period, file_name = compare(res)
        try:
            if  file_name[:4] != "text":
                continue
            result = pd.read_csv(os.path.join("./", "results", model, task, segment, file_name))
            f.write(f"{task},{model},{note},{period},{result['PRAUC'].iloc[0]:.3f},{result['ROCAUC'].iloc[0]:.3f}\n")
        except:
            print(os.path.join("./", model, "results", task, note, res))
            continue
