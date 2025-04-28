"""
Goal
---
1. Read test results from log.txt files
2. Compute mean and std across different folders (seeds)
3. Compute all datasets' accuracy and h-mean
4. Save the results to an Excel file
Usage
---
Assume the output files are saved under output/my_experiment,
which contains results of different seeds, e.g.,

my_experiment/
    seed1/
        log.txt
    seed2/
        log.txt
    seed3/
        log.txt

Run the following command from the root directory:

$ python tools/parse_test_res.py output/my_experiment

Add --ci95 to the argument if you wanna get 95% confidence
interval instead of standard deviation:

$ python tools/parse_test_res.py output/my_experiment --ci95

If my_experiment/ has the following structure,

my_experiment/
    exp-1/
        seed1/
            log.txt
            ...
        seed2/
            log.txt
            ...
        seed3/
            log.txt
            ...
    exp-2/
        ...
    exp-3/
        ...

Run

$ python tools/parse_test_res.py output/my_experiment --multi-exp
"""
import re
import numpy as np
import os.path as osp
import argparse
import pandas as pd
from collections import OrderedDict, defaultdict

from dassl.utils import check_isfile, listdir_nohidden


b2n_dataset = [
                "imagenet",
                "caltech101",
                "fgvc_aircraft",
                "oxford_flowers",
                "dtd",
                "eurosat",
                "food101",
                "oxford_pets",
                "stanford_cars",
                "sun397",
                "ucf101",
               ]
cross_dataset = [
                "caltech101",
                "fgvc_aircraft",
                "oxford_flowers",
                "dtd",
                "eurosat",
                "food101",
                "oxford_pets",
                "stanford_cars",
                "sun397",
                "ucf101",
               ]
dg_dataset = [  
                "imagenet",
                "imagenetv2",
                "imagenet_sketch",
                "imagenet_a",
                "imagenet_r",
             ]
def compute_ci95(res):
    return 1.96 * np.std(res) / np.sqrt(len(res))


def parse_function(*metrics, directory="", args=None, end_signal=None):
    print(f"Parsing files in {directory}")
    output_results = OrderedDict()
    output_results['accuracy'] = 0.0

    try:
        subdirs = listdir_nohidden(directory, sort=True)
    except:
        print("no folder")
        return output_results

    # subdirs = [directory]
    outputs = []

    for subdir in subdirs:
        fpath = osp.join(directory, subdir, "log.txt")
        # fpath = osp.join(directory, "log.txt")
        assert check_isfile(fpath)
        good_to_go = False
        output = OrderedDict()

        with open(fpath, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()

                if line == end_signal:
                    good_to_go = True
                
                
                for metric in metrics:
                    match = metric["regex"].search(line)
                    if match and good_to_go:
                        if "file" not in output:
                            output["file"] = fpath
                        num = float(match.group(1))
                        name = metric["name"]
                        output[name] = num

        if output:
            outputs.append(output)

    
    
    if len(outputs) <= 0:
        print("Nothing found in :")
        print(directory)
        return output_results 

    metrics_results = defaultdict(list)

    for output in outputs:
        msg = ""
        for key, value in output.items():
            if isinstance(value, float):
                msg += f"{key}: {value:.2f}%. "
            else:
                msg += f"{key}: {value}. "
            if key != "file":
                metrics_results[key].append(value)
        print(msg)

    

    print("===")
    print(f"Summary of directory: {directory}")
    for key, values in metrics_results.items():
        avg = np.mean(values)
        std = compute_ci95(values) if args.ci95 else np.std(values)
        print(f"* {key}: {avg:.2f}% +- {std:.2f}%")
        output_results[key] = avg
    print("===")

    return output_results


def main(args, end_signal):
    metric = {
        "name": args.keyword,
        "regex": re.compile(fr"\* {args.keyword}: ([\.\deE+-]+)%"),
    }

    if args.type == "base2new":
        all_dataset = b2n_dataset
        final_results = defaultdict(list)
        final_results1 = defaultdict(list)
        pattern = r'\b(' + '|'.join(map(re.escape, all_dataset)) + r')\b'
        # 替换匹配到的单词为 '{}'
        p=args.directory
        path_str = re.sub(pattern, "{}", p)
        all_dic = [path_str.format(dataset)for dataset in all_dataset]
        
        all_dic1 = []
        if "train_base" in all_dic[0]:
            for p in all_dic:
                
                all_dic1.append(p.replace("train_base", "test_new"))
        
        elif "test_new" in all_dic[0]:
            for p in all_dic:
                
                
                all_dic1.append(p.replace("test_new", "train_base"))

            temp = all_dic
            all_dic = all_dic1
            all_dic1= temp
            
        for i, directory in enumerate(all_dic):
            results = parse_function(
                metric, directory=directory, args=args, end_signal=end_signal
            )
            for key, value in results.items():
                final_results[key].append(value)
            
        for i, directory in enumerate(all_dic1):
            results1 = parse_function(
                metric, directory=directory, args=args, end_signal=end_signal
            )
            for key, value in results1.items():
                final_results1[key].append(value)
        
        
        output_data = []
        for i in range(len(all_dataset)):
            base = final_results['accuracy'][i]
            new  = final_results1['accuracy'][i]
            try:
                h = 2 / (1/base + 1/new)
            except: 
                h = 0
            result = {
                'Dataset': all_dataset[i],
                'Base Accuracy': base,
                'New Accuracy': new,
                'H-Mean': h
            }
            output_data.append(result)
            print(f"{all_dataset[i]:<20}: base: {base:>6.2f}  new: {new:>6.2f}  h: {h:>6.2f}")

        output_df = pd.DataFrame(output_data)

        # 将结果保存到 Excel
        output_file = "form_results_base2new.xlsx"
        output_df.to_excel(output_file, index=False)


        print("Average performance:")
        
        for key, values in final_results.items():
            avg_base = np.mean(values)
            print('base')
            print(f"* {key}: {avg_base:.2f}%")

        for key, values in final_results1.items():
            avg_new = np.mean(values)
            print('new')
            print(f"* {key}: {avg_new:.2f}%")
        
        try:
            avg_h = 2 / (1/avg_base + 1/avg_new)
        except:
            avg_h = 0
        print(f'h: {avg_h:.2f}%')
    else:
        if args.type == "fewshot":
            all_dataset = b2n_dataset
        elif args.type == "cross":
            all_dataset = cross_dataset
        elif args.type == "dg":
            all_dataset = dg_dataset

        final_results = defaultdict(list)
        pattern = r'\b(' + '|'.join(map(re.escape, all_dataset)) + r')\b'
        p=args.directory
        path_str = re.sub(pattern, "{}", p)
        all_dic = [path_str.format(dataset)for dataset in all_dataset]
    

        for i, directory in enumerate(all_dic):
            results = parse_function(
                metric, directory=directory, args=args, end_signal=end_signal
            )
            for key, value in results.items():
                final_results[key].append(value)
           
        output_data = []
        for i in range(len(all_dataset)):
            base = final_results['accuracy'][i]
             
            result = {
                'Dataset': all_dataset[i],
                'Accuracy': base,
            }
            output_data.append(result)
            print(f"{all_dataset[i]:<20}: Accuracy: {base:>6.2f}")

        output_df = pd.DataFrame(output_data)

        # 将结果保存到 Excel
        output_file = "form_results_"+args.type+".xlsx"
        output_df.to_excel(output_file, index=False)


        print("Average performance:")

        for key, values in final_results.items():
            avg_base = np.mean(values)
            print(f"* {key}: {avg_base:.2f}%")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="path to directory")
    parser.add_argument("-type", type=str, 
                    choices=['base2new', 'fewshot', 'cross', 'dg'],  # 添加参数校验
                    help="task type:base2new, fewshot, cross, dg")
    parser.add_argument(
        "--ci95", action="store_true", help=r"compute 95\% confidence interval"
    )
    parser.add_argument("--test-log", action="store_true", help="parse test-only logs")
    parser.add_argument(
        "--multi-exp", action="store_true", help="parse multiple experiments"
    )
    parser.add_argument(
        "--keyword", default="accuracy", type=str, help="which keyword to extract"
    )
    args = parser.parse_args()
   
    end_signal = "=> result"
    if args.test_log:
        end_signal = "=> result"

    main(args, end_signal)

