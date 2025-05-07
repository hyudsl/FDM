import os
import json

from metric_functions.fid import create_dataset_stats, cal_fid, remove_dataset_stats

def main():
    dataset_root_path = './data/gt'
    gen_root_path = './generated'
    result_root_path = './metric_results'

    save_path = os.path.join(result_root_path, "fid.json")

    if os.path.isfile(save_path):
        with open(save_path, 'r') as f:
            output = json.load(f)
    else:
        output = {}

    dataset_list = os.listdir(dataset_root_path)

    for dataset in dataset_list:
        _gen_path = os.path.join(gen_root_path, dataset)
        exist = os.path.isdir(_gen_path)
        assert exist, "%s not in generate folder" % (dataset)

    for dataset in dataset_list:
        dataset_path = os.path.join(dataset_root_path, dataset)
        gen_path = os.path.join(gen_root_path, dataset)

        create_dataset_stats(dataset, dataset_path)

        if dataset in output.keys():
            dataset_result = output[dataset]
        else:
            dataset_result = {}

        model_list = os.listdir(gen_path)
        for model_name in model_list:
            if (dataset in output.keys()) and model_name in output[dataset].keys():
                print(dataset," ",model_name," already exist")
                continue
            gen_model_path = os.path.join(gen_path, model_name)

            model_result = {}

            sub_list = os.listdir(gen_model_path)
            for sub_folder in sub_list:
                gen_sub_path = os.path.join(gen_model_path, sub_folder, 'inpainted')
                if not os.path.isdir(gen_sub_path):
                    print(dataset," ",model_name,sub_folder," does not have inpainted folder")
                    continue
                fid = cal_fid(dataset, gen_sub_path)
                print(fid)
                model_result[sub_folder] = fid

            dataset_result[model_name] = model_result

        remove_dataset_stats(dataset)

        output[dataset] = dataset_result

    with open(save_path, "w") as json_file:
        json.dump(output, json_file, indent=4)

if __name__ == '__main__':
    main()