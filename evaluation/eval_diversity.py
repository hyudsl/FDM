import os
import json

from metric_functions.lpips import calculate_lpips_diversity

def main():
    dataset_root_path = './data/gt'
    gen_root_path = './generated'
    result_root_path = './metric_results'

    save_path = os.path.join(result_root_path, "diversity.json")

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
        if dataset == 'places2':
            continue
        gt_path = os.path.join(dataset_root_path, dataset)
        gen_path = os.path.join(gen_root_path, dataset)

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
                print(dataset, model_name, sub_folder)
                gen_sub_path = os.path.join(gen_model_path, sub_folder, 'inpainted')
                if not os.path.isdir(gen_sub_path):
                    print(dataset," ",model_name,sub_folder," does not have inpainted folder")
                    continue
                lpips = calculate_lpips_diversity(path1=gt_path,path2=gen_sub_path)
                print(lpips)
                model_result[sub_folder] = lpips

            if len(model_result.keys()) > 0:
                dataset_result[model_name] = model_result

        if len(dataset_result.keys()) > 0:
            output[dataset] = dataset_result

    with open(save_path, "w") as json_file:
        json.dump(output, json_file, indent=4)

if __name__ == '__main__':
    main()