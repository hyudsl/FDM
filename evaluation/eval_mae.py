import os
import json

from metric_functions.mae import get_mae

def main():
    dataset_root_path = './data/gt'
    gen_root_path = './generated'
    result_root_path = './metric_results'

    save_path = os.path.join(result_root_path, "mae.json")

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
        gen_path = os.path.join(gen_root_path, dataset)

        if dataset in output.keys():
            dataset_result = output[dataset]
        else:
            dataset_result = {}

        gt_root_path = os.path.join(dataset_root_path, dataset)

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
                mae = 0.0
                gen_sub_path = os.path.join(gen_model_path, sub_folder, 'inpainted')
                image_list = os.listdir(gen_sub_path)
                for img in image_list:
                    gt_path = os.path.join(gt_root_path, img)
                    img_path = os.path.join(gen_sub_path, img)
                    mae += get_mae(gt_path, img_path).item()
                model_result[sub_folder] = mae/len(image_list)
                print(mae/len(image_list))

            dataset_result[model_name] = model_result

        output[dataset] = dataset_result

    with open(save_path, "w") as json_file:
        json.dump(output, json_file, indent=4)

if __name__ == '__main__':
    main()