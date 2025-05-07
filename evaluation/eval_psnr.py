import os
import json

from metric_functions.psnr import get_PSNR

def main():
    dataset_root_path = './data/gt'
    gen_root_path = './generated'
    result_root_path = './metric_results'

    save_path = os.path.join(result_root_path, "psnr.json")

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
                psnr = 0.0
                gen_sub_path = os.path.join(gen_model_path, sub_folder, 'inpainted')
                image_list = os.listdir(gen_sub_path)
                for img in image_list:
                    gt_path = os.path.join(gt_root_path, img)
                    img_path = os.path.join(gen_sub_path, img)
                    psnr += get_PSNR(gt_path, img_path, tool='skimage').item()
                model_result[sub_folder] = psnr/len(image_list)

            dataset_result[model_name] = model_result

        output[dataset] = dataset_result

    with open(save_path, "w") as json_file:
        json.dump(output, json_file, indent=4)

if __name__ == '__main__':
    main()