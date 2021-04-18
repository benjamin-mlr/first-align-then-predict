import argparse
import json
import numpy as np
import torch.nn as nn

from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer, AutoConfig

from hidden_similarity.utils import load_data, get_hidden_representation
from hidden_similarity.cka import kernel_CKA


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    #
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--source_lang_dataset", type=str, default="en_pud", required=False)
    parser.add_argument("--model", type=str, default="bert-base-multilingual-cased", required=False)
    parser.add_argument("--metric", type=str, default="cka", required=False)
    parser.add_argument("--line_filter", type=str, default=None, required=False)

    parser.add_argument("--report_dir", type=str, default=".", required=False)
    parser.add_argument("--dataset_suffix", type=str, default="-ud-test.conllu", required=False)
    parser.add_argument("--n_sent_total", type=int, default=800, required=False)
    parser.add_argument("--max_seq_len", type=int, default=200, required=False)
    parser.add_argument('--target_lang_list', default=["de_pud"], nargs='+')

    args = parser.parse_args()
    args.data_dir = Path(args.data_dir)
    args.dataset_suffix = args.dataset_suffix.strip()

    # define function. given 2 parallel corpora --> get hidden representations with mBert
    # and output similarity per layers
    config = AutoConfig.from_pretrained(args.model, output_hidden_states=True)
    model = AutoModel.from_pretrained(args.model, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    dataset = [args.source_lang_dataset ]+ args.target_lang_list
    verbose = 0

    # Load data
    data_target_ls = [load_data(args.data_dir/f"{target}{args.dataset_suffix}", id_end=args.n_sent_total, line_filter=args.line_filter, verbose=verbose) for target in dataset]
    data_target_dic = OrderedDict([(lang, data) for lang, data in zip(dataset, data_target_ls)])

    src = f"{args.source_lang_dataset}"

    data_src = data_target_dic[src]

    for _data_target in data_target_dic:
        assert len(data_target_dic[_data_target]) == len(data_src), f"Should have as much sentences on both sides en:{len(data_src)} target:{len(data_target_dic[_data_target])}"
        # just to get the key
    metric = args.metric

    if metric == "cos":
        batch_size = 1
    elif metric == "cka":
        # cka is computed on batch of samples , we split the dataset in 4 to get the batch size
        batch_size = len(data_src) // 4
    else:
        raise(Exception(f"{metric} not supported"))

    studied_ind = 0
    src_lang = src
    src_lang_ls = [src]

    cosine_sent_to_src = OrderedDict([(src_lang+"-"+lang, OrderedDict()) for src_lang in src_lang_ls for lang in data_target_dic.keys()])

    output_dic = True
    pad_below_max_len = False
    max_len = args.max_seq_len

    n_batch = len(data_src) // batch_size
    cosinus = nn.CosineSimilarity(dim=1)

    for i_data in tqdm(range(n_batch), desc="Processing batch"):

        data_src = load_data(args.data_dir/f"{src}{args.dataset_suffix}", line_filter=args.line_filter, id_end=args.n_sent_total,
                             verbose=verbose)
        batch_src = data_src[i_data:i_data+batch_size]
        # get hidden states for source language
        all = get_hidden_representation(batch_src, model, tokenizer, pad_below_max_len=pad_below_max_len, max_len=max_len, output_dic=output_dic)
        analysed_batch_dic_en = all[studied_ind]
        i_lang = 0

        for lang, target in data_target_dic.items():

            i_lang += 1
            target_batch = target[i_data:i_data + batch_size]

            all = get_hidden_representation(target_batch, model, tokenizer, pad_below_max_len=pad_below_max_len,
                                            max_len=max_len, output_dic=output_dic)

            former_layer, former_mean_target, former_mean_src = None, None, None
            analysed_batch_dic_target = all[0]
            for layer in analysed_batch_dic_target:
                # get average for sentence removing first and last special tokens
                if output_dic:
                    mean_over_sent_src = []
                    mean_over_sent_target = []
                    mean_over_sent_target_origin = []
                    mean_over_sent_src_origin = []
                    for i_sent in range(len(analysed_batch_dic_en[layer])):
                        # removing special characters first and last and
                        mean_over_sent_src.append(
                            np.array(analysed_batch_dic_en[layer][i_sent][0, 1:-1, :].mean(dim=0).cpu()))
                        mean_over_sent_target.append(
                            np.array(analysed_batch_dic_target[layer][i_sent][0, 1:-1, :].mean(dim=0).cpu()))

                    mean_over_sent_src = np.array(mean_over_sent_src)
                    mean_over_sent_target = np.array(mean_over_sent_target)

                else:
                    mean_over_sent_src = analysed_batch_dic_en[layer][:, 1:-1, :].mean(dim=1).cpu()
                    mean_over_sent_target = analysed_batch_dic_target[layer][:, 1:-1, :].mean(dim=1).cpu()

                if layer not in cosine_sent_to_src[src_lang + "-" + lang]:
                    cosine_sent_to_src[src_lang + "-" + lang][layer] = []

                if metric == "cka":
                    mean_over_sent_src = np.array(mean_over_sent_src)
                    mean_over_sent_target = np.array(mean_over_sent_target)

                    cosine_sent_to_src[src_lang + "-" + lang][layer].append(kernel_CKA(mean_over_sent_src, mean_over_sent_target))
                    if verbose:
                        print(f"Measured {metric} {layer} {kernel_CKA(mean_over_sent_src, mean_over_sent_target)} ")
                elif metric == "cos":
                    cosine_sent_to_src[src_lang + "-" + lang][layer].append(cosinus(mean_over_sent_src, mean_over_sent_target).item())

                former_layer = layer
                former_mean_target = mean_over_sent_target
                former_mean_src = mean_over_sent_src

    report = {}
    for src_trg, cosine_ls in cosine_sent_to_src.items():
        if src_trg != src_lang+'-'+src_lang:
            print(f"{metric} between {src_trg.split('-')[0]} and {src_trg.split('-')[1]}")
            if src_trg not in report:
                report[src_trg] = {}
            for layer in cosine_ls:
                if layer not in report[src_trg]:
                    report[src_trg][layer] = {}
                report[src_trg][layer][f"{metric}_mean"] = np.mean(cosine_sent_to_src[src_trg][layer])
                report[src_trg][layer][f"{metric}_std"] = np.std(cosine_sent_to_src[src_trg][layer])
                report[src_trg][layer][f"{metric}_batch_size"] = batch_size
                report[src_trg][layer][f"{metric}_n_batch"] = len(cosine_sent_to_src[src_trg][layer])
                print(f"Mean {metric} between {src_trg} for {layer} is {np.mean(cosine_sent_to_src[src_trg][layer]):0.3f} std:{np.std(cosine_sent_to_src[src_trg][layer]):0.3f} measured on {len(cosine_sent_to_src[src_trg][layer])} samples of size {batch_size}" )

    report_file = Path(args.report_dir)/f"{metric}-similarity-{src_lang}-trg.json"
    json.dump(report, open(report_file, "w"), indent=4, sort_keys=True)
    print(f"Similarity saved in {report_file}")

