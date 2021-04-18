# First Align, then Predict: Understanding the Cross-Lingual Ability of Multilingual BERT 

This repository includes pointers and scripts to reproduce the experiments presented in our paper [First Align, then Predict: Understanding the Cross-Lingual Ability of Multilingual BERT](https://arxiv.org/abs/2101.11109) accepted at [EACL 2021](https://2021.eacl.org/)


## Setting up 

`cd ./first-align-then-predict`

`bash install.sh` 

`conda activate align-then-predict` 

## Computing Cross-Lingual Similarity  

We measure mBERT's hidden representation similarity between source and target languages with the [Central Kernel Alignment metric (CKA)](https://arxiv.org/abs/1905.00414) 

### Downloading parallel data

In our paper, we use the parrallel sentences provided by the PUD UD treebanks.  

The PUD treebanks are available here: 

`curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3424{/ud-treebanks-v2.7.tgz,/ud-documentation-v2.7.tgz,/ud-tools-v2.7.tgz}`

After uncompressing them, we can use the `./{lang_src}-ud-test.conllu` and `./{lang_trg}-ud-test.conllu` as parrallel sentences between `lang_src` and `lang_trg`. 


### Computing Cross-Lingual Similarity with the CKA metric 


```
python ./measure_similarity.py \
--data_dir ./data/ \                # location of parrallel data in the source and target languages
--source_lang_dataset en_pud \      # prefix of the dataset name of the source language
--target_lang_list fr_pud de_pud \  # list of prefix of the dataset name of the source language
--dataset_suffix ' -ud-test.conllu'\# suffix of the dataset names (should be the same for all source and target languages e.g. en_pud-ud-test.conllu )
--line_filter '# text =' \          # if we work with conllu files, we filter-in only the raw sentences starting with '# text =' 
--report_dir ./  \                  # directory where a json file will be stored with the similarity metric
--n_sent_total 100 \                # how many parrallel sentences picked from each file (it will sample the n_sent_total top sentences)
```

This script will printout and write in report_dir the CKA score between the source language and each target language for each layer hidden layer of mBERT. 


NB: 
- We assume that each dataset will follow the template: args.data_dir/{dataset_name}{dataset_suffix}
- To measure the similarity , the dataset between the source and the target languages should be aligned at the sentence level (for instance `en_pud-ud-test.conllu` and `de_pud-ud-test.conllu` are aligned). 


 


# How to cite 

If you extend or use this work, please cite:

```
@misc{muller2021align,
      title={First Align, then Predict: Understanding the Cross-Lingual Ability of Multilingual BERT}, 
      author={Benjamin Muller and Yanai Elazar and Benoît Sagot and Djamé Seddah},
      year={2021},
      eprint={2101.11109},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```