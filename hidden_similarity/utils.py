import torch
from collections import OrderedDict


def load_data(dir, data=None, line_filter=None, id_start=0, id_end=100,  verbose=1):
    if data is None:
        data = []
    if verbose:
        print(f"Loading {dir}")
    n_sent = 0
    n_appending = 0
    with open(dir, "r") as f:
        for line in f:
            if line_filter is not None:
                if not line.startswith(line_filter):
                    continue
                line = line[len(line_filter):]
            line = line.strip()
            if n_sent >= id_start and n_sent < id_end:
                n_appending += 1
                data.append(line)
            n_sent += 1
    try:
        assert n_appending == (id_end-id_start)
    except Exception as e:
        print(e)
        return None
    if verbose:
        print(f"Loaded {n_appending} form {dir}  ")

    return data




def get_batch_per_layer_head(attention, layer_head_att_batch=None, head=True):
    """

    :param attention:
    :param layer_head_att_batch:  if not None, will append to it as a list of tensor
    :return:
    """
    if layer_head_att_batch is None:
        layer_head_att_batch = OrderedDict()
    for i_layer in range(len(attention)):
        if head:
            for i_head in range(attention[0].size(1)):
                if f"layer_{i_layer}-head_{i_head}" not in layer_head_att_batch:
                    layer_head_att_batch[f"layer_{i_layer}-head_{i_head}"] = []

                layer_head_att_batch[f"layer_{i_layer}-head_{i_head}"].append(attention[i_layer][:, i_head].detach())
        else:
            if f"layer_{i_layer}" not in layer_head_att_batch:
                layer_head_att_batch[f"layer_{i_layer}"] = []
            layer_head_att_batch[f"layer_{i_layer}"].append(attention[i_layer][:].detach())
    return layer_head_att_batch


def get_hidden_representation(data, model, tokenizer,
                              pad="[PAD]", max_len=100, pad_below_max_len=False, output_dic=True):
    """
    get hidden representation (ie contetualized vector at the word level : add it as list or padded tensor : output[attention|layer|layer_head]["layer_x"] list or tensor)
    :param data: list of raw text
    :param pad: will add padding below max_len
    :return: output a dictionary (if output_dic) or a tensor (if not output_dic) : of contextualized representation at the word level per layer/layer_head
    """
    model.eval()
    special_start = tokenizer.cls_token
    special_end = tokenizer.sep_token
    layer_head_att_tensor_dic = OrderedDict()
    layer_hidden_state_tensor_dic = OrderedDict()
    layer_head_hidden_state_tensor_dic = OrderedDict()
    layer_head_att_batch_dic = OrderedDict()
    layer_head_hidden_state_dic = OrderedDict()
    layer_hidden_state_dic = OrderedDict()

    for seq in data:
        seq = special_start + " " + seq + " " + special_end
        tokenized = tokenizer.encode(seq)
        if len(tokenized) >= max_len:
            tokenized = tokenized[:max_len - 1]
            tokenized += tokenizer.encode(special_end)
        mask = [1 for _ in range(len(tokenized))]
        real_len = len(tokenized)
        if pad_below_max_len:
            if len(tokenized) < max_len:
                for _ in range(max_len - len(tokenized)):
                    tokenized += tokenizer.encode(pad)
                    mask.append(0)
            assert len(tokenized) == max_len
        assert len(tokenized) <= max_len + 2

        encoded = torch.tensor(tokenized).unsqueeze(0)
        inputs = OrderedDict([("wordpieces_inputs_words", encoded)])
        attention_mask = OrderedDict([("wordpieces_inputs_words", torch.tensor(mask).unsqueeze(0))])
        assert real_len
        if torch.cuda.is_available():
            inputs["wordpieces_inputs_words"] = inputs["wordpieces_inputs_words"].cuda()

            attention_mask["wordpieces_inputs_words"] = attention_mask["wordpieces_inputs_words"].cuda()

        model_output = model(input_ids=inputs['wordpieces_inputs_words'], attention_mask=attention_mask['wordpieces_inputs_words'])

        # getting the hidden states per layer
        hidden_state_per_layer = model_output[2]
        assert len(hidden_state_per_layer) == 12 + 1, "hidden state wrong length"
        assert hidden_state_per_layer[0].size()[-1] == 768, "hidden state wrong shape"

        layer_hidden_state_dic = get_batch_per_layer_head(hidden_state_per_layer, layer_hidden_state_dic, head=False)
    output = ()
    if output_dic:
        if len(layer_hidden_state_dic) > 0:
            output = output + (layer_hidden_state_dic,)
        if len(layer_head_att_batch_dic) > 0:
            output = output + (layer_head_att_batch_dic,)
        if len(layer_head_hidden_state_dic) > 0:
            output = output + (layer_head_hidden_state_dic,)
    else:
        assert pad_below_max_len
        if len(layer_hidden_state_dic) > 0:
            for key in layer_hidden_state_dic:
                layer_hidden_state_tensor_dic[key] = torch.cat(layer_hidden_state_dic[key], 0)
            output = output + (layer_hidden_state_tensor_dic,)
        if len(layer_head_att_batch_dic) > 0:
            for key in layer_head_att_batch_dic:
                layer_head_att_tensor_dic[key] = torch.cat(layer_head_att_batch_dic[key], 0)
            output = output + (layer_head_att_tensor_dic,)
        if len(layer_head_hidden_state_dic) > 0:
            for key in layer_head_hidden_state_dic:
                layer_head_hidden_state_tensor_dic[key] = torch.cat(layer_head_hidden_state_dic[key], 0)
            output = output + (layer_head_hidden_state_tensor_dic,)

    return output
