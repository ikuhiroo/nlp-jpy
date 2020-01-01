import torch
from collections import defaultdict


def get_attention(model, tokenizer, text, include_queries_and_keys=False):
    """Compute representation of the attention to pass to the d3 visualization

    Args:
      model: pytorch_transformers model
      tokenizer: pytorch_transformers tokenizer
      text: Input text
      include_queries_and_keys: Indicates whether to include queries/keys in results

    Returns:
      Dictionary of attn representations with the structure:
      {
        'left_text': list of source tokens, to be displayed on the left of the vis
        'right_text': list of target tokens, to be displayed on the right of the vis
        'attn': list of attention matrices, one for each layer. Each has shape (num_heads, source_seq_len, target_seq_len)
        'queries' (optional): list of query vector arrays, one for each layer. Each has shape (num_heads, source_seq_len, vector_size)
        'keys' (optional): list of key vector arrays, one for each layer. Each has shape (num_heads, target_seq_len, vector_size)
      }
    """
    print("A")
    # Prepare inputs to model
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.decode([t]).strip() for t in token_ids]
    tokens_tensor = torch.tensor(token_ids).unsqueeze(0)

    # Call model to get attention data
    model.eval()
    _, _, attn_data_list = model(tokens_tensor)

    # Format attention data for visualization
    all_attns = []
    all_queries = []
    all_keys = []
    for layer, attn_data in enumerate(attn_data_list):
        # assume batch_size=1; output shape = (num_heads, seq_len, seq_len)
        attn = attn_data['attn'][0]
        all_attns.append(attn.tolist())
        if include_queries_and_keys:
            # assume batch_size=1; output shape = (num_heads, seq_len, vector_size)
            queries = attn_data['queries'][0]
            all_queries.append(queries.tolist())
            # assume batch_size=1; output shape = (num_heads, seq_len, vector_size)
            keys = attn_data['keys'][0]
            all_keys.append(keys.tolist())
    results = {
        'attn': all_attns,
        'left_text': tokens,
        'right_text': tokens
    }
    if include_queries_and_keys:
        results.update({
            'queries': all_queries,
            'keys': all_keys,
        })
    return {'all': results}


def get_attention_bert(model, tokenizer, sentence_a, sentence_b, masked, mask_id, bert_version, include_queries_and_keys=False):
    """Compute representation of the attention for BERT to pass to the d3 visualization

    Args:
      model: BERT model
      tokenizer: BERT tokenizer
      sentence_a: Sentence A string
      sentence_b: Sentence B string
      include_queries_and_keys: Indicates whether to include queries/keys in results

    Returns:
      Dictionary of attn representations with the structure:
      {
        'all': All attention (source = AB, target = AB)
        'aa': Sentence A self-attention (source = A, target = A)
        'bb': Sentence B self-attention (source = B, target = B)
        'ab': Sentence A -> Sentence B attention (source = A, target = B)
        'ba': Sentence B -> Sentence A attention (source = B, target = A)
      }
      where each value is a dictionary:
      {
        'left_text': list of source tokens, to be displayed on the left of the vis
        'right_text': list of target tokens, to be displayed on the right of the vis
        'attn': list of attention matrices, one for each layer. Each has shape [num_heads, source_seq_len, target_seq_len]
        'queries' (optional): list of query vector arrays, one for each layer. Each has shape (num_heads, source_seq_len, vector_size)
        'keys' (optional): list of key vector arrays, one for each layer. Each has shape (num_heads, target_seq_len, vector_size)
      }
    """
    
    """Juman++Tokenizer"""
    from pyknp import Juman
    class JumanTokenizer():
        def __init__(self):
            self.juman = Juman()

        def tokenize(self, text):
            result = self.juman.analysis(text)
            return [mrph.midasi for mrph in result.mrph_list()]
    
    """MecabTokenizer"""
    import MeCab
    class MecabTokenizer():
        def __init__(self):
            self.mecab = MeCab.Tagger(
                "-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
            self.mecab.parse("")

        def tokenize(self, text):
            node = self.mecab.parseToNode(text)
            result = []
            while node:
                result.append(node.surface)
                node = node.next
            return result
        
    if bert_version == 'bert-mecab':
        Mecab_Tokenizer = MecabTokenizer()
        tokens_a = Mecab_Tokenizer.tokenize(sentence_a)
        tokens_b = Mecab_Tokenizer.tokenize(sentence_b)
        tokens_a = tokenizer.tokenize(" ".join(tokens_a))
        tokens_b = tokenizer.tokenize(" ".join(tokens_b))
        tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        tokens_b = tokens_b + ['[SEP]']
        
    elif bert_version == 'bert-juman':
        juman_tokenizer = JumanTokenizer()
        tokens_a = juman_tokenizer.tokenize(sentence_a)
        tokens_b = juman_tokenizer.tokenize(sentence_b)
        tokens_a = tokenizer.tokenize(" ".join(tokens_a))
        tokens_b = tokenizer.tokenize(" ".join(tokens_b))
        tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        tokens_b = tokens_b + ['[SEP]']
    else:
        """InitTokenizer"""
        tokens_a = ['[CLS]'] + tokenizer.tokenize(sentence_a)  + ['[SEP]']
        tokens_b = tokenizer.tokenize(sentence_b) + ['[SEP]']
        
    if masked:
        for i in mask_id:
            if i < len(tokens_a):
                tokens_a[i] = '[MASK]'
            else:
                tokens_b[i - len(tokens_a)] = '[MASK]'
    token_ids = tokenizer.convert_tokens_to_ids(tokens_a + tokens_b)
    tokens_tensor = torch.tensor([token_ids])
    token_type_tensor = torch.LongTensor(
        [[0] * len(tokens_a) + [1] * len(tokens_b)])

    # Call model to get attention data
    model.eval()
    _, _, attn_data_list = model(
        tokens_tensor, token_type_ids=token_type_tensor)

    # Populate map with attn data and, optionally, query, key data
    keys_dict = defaultdict(list)
    queries_dict = defaultdict(list)
    attn_dict = defaultdict(list)
    # Positions corresponding to sentence A in input
    slice_a = slice(0, len(tokens_a))
    # Position corresponding to sentence B in input
    slice_b = slice(len(tokens_a), len(tokens_a) + len(tokens_b))
    for layer, attn_data in enumerate(attn_data_list):
        # Process attention
        # assume batch_size=1; shape = [num_heads, source_seq_len, target_seq_len]
        attn = attn_data['attn'][0]
        attn_dict['all'].append(attn.tolist())
        # Append A->A attention for layer, across all heads
        attn_dict['aa'].append(attn[:, slice_a, slice_a].tolist())
        # Append B->B attention for layer, across all heads
        attn_dict['bb'].append(attn[:, slice_b, slice_b].tolist())
        # Append A->B attention for layer, across all heads
        attn_dict['ab'].append(attn[:, slice_a, slice_b].tolist())
        # Append B->A attention for layer, across all heads
        attn_dict['ba'].append(attn[:, slice_b, slice_a].tolist())
        # Process queries and keys
        if include_queries_and_keys:
            # assume batch_size=1; shape = [num_heads, seq_len, vector_size]
            queries = attn_data['queries'][0]
            # assume batch_size=1; shape = [num_heads, seq_len, vector_size]
            keys = attn_data['keys'][0]
            queries_dict['all'].append(queries.tolist())
            keys_dict['all'].append(keys.tolist())
            queries_dict['a'].append(queries[:, slice_a, :].tolist())
            keys_dict['a'].append(keys[:, slice_a, :].tolist())
            queries_dict['b'].append(queries[:, slice_b, :].tolist())
            keys_dict['b'].append(keys[:, slice_b, :].tolist())

    results = {
        'all': {
            'attn': attn_dict['all'],
            'left_text': tokens_a + tokens_b,
            'right_text': tokens_a + tokens_b
        },
        'aa': {
            'attn': attn_dict['aa'],
            'left_text': tokens_a,
            'right_text': tokens_a
        },
        'bb': {
            'attn': attn_dict['bb'],
            'left_text': tokens_b,
            'right_text': tokens_b
        },
        'ab': {
            'attn': attn_dict['ab'],
            'left_text': tokens_a,
            'right_text': tokens_b
        },
        'ba': {
            'attn': attn_dict['ba'],
            'left_text': tokens_b,
            'right_text': tokens_a
        }
    }
    if include_queries_and_keys:
        results['all'].update({
            'queries': queries_dict['all'],
            'keys': keys_dict['all'],
        })
        results['aa'].update({
            'queries': queries_dict['a'],
            'keys': keys_dict['a'],
        })
        results['bb'].update({
            'queries': queries_dict['b'],
            'keys': keys_dict['b'],
        })
        results['ab'].update({
            'queries': queries_dict['a'],
            'keys': keys_dict['b'],
        })
        results['ba'].update({
            'queries': queries_dict['b'],
            'keys': keys_dict['a'],
        })
    return results
