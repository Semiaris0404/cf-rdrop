"""Offline smoke-test helpers: a tiny random RoBERTa and a word-level tokenizer.
Only used with `train.py --tiny`; never for reported results."""
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaForSequenceClassification


def tiny_model_and_tok(vocab_size: int = 5000):
    import pickle, string
    words = ["<s>", "<pad>", "</s>", "<unk>"] + list(string.ascii_lowercase)
    try:
        with open("data/cache.pkl", "rb") as f:
            d = pickle.load(f)
        from collections import Counter
        c = Counter(w for o, _, cf, _, _ in d["cad_train_pairs"] for w in (o + " " + cf).lower().split())
        words += [w for w, _ in c.most_common(vocab_size - len(words))]
    except FileNotFoundError:
        pass
    vocab = {w: i for i, w in enumerate(dict.fromkeys(words))}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>",
                                  pad_token="<pad>", unk_token="<unk>")
    cfg = RobertaConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
                        num_attention_heads=2, intermediate_size=64, max_position_embeddings=600,
                        pad_token_id=1, num_labels=2)
    return RobertaForSequenceClassification(cfg), tok
