import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.generation_utils import GenerationMixin
from src.model.attention import WordDefnAttention
from src.model.highway_network import HighwayNetwork
from src.utils.file_util import load_json_file


class MaskedConditionalGenerationModelHF(nn.Module, GenerationMixin):
    """
    This class defines the BART-UCD model as a masked conditional generation model.
    """
    def __init__(self, config):
        super(MaskedConditionalGenerationModelHF, self).__init__()
        self.config = config
        self.device = config.DEVICE
        # Initialize BART backbone
        self.bart = BartForConditionalGeneration.from_pretrained(config.PRETRAINED_CONDGEN_MODEL_NAME,
                                                                force_bos_token_to_be_generated=True)
        # Expand the vocab to the special POS tag vocabs
        pos_tags = load_json_file(config.PATH_TO_EXTRA_VOCAB)
        tokenizer = BartTokenizer.from_pretrained(config.PRETRAINED_CONDGEN_MODEL_NAME)
        num_added_toks = tokenizer.add_tokens(['pos_'+k for k in pos_tags.keys()])
        print(f'Added {num_added_toks} POS tokens for BART model!')
        self.bart.resize_token_embeddings(len(tokenizer))
        # Initialize other layers.
        self.word_defn_attn_layer = WordDefnAttention(config)
        self.highway_network = HighwayNetwork(config)
        self.embedding_fusion_layer = nn.Linear(config.PRETRAINED_CONDGEN_EMBED_DIM + config.PRETRAINED_SENT_EMBED_DIM,
                                                config.PRETRAINED_CONDGEN_EMBED_DIM)

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        return self.bart.resize_token_embeddings(new_num_tokens)

    def _resize_final_logits_bias(self, new_num_tokens: int, old_num_tokens: int) -> None:
        self.bart._resize_final_logits_bias(new_num_tokens, old_num_tokens)

    def encoder_forward(self, input_ids,
                        attention_mask=None,
                        word_defn_embed=None,
                        num_word_defns=None,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=False):
        # 1. BART encoder to generate contextualized embeddings
        encoder_outputs = self.bart.get_encoder()(input_ids=input_ids,
                                                  attention_mask=attention_mask,
                                                  output_attentions=output_attentions,
                                                  output_hidden_states=output_hidden_states,
                                                  return_dict=return_dict)
        # 2. Meaning selection with attention mechanism
        # 2.1 select masked representations
        word_masked_embed = encoder_outputs[0][input_ids == self.config.MASK_IDX]

        # 2.2 apply attention to get word meaning embed
        attn_w = self.word_defn_attn_layer(word_masked_embed, word_defn_embed, num_word_defns)
        # batch_size, 1, en_hidden_size
        word_defn_embed = attn_w.bmm(word_defn_embed).squeeze(1)

        # 3. Embedding fusion with highway network
        # 3.1 apply highway network to combine word meaning embed
        word_embed = self.highway_network(torch.cat((word_masked_embed, word_defn_embed), -1))
        word_embed = self.embedding_fusion_layer(word_embed)
        # 3.2 replace the original input hidden representation
        encoder_outputs[0][input_ids == self.config.MASK_IDX] = word_embed

        return encoder_outputs

    def forward(
        self,
        input_ids,
        attention_mask=None,
        word_defn_embed=None,
        num_word_defns=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=False,
    ):

        # 1.Encoding (with modified BART's encoder)
        if not encoder_outputs:
            encoder_outputs = self.encoder_forward(
                input_ids,
                attention_mask,
                word_defn_embed,
                num_word_defns,
                output_attentions,
                output_hidden_states,
                False)

        # 2. Decoding with BART's decoder
        decoder_outputs = self.bart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            use_cache=False,
            return_dict=True
        )

        return decoder_outputs

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        return self.bart.prepare_inputs_for_generation(decoder_input_ids, past,
                                                       attention_mask, use_cache,
                                                       encoder_outputs, **kwargs)

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        return self.bart.adjust_logits_during_generation(logits, cur_len, max_length)

    @staticmethod
    def _force_token_id_to_be_generated(scores, token_id) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
        scores[:, [x for x in range(scores.shape[1]) if x != token_id]] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = []
        for layer_past in past:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: _reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)
        return reordered_past

    def get_encoder(self):
        return self.encoder_forward

    def get_decoder(self):
        return self.bart.decoder

    def get_output_embeddings(self):
        return self.bart.get_output_embeddings()


def _reorder_buffer(attn_cache, new_order):
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None:
            attn_cache[k] = input_buffer_k.index_select(0, new_order)
    return attn_cache
