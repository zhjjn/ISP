from tqdm import tqdm
import torch
from src.model.masked_conditional_generation_model_huggingface import MaskedConditionalGenerationModelHF
from src.utils.data_util import TestDataHandler
from src.utils.model_util import load_init_model, count_parameters
from src.utils.eval_util import post_process
from src.utils.file_util import write_json_file
from config import Config
from transformers import BartTokenizer


def inference():
    data_handler = TestDataHandler()

    # Manage and initialize model
    # ---------------------------------------------------------------------------------
    # Initialize and load model
    model, _, _, _ = load_init_model(MaskedConditionalGenerationModelHF, data_handler.config)
    print(f"Total number of parameters: {count_parameters(model)}")  # 423,161,344
    tok = BartTokenizer.from_pretrained(Config.PRETRAINED_CONDGEN_MODEL_NAME)
    paraphrased_sentences = []
    bbar = tqdm(enumerate(data_handler.testset_generator), ncols=100, leave=False,
                total=data_handler.config.num_batch_test)
    # Run model inference
    for idx, data in bbar:
        torch.cuda.empty_cache()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=data['input_ids'],
                attention_mask=data['attention_mask'],
                word_defn_embed=data['word_defn_embeds'],
                num_word_defns=data['num_word_defns'],
            )
            y_true, y_pred = data['input_ids'], outputs
            y_true, y_pred = post_process(y_true, y_pred, data_handler.config)
            y_pred = tok.batch_decode(y_pred, skip_special_tokens=True)
            paraphrased_sentences.append(y_pred)

        bbar.set_description("Phase: [Inference] | Progress: {}/{} |".format(idx, data_handler.config.num_batch_test))
    # Save inference results.
    write_json_file(Config.PATH_TO_SAVE_INFERENCE_RESULTS, paraphrased_sentences)


if __name__ == '__main__':
    inference()
