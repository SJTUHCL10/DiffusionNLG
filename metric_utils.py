import torch
import transformers

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from rouge import Rouge
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def bleu_metric(diffusion_model, eval_dataset, rev_tokenizer, batch_size=256):
    """
    generate sentences and metric with bleu score
    :param diffusion_model: diffusion model with conditional encoder trained from-scratch
    :param eval_dataset: pytorch dataset
    :param rev_tokenizer: token_id to token
    :param batch_size: batch size of evaluate dataloader
    :return: ([list] bleu score, [list] generated examples)
    """
    device = diffusion_model.model.device
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    generated_questions = []
    reference_questions = []

    for eval_batch in tqdm(eval_dataloader):
        x_T = torch.randn(size=(eval_batch['question1_input_ids'].shape[0],
                                diffusion_model.config.max_position_embeddings,
                                diffusion_model.config.word_embedding_dim))
        final_hidden_state = diffusion_model.sample(x_T.to(device),
                                                    src_ids=eval_batch['question1_input_ids'].to(device),
                                                    src_attention_mask=eval_batch['question1_attention_mask'].to(device),
                                                    return_hidden_states=False,
                                                    verbose=False)
        with torch.no_grad():
            sampled_ids = diffusion_model.rounding(final_hidden_state).cpu()

        generated_questions += [[rev_tokenizer[token_id.item()] for token_id in sampled_id] for sampled_id in sampled_ids]
        reference_questions += [[rev_tokenizer[id.item()] for id in ids] for ids in eval_batch['question2_input_ids']]

    bleu_score = []
    for ref, generate in zip(reference_questions, generated_questions):
        bleu_score.append(sentence_bleu([list(filter(lambda x:x not in ['[PAD]','[START]','[END]'], ref))],
                                        list(filter(lambda x:x not in ['[PAD]','[START]','[END]'], generate))))

    return bleu_score, generated_questions


def calculate_bleu(generated_questions, dataset, rev_tokenizer):
    """
    calculate BLEU metric
    :param generated_questions: list[token_list]
    :param dataset: pytorch dataset
    :param rev_tokenizer: token_id to token dict
    :return: {"bleu": val_list, "self_bleu": val_list}
    """
    bleu, self_bleu = [],[]
    for ind, sample in enumerate(dataset):
        src_question = [rev_tokenizer[id.item()] for id in sample['question1_input_ids']]
        src_question = list(filter(lambda x:x not in ['[PAD]','[START]','[END]'], src_question))
        tgt_question = [rev_tokenizer[id.item()] for id in sample['question2_input_ids']]
        tgt_question = list(filter(lambda x:x not in ['[PAD]','[START]','[END]'], tgt_question))
        bleu.append(sentence_bleu([tgt_question], generated_questions[ind]))
        self_bleu.append(sentence_bleu([src_question], generated_questions[ind]))

    return {"bleu": bleu, "self_bleu": self_bleu}


def calculate_rouge(generated_questions, dataset, rev_tokenizer):
    rouge = Rouge()
    rouge_scores = []
    for ind, sample in enumerate(dataset):
        tgt_question = [rev_tokenizer[id.item()] for id in sample['question2_input_ids']]
        tgt_question = list(filter(lambda x:x not in ['[PAD]','[START]','[END]'], tgt_question))
        rouge_scores += rouge.get_scores(" ".join(generated_questions[ind]), " ".join(tgt_question))

    return rouge_scores
