import sys
import os
import time
import argparse
import numpy as np
import torch
import transformers

from transformers import BertConfig
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models import BertForDiffusion, DiffusionLM, ConditionalDiffusionLM
from data_utils import load_qqp_dataset_and_tokenizer_from_disk, QQPParaphraseDataset, load_split_qqp_dataset_and_tokenizer_from_disk
from noise_schedule import get_named_beta_schedule
from train_utils import train_conditional, evaluate_conditional
from metric_utils import calculate_bleu, calculate_rouge


def main():
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument("--dataset", type=str, default='qqp', help='dataset name: qqp for paraphrase')
    parser.add_argument('--data_path', type=str, default='data', help='path of qqp dataset for paraphrase')

    # training args
    parser.add_argument('--device', type=str, default="cuda:0",)
    parser.add_argument('--batch_size', type=int, default=64,)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epoch', type=int, default=50)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_warmup_steps', type=int, default=100)
    parser.add_argument('--save_dir', type=str, help='save model.state_dict if provided save_dir')
    parser.add_argument('--save_name', type=str, help='file name like model.pth')
    parser.add_argument('--metric', dest='metric', action='store_true', help='calculate metric after training (default True)')
    parser.add_argument('--no_metric', dest='metric', action='store_false', help='do not calculate metric after training')
    parser.set_defaults(metric=True)

    # model size args
    parser.add_argument('--T', type=int, default=2000, help='diffusion time steps')
    parser.add_argument('--noise_schedule', type=str, default='cosine', help='noise schedule: cosine, linear, or sqrt')
    parser.add_argument('--word_embedding_dim', type=int, help='dimension of word embedding')
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_hidden_layers', type=int, default=4)
    parser.add_argument('--num_attention_heads', type=int, default=8)
    parser.add_argument('--intermediate_size', type=int, default=2048)
    parser.add_argument('--max_position_embeddings', type=int, default=32)

    # model type args
    parser.add_argument('--emb_type', type=str, default='learned', help='embedding type of diffusion model: learned, randn, or bit')
    parser.add_argument('--use_shared_weight', dest='use_shared_weight', action='store_true', help='share learned embedding weights with rounding (default True)')
    parser.add_argument('--use_unshared_weight', dest='use_shared_weight', action='store_false', help='do not share learned embedding weights with rounding')
    parser.set_defaults(use_shared_weight=True)
    parser.add_argument('--lm_head_bias', action='store_true', help='rounding with nn.Linear(bias=True) (default False)')
    parser.set_defaults(lm_head_bias=False)
    parser.add_argument('--add_emb_noise', action='store_true', help='add noise to p(x_0 | w) (default False)')
    parser.set_defaults(add_emb_noise=False)
    parser.add_argument('--use_attention_mask', action='store_true', help='add attention mask to the input of bert encoder (default False)')
    parser.set_defaults(use_attention_mask=False)
    parser.add_argument('--self_condition', dest='self_condition', action='store_true', help='use self-condition (default True)')
    parser.add_argument('--no_self_condition', dest='self_condition', action='store_false', help='do not use self-condition')
    parser.set_defaults(self_condition=True)
    parser.add_argument('--encoder_type', type=str, default='frozen', help='type of encoder: frozen, fine-tune, or from-scratch')
    parser.add_argument('--encoder_name_or_path', type=str, default='bert-base-uncased', help='pretrained encoder type')

    args = parser.parse_args()
    print(args)

    # load dataset
    if args.dataset == 'qqp':
        # tokenized_qqp_train, tokenized_qqp_eval, tokenizer = load_qqp_dataset_and_tokenizer_from_disk(data_path=args.data_path)
        # rev_tokenizer = {v: k for k, v in tokenizer.items()}
        train_dataset, eval_dataset, tokenizer = load_split_qqp_dataset_and_tokenizer_from_disk(data_path=args.data_path)
        print("Tokenizer vocab size:", len(tokenizer))

        # train_dataset = QQPParaphraseDataset(dataset=tokenized_qqp_train, random_swap=True)
        print("Training set size:", len(train_dataset))
        # eval_dataset = QQPParaphraseDataset(dataset=tokenized_qqp_eval, random_swap=False)
        print("Evaluation set size:", len(eval_dataset))

    else:
        raise NotImplementedError

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    config = BertConfig(vocab_size=len(tokenizer),
                        hidden_size=args.hidden_size,
                        num_hidden_layers=args.num_hidden_layers,
                        num_attention_heads=args.num_attention_heads,
                        intermediate_size=args.intermediate_size,
                        max_position_embeddings=args.max_position_embeddings,
                        pad_token_id=tokenizer['[PAD]'])

    config.T = args.T

    if args.word_embedding_dim is not None:
        config.word_embedding_dim = args.word_embedding_dim

    print(config)

    betas = torch.Tensor(get_named_beta_schedule(schedule_name=args.noise_schedule, num_diffusion_timesteps=config.T))

    device = args.device

    # build model
    diffusion_model = ConditionalDiffusionLM(config=config,
                                             betas=betas,
                                             emb_type=args.emb_type,
                                             self_condition=args.self_condition,
                                             use_attention_mask=args.use_attention_mask,
                                             add_emb_noise=args.add_emb_noise,
                                             use_shared_weight=args.use_shared_weight,
                                             lm_head_bias=args.lm_head_bias,
                                             conditional_gen=True,
                                             encoder_type=args.encoder_type,
                                             encoder_name_or_path=args.encoder_name_or_path,
                                             ).to(device)

    print("Diffusion model #parameters:")
    print(sum([p.numel() for p in diffusion_model.parameters()]))

    print("Diffusion model #trainable parameters")
    print(sum([p.numel() for p in filter(lambda p:p.requires_grad, diffusion_model.parameters())]))

    optimizer = torch.optim.AdamW(filter(lambda p:p.requires_grad, diffusion_model.parameters()),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=args.num_warmup_steps,
                                                num_training_steps=args.num_epoch*len(train_dataloader))

    # begin to train
    loss_terms_dict_lst = []
    progress_bar = tqdm(range(args.num_epoch*len(train_dataloader)))

    for epoch in range(args.num_epoch):
        print("epoch:", epoch+1)
        loss_terms_dict_lst.append(train_conditional(diffusion_model=diffusion_model,
                                         dataloader=train_dataloader,
                                         optimizer=optimizer,
                                         scheduler=scheduler,
                                         progress_bar=progress_bar,
                                         verbose=True,
                                         print_steps=len(train_dataloader)))
        evaluate_conditional(diffusion_model=diffusion_model, dataloader=eval_dataloader)

    print("Training completed.")

    if args.save_dir is not None:
        save_name = args.save_name if (args.save_name is not None) else time.strftime("%Y%m%d_%H%M")
        save_dir = os.path.join(args.save_dir, save_name)
        torch.save(diffusion_model.state_dict(), save_dir)
        print("Model saved to", save_dir)

    if args.metric:
        diffusion_model.eval()
        sampling_configs = [(500,1), (200,1), (200,5),
                            (20,1), (20,5), (20,10),
                            (2,1), (2,5), (2,10),
                            (1,1), (1,5), (1,10)]   # (ddim_steps, mbr)
        rev_tokenizer = {v: k for k, v in tokenizer.items()}

        for (sampling_timesteps, mbr) in sampling_configs:
            generated_questions = diffusion_model.generate(
                dataset = eval_dataset,
                rev_tokenizer=rev_tokenizer,
                sampling_timesteps=sampling_timesteps,
                eta=0,
                mbr=mbr,
                verbose=True,
            )
            
            print("DDIM",sampling_timesteps,"MBR",mbr)
            bleu_dict = calculate_bleu(generated_questions, eval_dataset, rev_tokenizer)
            print("BLEU score: ", sum(bleu_dict["bleu"])/len(bleu_dict["bleu"]))
            print("Self-BLEU: ", sum(bleu_dict["self_bleu"])/len(bleu_dict["self_bleu"]))
            rouge_scores = calculate_rouge(generated_questions, eval_dataset, rev_tokenizer)
            rouge_l_f = [d['rouge-l']['f'] for d in rouge_scores]
            print("Rouge-L(f) :", sum(rouge_l_f)/len(rouge_l_f))


if __name__ == '__main__':
    main()