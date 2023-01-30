import sys
import os
import argparse
import numpy as np
import torch
import transformers

from transformers import BertConfig
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models import BertForDiffusion, DiffusionLM
from data_utils import load_e2enlg_dataset_and_tokenizer, E2enlgDataset, \
    load_rocstories_dataset_and_tokenizer, RocstoriesDataset
from noise_schedule import get_named_beta_schedule
from train_utils import train, evaluate


def main():
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument("--dataset", type=str, default='rocstories', help='dataset name: e2enlg or rocstories')
    parser.add_argument('--max_len', type=int, default=64, help='maximum sequence length')
    parser.add_argument('--vocab_threshold', type=int, default=10,
                        help='token occurrence time < threshold will be treated as [UNK]')
    parser.add_argument('--test_size', type=float, default=0.1, help='size of evaluation dataset')

    # training args
    parser.add_argument('--device', type=str, default="cuda:0",)
    parser.add_argument('--batch_size', type=int, default=64,)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epoch', type=int, default=50)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_warmup_steps', type=int, default=100)

    # model size args
    parser.add_argument('--T', type=int, default=2000, help='diffusion time steps')
    parser.add_argument('--noise_schedule', type=str, default='cosine', help='noise schedule: cosine, linear, or sqrt')
    parser.add_argument('--word_embedding_dim', type=int, help='dimension of word embedding')
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_hidden_layers', type=int, default=4)
    parser.add_argument('--num_attention_heads', type=int, default=8)
    parser.add_argument('--intermediate_size', type=int, default=2048)

    # model type args
    parser.add_argument('--emb_type', type=str, default='learned', help='embedding type of diffusion model: learned, randn, or bit')
    parser.add_argument('--use_shared_weight', type=bool, default=True, help='share learned embedding weights when rounding')
    parser.add_argument('--lm_head_bias', type=bool, default=False, help='rounding with nn.Linear(bias=True)')
    parser.add_argument('--add_emb_noise', type=bool, default=False, help='add noise to p(x_0 | w)')
    parser.add_argument('--use_attention_mask', type=bool, default=False, help='add attention mask to the input of bert encoder')
    parser.add_argument('--self_condition', type=bool, default=True, help='use self-condition')

    args = parser.parse_args()

    # load dataset
    if args.dataset == 'rocstories':
        tokenized_rocstories_dataset, tokenizer = load_rocstories_dataset_and_tokenizer(max_len=args.max_len, vocab_threshold=args.vocab_threshold)

        rev_tokenizer = {v: k for k, v in tokenizer.items()}

        train_set, eval_set = train_test_split(tokenized_rocstories_dataset, test_size=args.test_size, shuffle=True)

        train_dataset = RocstoriesDataset(data_lst=train_set['input_ids'], attention_mask_lst=train_set['attention_mask'])
        print("Training set size:",len(train_dataset))
        eval_dataset = RocstoriesDataset(data_lst=eval_set['input_ids'], attention_mask_lst=eval_set['attention_mask'])
        print("Evaluation set size:", len(eval_dataset))

    elif args.dataset == 'e2enlg':
        tokenized_e2enlg_dataset, tokenizer = load_e2enlg_dataset_and_tokenizer(max_len=args.max_len, vocab_threshold=args.vocab_threshold)

        rev_tokenizer = {v: k for k, v in tokenizer.items()}

        train_set, eval_set = train_test_split(tokenized_e2enlg_dataset, test_size=args.test_size, shuffle=True)

        train_dataset = E2enlgDataset(data_lst=train_set['input_ids'], attention_mask_lst=train_set['attention_mask'])
        print("Training set size:", len(train_dataset))
        eval_dataset = E2enlgDataset(data_lst=eval_set['input_ids'], attention_mask_lst=eval_set['attention_mask'])
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
                        max_position_embeddings=args.max_len,
                        pad_token_id=tokenizer['[PAD]'])

    config.T = args.T

    if hasattr(args, 'word_embedding_dim'):
        config.word_embedding_dim = args.word_embedding_dim

    print(config)

    betas = torch.Tensor(get_named_beta_schedule(schedule_name=args.noise_schedule, num_diffusion_timesteps=config.T))

    device = args.device

    # build model
    diffusion_model = DiffusionLM(config=config,
                                  betas=betas,
                                  emb_type=args.emb_type,
                                  self_condition=args.self_condition,
                                  use_attention_mask=args.use_attention_mask,
                                  add_emb_noise=args.add_emb_noise,
                                  use_shared_weight=args.use_shared_weight,
                                  lm_head_bias=args.lm_head_bias,
                                  ).to(device)

    print("Diffusion model number of parameters:")
    print(sum([p.numel() for p in diffusion_model.parameters()]))

    optimizer = torch.optim.AdamW(diffusion_model.parameters(),
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
        loss_terms_dict_lst.append(train(diffusion_model=diffusion_model,
                                         dataloader=train_dataloader,
                                         optimizer=optimizer,
                                         scheduler=scheduler,
                                         progress_bar=progress_bar,
                                         verbose=True))
        evaluate(diffusion_model=diffusion_model, dataloader=eval_dataloader)

    print("Training completed. Begin to sampling...")

    x_T = torch.randn(size=(args.batch_size, args.max_len, diffusion_model.config.word_embedding_dim))
    if args.use_attention_mask:
        print("sampling without attention mask")
    x_0, hidden_states = diffusion_model.sample(x_T.to(device), return_hidden_states=True, verbose=False)
    

if __name__ == '__main__':
    main()