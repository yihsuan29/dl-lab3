import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.args = args
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim,self.scheduler = self.configure_optimizers()
        self.prepare_training()
        
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for imgs in tqdm(dataloader):
            x = imgs.to(self.args.device)
            self.optim.zero_grad()
            # get y_pred and gt y
            logits, z_indices = self.model(x)
            
            # reshape to (batch_size * num_tokens, token_size)
            logits = logits.view(-1, logits.size(-1))
            z_indices = F.one_hot(z_indices.view(-1), num_classes = logits.size(-1)).float()
            
            loss = F.cross_entropy(logits, z_indices)
            total_loss += loss.item()
            loss.backward()
            self.optim.step()
        
        total_loss /= len(dataloader)
        print(f"Training Loss: {total_loss}")
        return total_loss

    @torch.no_grad()
    def eval_one_epoch(self, dataloader):
        total_loss = 0
        for imgs in tqdm(dataloader):
            x = imgs.to(self.args.device)
            # get y_pred and gt y
            logits, z_indices = self.model(x)
            
            # reshape to (batch_size * num_tokens, token_size)
            logits = logits.view(-1, logits.size(-1))
            z_indices = F.one_hot(z_indices.view(-1), num_classes = logits.size(-1)).float()
            
            loss = F.cross_entropy(logits, z_indices)
            total_loss += loss.item()
        
        total_loss /= len(dataloader)
        print(f"Validation Loss: {total_loss}")
        return total_loss

    def configure_optimizers(self):
        """
        source: minGPT, link: https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
        goal: using "weight-decay" to prevent overfitting
        note: only Linear Layer need weight-decay
        """
        
        decay = set()
        no_decay = set()
        # seperate 2 class: weight-decay or not
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        """
        mn: module name, m: module
        pn: parameter name, p: parameter
        fpn: distinguish duplicate parameter names
        """
        for mn, m in self.model.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn 
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.model.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        # assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        # assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
        #                                            % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.args.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.args.learning_rate, betas=self.args.betas)
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps+1)/self.args.warmup_steps,1))

        return optimizer,scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./lab3_dataset/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./lab3_dataset/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=5, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=5e-4, help='Learning rate.')    
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.95), help='Betas for Adam.')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay for Adam.')
    parser.add_argument('--warmup-steps', type=int, default=50, help='Warmup steps.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
    writer = SummaryWriter()
    
#TODO2 step1-5:    
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        print(f"Epoch: {epoch}, lr: {train_transformer.scheduler.get_last_lr()[0]}")
        train_loss = train_transformer.train_one_epoch(train_loader)
        valid_loss = train_transformer.eval_one_epoch(val_loader)
        
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Valid", valid_loss, epoch)
        if epoch % args.save_per_epoch == 0:
            print(f"Saving epoch {epoch} ...")
            torch.save(train_transformer.model.transformer.state_dict(), f"transformer_checkpoints/epoch_{epoch}.pt")
        train_transformer.scheduler.step()
        