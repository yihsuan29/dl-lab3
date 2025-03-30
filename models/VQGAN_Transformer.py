import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = 0.3 #configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq(codebook_mapping)
    @torch.no_grad()
    def encode_to_z(self, x):
        zq, codebook_indices, q_loss = self.vqgan.encode(x)
        return zq, codebook_indices
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return lambda ratio: 1 - ratio
        elif mode == "cosine":
            return lambda ratio: math.cos(math.pi/2 * ratio)
        elif mode == "square":
            return lambda ratio: 1 - ratio**2
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x):
        # get ground truth, and reshape as (batch_size, num_img_tokens)
        _, z_indices= self.encode_to_z(x) 
        z_indices = z_indices.view(-1, self.num_image_tokens)
        
        # set the mask ratio at most 50%
        mask_ratio = torch.rand(1)
        mask_ratio = (mask_ratio * 0.5).item()
        
        # randomize the mask and set those < mask_ratio as masked 
        mask = torch.rand_like(z_indices, dtype = float)
        mask = mask < mask_ratio
        
        # since there are 0~1023 entries in codebook => the masked one = 1024
        masked_z = torch.where(mask, 1024, z_indices)

        #transformer predict the probability of tokens
        logits = self.transformer(masked_z)
        return logits, z_indices
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, z_indices, mask, ratio, mask_num):
        masked_z = torch.where(mask, 1024, z_indices)        
        logits = self.transformer(masked_z)
        
        #Apply softmax to convert logits into a probability distribution across the last dimension.
        logits = torch.nn.functional.softmax(logits, dim = -1).squeeze(0)        

        #FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = torch.max(logits, dim = -1)

        #predicted probabilities add temperature annealing gumbel noise as confidence
        g = -torch.empty_like(z_indices_predict_prob).exponential_().log()  # gumbel noise
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g
        
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        #sort the confidence for the rank 
        #define how much the iteration remain predicted tokens by mask scheduling
        ##At the end of the decoding process, add back the original(non-masked) token values
        confidence_new = torch.where(mask, confidence, float('inf'))
        z_sort = torch.argsort(confidence_new, descending = False)
        n = int(ratio * mask_num)
        threshold = confidence_new[0, z_sort[0,n]]
        
        z_indices_predict = torch.where(mask, z_indices_predict, z_indices)        
        mask_bc=confidence_new < threshold
        return z_indices_predict.squeeze(0), mask_bc
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
