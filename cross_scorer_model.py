import transformers
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
import torch

from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
# To use as the model for DialogMLM
from transformers import BertForMaskedLM

import torch.nn.functional as F

import spacy
import transformers
import torch.nn as nn


class CrossScorerCrossEncoder(nn.Module):

    def __init__(self, transformer): #, tokenizer):
        """

        """
        super(CrossScorerCrossEncoder, self).__init__()

        self.cross_encoder = transformer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

        # Binary Head
        self.l1 = torch.nn.Linear(768, 512)
        self.relu = torch.nn.ELU()
        self.l2 = torch.nn.Linear(512,1)

        self.encoder_type = "cross"    


    def score_forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_attentions=False
        ):  


        output = self.cross_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pair_reps = output.last_hidden_state[:,0,:]
        score = self.l2(self.relu(self.l1(pair_reps)))
        
        if output_attentions and return_attentions:
            return score.sigmoid().squeeze(), output.attentions

        return score


    def cl_loss(self, pair_scores, labels):
        BSZ = pair_scores.size(0) # BSZ=2 * 4 (# Pos + # Neg) = 8 
        BSZ = int(BSZ/(4))

        pair_scores= list(pair_scores.tensor_split(BSZ, dim=0) )
        pair_scores = torch.stack(pair_scores)
        
        
        gap_1_loss_fct = nn.MarginRankingLoss(margin=0.5)
        gap_2_loss_fct = nn.MarginRankingLoss(margin=1.0)
              
        mq_scores = pair_scores[:,1] # 1
        lq_scores = pair_scores[:,2:-1] # 2

        # Use torch.clone to match Positive to Negatives
        hq_scores = pair_scores[:,0] #.repeat(1,neg_scores.size(-1)).flatten()
        # 6
        #target = torch.ones(pos_scores.size()).to(self.device)
        
        hq_mq_loss = gap_1_loss_fct(
                hq_scores.flatten(), 
                mq_scores.flatten(), 
                torch.ones(mq_scores.flatten().size()).to(self.device))
        mq_lq_loss = gap_1_loss_fct(
                mq_scores.repeat(1,lq_scores.size(-1)).flatten(), 
                lq_scores.flatten(), 
                torch.ones(lq_scores.flatten().size()).to(self.device))
        hq_lq_loss = gap_2_loss_fct(
                hq_scores.repeat(1,lq_scores.size(-1)).flatten(), 
                lq_scores.flatten(), 
                torch.ones(lq_scores.flatten().size()).to(self.device))
        
        mismatch_scores = pair_scores[:,-1]
        hq_mismatch_loss =  gap_2_loss_fct(
                        hq_scores.flatten(), 
                        mismatch_scores.flatten(), 
                        torch.ones(mismatch_scores.flatten().size()).to(self.device))
        mq_mismatch_loss = gap_1_loss_fct(
                mq_scores.flatten(), 
                mismatch_scores.flatten(), 
                torch.ones(mismatch_scores.flatten().size()).to(self.device))
        mismatch_loss = hq_mismatch_loss + mq_mismatch_loss 

        # NOTE: comment out the mismatch loss for ablation of prompt-aware loss
        loss = hq_mq_loss + mq_lq_loss + hq_lq_loss  + mismatch_loss 
        return loss

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        random = False
        ):
 
        pair_scores = self.score_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ).sigmoid().squeeze()

        if True:
            loss_fct = torch.nn.MSELoss()

            BSZ = pair_scores.size(0) # BSZ=2 * 4 (# Pos + # Neg) = 8 
            #print("size(0)", BSZ)
            BSZ = int(BSZ/(4+1))
            #print("bsz", BSZ)

    
            label = torch.zeros(5).float() #.float() 
            label[0] = 1.0
            label[1] = 0.5
            labels = torch.cat( [ label for x in range(BSZ)], -1).float().to(self.device)
    

            if True:
                """
                # remove every 5th item from pair_scores
                # THIS IS FOR ABLATION STUDY OF naive_regression prompt loss
                # BECAUSE 5th item is the mismatch score
                """

                # BSZ == 2 or BSZ == 4
                
                import numpy as np
                idx = np.array([i for i in range(len(pair_scores)) if i%5!=4])
                pair_scores = pair_scores[idx]
                labels = labels[idx]
                # print(pair_scores.size())
                # print(labels.size())
                # print()

            #labels = torch.cat( [ label for x in range(BSZ)], -1).float().to(self.device)
            """
            here taking care of prompt-aware terms for ablation
            """

   
            reg_loss = loss_fct(pair_scores, labels)
            return SequenceClassifierOutput(
                loss=reg_loss,
                logits=pair_scores,
                )


        # BSZ = pair_scores.size(0) # BSZ=2 * 4 (# Pos + # Neg) = 8 
        # BSZ = int(BSZ/4)

        # label = torch.zeros(4).long()
        # label[0] = 1
        # labels = torch.cat( [ label for x in range(BSZ)], -1).float().to(self.device)
        labels = None
        # 2 != 4

        #pred_loss_fct = torch.nn.BCEWithLogitsLoss()

        #pred_loss = pred_loss_fct(pair_scores, labels)
        
        # TODO: For both loss functions,
        #       Add Prompt-Switch Loss 
        if not random:
            cl_loss = self.cl_loss(pair_scores, labels)
        else:
            cl_loss = self.cl_loss_all_random(pair_scores, labels)
   

        loss =   cl_loss 
        return SequenceClassifierOutput(
                loss=loss,
                logits=pair_scores,
                )
 