import numpy as np
import random

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class AudioVisualModel(nn.Module):
    def __init__(self):
        super(AudioVisualModel, self).__init__()

        self.input_sizes = input_sizes = [1, 35, 259]
        self.hidden_sizes = hidden_sizes = [int(1), 35, 259]
      
        self.tanh = nn.Tanh()
        
        rnn = nn.LSTM 
        # defining modules - two layer bidirectional LSTM with layer norm in between
        
        self.vrnn1 = rnn(35, hidden_sizes[1], bidirectional=True)
        
        self.arnn1 = rnn(input_sizes[2], hidden_sizes[2], bidirectional=True)
        
        self.project_v = nn.Sequential()
  
        self.project_v.add_module('project_v', nn.Linear(in_features=70, 
                                    out_features=128))
        self.project_v.add_module('project_v_activation', nn.ReLU())
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(128))
        '''
        RuntimeError: mat1 and mat2 shapes cannot be multiplied (60x518 and 148x128)
        '''
        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=518, 
                        out_features=128))
        self.project_a.add_module('project_a_activation', nn.ReLU())
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(128))
        
        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1', nn.Linear(in_features=128, 
                                                                    out_features=128))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())
        
        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3', nn.Linear(in_features=128, 
                                                                out_features=128))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())
        
        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=128, 
                                                            out_features=128))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())

        self.fusion = nn.Sequential()
      
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=128*4, 
                out_features=128*3))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(0.5))
        self.fusion.add_module('fusion_layer_1_activation', nn.ReLU())
        
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=128*3, 
                out_features= 8))

        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1]*2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2]*2,))

        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        
    def extract_features(self, sequence, rnn1):
        
        packed_h1, (final_h1, _) = rnn1(sequence)    
    
        return final_h1

    def shared_private(self, utterance_v, utterance_a):
        
        # Projecting to same sized space
        # self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        utterance_v = self.project_v(utterance_v)
        
        utterance_a = self.project_a(utterance_a)
        
        # Private-shared components
        # self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_v = self.private_v(utterance_v)
        self.utt_private_a = self.private_a(utterance_a)

        # self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)
        self.utt_shared_a = self.shared(utterance_a)

    def forward(self, video, acoustic):
        batch_size = 60
        
        # extract features from visual modality
        final_h1v = self.extract_features(video, self.vrnn1)
        permuted=final_h1v.permute(1,0,2) #(60, 2, 35)
        utterance_video = permuted.contiguous().view(batch_size, -1)
        
        # extract features from acoustic modality
        final_h1a = self.extract_features(acoustic, self.arnn1)
        utterance_audio = final_h1a.permute(1, 0, 2).contiguous().view(batch_size, -1)

        # Shared-private encoders
        self.shared_private(utterance_video, utterance_audio)
         
        # 1-LAYER TRANSFORMER FUSION
        
        h = torch.stack((self.utt_private_v, self.utt_private_a, 
                            self.utt_shared_v,  self.utt_shared_a), dim=0)
        
        h = self.transformer_encoder(h)
       
        h = torch.cat((h[0], h[1], h[2], h[3]), dim=1)
        
        o = self.fusion(h)
        
        return o
