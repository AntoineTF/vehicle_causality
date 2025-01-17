import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import special
from torch import optim
from torch.distributions import MultivariateNormal, Laplace
from torch.optim.lr_scheduler import MultiStepLR

from unitraj.models.base_model.base_model import BaseModel


class MapEncoderCNN(nn.Module):
    '''
    Regular CNN encoder for road image.
    '''

    def __init__(self, d_k=64, dropout=0.1, c=10):
        super(MapEncoderCNN, self).__init__()
        self.dropout = dropout
        self.c = c
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        # MAP ENCODER
        fm_size = 7
        self.map_encoder = nn.Sequential(
            init_(nn.Conv2d(3, 32, kernel_size=4, stride=1)), nn.ReLU(),
            init_(nn.Conv2d(32, 32, kernel_size=4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(32, 32, kernel_size=3, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(32, 32, kernel_size=3, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(32, fm_size * self.c, kernel_size=2, stride=2)), nn.ReLU(),
            nn.Dropout2d(p=self.dropout)
        )
        self.map_feats = nn.Sequential(
            init_(nn.Linear(7 * 7 * fm_size, d_k)), nn.ReLU(),
            init_(nn.Linear(d_k, d_k)), nn.ReLU(),
        )
        self.fisher_information = None
        self.optimal_params = None

    def forward(self, roads):
        '''
        :param roads: road image with size (B, 128, 128, 3)
        :return: road features, with one for every mode (B, c, d_k)
        '''
        B = roads.size(0)  # batch size
        return self.map_feats(self.map_encoder(roads).view(B, self.c, -1))


class MapEncoderPts(nn.Module):
    '''
    This class operates on the road lanes provided as a tensor with shape
    (B, num_road_segs, num_pts_per_road_seg, k_attr+1)
    '''

    def __init__(self, d_k, map_attr=3, dropout=0.1):
        super(MapEncoderPts, self).__init__()
        self.dropout = dropout
        self.d_k = d_k
        self.map_attr = map_attr
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.road_pts_lin = nn.Sequential(init_(nn.Linear(map_attr, self.d_k)))
        self.road_pts_attn_layer = nn.MultiheadAttention(self.d_k, num_heads=8, dropout=self.dropout)
        self.norm1 = nn.LayerNorm(self.d_k, eps=1e-5)
        self.norm2 = nn.LayerNorm(self.d_k, eps=1e-5)
        self.map_feats = nn.Sequential(
            init_(nn.Linear(self.d_k, self.d_k)), nn.ReLU(), nn.Dropout(self.dropout),
            init_(nn.Linear(self.d_k, self.d_k)),
        )

    def get_road_pts_mask(self, roads):
        road_segment_mask = torch.sum(roads[:, :, :, -1], dim=2) == 0
        road_pts_mask = (1.0 - roads[:, :, :, -1]).type(torch.BoolTensor).to(roads.device).view(-1, roads.shape[2])
        road_pts_mask = road_pts_mask.masked_fill((road_pts_mask.sum(-1) == roads.shape[2]).unsqueeze(-1), False) # Ensures no NaNs due to empty rows.
        return road_segment_mask, road_pts_mask

    def forward(self, roads, agents_emb):
        '''
        :param roads: (B, S, P, k_attr+1)  where B is batch size, S is num road segments, P is
        num pts per road segment.
        :param agents_emb: (T_obs, B, d_k) where T_obs is the observation horizon. THis tensor is obtained from
        AutoBot's encoder, and basically represents the observed socio-temporal context of agents.
        :return: embedded road segments with shape (S)
        '''
        B = roads.shape[0]
        S = roads.shape[1]
        P = roads.shape[2]
        road_segment_mask, road_pts_mask = self.get_road_pts_mask(roads)
        road_pts_feats = self.road_pts_lin(roads[:, :, :, :self.map_attr]).view(B * S, P, -1).permute(1, 0, 2)

        # Combining information from each road segment using attention with agent contextual embeddings as queries.
        agents_emb = agents_emb[-1].unsqueeze(2).repeat(1, 1, S, 1).view(-1, self.d_k).unsqueeze(0)
        road_seg_emb = self.road_pts_attn_layer(query=agents_emb, key=road_pts_feats, value=road_pts_feats,
                                                key_padding_mask=road_pts_mask)[0]
        road_seg_emb = self.norm1(road_seg_emb)
        road_seg_emb2 = road_seg_emb + self.map_feats(road_seg_emb)
        road_seg_emb2 = self.norm2(road_seg_emb2)
        road_seg_emb = road_seg_emb2.view(B, S, -1)

        return road_seg_emb.permute(1, 0, 2), road_segment_mask


def init(module, weight_init, bias_init, gain=1):
    '''
    This function provides weight and bias initializations for linear layers.
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class OutputModel(nn.Module):
    '''
    This class operates on the output of AutoBot-Ego's decoder representation. It produces the parameters of a
    bivariate Gaussian distribution.
    '''

    def __init__(self, d_k=64):
        super(OutputModel, self).__init__()
        self.d_k = d_k
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.observation_model = nn.Sequential(
            init_(nn.Linear(d_k, d_k)), nn.ReLU(),
            init_(nn.Linear(d_k, d_k)), nn.ReLU(),
            init_(nn.Linear(d_k, 5))
        )
        self.min_stdev = 0.01

    def forward(self, agent_decoder_state):
        T = agent_decoder_state.shape[0]
        BK = agent_decoder_state.shape[1]
        pred_obs = self.observation_model(agent_decoder_state.reshape(-1, self.d_k)).reshape(T, BK, -1)

        x_mean = pred_obs[:, :, 0]
        y_mean = pred_obs[:, :, 1]
        x_sigma = F.softplus(pred_obs[:, :, 2]) + self.min_stdev
        y_sigma = F.softplus(pred_obs[:, :, 3]) + self.min_stdev
        rho = torch.tanh(pred_obs[:, :, 4]) * 0.9  # for stability
        return torch.stack([x_mean, y_mean, x_sigma, y_sigma, rho], dim=2)


# class AutoBotEgo(nn.Module):
class AutoBotEgo(BaseModel):
    '''
    AutoBot-Ego Class.
    '''

    def __init__(self, config, k_attr=2, map_attr=2):

        super(AutoBotEgo, self).__init__(config)

        self.config = config
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.T = config['future_len']
        self.past = config['past_len']
        self.fisher_information = None
        self.map_attr = map_attr
        self.k_attr = k_attr
        self.d_k = config['hidden_size'] #128
        self.c = config['num_modes']

        self.L_enc = config['num_encoder_layers']
        self.dropout = config['dropout']
        self.num_heads = config['tx_num_heads']
        self.L_dec = config['num_decoder_layers']
        self.tx_hidden_size = config['tx_hidden_size']
        # Added for the causality
        self.return_embeddings = config["return_embeddings"]
        self.reg_type = config["reg_type"]
        self.contrastive_weight = config["contrastive_weight"]
        self.ranking_weight = config["ranking_weight"]
        self.max_num_agents = config["max_num_agents"]

        # INPUT ENCODERS
        self.agents_dynamic_encoder = nn.Sequential(init_(nn.Linear(self.k_attr, self.d_k)))

        # ============================== AutoBot-Ego ENCODER ==============================
        self.social_attn_layers = []
        self.temporal_attn_layers = []
        for _ in range(self.L_enc):
            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads, dropout=self.dropout,
                                                          dim_feedforward=self.tx_hidden_size)
            self.social_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads, dropout=self.dropout,
                                                          dim_feedforward=self.tx_hidden_size)
            self.temporal_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

        self.temporal_attn_layers = nn.ModuleList(self.temporal_attn_layers)
        self.social_attn_layers = nn.ModuleList(self.social_attn_layers)

        # ============================== MAP ENCODER ==========================
        self.map_encoder = MapEncoderPts(d_k=self.d_k, map_attr=self.map_attr, dropout=self.dropout)
        self.map_attn_layers = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=0.3)
        
        # ============================== Contrastive Projector (Added for the causality) ==========================
        if self.return_embeddings:
            input_dim = self.d_k * self.past
            self.contrastive_projector = nn.Sequential(
                torch.nn.Linear(input_dim, input_dim // 8),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(input_dim // 8, input_dim // 64)
            )

        # ============================== AutoBot-Ego DECODER ==============================
        self.Q = nn.Parameter(torch.Tensor(self.T, 1, self.c, self.d_k), requires_grad=True)
        nn.init.xavier_uniform_(self.Q)

        self.tx_decoder = []
        for _ in range(self.L_dec):
            self.tx_decoder.append(nn.TransformerDecoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                              dropout=self.dropout,
                                                              dim_feedforward=self.tx_hidden_size))
        self.tx_decoder = nn.ModuleList(self.tx_decoder)

        # ============================== Positional encoder ==============================
        self.pos_encoder = PositionalEncoding(self.d_k, dropout=0.0, max_len=self.past)

        # ============================== OUTPUT MODEL ==============================
        self.output_model = OutputModel(d_k=self.d_k)

        # ============================== Mode Prob prediction (P(z|X_1:t)) ==============================
        self.P = nn.Parameter(torch.Tensor(self.c, 1, self.d_k), requires_grad=True)  # Appendix C.2.
        nn.init.xavier_uniform_(self.P)

        self.mode_map_attn = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads)

        self.prob_decoder = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=self.dropout)
        self.prob_predictor = init_(nn.Linear(self.d_k, 1))

        self.criterion = Criterion(self.config)

        self.fisher_information = None
        self.optimal_params = None

    def generate_decoder_mask(self, seq_len, device):
        ''' For masking out the subsequent info. '''
        subsequent_mask = (torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1)).bool()
        return subsequent_mask

    def process_observations(self, ego, agents):
        '''
        :param observations: (B, T, N+2, A+1) where N+2 is [ego, other_agents, env]
        :return: a tensor of only the agent dynamic states, active_agent masks and env masks.
        '''
        # ego stuff
        ego_tensor = ego[:, :, :self.k_attr]
        env_masks_orig = ego[:, :, -1]
        env_masks = (1.0 - env_masks_orig).to(torch.bool)
        env_masks = env_masks.unsqueeze(1).repeat(1, self.c, 1).view(ego.shape[0] * self.c, -1)
        
        # Agents stuff
        temp_masks = torch.cat((torch.ones_like(env_masks_orig.unsqueeze(-1)), agents[:, :, :, -1]), dim=-1)
        opps_masks = (1.0 - temp_masks).to(torch.bool)  # only for agents.
        opps_tensor = agents[:, :, :, :self.k_attr]  # only opponent states

        return ego_tensor, opps_tensor, opps_masks, env_masks
    
    def process_observations_cf(self, ego):
        
        ego_tensor = ego[:, :, :self.k_attr]
        env_masks_orig = ego[:, :, -1]
        
        pred_masks = (1.0 - env_masks_orig).to(torch.bool)
        pred_masks = pred_masks.unsqueeze(1).repeat(1, self.c, 1).view(ego.shape[0] * self.c, -1)
        
        env_masks = torch.ones_like(env_masks_orig.unsqueeze(-1))
        env_masks = (1.0 - env_masks).to(torch.bool)
        
        
        return ego_tensor, env_masks, pred_masks
        
    def temporal_attn_fn(self, agents_emb, agent_masks, layer):
        '''
        :param agents_emb: (T, B, N, H)
        :param agent_masks: (B, T, N)
        :return: (T, B, N, H)
        '''
        T_obs = agents_emb.size(0)
        B = agent_masks.size(0)
        num_agents = agent_masks.size(2)
        temp_masks = agent_masks.permute(0, 2, 1).reshape(-1, T_obs)
        temp_masks = temp_masks.masked_fill((temp_masks.sum(-1) == T_obs).unsqueeze(-1), False)
        agents_temp_emb = layer(self.pos_encoder(agents_emb.reshape(T_obs, B * (num_agents), -1)),
                                src_key_padding_mask=temp_masks)
        return agents_temp_emb.view(T_obs, B, num_agents, -1)

    def social_attn_fn(self, agents_emb, agent_masks, layer):
        '''
        :param agents_emb: (T, B, N, H)
        :param agent_masks: (B, T, N)
        :return: (T, B, N, H)
        '''
        T_obs, B, num_agents, dim = agents_emb.shape
        agents_emb = agents_emb.permute(2, 1, 0, 3).reshape(num_agents, B * T_obs, -1)
        agents_soc_emb = layer(agents_emb, src_key_padding_mask=agent_masks.view(-1, num_agents))
        agents_soc_emb = agents_soc_emb.view(num_agents, B, T_obs, -1).permute(2, 1, 0, 3)
        return agents_soc_emb
    


    def _forward(self, inputs):
        """
        Forward pass for the model, processing both factual and counterfactual embeddings if requested.

        Args:
            inputs (dict): Dictionary containing inputs for the model, including:
                - ego_in: Tensor [B, T_obs, k_attr+1] with the last value being the existence mask.
                - agents_in: Tensor [B, T_obs, M-1, k_attr+1] with the last value being the existence mask.
                - roads: Road network representation as a tensor.

        Returns:
            dict: Output containing predicted probabilities and trajectories.
            dict: Embeddings if `return_embeddings` is True.
            list: Counterfactual predictions if `return_embeddings` is True.
        """
        # Unpacking inputs
        ego_in, agents_in, roads = inputs['ego_in'], inputs['agents_in'], inputs['roads']
        B = ego_in.size(0)
        T = ego_in.size(1)

        # Encode all input observations
        ego_tensor, _agents_tensor, opps_masks, env_masks = self.process_observations(ego_in, agents_in) 
        
        # Combine ego and agent embeddings
        agents_tensor = torch.cat((ego_tensor.unsqueeze(2), _agents_tensor), dim=2) 
        agents_emb = self.agents_dynamic_encoder(agents_tensor).permute(1, 0, 2, 3)
        
        # Process through AutoBot's encoder (applying temporal and social attention layers)
        for i in range(self.L_enc):
            agents_emb = self.temporal_attn_fn(agents_emb, opps_masks, layer=self.temporal_attn_layers[i])
            agents_emb = self.social_attn_fn(agents_emb, opps_masks, layer=self.social_attn_layers[i])
        
         # Select ego-agent embeddings
        ego_soctemp_emb = agents_emb[:, :, 0] 
        
        # Process map features
        orig_map_features, orig_road_segs_masks = self.map_encoder(roads, ego_soctemp_emb)
        map_features = orig_map_features.unsqueeze(2).repeat(1, 1, self.c, 1).view(-1, B * self.c, self.d_k)
        road_segs_masks = orig_road_segs_masks.unsqueeze(1).repeat(1, self.c, 1).view(B * self.c, -1)
        
        # Prepare embeddings if requested
        all_embeds = {}
        input_dim = self.d_k * self.past # self.past = 21 = T
        if self.return_embeddings:
            factual_input = ego_soctemp_emb.permute(1, 0, 2).reshape(B, -1)  # [B, T * d_k]
            ret_embeds = self.contrastive_projector(factual_input)  # [B, input_dim // 64]
            ret_embeds = nn.functional.normalize(ret_embeds, p=2, dim=1)
            all_embeds['factuals'] = ret_embeds
            
            # Counterfactual processing
            ego_in_cf = inputs["ego_in_cf"] # B,14,T,3
            nb_cf = inputs["nb_cf"]
            all_embeds["cf"] = []
            cf_predictions = []
            max_nb_cf = self.max_num_agents + 1 
            
            #Looping over counterfactual scenarios - Potential optimization for vectorization
            for batch_idx in range(B):
                num_cf = nb_cf[batch_idx].item()
                cf_sliced = ego_in_cf[batch_idx]
                with torch.no_grad():
                    ego_cf_tensor, env_masks_cf, pred_masks = self.process_observations_cf(cf_sliced)
                    ego_cf_tensor = ego_cf_tensor.unsqueeze(2)
                    cf_ego_emb = self.agents_dynamic_encoder(ego_cf_tensor).permute(1, 0, 2, 3)

                    for i in range(self.L_enc):
                        cf_ego_emb = self.temporal_attn_fn(cf_ego_emb, env_masks_cf, layer=self.temporal_attn_layers[i])
                        cf_ego_emb = self.social_attn_fn(cf_ego_emb, env_masks_cf, layer=self.social_attn_layers[i]) 
                    cf_ego_emb = cf_ego_emb.squeeze(2)
                    
                    cf_ret_embeds = self.contrastive_projector(cf_ego_emb.permute(1, 0, 2).reshape(max_nb_cf, -1))  
                    cf_ret_embeds = nn.functional.normalize(cf_ret_embeds, p=2, dim=1)
                    all_embeds["cf"].append(cf_ret_embeds)
                
                # Predict counterfactuals
                cf_out_seq = self.Q.repeat(1, max_nb_cf, 1, 1).view(self.T, max_nb_cf * self.c, -1)
                cf_context = cf_ego_emb.unsqueeze(2).repeat(1, 1, self.c, 1) 
                cf_context = cf_context.view(-1, max_nb_cf * self.c, self.d_k)
                
                # Prepare map features for counterfactuals
                cf_map_features = orig_map_features.unsqueeze(2).repeat(1, 1, self.c, 1).view(-1, max_nb_cf  * self.c, self.d_k)
                cf_road_segs_masks = orig_road_segs_masks.unsqueeze(1).repeat(1, self.c, 1).view(max_nb_cf * self.c, -1)
                
                # Generate decoder mask for counterfactual prediction
                cf_time_masks = self.generate_decoder_mask(seq_len=self.T, device=cf_sliced.device)

                env_masks_cf = env_masks_cf.squeeze(2)
                for d in range(self.L_dec):
                    cf_dec_emb_map = self.map_attn_layers(query=cf_out_seq, key=cf_map_features, value=cf_map_features,
                                                        key_padding_mask=cf_road_segs_masks)[0]
                    cf_out_seq = cf_out_seq + cf_dec_emb_map
                    cf_out_seq = self.tx_decoder[d](cf_out_seq, cf_context, tgt_mask=cf_time_masks, memory_key_padding_mask=pred_masks)

                # Generate counterfactual predictions
                cf_out_dists = self.output_model(cf_out_seq).reshape(self.T, max_nb_cf, self.c, -1).permute(2, 0, 1, 3)
                                    
                cf_predictions.append({
                    #'predicted_probability': cf_mode_probs,
                    'predicted_trajectory': cf_out_dists.permute(2, 0, 1, 3)
                })

        # Repeat the tensors for the number of modes for efficient forward pass.
        context = ego_soctemp_emb.unsqueeze(2).repeat(1, 1, self.c, 1)
        context = context.view(-1, B * self.c, self.d_k)

        # AutoBot-Ego Decoding
        out_seq = self.Q.repeat(1, B, 1, 1).view(self.T, B * self.c, -1)
        time_masks = self.generate_decoder_mask(seq_len=self.T, device=ego_in.device)
        
        for d in range(self.L_dec):
            ego_dec_emb_map = self.map_attn_layers(query=out_seq, key=map_features, value=map_features,key_padding_mask=road_segs_masks)[0]
            out_seq = out_seq + ego_dec_emb_map
            out_seq = self.tx_decoder[d](out_seq, context, tgt_mask=time_masks, memory_key_padding_mask=env_masks)
        out_dists = self.output_model(out_seq).reshape(self.T, B, self.c, -1).permute(2, 0, 1, 3)

        # Mode probability prediction
        mode_params_emb = self.P.repeat(1, B, 1)
        mode_params_emb = self.prob_decoder(query=mode_params_emb, key=ego_soctemp_emb, value=ego_soctemp_emb)[0]
        mode_params_emb = self.mode_map_attn(query=mode_params_emb, key=orig_map_features, value=orig_map_features,
                                             key_padding_mask=orig_road_segs_masks)[0] + mode_params_emb
        mode_probs = F.softmax(self.prob_predictor(mode_params_emb).squeeze(-1), dim=0).transpose(0, 1)

        output = {}
        output['predicted_probability'] = mode_probs
        output['predicted_trajectory'] = out_dists.permute(2, 0, 1,3)  # [c, T, B, 5] to [B, c, T, 5] to be able to parallelize code
        if self.return_embeddings:
            return output, all_embeds, cf_predictions
        else :
            return output, None, None

    def forward(self, batch):
        """
        Forward pass for the model. Processes batch data, computes loss, and applies optional regularization.

        Parameters:
        - batch (dict): Batch containing input data for the model.

        Returns:
        - output (dict): Model predictions.
        - loss (tensor): Total loss including optional regularization.
        - all_embeds (dict): Embeddings if `return_embeddings` is True.
        - regularizer (dict): Regularization loss information if applicable.
        - cf_predictions (list): Counterfactual predictions if applicable.
        - causal_effect_matrix (tensor): Causal effect measurements if applicable.
        - nb_cf (tensor): Number of counterfactuals per batch element.
        """
         # Initialize model input dictionary
        model_input = {}
        inputs = batch['input_dict']
        dataset_name = inputs["dataset_name"]
        print("the dataset name is:", dataset_name)
        print(f"the shape of the name is {np.shape(dataset_name)}\nThe one of the inputs trajs is {np.shape(inputs['obj_trajs'])}")
        dataset_name_cf = inputs["dataset_name_cf"]
        print(f"CF: the shape of the name is {np.shape(dataset_name_cf)}\nThe one of the inputs trajs is {np.shape(inputs['obj_trajs_cf'])}")
        
        agents_in, agents_mask, roads = inputs['obj_trajs'], inputs['obj_trajs_mask'], inputs['map_polylines']
        
        # Prepare the ego vehicle inputs using tensor indexing
        ego_in = torch.gather(agents_in, 1, inputs['track_index_to_predict'].view(-1, 1, 1, 1).repeat(1, 1,
                                                                                                      *agents_in.shape[
                                                                                                       -2:])).squeeze(1)
        ego_mask = torch.gather(agents_mask, 1, inputs['track_index_to_predict'].view(-1, 1, 1).repeat(1, 1,
                                                                                                       agents_mask.shape[
                                                                                                           -1])).squeeze(1)
        
        # Combine input features and masks
        agents_in = torch.cat([agents_in[..., :2], agents_mask.unsqueeze(-1)], dim=-1)
        agents_in = agents_in.transpose(1, 2)
        ego_in = torch.cat([ego_in[..., :2], ego_mask.unsqueeze(-1)], dim=-1)
        roads = torch.cat([inputs['map_polylines'][..., :2], inputs['map_polylines_mask'].unsqueeze(-1)], dim=-1)
        
        # Add to model input dictionary
        model_input['ego_in'] = ego_in 
        model_input['agents_in'] = agents_in
        model_input['roads'] = roads
        
        
        # Counterfactual embeddings preparation
        if self.return_embeddings:
        
            futur_ego_in_cf, futur_ego_mask_cf = inputs["obj_trajs_future_state_cf"], inputs["obj_trajs_future_mask_cf"]
            futur_agents_in, futur_agents_mask = inputs["obj_trajs_future_state"], inputs["obj_trajs_future_mask"]
            ego_in_cf, ego_mask_cf = inputs["obj_trajs_cf"], inputs["obj_trajs_mask_cf"] 
            nb_cf = inputs["nb_cf"]

            # Removing the 3rd dimension since there is only one el
            ego_in_cf = ego_in_cf.squeeze(2)  # B,14,T,39
            ego_mask_cf = ego_mask_cf.squeeze(2)  # B,14,1,T
            ego_in_cf = torch.cat([ego_in_cf[..., :2], ego_mask_cf.unsqueeze(-1)], dim=-1) # B,14,T,3 
            
            # Future state processing
            futur_ego_in_cf = futur_ego_in_cf.squeeze(2)
            futur_ego_mask_cf = ego_mask_cf.squeeze(2)
            futur_ego_in = torch.gather(futur_agents_in, 1, inputs['track_index_to_predict'].view(-1, 1, 1, 1).repeat(1, 1,
                                                                                                      *futur_agents_in.shape[
                                                                                                       -2:])).squeeze(1)
            futur_ego_mask = torch.gather(futur_agents_mask, 1, inputs['track_index_to_predict'].view(-1, 1, 1).repeat(1, 1,
                                                                                                       futur_agents_mask.shape[
                                                                                                           -1])).squeeze(1)
            
            futur_agents_in = torch.cat([futur_agents_in[..., :2], futur_agents_mask.unsqueeze(-1)], dim=-1)
            futur_agents_in = futur_agents_in.transpose(1, 2)
            futur_ego_in = torch.cat([futur_ego_in[..., :2], futur_ego_mask.unsqueeze(-1)], dim=-1)

            # Update model input with counterfactuals
            model_input["ego_in_cf"] = ego_in_cf
            model_input["nb_cf"] = nb_cf
            
            # Compute causal effect matrix
            B = ego_in.size(0)
            ego_in = ego_in[..., :2]  # Shape: (B, 21, 3)
            futur_ego_in = futur_ego_in[..., :2]  # Shape: (B, 21, 3)
            full_ego = torch.cat((ego_in, futur_ego_in), dim=1)
            assert full_ego.shape == (B, 81, 2), f"Unexpected merged shape: {full_ego.shape}"
            
            # Loop preparation
            causal_effect_matrix = []
            max_num_cf = max(nb_cf).item()
            
            for batch_idx in range(B):
                num_cf = nb_cf[batch_idx].item()
                cf_sliced = ego_in_cf[batch_idx, :num_cf]
                fut_cf_sliced = futur_ego_in_cf[batch_idx, :num_cf]
                
                # Assertion for debuging
                assert cf_sliced.shape == (num_cf, 21, 3), f"Unexpected shape: {cf_sliced.shape}"
                assert fut_cf_sliced.shape == (num_cf,60,4), f"Unexpected shape: {fut_cf_sliced.shape}"
                
                # Select the first two features from both arrays
                cf_sliced_selected = cf_sliced[..., :2]  
                fut_cf_sliced_selected = fut_cf_sliced[..., :2] 
                
                # Concatenate past and future trajectories
                full_ego_cf = torch.cat((cf_sliced_selected, fut_cf_sliced_selected), dim=1)  # Shape: (num_cf, 81, 2)
                
                f_traj = full_ego[batch_idx]
                
                # Calculate L2 distance for causal effect
                l2_distances = torch.norm(full_ego_cf - f_traj.unsqueeze(0), dim=2)  
                l2_distances_sum = l2_distances.sum(dim=1)  

                # Pad causal matrix to uniform size
                padded_l2_distances_sum = torch.zeros(max_num_cf, device=full_ego.device)
                padded_l2_distances_sum[:num_cf] = l2_distances_sum

                causal_effect_matrix.append(padded_l2_distances_sum)
                
            # Convert the list to a tensor for further processing
            causal_effect_matrix = torch.stack(causal_effect_matrix)  # Shape: (B, max_num_cf)

            assert causal_effect_matrix.shape == (B, max_num_cf), f"Unexpected shape: {causal_effect_matrix.shape}"
                
        # Computing the causal effect matrix (Use if no access to vizualisation of the simulation for exemple)
        # self.analyze_causal_effects(causal_effect_matrix, nb_cf)
        
        # Forward pass through the core model
        output, all_embeds, cf_predictions = self._forward(model_input)
        loss = self.get_loss(batch, output)
            
        # Apply regularization if embeddings are required
        if self.return_embeddings and self.reg_type in ["ranking", "contrastive"]:
            factual_embeds = all_embeds['factuals']
            cf_embeds = all_embeds['cf']
            regularizer = {}
            
            if self.reg_type == "ranking":
                reg_loss = self.calc_ranking_loss(factual_embeds, cf_embeds,causal_effect_matrix, nb_cf, margin = 0.001, ranking_weight=self.ranking_weight)
                loss += reg_loss  
                regularizer["name"] = 'ranking_loss'
                regularizer["value"] = reg_loss
                 
            elif self.reg_type == "contrastive":
                reg_loss = self.calc_contrastive_loss(factual_embeds, cf_embeds,causal_effect_matrix, nb_cf ,tau=0.5, contrastive_weight=self.contrastive_weight)
                loss += reg_loss
                regularizer["name"] = "contrastive_loss"
                regularizer["value"] = reg_loss
            
            return output, loss, all_embeds, regularizer, cf_predictions, causal_effect_matrix, nb_cf
        
        elif self.return_embeddings and self.reg_type == "no_reg":
            regularizer = {}
            regularizer["name"] = "no_reg"
            regularizer["value"] = -1
            return output, loss, all_embeds, regularizer, cf_predictions, causal_effect_matrix, nb_cf
        
        # Return results when embeddings are not requested
        return output, loss, None, None, None, None, None
    
    def analyze_causal_effects(self, causal_effect_matrix, nb_cf):
        """
        Analyzes the distribution of the causal effect matrix to help select thresholds.
        
        :param causal_effect_matrix: Tensor [B, max_num_cf] with causal effect values (padded).
        :param nb_cf: Tensor [B] containing the number of valid counterfactuals for each batch.
        :return: Recommended thresholds for non-causal and causal classification.
        """
         # Mask creation to filter valid entries
        valid_values = []
        for batch_idx in range(causal_effect_matrix.shape[0]):
            num_cf = nb_cf[batch_idx].item()
            valid_values.append(causal_effect_matrix[batch_idx, :num_cf])

        # Convert the list of tensors to a single tensor
        all_values = torch.cat(valid_values)

        # Compute descriptive statistics
        mean = torch.mean(all_values).item()
        std = torch.std(all_values).item()
        median = torch.median(all_values).item()
        percentile_5 = torch.quantile(all_values, 0.05).item()
        percentile_95 = torch.quantile(all_values, 0.95).item()

    
    def calc_contrastive_loss(self, factual_embeddings, counterfactual_embeddings, causal_effect_matrix, nb_cf,tau, contrastive_weight):
        """
        Calculates contrastive loss using factual and counterfactual embeddings and a causal effect matrix.
        Implements an InfoNCE loss where factual embeddings act as anchors, positive pairs are low-causal counterfactuals,
        and negative pairs are high-causal counterfactuals.

        :param factual_embeddings: Tensor [B, embedding_dim] - Factual embeddings for each batch.
        :param counterfactual_embeddings: List of tensors, each of shape [nb_cf, embedding_dim].
        :param causal_effect_matrix: Tensor [B, max_nb_cf] - Causal effect values for each batch and counterfactual.
        :param nb_cf: Tensor [B] - Number of valid counterfactuals for each batch.
        :param tau: Float - Temperature parameter for contrastive scaling.
        :param contrastive_weight: Float - Weight for the contrastive loss component.
        :return: Contrastive loss (scalar).
        """
        batch_size = factual_embeddings.shape[0]
        contrastive_losses = []

        for i in range(batch_size):
            # Extract current factual embedding
            q = factual_embeddings[i]  # Shape: [embedding_dim]
            
            # Extract the valid counterfactual embeddings and causal effects for the batch
            cf_embeddings = counterfactual_embeddings[i]  # Shape: [nb_cf, embedding_dim]
            causal_effects = causal_effect_matrix[i][:nb_cf[i].item()]  # Shape: [nb_cf]
            
            # Skip if no valid counterfactuals exist
            if causal_effects.numel() == 0:
                continue
            
            # Sort counterfactuals based on causal effect values
            sorted_indices = torch.argsort(causal_effects)
            sorted_cf_embeddings = cf_embeddings[sorted_indices]
            sorted_effects = causal_effects[sorted_indices]

            # Extract positive and negative indices
            positives = (sorted_effects <= 40).nonzero(as_tuple=True)[0]  # Non-causal pairs
            negatives = (sorted_effects >= 450).nonzero(as_tuple=True)[0]  # Strong causal pairs

            # Check if valid pairs exist for contrastive loss calculation
            if len(positives) > 0 and len(negatives) > 0:
                # Select a random positive and all negatives
                positive_idx = torch.randint(0, len(positives), (1,)).item()
                k_plus = sorted_cf_embeddings[positives[positive_idx]]
                k_negs = sorted_cf_embeddings[negatives]

                # Compute the numerator for the InfoNCE loss (factual-positive similarity)
                numerator = torch.exp(torch.dot(q, k_plus) / tau)
                
                # Compute the denominator (including all negative pairs)
                denominator = numerator + torch.sum(torch.exp(torch.matmul(k_negs, q) / tau))
                
                # Compute the contrastive loss for the batch
                loss = -torch.log(numerator / denominator)
                contrastive_losses.append(loss)

        # Aggregate loss across batches
        if len(contrastive_losses) == 0:
            # Return zero loss if no valid pairs were found
            return torch.tensor(0.0, device=factual_embeddings.device) 
        else:
            contrastive_loss = torch.mean(torch.stack(contrastive_losses)) * contrastive_weight
            return contrastive_loss

    
    def calc_ranking_loss(self, factual_embeddings, counterfactual_embeddings, causal_effect_matrix, nb_cf, margin, ranking_weight):
        """
        Calculates a margin-based ranking loss for factual and counterfactual embeddings based on causal effect values.

        The goal is to ensure that embeddings with lower causal effects (non-causal) are ranked closer to the factual 
        embeddings than those with higher causal effects (strong causal).

        :param factual_embeddings: Tensor [B, embedding_dim] - Factual embeddings for each batch.
        :param counterfactual_embeddings: List of tensors, each [nb_cf, embedding_dim].
        :param causal_effect_matrix: Tensor [B, max_nb_cf] - Causal effect values for each batch and counterfactual.
        :param nb_cf: Tensor [B] - Number of valid counterfactuals for each batch.
        :param margin: Float - Margin value for the margin ranking loss.
        :param ranking_weight: Float - Weight to scale the ranking loss.
        :return: Scalar - Aggregated ranking loss across the batch.
        """
        batch_size = factual_embeddings.shape[0]
        ranking_losses = []
        loss_fn = torch.nn.MarginRankingLoss(margin=margin)

        for batch_idx in range(batch_size):
            # Retrieve the factual embedding for this batch
            factual_embedding = factual_embeddings[batch_idx]  # Shape: [embedding_dim]

            # Retrieve valid counterfactual embeddings and causal effects
            num_valid_cf = nb_cf[batch_idx].item()
            if num_valid_cf == 0:
                continue  # Skip batches without valid counterfactuals

            cf_embeddings = counterfactual_embeddings[batch_idx][:num_valid_cf]  # [nb_cf, embedding_dim]
            causal_effects = causal_effect_matrix[batch_idx][:num_valid_cf]  # [nb_cf]

            # Calculate similarity (using dot product as distance measure)
            similarities = torch.matmul(cf_embeddings, factual_embedding)  # [nb_cf]

            # Sort based on causal effects (not similarity)
            sorted_indices = torch.argsort(causal_effects, descending=False)
            sorted_similarities = similarities[sorted_indices]
            sorted_causal_effects = causal_effects[sorted_indices]

            # Identify positive and negative pairs for ranking loss based on causal thresholds
            positive_indices = (sorted_causal_effects <= 40).nonzero(as_tuple=True)[0]
            negative_indices = (sorted_causal_effects >= 450).nonzero(as_tuple=True)[0]

            # Ensure valid positive and negative pairs exist
            if len(positive_indices) == 0 or len(negative_indices) == 0:
                continue

            # Randomly select one positive and one negative pair for ranking loss calculation
            pos_idx = positive_indices[torch.randint(0, len(positive_indices), (1,)).item()]
            neg_idx = negative_indices[torch.randint(0, len(negative_indices), (1,)).item()]

            # Compute the ranking loss (ensuring the positive pair is ranked closer than the negative pair)
            positive_sim = sorted_similarities[pos_idx].unsqueeze(0)  # Positive pair similarity
            negative_sim = sorted_similarities[neg_idx].unsqueeze(0)  # Negative pair similarity

            # Loss is positive if the negative pair is ranked closer than the positive pair
            loss = loss_fn(positive_sim, negative_sim, torch.tensor([1.0], device=factual_embeddings.device))
            ranking_losses.append(loss)

        # Aggregate ranking loss across the batch or return zero if no pairs found
        if len(ranking_losses) == 0:
            return torch.tensor(0.0, device=factual_embeddings.device)  # No valid pairs found
        else:
            return torch.mean(torch.stack(ranking_losses)) * ranking_weight

    
    
    def get_loss(self, batch, prediction):
        inputs = batch['input_dict']
        ground_truth = torch.cat([inputs['center_gt_trajs'][..., :2], inputs['center_gt_trajs_mask'].unsqueeze(-1)],
                                 dim=-1)
        loss = self.criterion(prediction, ground_truth, inputs['center_gt_final_valid_idx'])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['learning_rate'], eps=0.0001)
        scheduler = MultiStepLR(optimizer, milestones=self.config['learning_rate_sched'], gamma=0.5,
                                verbose=True)
        return [optimizer], [scheduler]


class Criterion(nn.Module):
    def __init__(self, config):
        super(Criterion, self).__init__()
        self.config = config

    def forward(self, out, gt, center_gt_final_valid_idx):

        return self.nll_loss_multimodes(out, gt, center_gt_final_valid_idx)

    def get_BVG_distributions(self, pred):
        B = pred.size(0)
        T = pred.size(1)
        mu_x = pred[:, :, 0].unsqueeze(2)
        mu_y = pred[:, :, 1].unsqueeze(2)
        sigma_x = pred[:, :, 2]
        sigma_y = pred[:, :, 3]
        rho = pred[:, :, 4]

        # Create the base covariance matrix for a single element
        cov = torch.stack([
            torch.stack([sigma_x ** 2, rho * sigma_x * sigma_y], dim=-1),
            torch.stack([rho * sigma_x * sigma_y, sigma_y ** 2], dim=-1)
        ], dim=-2)

        # Expand this base matrix to match the desired shape
        biv_gauss_dist = MultivariateNormal(loc=torch.cat((mu_x, mu_y), dim=-1), covariance_matrix=cov,validate_args=False)
        return biv_gauss_dist

    def get_Laplace_dist(self, pred):
        return Laplace(pred[:, :, :2], pred[:, :, 2:4],validate_args=False)

    def nll_pytorch_dist(self, pred, data, mask, rtn_loss=True):
        # biv_gauss_dist = get_BVG_distributions(pred)
        biv_gauss_dist = self.get_Laplace_dist(pred)
        num_active_per_timestep = mask.sum()
        data_reshaped = data[:, :, :2]
        if rtn_loss:
            # return (-biv_gauss_dist.log_prob(data)).sum(1)  # Gauss
            return ((-biv_gauss_dist.log_prob(data_reshaped)).sum(-1) * mask).sum(1)  # Laplace
        else:
            # return (-biv_gauss_dist.log_prob(data)).sum(-1)  # Gauss
            # need to multiply by masks
            # return (-biv_gauss_dist.log_prob(data_reshaped)).sum(dim=(1, 2))  # Laplace
            return ((-biv_gauss_dist.log_prob(data_reshaped)).sum(dim=2) * mask).sum(1)  # Laplace

    def nll_loss_multimodes(self, output, data, center_gt_final_valid_idx):
        """NLL loss multimodes for training. MFP Loss function
        Args:
          pred: [K, T, B, 5]
          data: [B, T, 5]
          modes_pred: [B, K], prior prob over modes
          noise is optional
        """
        modes_pred = output['predicted_probability']
        pred = output['predicted_trajectory'].permute(1, 2, 0, 3)
        mask = data[..., -1]

        entropy_weight = self.config['entropy_weight']
        kl_weight = self.config['kl_weight']
        use_FDEADE_aux_loss = self.config['use_FDEADE_aux_loss']

        modes = len(pred)
        nSteps, batch_sz, dim = pred[0].shape

        log_lik_list = []
        with torch.no_grad():
            for kk in range(modes):
                nll = self.nll_pytorch_dist(pred[kk].transpose(0, 1), data, mask, rtn_loss=False)
                log_lik_list.append(-nll.unsqueeze(1))  # Add a new dimension to concatenate later

        # Concatenate the list to form the log_lik tensor
            log_lik = torch.cat(log_lik_list, dim=1)

            priors = modes_pred
            log_priors = torch.log(priors)
            log_posterior_unnorm = log_lik + log_priors

            # Compute logsumexp for normalization, ensuring no in-place operations
            logsumexp = torch.logsumexp(log_posterior_unnorm, dim=-1, keepdim=True)
            log_posterior = log_posterior_unnorm - logsumexp

            # Compute the posterior probabilities without in-place operations
            post_pr = torch.exp(log_posterior)
            # Ensure post_pr is a tensor on the correct device
            post_pr = post_pr.to(data.device)

        # Compute loss.
        loss = 0.0
        for kk in range(modes):
            nll_k = self.nll_pytorch_dist(pred[kk].transpose(0, 1), data, mask, rtn_loss=True) * post_pr[:, kk]
            loss += nll_k.mean()

        # Adding entropy loss term to ensure that individual predictions do not try to cover multiple modes.
        entropy_vals = []
        for kk in range(modes):
            entropy_vals.append(self.get_BVG_distributions(pred[kk]).entropy())
        entropy_vals = torch.stack(entropy_vals).permute(2, 0, 1)
        entropy_loss = torch.mean((entropy_vals).sum(2).max(1)[0])
        loss += entropy_weight * entropy_loss

        # KL divergence between the prior and the posterior distributions.
        kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')  # type: ignore
        kl_loss = kl_weight * kl_loss_fn(torch.log(modes_pred), post_pr)

        # compute ADE/FDE loss - L2 norms with between best predictions and GT.
        if use_FDEADE_aux_loss:
            adefde_loss = self.l2_loss_fde(pred, data, mask)
        else:
            adefde_loss = torch.tensor(0.0).to(data.device)

        # post_entropy
        final_loss = loss + kl_loss + adefde_loss

        return final_loss

    def l2_loss_fde(self, pred, data, mask):

        fde_loss = (torch.norm((pred[:, -1, :, :2].transpose(0, 1) - data[:, -1, :2].unsqueeze(1)), 2, dim=-1) * mask[:,
                                                                                                                 -1:])
        ade_loss = (torch.norm((pred[:, :, :, :2].transpose(1, 2) - data[:, :, :2].unsqueeze(0)), 2,
                               dim=-1) * mask.unsqueeze(0)).mean(dim=2).transpose(0, 1)
        loss, min_inds = (fde_loss + ade_loss).min(dim=1)
        return 100.0 * loss.mean()


