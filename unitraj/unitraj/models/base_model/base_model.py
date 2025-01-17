import json

import os
import csv

import numpy as np
import pytorch_lightning as pl
import torch
import wandb

import unitraj.datasets.common_utils as common_utils
import unitraj.utils.visualization as visualization


class BaseModel(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.pred_dicts = []

        if config.get('eval_nuscenes', False):
            self.init_nuscenes()

    def init_nuscenes(self):
        if self.config.get('eval_nuscenes', False):
            from nuscenes import NuScenes

            from nuscenes.eval.prediction.config import PredictionConfig

            from nuscenes.prediction import PredictHelper
            nusc = NuScenes(version='v1.0-trainval', dataroot=self.config['nuscenes_dataroot'])

            # Prediction helper and configs:
            self.helper = PredictHelper(nusc)

            with open('models/base_model/nuscenes_config.json', 'r') as f:
                pred_config = json.load(f)
            self.pred_config5 = PredictionConfig.deserialize(pred_config, self.helper)

    def forward(self, batch):
        """
        Forward pass for the model
        :param batch: input batch
        :return: prediction: {
                'predicted_probability': (batch_size,modes)),
                'predicted_trajectory': (batch_size,modes, future_len, 2)
                }
                loss (with gradient)
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        #rajouter nb_cf en training step et dans login, la fonction  ACEs compile pas
        
        prediction, loss, all_embeds, regularizer, cf_predictions, causal_effects, nb_cf = self.forward(batch)
        self.compute_official_evaluation(batch, prediction)
        if self.return_embeddings:
            f_actual = prediction["predicted_trajectory"]
            f_counterfactual = [cf["predicted_trajectory"] for cf in cf_predictions]
            self.log_info(batch = batch, 
                        batch_idx = batch_idx, 
                        prediction = prediction, 
                        f_actual = f_actual, 
                        f_counterfactual = f_counterfactual,
                        causal_effects = causal_effects, 
                        regularizer = regularizer,
                        nb_cf = nb_cf,
                        status='train'
                        )
        else:
            self.log_info(batch, batch_idx, prediction, status='train')
        return loss

    # def validation_step(self, batch, batch_idx):
    #     prediction, loss = self.forward(batch)
    #     self.compute_official_evaluation(batch, prediction)
    #     self.log_info(batch, batch_idx, prediction, status='val')
    #     return loss
    
    def validation_step(self,  batch, batch_idx):
        prediction, loss, all_embeds, regularizer, cf_predictions, causal_effects, nb_cf = self.forward(batch)
        self.compute_official_evaluation(batch, prediction)
        if self.return_embeddings:
            f_actual = prediction["predicted_trajectory"]
            f_counterfactual = [cf["predicted_trajectory"] for cf in cf_predictions]
            self.log_info(batch = batch, 
                        batch_idx = batch_idx, 
                        prediction = prediction, 
                        f_actual = f_actual, 
                        f_counterfactual = f_counterfactual,
                        causal_effects = causal_effects, 
                        regularizer = regularizer,
                        nb_cf = nb_cf,
                        status='val'
                        )
        else:
            self.log_info(batch, batch_idx, prediction, status='val')
        
        return loss

    def on_validation_epoch_end(self):
        if self.config.get('eval_waymo', False):
            metric_results, result_format_str = self.compute_metrics_waymo(self.pred_dicts)
            print(metric_results)
            print(result_format_str)

        elif self.config.get('eval_nuscenes', False):
            import os
            os.makedirs('submission', exist_ok=True)
            json.dump(self.pred_dicts, open(os.path.join('submission', "evalai_submission.json"), "w"))
            metric_results = self.compute_metrics_nuscenes(self.pred_dicts)
            print('\n', metric_results)
            
        elif self.config.get('eval_argoverse2', False):
            metric_results = self.compute_metrics_av2(self.pred_dicts)
            
        self.pred_dicts = []

    def configure_optimizers(self):
        raise NotImplementedError

    def compute_metrics_nuscenes(self, pred_dicts):
        from nuscenes.eval.prediction.compute_metrics import compute_metrics
        metric_results = compute_metrics(pred_dicts, self.helper, self.pred_config5)
        return metric_results

    def compute_metrics_waymo(self, pred_dicts):
        from unitraj.models.base_model.waymo_eval import waymo_evaluation
        try:
            num_modes_for_eval = pred_dicts[0]['pred_trajs'].shape[0]
        except:
            num_modes_for_eval = 6
        metric_results, result_format_str = waymo_evaluation(pred_dicts=pred_dicts,
                                                             num_modes_for_eval=num_modes_for_eval)

        metric_result_str = '\n'
        for key in metric_results:
            metric_results[key] = metric_results[key]
            metric_result_str += '%s: %.4f \n' % (key, metric_results[key])
        metric_result_str += '\n'
        metric_result_str += result_format_str

        return metric_result_str, metric_results

    def compute_metrics_av2(self, pred_dicts):
        from unitraj.models.base_model.av2_eval import argoverse2_evaluation
        try:
            num_modes_for_eval = pred_dicts[0]['pred_trajs'].shape[0]
        except:
            num_modes_for_eval = 6
        metric_results = argoverse2_evaluation(pred_dicts=pred_dicts,
                                               num_modes_for_eval=num_modes_for_eval)
        self.log('val/av2_official_minADE6', metric_results['min_ADE'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/av2_official_minFDE6', metric_results['min_FDE'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/av2_official_brier_minADE', metric_results['brier_min_ADE'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/av2_official_brier_minFDE', metric_results['brier_min_FDE'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/av2_official_miss_rate', metric_results['miss_rate'], prog_bar=True, on_step=False, on_epoch=True)
        
        # metric_result_str = '\n'
        # for key, value in metric_results.items():
        #     metric_result_str += '%s: %.4f\n' % (key, value)
        # metric_result_str += '\n'
        # print(metric_result_str)
        return metric_results
        
    def compute_official_evaluation(self, batch_dict, prediction):
        if self.config.get('eval_waymo', False):

            input_dict = batch_dict['input_dict']
            pred_scores = prediction['predicted_probability']
            pred_trajs = prediction['predicted_trajectory']
            center_objects_world = input_dict['center_objects_world'].type_as(pred_trajs)
            num_center_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape

            pred_trajs_world = common_utils.rotate_points_along_z_tensor(
                points=pred_trajs.reshape(num_center_objects, num_modes * num_timestamps, num_feat),
                angle=center_objects_world[:, 6].reshape(num_center_objects)
            ).reshape(num_center_objects, num_modes, num_timestamps, num_feat)
            pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2] + input_dict['map_center'][:,
                                                                                         None, None, 0:2]

            pred_dict_list = []

            for bs_idx in range(batch_dict['batch_size']):
                single_pred_dict = {
                    'scenario_id': input_dict['scenario_id'][bs_idx],
                    'pred_trajs': pred_trajs_world[bs_idx, :, :, 0:2].cpu().numpy(),
                    'pred_scores': pred_scores[bs_idx, :].cpu().numpy(),
                    'object_id': input_dict['center_objects_id'][bs_idx],
                    'object_type': input_dict['center_objects_type'][bs_idx],
                    'gt_trajs': input_dict['center_gt_trajs_src'][bs_idx].cpu().numpy(),
                    'track_index_to_predict': input_dict['track_index_to_predict'][bs_idx].cpu().numpy()
                }
                pred_dict_list.append(single_pred_dict)

            assert len(pred_dict_list) == batch_dict['batch_size']

            self.pred_dicts += pred_dict_list

        elif self.config.get('eval_nuscenes', False):
            from nuscenes.eval.prediction.data_classes import Prediction
            input_dict = batch_dict['input_dict']
            pred_scores = prediction['predicted_probability']
            pred_trajs = prediction['predicted_trajectory']
            center_objects_world = input_dict['center_objects_world'].type_as(pred_trajs)

            num_center_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape

            pred_trajs_world = common_utils.rotate_points_along_z_tensor(
                points=pred_trajs.reshape(num_center_objects, num_modes * num_timestamps, num_feat),
                angle=center_objects_world[:, 6].reshape(num_center_objects)
            ).reshape(num_center_objects, num_modes, num_timestamps, num_feat)
            pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2] + input_dict['map_center'][:,
                                                                                         None, None, 0:2]
            pred_dict_list = []

            for bs_idx in range(batch_dict['batch_size']):
                single_pred_dict = {
                    'instance': input_dict['scenario_id'][bs_idx].split('_')[1],
                    'sample': input_dict['scenario_id'][bs_idx].split('_')[2],
                    'prediction': pred_trajs_world[bs_idx, :, 4::5, 0:2].cpu().numpy(),
                    'probabilities': pred_scores[bs_idx, :].cpu().numpy(),
                }

                pred_dict_list.append(
                    Prediction(instance=single_pred_dict["instance"], sample=single_pred_dict["sample"],
                               prediction=single_pred_dict["prediction"],
                               probabilities=single_pred_dict["probabilities"]).serialize())

            self.pred_dicts += pred_dict_list
        
        elif self.config.get('eval_argoverse2', False):

            input_dict = batch_dict['input_dict']
            pred_scores = prediction['predicted_probability']
            pred_trajs = prediction['predicted_trajectory']
            center_objects_world = input_dict['center_objects_world'].type_as(pred_trajs)
            num_center_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape

            pred_trajs_world = common_utils.rotate_points_along_z_tensor(
                points=pred_trajs.reshape(num_center_objects, num_modes * num_timestamps, num_feat),
                angle=center_objects_world[:, 6].reshape(num_center_objects)
            ).reshape(num_center_objects, num_modes, num_timestamps, num_feat)
            pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2] + input_dict['map_center'][:,
                                                                                         None, None, 0:2]

            pred_dict_list = []

            for bs_idx in range(batch_dict['batch_size']):
                single_pred_dict = {
                    'scenario_id': input_dict['scenario_id'][bs_idx],
                    'pred_trajs': pred_trajs_world[bs_idx, :, :, 0:2].cpu().numpy(),
                    'pred_scores': pred_scores[bs_idx, :].cpu().numpy(),
                    'object_id': input_dict['center_objects_id'][bs_idx],
                    'object_type': input_dict['center_objects_type'][bs_idx],
                    'gt_trajs': input_dict['center_gt_trajs_src'][bs_idx].cpu().numpy(),
                    'track_index_to_predict': input_dict['track_index_to_predict'][bs_idx].cpu().numpy()
                }
                pred_dict_list.append(single_pred_dict)

            assert len(pred_dict_list) == batch_dict['batch_size']

            self.pred_dicts += pred_dict_list

    def log_info(self, batch, batch_idx, prediction, f_actual = None, 
                f_counterfactual = None,causal_effects = None ,regularizer = None, nb_cf = None,status='train'):
        """
        Logs various metrics and losses for a batch during training or validation.

        Parameters:
        - batch: dict, input batch containing ground truth and input data.
        - batch_idx: int, index of the current batch.
        - prediction: dict, model predictions including trajectory and probability outputs.
        - f_actual: torch.Tensor, factual embeddings for the batch.
        - f_counterfactual: torch.Tensor, counterfactual embeddings for the batch.
        - causal_effects: torch.Tensor, causal effect values for the batch.
        - regularizer: dict, containing the regularization loss and its name.
        - nb_cf: torch.Tensor, number of counterfactuals for each batch.
        - status: str, either 'train' or 'val' indicating the current mode.
        """
    
        # Extract ground truth trajectories and masks
        inputs = batch['input_dict']
        gt_traj = inputs['center_gt_trajs'].unsqueeze(1)  # .transpose(0, 1).unsqueeze(0)
        print("gt_traj:", gt_traj.shape)
        gt_traj_mask = inputs['center_gt_trajs_mask'].unsqueeze(1)
        center_gt_final_valid_idx = inputs['center_gt_final_valid_idx']

        # Extract predicted trajectories and probabilities
        predicted_traj = prediction['predicted_trajectory']
        predicted_prob = prediction['predicted_probability'].detach().cpu().numpy()
       
        # --- ADE Calculation ---
        ade_diff = torch.norm(predicted_traj[:, :, :, :2] - gt_traj[:, :, :, :2], 2, dim=-1)
        ade_losses = torch.sum(ade_diff * gt_traj_mask, dim=-1) / torch.sum(gt_traj_mask, dim=-1)
        ade_losses = ade_losses.cpu().detach().numpy()
        minade = np.min(ade_losses, axis=1)
        
        # --- FDE Calculation ---
        bs, modes, future_len = ade_diff.shape
        center_gt_final_valid_idx = center_gt_final_valid_idx.view(-1, 1, 1).repeat(1, modes, 1).to(torch.int64)
        fde = torch.gather(ade_diff, -1, center_gt_final_valid_idx).cpu().detach().numpy().squeeze(-1)
        minfde = np.min(fde, axis=-1)

        # Identify the best mode based on minimum FDE
        best_fde_idx = np.argmin(fde, axis=-1)
        predicted_prob = predicted_prob[np.arange(bs), best_fde_idx]
        miss_rate = (minfde > 2.0)
        brier_fde = minfde + np.square(1 - predicted_prob)

        # --- Loss Dictionary ---
        loss_dict = {
            'minADE6': minade,
            'minFDE6': minfde,
            'miss_rate': miss_rate.astype(np.float32),
            'brier_fde': brier_fde}
        
         # --- ACE Calculation (if embeddings are used) ---
        if self.return_embeddings:    
            NC_ACE, DC_ACE, ACE = self.ACEs(f_actual, f_counterfactual, causal_effects, nb_cf)

            loss_dict.update({
                'avg_NC_ACE': np.float32(NC_ACE),
                'avg_DC_ACE': np.float32(DC_ACE),
                'ACE-avg_ACE': np.float32(ACE),
                regularizer["name"]: regularizer["value"],
            })

        # --- Dataset-Specific Logging ---
        important_metrics = list(loss_dict.keys())
        new_dict = {}
        dataset_names = inputs['dataset_name']
        unique_dataset_names = np.unique(dataset_names)
        
        # Log metrics per dataset
        for dataset_name in unique_dataset_names:
            batch_idx_for_this_dataset = np.argwhere([n == str(dataset_name) for n in dataset_names])[:, 0]
            for key in loss_dict.keys():
                if isinstance(loss_dict[key], (list, np.ndarray)):
                    new_dict[dataset_name + '/' + key] = loss_dict[key][batch_idx_for_this_dataset]
                else:
                    # Directly assign the scalar value if it's not an array
                    new_dict[dataset_name + '/' + key] = loss_dict[key]
                    
        loss_dict.update(new_dict)
        
        # --- Normalize and Finalize Loss Dictionary ---
        size_dict = {key: (len(value) if isinstance(value, (list, np.ndarray)) else 1) 
                    for key, value in loss_dict.items()}
        loss_dict = {
            key: value.detach().cpu().numpy().mean() if isinstance(value, torch.Tensor) 
            else np.mean(value) if isinstance(value, (list, np.ndarray)) 
            else value
            for key, value in loss_dict.items()
        }

        #if self.return_embeddings:
            # --- Logging to CSV (if embeddings are used) ---
            #csv_path = os.path.join("training_logs_whole_sim.csv")  # Path for CSV logging
            #self.log_metrics_to_csv(csv_path, batch_idx, loss_dict)  # Log to CSV
        
        # --- Log to WandB or Console ---
        for k, v in loss_dict.items():
            if isinstance(v, np.float64):
                v = np.float32(v)
            self.log(status + "/" + k, v, on_step=False, on_epoch=True, sync_dist=False, batch_size=size_dict[k])
            
        return
    
    def log_metrics_to_csv(self,file_path, epoch, metrics_dict):
        """
        Logs metrics to a CSV file. If the file doesn't exist, it creates a new one with headers.

        Parameters:
        - file_path (str): Path to the CSV file where metrics will be stored.
        - epoch (int): Current epoch number for logging.
        - metrics_dict (dict): Dictionary containing the metrics to be logged, where keys are metric names, and values are the metric values.
        """
        
        # Check if the file already exists to determine whether to write headers
        file_exists = os.path.isfile(file_path)
        
        try: 
            # Open the file in append mode ('a') to keep adding entries after each epoch
            with open(file_path, mode='a') as file:
                # Initialize the CSV writer with the keys as column headers
                writer = csv.DictWriter(file, fieldnames=['epoch'] + list(metrics_dict.keys()))
                
                # Write the header only if the file is new
                if not file_exists:
                    writer.writeheader()
                
                 # Prepare the row with the epoch and metrics
                row = {'epoch': epoch}
                row.update(metrics_dict)
                
                # Write the metrics row to the file
                writer.writerow(row)
                
        except Exception as e:
            print(f"Error while writing metrics to CSV: {e}")
    
    def ACEs(self, f_actual, f_counterfactual, causal_effects, nb_cf, non_causal_thresh=40, causal_thresh=450):
        """
        Calculate Average Causal Effects (ACE) using factual and counterfactual outputs.
        Excludes indirect causal pairs (IC) and computes ACE for non-causal and direct causal pairs.

        Parameters:
        - f_actual (Tensor): Shape [B, T, 2]. Factual predictions for each batch.
        - f_counterfactual (list of Tensors): Each [num_cf, T, 2]. Counterfactual predictions for each batch.
        - causal_effects (Tensor): Shape [B, num_cf]. Causal effect values for each counterfactual.
        - nb_cf (Tensor): Number of counterfactuals per batch.
        - non_causal_thresh (float): Threshold for non-causal classification. Default is 40.
        - causal_thresh (float): Threshold for causal classification. Default is 450.

        Returns:
        - avg_NC_ACE (float): Average Non-Causal ACE across batches.
        - avg_DC_ACE (float): Average Direct Causal ACE across batches.
        - avg_ACE (float): Overall Average ACE across all pairs and batches.
        """

        # Initialize lists to store ACE values for each batch
        NC_ACE, DC_ACE, all_ACE = [], [], []

        # Iterate over each batch for ACE calculation
        for batch_idx in range(len(f_actual)):
            # Get the number of counterfactuals for the current batch
            num_cf = nb_cf[batch_idx].item()
            
            # Extract factual and counterfactual predictions
            f_pred = f_actual[batch_idx]  # [T, 2]
            cf_preds = f_counterfactual[batch_idx][:num_cf]  # [num_cf, T, 2]
            causal_effects_batch = causal_effects[batch_idx, :num_cf]

            # Calculate the sensitivity (L2 distance) between factual and counterfactual trajectories
            # Sensitivity measures how far counterfactual trajectories deviate from factual ones
            sensitivities = torch.norm(cf_preds - f_pred.unsqueeze(0), dim=-1).mean(dim=(1, 2))

            # Identify non-causal and direct causal pairs using causal effect thresholds
            NC_mask = causal_effects_batch <= non_causal_thresh
            DC_mask = causal_effects_batch >= causal_thresh

            # Compute and store Non-Causal ACE if valid pairs exist
            if NC_mask.sum() > 0:
                nc_values = torch.abs(sensitivities[NC_mask] - causal_effects_batch[NC_mask])
                NC_ACE.append(nc_values.mean())  

            # Compute and store Direct Causal ACE if valid pairs exist
            if DC_mask.sum() > 0:
                dc_values = torch.abs(sensitivities[DC_mask] - causal_effects_batch[DC_mask])
                DC_ACE.append(dc_values.mean())

            # Compute and store overall ACE for the batch (average across all counterfactuals)
            all_ACE.append(torch.abs(sensitivities - causal_effects_batch).mean())

        # Convert lists to tensors, defaulting to zero if no valid pairs were found
        NC_ACE = torch.stack(NC_ACE) if NC_ACE else torch.tensor([0.0], device=f_actual.device)
        DC_ACE = torch.stack(DC_ACE) if DC_ACE else torch.tensor([0.0], device=f_actual.device)
        all_ACE = torch.stack(all_ACE) if all_ACE else torch.tensor([0.0], device=f_actual.device)

        # Compute final averages and detach from the computation graph
        avg_NC_ACE = NC_ACE.mean().detach().cpu().item()
        avg_DC_ACE = DC_ACE.mean().detach().cpu().item()
        avg_ACE = all_ACE.mean().detach().cpu().item()
        

        # Return scalar values representing the average ACE metrics
        return avg_NC_ACE, avg_DC_ACE, avg_ACE
    
    
