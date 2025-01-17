import torch
import numpy as np

def calc_contrastive_loss(embeds, causal_effects, data_splits, contrastive_weight=1.0, tau=0.2):
    infonces = []
    for sample_id in range(len(causal_effects)):
        q = embeds[data_splits[sample_id]]
        positives = np.where(causal_effects[sample_id].cpu().numpy() <= 0.02)[0]
        if len(positives) > 0:
            positive_id = np.random.choice(positives, 1)[0]
            k_plus = embeds[data_splits[sample_id] + 1 + positive_id]
            k_negs = embeds[data_splits[sample_id] + 1 + np.where(causal_effects[sample_id].cpu().numpy() >= 0.1)[0]]

            numerator = torch.matmul(q, k_plus) / tau
            denominator = torch.exp(numerator) + torch.sum(torch.exp(torch.matmul(k_negs, q) / tau))
            infonces.append(-numerator + torch.log(denominator))
    if len(infonces) == 0:
        return torch.Tensor(1)
    contrastive_loss = torch.mean(torch.stack(infonces)) * contrastive_weight
    # breakpoint()
    return contrastive_loss

def calc_ranking_loss(embeds, causal_effects, data_splits, ranking_weight=1.0, margin=0.001):
    ranking_losses = []
    for sample_id in range(len(causal_effects)):
        # Compute the distance of counterfactuals to the factual scene
        q = embeds[data_splits[sample_id]]
        keys = embeds[data_splits[sample_id] + 1:data_splits[sample_id + 1]]
        keys = keys[torch.argsort(causal_effects[sample_id])]
        dists = torch.matmul(keys, q)

        # Compute the difference of causal effect matrix
        ce = causal_effects[sample_id][torch.argsort(causal_effects[sample_id])]
        ce_row = ce.unsqueeze(1)
        ce_col = ce.unsqueeze(0)
        diff_ce = ce_col - ce_row

        # Create a mask for values between 0.2 and 0.5
        mask = (diff_ce >= 0.2) & (diff_ce <= 0.5)

        # Take the indices of the columns that are valid according to the mask
        cols = torch.arange(mask.size(1)).expand_as(mask).to(mask.device)
        valid_cols = cols[mask]

        # Calculate the number of valid columns per row which is the number of elements in the valid_cols for each row
        lengths = mask.sum(dim=1)
        if lengths.max().item() == 0:
            continue  # The mask is empty, skip this sample

        # Make the offsets for valid_cols, with offset[i] pointing out to the first element in valid_cols for row i
        offsets = torch.cat((torch.tensor([0]).to(mask.device), lengths.cumsum(dim=0)[:-1]))

        # Create a random index for each row and select the corresponding column from valid_cols
        rand_indices = torch.randint_like(lengths, 0, lengths.max().item()).to(mask.device)
        rand_indices = rand_indices % lengths
        selection = (offsets + rand_indices).long()
        selected_cols = valid_cols[selection[lengths > 0]]

        # Calculate the ranking loss
        ranking_losses.append(torch.nn.MarginRankingLoss(margin=margin)(dists[selected_cols], dists[lengths > 0], torch.ones((lengths > 0).sum()).to(dists.device)))

    if len(ranking_losses) == 0:
        return torch.Tensor(1)
    ranking_loss = torch.mean(torch.stack(ranking_losses)) * ranking_weight
    # breakpoint()
    return ranking_loss

def nll_loss_multimodes(pred, data, modes_pred, entropy_weight=1.0, kl_weight=1.0, use_FDEADE_aux_loss=True):
    """NLL loss multimodes for training. MFP Loss function
    Args:
      pred: [K, T, B, 5]
      data: [B, T, 5]
      modes_pred: [B, K], prior prob over modes
      noise is optional
    """
    modes = len(pred)
    nSteps, batch_sz, dim = pred[0].shape

    # compute posterior probability based on predicted prior and likelihood of predicted trajectory.
    log_lik = np.zeros((batch_sz, modes))
    with torch.no_grad():
        for kk in range(modes):
            nll = nll_pytorch_dist(pred[kk].transpose(0, 1), data, rtn_loss=False)
            log_lik[:, kk] = -nll.cpu().numpy()

    priors = modes_pred.detach().cpu().numpy()
    log_posterior_unnorm = log_lik + np.log(priors)
    log_posterior = log_posterior_unnorm - special.logsumexp(log_posterior_unnorm, axis=-1).reshape((batch_sz, -1))
    post_pr = np.exp(log_posterior)
    post_pr = torch.tensor(post_pr).float().to(data.device)
    post_entropy = torch.mean(D.Categorical(post_pr).entropy()).item()

    # Compute loss.
    loss = 0.0
    for kk in range(modes):
        nll_k = nll_pytorch_dist(pred[kk].transpose(0, 1), data, rtn_loss=True) * post_pr[:, kk]
        loss += nll_k.mean()

    # Adding entropy loss term to ensure that individual predictions do not try to cover multiple modes.
    entropy_vals = []
    for kk in range(modes):
        entropy_vals.append(get_BVG_distributions(pred[kk]).entropy())
    entropy_vals = torch.stack(entropy_vals).permute(2, 0, 1)
    entropy_loss = torch.mean((entropy_vals).sum(2).max(1)[0])
    loss += entropy_weight * entropy_loss

    # KL divergence between the prior and the posterior distributions.
    kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')  # type: ignore
    kl_loss = kl_weight*kl_loss_fn(torch.log(modes_pred), post_pr)

    # compute ADE/FDE loss - L2 norms with between best predictions and GT.
    if use_FDEADE_aux_loss:
        adefde_loss = l2_loss_fde(pred, data)
    else:
        adefde_loss = torch.tensor(0.0).to(data.device)

    return loss, kl_loss, post_entropy, adefde_loss