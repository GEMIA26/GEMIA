import torch
from src.models import *
from src.utils import AverageMeterSet, absolute_recall_mrr_ndcg_for_ks
from tqdm import tqdm


def train(
    model,
    optimizer,
    dataloader,
    criterion,
    scheduler,
    alpha: float = None,
    beta: float = None,
    device: str = "cpu",
):
    def filter_data(t, mask):
        # (B, L, D) -> (B*L, D) -> (N, D) -> (N, 1, D)
        return t.reshape(-1, t.shape[-1])[mask].unsqueeze(1)

    model.train()

    total_loss = 0
    img_loss_list = []
    text_loss_list = []
    L_rec_list = []

    with tqdm(dataloader) as t:
        for tokens, labels, img_emb, gen_img_emb, text_emb, qeury_emb in t:
            optimizer.zero_grad()

            tokens = tokens.to(device)
            labels = labels.to(device)
            img_emb = img_emb.to(device)
            gen_img_emb = gen_img_emb.to(device)
            text_emb = text_emb.to(device)
            qeury_emb = qeury_emb.to(device)

            ## Forward
            (pred_img, pred_text, logits) = model(tokens, img_emb, text_emb)

            tokens_flat = tokens.reshape(-1)
            valid_mask = tokens_flat != 0  
            valid_ids = tokens_flat[valid_mask]  
            ## cal loss
            p_img = filter_data(pred_img, valid_mask)
            i_emb = filter_data(img_emb, valid_mask)
            p_txt = filter_data(pred_text, valid_mask)
            t_emb = filter_data(text_emb, valid_mask)
            q_emb = filter_data(qeury_emb, valid_mask)
            g_img = filter_data(gen_img_emb, valid_mask)

            L_cl2_V = clip_loss(p_img, i_emb, valid_ids, model.logit_scale)
            L_cl2_T = clip_loss(p_txt, t_emb, valid_ids, model.logit_scale)
            L_cl_T = clip_loss(p_txt, q_emb, valid_ids, model.logit_scale)
            L_cl_V = clip_loss(p_img, g_img, valid_ids, model.logit_scale)

            img_loss_list.append(alpha * L_cl_V + beta * L_cl2_V)
            text_loss_list.append(alpha * L_cl_T + beta * L_cl2_T)

            logits[:, :, 0] = -1e9  # remove padding index

            L_rec = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            L_rec_list.append(L_rec)

            contra_loss = img_loss_list[-1] + text_loss_list[-1]
            loss = L_rec_list[-1] + contra_loss

            t.set_postfix({"loss": loss.item()})

            total_loss += loss.item()

            loss.backward()
            optimizer.step()

    if scheduler is not None:
        scheduler.step()

    return (
        total_loss / len(dataloader),
        torch.mean(torch.tensor(img_loss_list)),
        torch.mean(torch.tensor(text_loss_list)),
        torch.mean(torch.tensor(L_rec_list)),
    )


def eval(
    model,
    dataloader,
    device: str = "cpu",
):
    model.eval()

    def _update_meter_set(meter_set, metrics):
        for k, v in metrics.items():
            meter_set.update(k, v)

    def _update_dataloader_metrics(tqdm_dataloader, meter_set):
        description_metrics = ["N%d" % k for k in [1, 10, 20]] + [
            "R%d" % k for k in [1, 10, 20]
        ]
        description = "Eval: " + ", ".join(s + " {:.4f}" for s in description_metrics)
        description = description.format(
            *(meter_set[k].avg for k in description_metrics)
        )
        tqdm_dataloader.set_description(description)

    average_meter_set = AverageMeterSet()

    with torch.no_grad():
        with tqdm(dataloader) as t:
            for tokens, labels, img_emb, gen_img_emb, text_emb, qeury_emb in t:
                tokens = tokens.to(device)
                labels = labels.to(device)
                img_emb = img_emb.to(device)
                gen_img_emb = gen_img_emb.to(device)
                text_emb = text_emb.to(device)
                qeury_emb = qeury_emb.to(device)

                ## forward
                (_, _, logits) = model(tokens, img_emb, text_emb)

                logits = logits[:, -1, :]
                labels = labels.squeeze()

                if isinstance(model, (AGMM)):
                    for u, i in enumerate(tokens):
                        logits[u, i] = -1e9 
                    logits[:, 0] = -1e9

                metrics = absolute_recall_mrr_ndcg_for_ks(logits, labels)
                _update_meter_set(average_meter_set, metrics)
                _update_dataloader_metrics(t, average_meter_set)

    average_metrics = average_meter_set.averages()

    return average_metrics
