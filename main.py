# flake8: noqa
import argparse
import os
# import dotenv
import torch
import torch.nn as nn
import wandb
from src import datasets as DS
from src.models import * 
from src.train import eval, train
from src.utils import (
    append_log,
    dump_yaml,
    get_config,
    get_timestamp,
    load_json,
    mk_dir
)
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader


def main(settings):
    ############# SETTING #############
    # seed_everything(settings["seed"])
    timestamp = get_timestamp()

    models = {"AGMM": AGMM}

    model_name: str = settings["model_name"]
    model_args: dict = settings["model_arguments"]
    model_dataset: dict = settings["model_dataset"]
    data_name: dict = settings["data_name"]

    save_path = f"./model/{timestamp}"
    data_path = f"./data/{data_name}"

    mk_dir("./model")
    mk_dir("./data")
    mk_dir(save_path)

    settings["experiment_name"] = f"work-{timestamp}_{model_name}_{data_name}"
    dump_yaml(f"{save_path}/setting.yaml", settings)

    ############ SET HYPER PARAMS #############
    ## TRAIN ##
    lr = settings["lr"]
    epoch = settings["epoch"]
    batch_size = settings["batch_size"]
    num_workers = settings["num_workers"]

    ## ETC ##
    n_cuda = settings["n_cuda"]

    ############ WANDB INIT #############
    print("--------------- Wandb SETTING ---------------")
    if settings["use_wandb"]:
        WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
        wandb.login(key=WANDB_API_KEY)
        wandb.init(
            entity=os.environ.get("WANDB_ENTITY"),
            project="Baseline",
            name=settings["experiment_name"],
            mode=os.environ.get("WANDB_MODE"),
        )
        wandb.log(settings)

    ############# LOAD DATASET #############
    print("-------------LOAD DATA-------------")
    metadata = load_json(f"{data_path}/uniqued_metadata.json")
    test_data = torch.load(f"{data_path}/uniqued_test_data.pt")
    _parameter = {"max_len": model_args["max_len"]}

    _parameter["img_emb"] = torch.load(
        f"{data_path}/idx_img_emb_map.pt", weights_only=False
    )
    _parameter["text_emb"] = torch.load(
        f"{data_path}/idx_text_emb_map.pt", weights_only=False
    )
    model_args["d_V"] = _parameter["text_emb"][0].shape[-1]
    model_args["d_T"] = _parameter["img_emb"][0].shape[-1]

    text_embedding = torch.stack(list(_parameter["text_emb"].values()), dim=0)  # 512
    img_embedding = torch.stack(list(_parameter["img_emb"].values()), dim=0)  # 512

    _parameter["num_user"] = metadata["num of user"]
    _parameter["num_item"] = metadata["num of item"]
    model_args["num_user"] = metadata["num of user"]
    model_args["num_item"] = metadata["num of item"]

    print(f'NUM USER : {_parameter["num_user"]} | NUM ITEM : {_parameter["num_item"]}')

    print("-------------COMPLETE LOAD DATA-------------")
    train_dataset_class_ = getattr(DS, f"{model_dataset}TrainDataset")
    valid_dataset_class_ = getattr(DS, f"{model_dataset}ValidDataset")
    test_dataset_class_ = getattr(DS, f"{model_dataset}TestDataset")

    train_dataset = train_dataset_class_(
        user_seq=test_data,
        **_parameter,
    )

    valid_dataset = valid_dataset_class_(
        user_seq=test_data,
        **_parameter,
    )
    test_dataset = test_dataset_class_(
        user_seq=test_data,
        **_parameter,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=settings["shuffle"],
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers
    )

    ############# SETTING FOR TRAIN #############
    device = f"cuda:{n_cuda}" if n_cuda != "cpu" else "cpu"

    ## MODEL INIT ##
    model_class_ = models[model_name]

    model = model_class_(
        **model_args,
        device=device,
    ).to(device)

    with torch.no_grad():
        model.text_emb.weight[1:].copy_(text_embedding)
        model.text_emb.weight[0].zero_()
        model.img_emb.weight[1:].copy_(img_embedding)
        model.img_emb.weight[0].zero_()
        model.text_emb.weight.requires_grad = False
        model.img_emb.weight.requires_grad = False

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    no_decay = [
        "bias",
    ]
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": settings["weight_decay"],
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = Adam(params=optimizer_grouped_parameters, lr=lr)

    early_stopping = EarlyStopping(patience=10, delta=0)

    if settings["scheduler"] == "LambdaLR":
        scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda epoch: settings["scheduler_rate"]
            ** (epoch / settings["lr_step"]),
        )
    best_valid_metrics, best_test_metrics = {}, {}
    best_valid_metrics["best_valid_R10"] = -1
    img_loss_list = []
    text_loss_list = []
    rec_loss_list = []

    ############# TRAIN AND EVAL #############
    for i in range(epoch):
        print("-------------TRAIN-------------")
        train_loss, img_loss, text_loss, rec_loss = train(
            model=model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            criterion=criterion,
            scheduler=scheduler,
            alpha=settings["alpha"],
            beta=settings["beta"],
            device=device,
        )

        print(
            f"epoch: {i+1:3d} | train_loss: {train_loss:.4f} | "
            f"img_loss: {img_loss:.4f} | text_loss: {text_loss:.4f} | rec_loss: {rec_loss:.4f}"
        )

        img_loss_list.append(img_loss)
        text_loss_list.append(text_loss)
        rec_loss_list.append(rec_loss)

        if settings["use_wandb"]:
            wandb.log(
                {
                    "loss": train_loss,
                    "epoch_img_loss": img_loss,
                    "epoch_rec_loss": rec_loss,
                    "epoch_text_loss": text_loss,
                    "epoch": i + 1,
                }
            )

        if i % settings["valid_step"] == 0:
            print("-------------VALID-------------")
            valid_metrics = eval(
                model=model,
                dataloader=valid_dataloader,
                device=device,
            )
            print(
                (
                    f'valid_R1: {valid_metrics["R1"]} | valid_R10: {valid_metrics["R10"]} | valid_R20: {valid_metrics["R20"]} | '
                    f'valid_N1: {valid_metrics["N1"]} | valid_N10: {valid_metrics["N10"]} | valid_N20: {valid_metrics["N20"]}'
                )
            )
            if settings["use_wandb"]:
                wandb.log(
                    {
                        "epoch": i + 1,
                        "valid_R1": valid_metrics["R1"],
                        "valid_R10": valid_metrics["R10"],
                        "valid_R20": valid_metrics["R20"],
                        "valid_N1": valid_metrics["N1"],
                        "valid_N10": valid_metrics["N10"],
                        "valid_N20": valid_metrics["N20"],
                    }
                )

                if best_valid_metrics["best_valid_R10"] < valid_metrics["R10"]:
                    print(f"####### SAVE RESULT AT {i+1}#######")
                    torch.save(model.state_dict(), f"{save_path}/final_weight.pt")

                    if i % 1 == 0:
                        print("-------------EVAL-------------")
                        test_metrics = eval(
                            model=model,
                            dataloader=test_dataloader,
                            device=device,
                        )
                        print(
                            (
                                f'R1 : {test_metrics["R1"]} | R10 : {test_metrics["R10"]} | R20 : {test_metrics["R20"]} | '
                                f'N1 : {test_metrics["N1"]} | N10 : {test_metrics["N10"]} | N20 : {test_metrics["N20"]} '
                            )
                        )
                        test_metrics["epoch"] = i + 1

                        if settings["use_wandb"]:
                            wandb.log(test_metrics)

                        best_valid_metrics = valid_metrics
                        best_test_metrics = test_metrics
                        best_test_metrics["iter"] = i + 1
                        best_valid_metrics["iter"] = i + 1

                        best_valid_metrics = {
                            ("best_valid_" + k): v
                            for k, v in best_valid_metrics.items()
                        }
                        best_test_metrics = {
                            ("best_test_" + k): v for k, v in best_test_metrics.items()
                        }
                        append_log(f"{save_path}/eval.log", best_valid_metrics)
                        append_log(f"{save_path}/eval.log", best_test_metrics)
                        
        if early_stopping(valid_metrics["R10"]):
            print(f"\033[43mEARLY STOPPED!!\033[0m \ntriggered at epoch {i + 1}")
            break
        else:
            print(
                f"\033[43mKEEP GOING!!\033[0m \nEpoch {i + 1}: counter = {early_stopping.counter} | recall@10 = {valid_metrics['R10']}"
            )

    print("-------------FINAL EVAL-------------")
    if settings["use_wandb"]:
        wandb.log(best_test_metrics)
        wandb.log(best_valid_metrics)
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        action="store",
        required=False,
        default="Baby",
    )
    args = parser.parse_args()
    setting_yaml_path = f"./settings/{args.config}.yaml"
    settings = get_config(setting_yaml_path)

    main(settings)

