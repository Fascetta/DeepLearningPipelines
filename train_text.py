import argparse
from my_lib import *
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


def train_epoch(epoch, model, loader, criterion) -> dict:
    n_iters = len(loader)
    model.train()

    mean_loss = RunningMean()
    for iter, batch in enumerate(loader):
        input_ids = torch.stack(batch["input_ids"], 1).to(cfg.device)
        attention_mask = torch.stack(batch["attention_mask"], 1).to(cfg.device)
        label_batch = batch["label"].to(cfg.device)

        logits = model(input_ids, attention_mask)

        loss = criterion(logits, label_batch)

        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_sched.step()

        loss_value = loss.item()
        mean_loss.update(loss_value)

        if iter % int(n_iters * 0.10) == 0:
            lr_value = optim.param_groups[0]["lr"]
            print(
                f"{epoch} Train loss = {loss_value} ,  mean = {mean_loss}, learning_rate = {lr_value}"
            )

    results = {"mean_loss": mean_loss}

    return results


def eval_epoch(epoch, model, loader, criterion) -> dict:
    model.eval()
    y_pred_list = []
    prob_pred_list = []
    y_true_list = []
    mean_loss = RunningMean()

    for iter, batch in enumerate(loader):
        input_ids = torch.stack(batch["input_ids"], 1).to(cfg.device)
        attention_mask = torch.stack(batch["attention_mask"], 1).to(cfg.device)
        label_batch = batch["label"].to(cfg.device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)

        y_pred_list += list(torch.argmax(logits, dim=1).cpu().numpy())
        prob_pred_list += [torch.softmax(logits, dim=1).cpu().numpy()]

        y_true_list += list(label_batch.cpu().numpy())

        loss = criterion(logits, label_batch)

        mean_loss.update(loss.item())

    score = accuracy_score(y_true_list, y_pred_list)
    print(f"{epoch} Validation mean loss = {mean_loss}")
    print(f"{epoch} Validation Accuracy = {score}")

    # use vstack to stack vertically (B, num_classes) numpy arrays
    prob_pred_list = np.vstack(prob_pred_list)
    results = {"acc": score, "probabilities": prob_pred_list, "mean_loss": mean_loss}

    return results


def test_epoch(epoch, model, loader) -> dict:
    model.eval()
    y_pred_list = []
    prob_pred_list = []

    for iter, batch in enumerate(loader):
        input_ids = torch.stack(batch["input_ids"], 1).to(cfg.device)
        attention_mask = torch.stack(batch["attention_mask"], 1).to(cfg.device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)

        y_pred_list += list(torch.argmax(logits, dim=1).cpu().numpy())
        prob_pred_list += [torch.softmax(logits, dim=1).cpu().numpy()]

    # use vstack to stack vertically (B, num_classes) numpy arrays
    prob_pred_list = np.vstack(prob_pred_list)
    results = {"probabilities": prob_pred_list, "prediction": y_pred_list}

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=False, default=None)
    args = parser.parse_args()

    if args.cfg is not None:
        print("Loading config...")
        cfg = TextConfig.from_json(args.cfg)
    else:
        cfg = TextConfig()

    seed_everything(cfg.seed)
    # =============== Model ========================
    model = NLPClassificationModel(cfg.model_name, cfg.num_classes, cfg.pretrained)

    # ============= Dataset ========================
    df = pd.read_csv(cfg.csv_train_file)
    split = pd.read_csv(cfg.csv_split_file)
    if cfg.fold == -1:
        df_train = df
        evaluate = False
    else:
        df_train = df.loc[split["fold"] != cfg.fold, :]
        df_val = df.loc[split["fold"] == cfg.fold, :]
        evaluate = True

    df_test = pd.read_csv(cfg.csv_test_file)

    train_ds = dataset_from_pandas(df_train, cfg.model_name, cfg.batched, test=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        drop_last=True,
    )

    if evaluate:
        val_ds = dataset_from_pandas(df_val, cfg.model_name, cfg.batched, test=False)
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.test_batch_size,
            num_workers=cfg.num_workers,
            shuffle=False,
        )

    test_ds = dataset_from_pandas(df_test, cfg.model_name, cfg.batched, test=True)

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.test_batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
    )

    # ============= Optimizer ================
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    total_iterations = cfg.num_epochs * len(train_loader)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optim, T_max=total_iterations, eta_min=cfg.lr * 0.01
    )
    criterion = nn.CrossEntropyLoss()

    output_folder = f"outputs/{get_output_folder(cfg.project_name)}"
    os.makedirs(output_folder, exist_ok=True)

    # ============ Train Algorithm ============
    model.to(cfg.device)
    for epoch in range(cfg.num_epochs):
        train_results = train_epoch(epoch, model, train_loader, criterion)
        if evaluate:
            val_results = eval_epoch(epoch, model, val_loader, criterion)
            # Save validation prediction (to do error analysis or ensamble models)
            df_val_pred = df_val.copy()
            df_val.loc[:, [f"prob_{i}" for i in range(cfg.num_classes)]] = val_results[
                "probabilities"
            ]
            eval_score = val_results["acc"]

            df_val.to_csv(
                os.path.join(
                    output_folder,
                    f"fold_{cfg.fold}_valpred_{epoch}_{eval_score:.4f}.csv",
                ),
                index=False,
            )

        else:
            eval_score = -1

        torch.save(
            model.state_dict(),
            os.path.join(
                output_folder, f"fold_{cfg.fold}_model_{epoch}_{eval_score:.4f}.pth"
            ),
        )

        test_results = test_epoch(epoch, model, test_loader)

        # Save test prediction (useful for doing ensambles)
        df_test_pred = df_test.copy()
        df_test.loc[:, [f"prob_{i}" for i in range(cfg.num_classes)]] = test_results[
            "probabilities"
        ]
        df_test.to_csv(
            os.path.join(
                output_folder, f"fold_{cfg.fold}_testpred_{epoch}_{eval_score:.4f}.csv"
            ),
            index=False,
        )
