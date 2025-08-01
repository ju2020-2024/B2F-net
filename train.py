import torch
from tqdm import tqdm

def train(train_loader, model, optimizer, criterion,args):
    model.train()

    total_loss =[]

    total_loss_2 = []
    main_losses = []
    aux_losses = []

    for i, (inputs, labels) in tqdm(enumerate(train_loader)):
        seq_len = torch.sum(torch.max(torch.abs(inputs), dim=2)[0] > 0, 1)
        inputs = inputs[:, :torch.max(seq_len), :]
        inputs = inputs.float().to(args.device)
        labels = labels.float().to(args.device)

        out = model(inputs,seq_len)
        s1, s2, s3 = out["video_scores"]

        loss_main = criterion(s3, labels)
        loss1 = criterion(s1, labels)
        loss2 = criterion(s2, labels)
        loss_aux = loss1 + loss2
        loss = loss_main + args.gamma * loss_aux

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.append(loss)


    return sum(total_loss)/len(total_loss)
