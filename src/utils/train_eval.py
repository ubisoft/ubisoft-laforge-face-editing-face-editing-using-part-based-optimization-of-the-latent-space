import time
import torch
import torch.nn.functional as F


def run(model, train_loader, epochs, optimizer, scheduler, writer, device, part_verts, part_latent_size, beta=1e-3, ceta=1.0, save_all=False, dont_save=False):

    B = beta
    C = ceta
    last_loss = 1e10

    for epoch in range(1, epochs + 1):
        t_start = time.time()

        train_loss, rec_loss, kl_loss, control_loss = train(model, optimizer, train_loader, device,
                                                            B, C, part_verts, part_latent_size)

        t_duration = time.time() - t_start

        if scheduler != None:
            scheduler.step()

        info = {
            'current_epoch': epoch,
            'epochs': epochs,
            'train_loss': train_loss,
            'rec_loss': rec_loss,
            'kl_loss': kl_loss,
            'c_loss': control_loss,
            't_duration': t_duration
        }

        C *= 1.1
        writer.print_info(info)

        new_loss = round(train_loss, 5)
        if new_loss < last_loss or save_all:
            last_loss = new_loss
            if not dont_save:
                writer.save_checkpoint(model, optimizer, scheduler, epoch)


def train(model, optimizer, loader, device, B, C, part_verts, part_latent_size):
    model.train()

    kl_losses = 0
    rec_losses = 0
    total_loss = 0
    total_control = 0

    for data in loader:
        optimizer.zero_grad()
        for i in range(0, len(data)):
            data[i] = data[i].to(device)

        enc, mus, sigs = model.encoder(data)
        out = model.decoder(enc)
        loss, rec_loss, kl_loss, control_loss = loss_fn(
            out, data[0], mus, sigs, B, C, part_verts, part_latent_size, model, enc)

        loss.backward()
        optimizer.step()

        kl_losses += kl_loss.item()
        rec_losses += rec_loss.item()
        total_loss += loss.item()
        total_control += control_loss.item()

    return total_loss / len(loader), rec_losses / len(loader), kl_losses / len(loader), total_control / len(loader)


def loss_fn(out, x, mus, sigs, B, C, part_verts, part_latent_size, model, enc):
    rec_loss = F.l1_loss(out, x, reduction='mean')
    kl_loss = 0
    for mu, sig in zip(mus, sigs):
        kl_loss += torch.mean(-0.5 * torch.sum(1 + sig - mu ** 2 - sig.exp(), dim=1), dim=0)
    # ---------------------------
    control_loss = 0
    for idx, verts in enumerate(part_verts):
        pstart = idx * part_latent_size
        pend = idx * part_latent_size + part_latent_size
        new_enc = enc.detach().clone()
        new_enc[:, pstart:pend] = torch.FloatTensor(enc.shape[0], part_latent_size).uniform_(-10, 10)
        new_out = model.decoder(new_enc)
        new_out[:, verts] = out[:, verts].detach().clone()  # to make the loss zero
        control_loss += F.l1_loss(out, new_out, reduction='mean')
    # ---------------------------

    loss = rec_loss + (B * kl_loss) + (C * control_loss)
    return loss, rec_loss, kl_loss, control_loss
