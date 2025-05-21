
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscreteDiffusion(nn.Module):
    def __init__(self, steps, device):
        super(DiscreteDiffusion, self).__init__()
        self.steps = steps
        self.device = device

    def p_sample2(self, model, x_start, steps, sampling_noise=False):
        """
        Deterministic discrete denoising. Model learns to predict x_{t-1} from x_t.
        """
        x_t = x_start
        
        for i in reversed(range(steps)):
            t = torch.full((x_t.size(0),), i, dtype=torch.long).to(x_t.device)
            x_t = model(x_t, t)
        return x_t
    def p_sample4(self, model, x_start, steps, sampling_noise=False):
        x_t = x_start
        for i in reversed(range(steps)):
            t = torch.full((x_t.size(0),), i, dtype=torch.long).to(x_t.device)
            mask_ratio = (t.float() / self.steps).view(-1, 1)
            mask_ratio = 0.3
            mask = torch.rand_like(x_t) < mask_ratio
            mask_ratio = 0.3
            x_t_noised = x_t.clone()
            x_t_noised[mask] = 0.0
            x_t = model(x_t_noised, t)
        return x_t
    def p_sample(self, model, x_start, steps, sampling_noise=False):
        x_t = x_start
        for i in reversed(range(steps)):
            t = torch.full((x_t.size(0),), i, dtype=torch.long).to(x_t.device)
            # Sabit yüksek oranlı maskeleme (örneğin %30)
            #mask_ratio = torch.full((x_t.size(0), 1), 0.3, device=x_t.device)
            mask_ratio = (t.float() / self.steps).view(-1, 1)
            mask = torch.rand_like(x_t) < mask_ratio
            noise = torch.randn_like(x_t)
            x_t_noised = x_t.clone()
            x_t_noised[mask] = noise[mask]

            x_t = model(x_t_noised, t)
        return x_t


    def training_losses2(self, model, x_start, reweight=False):
        batch_size = x_start.size(0)
        t = torch.randint(0, self.steps, (batch_size,), device=x_start.device)
        mask = torch.rand_like(x_start) < (t.view(-1, 1) / self.steps)
        x_t = x_start.clone()
        noise = torch.randn_like(x_start)
        x_t[mask] = noise[mask]

        x_pred = model(x_t, t)
        loss = F.mse_loss(x_pred, x_start, reduction='mean')

        return {"loss": loss}
    def training_losses3(self, model, x_start, reweight=False):
        batch_size = x_start.size(0)
        t = torch.randint(0, self.steps, (batch_size,), device=x_start.device)

        # Random mask (discrete gibi davranır)
        mask_prob = (t.float() / self.steps).view(-1, 1)  # e.g. 0.2 -> %20 mask
        mask = torch.rand_like(x_start) < mask_prob
        noise = torch.randn_like(x_start)
        x_t = x_start.clone()
        x_t[mask] = noise[mask]

        x_pred = model(x_t, t)
        loss = F.mse_loss(x_pred, x_start, reduction='mean')
        return {"loss": loss}
    def training_losses5(self, model, x_start, reweight=False):
        batch_size = x_start.size(0)

        # Lazy init mask token
        if not hasattr(self, "mask_token"):
            self.mask_token = nn.Parameter(torch.randn(x_start.shape[1]).to(x_start.device), requires_grad=True)

        t = torch.randint(0, self.steps, (batch_size,), device=x_start.device)
        mask_ratio = (t.float() / self.steps).view(-1, 1)
        mask = torch.rand_like(x_start) < mask_ratio

        x_t = x_start.clone()
        x_t[mask] = self.mask_token[None, :].expand_as(x_start)[mask]

        x_pred = model(x_t, t)
        loss = 1 - F.cosine_similarity(x_pred, x_start, dim=-1).mean()
        return {"loss": loss}


    def training_losses7(self, model, x_start, reweight=False):
        batch_size, emb_dim = x_start.size()
        t = torch.randint(0, self.steps, (batch_size,), device=x_start.device)

        # 🔢 mask_ratio hesapla ve expand et
        mask_ratio = (t.float() / self.steps).view(-1, 1)  # [B, 1]

        # 🎯 user ve item parçalarını ayır
        emb_size = emb_dim // 2
        user_part = x_start[:, :emb_size].clone()
        item_part = x_start[:, emb_size:].clone()

        # 🎭 mask oluştur
        user_mask = torch.rand_like(user_part) < mask_ratio
        item_mask = torch.rand_like(item_part) < mask_ratio

        # 🚫 maskeleme uygula
        user_part[user_mask] = 0.0
        item_part[item_mask] = 0.0

        x_t = torch.cat([user_part, item_part], dim=1)

        x_pred = model(x_t, t)

        # 📉 sadece maskelenen kısımlarda loss hesapla
        mask = torch.cat([user_mask, item_mask], dim=1)
        loss = F.mse_loss(x_pred[mask], x_start[mask])

        return {"loss": loss}

    def training_losses8(self, model, x_start, reweight=False):
        batch_size, emb_dim = x_start.size()
        t = torch.randint(0, self.steps, (batch_size,), device=x_start.device)

        mask_ratio = (t.float() / self.steps).view(-1, 1)

        emb_size = emb_dim // 2
        user_part = x_start[:, :emb_size].clone()
        item_part = x_start[:, emb_size:].clone()

        user_mask = torch.rand_like(user_part) < mask_ratio
        item_mask = torch.rand_like(item_part) < mask_ratio

        # ❌ in-place yok, ✅ out-of-place:
        user_part = user_part.masked_fill(user_mask, 0.0)
        item_part = item_part.masked_fill(item_mask, 0.0)

        x_t = torch.cat([user_part, item_part], dim=1)
        x_pred = model(x_t, t)

        mask = torch.cat([user_mask, item_mask], dim=1)
        loss = F.mse_loss(x_pred[mask], x_start[mask])

        return {"loss": loss}
    
    def training_losses9(self, model, x_start, reweight=False):
        batch_size, emb_dim = x_start.size()
        t = torch.randint(0, self.steps, (batch_size,), device=x_start.device)

        # x_start'tan detached bir kopya al (in-place'e karşı koruma)
        x_start_detached = x_start.detach()

        mask_ratio = (t.float() / self.steps).view(-1, 1)

        emb_size = emb_dim // 2
        user_part = x_start_detached[:, :emb_size]
        item_part = x_start_detached[:, emb_size:]

        user_mask = torch.rand_like(user_part) < mask_ratio
        item_mask = torch.rand_like(item_part) < mask_ratio

        # maskeyi uygula ama orijinali bozmadan
        user_part_masked = user_part.masked_fill(user_mask, 0.0)
        item_part_masked = item_part.masked_fill(item_mask, 0.0)

        x_t = torch.cat([user_part_masked, item_part_masked], dim=1)

        x_pred = model(x_t, t)

        mask = torch.cat([user_mask, item_mask], dim=1)
        loss = F.mse_loss(x_pred[mask], x_start[mask])

        return {"loss": loss}
    def training_losses(self, model, x_start, reweight=False):
        batch_size, emb_dim = x_start.size()
        t = torch.randint(0, self.steps, (batch_size,), device=x_start.device)

        mask_ratio = (t.float() / self.steps).view(-1, 1)

        emb_size = emb_dim // 2
        user_part = x_start[:, :emb_size].clone()  # 🔧 in-place değil, clone ile güvenli
        item_part = x_start[:, emb_size:].clone()  # 🔧 in-place değil, clone ile güvenli

        user_mask = torch.rand_like(user_part) < mask_ratio
        item_mask = torch.rand_like(item_part) < mask_ratio

        user_part_masked = user_part.masked_fill(user_mask, 0.0)  # 🔧 in-place yerine masked_fill
        item_part_masked = item_part.masked_fill(item_mask, 0.0)  # 🔧 in-place yerine masked_fill

        x_t = torch.cat([user_part_masked, item_part_masked], dim=1)

        x_pred = model(x_t, t)

        mask = torch.cat([user_mask, item_mask], dim=1)
        loss = F.mse_loss(x_pred[mask], x_start[mask])  # 🔧 doğru gradient takibi için doğrudan kullanıldı

        return {"loss": loss}
