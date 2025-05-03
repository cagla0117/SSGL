

__all__ = ["SGL"]

import torch
from sklearn.cluster import KMeans
import numpy as np
import scipy.sparse as sp
from torch.serialization import save
import torch.sparse as torch_sp
import torch.nn as nn
import torch.nn.functional as F
from model.base import AbstractRecommender
from util.pytorch import inner_product, l2_loss
from util.pytorch import get_initializer
from util.common import Reduction
from data import PointwiseSamplerV2, PairwiseSamplerV2
import numpy as np
from time import time
from reckit import timer
import scipy.sparse as sp
from util.common import normalize_adj_matrix, ensureDir
from util.pytorch import sp_mat_to_sp_tensor
from reckit import randint_choice
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class UNetDenoiser(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.time_embed = SinusoidalPosEmb(dim)
        self.net = nn.Sequential(
            nn.Linear(dim + dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t.float())  # t: timestep
        x_in = torch.cat([x, t_emb], dim=-1)
        return self.net(x_in)

class DiffusionDenoiser(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )



    def forward(self, x, noise_level=0.1):
        noise = torch.randn_like(x) * noise_level
        x_noisy = x + noise
        x_recon = self.net(x_noisy)
        return x_recon


class _LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, norm_adj, n_layers):
        super(_LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.norm_adj = norm_adj
        self.n_layers = n_layers
        self.user_embeddings = nn.Embedding(self.num_users, self.embed_dim)
        self.item_embeddings = nn.Embedding(self.num_items, self.embed_dim)
        self.dropout = nn.Dropout(0.1)
        self._user_embeddings_final = None
        self._item_embeddings_final = None
        self.denoiser_user = DiffusionDenoiser(embed_dim)
        self.denoiser_item = DiffusionDenoiser(embed_dim)
        self.unet_user = UNetDenoiser(embed_dim)
        self.unet_item = UNetDenoiser(embed_dim)


        # # weight initialization
        # self.reset_parameters()
    def denoise_with_diffusion(self, x, unet, steps=10):
        batch_size = x.shape[0]
        for t in reversed(range(steps)):
            t_tensor = torch.full((batch_size,), t, device=x.device, dtype=torch.long)
            noise = torch.randn_like(x) * (0.1 + 0.9 * t / steps)  # β schedule
            x_noisy = x + noise
            x = unet(x_noisy, t_tensor)
        return x

    def reset_parameters(self, pretrain=0, init_method="uniform", dir=None):
        if pretrain:
            pretrain_user_embedding = np.load(dir + 'user_embeddings.npy')
            pretrain_item_embedding = np.load(dir + 'item_embeddings.npy')
            pretrain_user_tensor = torch.FloatTensor(pretrain_user_embedding).cuda()
            pretrain_item_tensor = torch.FloatTensor(pretrain_item_embedding).cuda()
            self.user_embeddings = nn.Embedding.from_pretrained(pretrain_user_tensor)
            self.item_embeddings = nn.Embedding.from_pretrained(pretrain_item_tensor)
        else:
            init = get_initializer(init_method)
            init(self.user_embeddings.weight)
            init(self.item_embeddings.weight)

    def forward(self, sub_graph1, sub_graph2, users, items, neg_items):
        user_embeddings, item_embeddings = self._forward_gcn(self.norm_adj)
        user_embeddings = self.denoiser_user(user_embeddings)
        item_embeddings = self.denoiser_item(item_embeddings)
        user_embeddings1, item_embeddings1 = self._forward_gcn(sub_graph1)
        user_embeddings2, item_embeddings2 = self._forward_gcn(sub_graph2)

        # Normalize embeddings learnt from sub-graph to construct SSL loss
        user_embeddings1 = F.normalize(user_embeddings1, dim=1)
        item_embeddings1 = F.normalize(item_embeddings1, dim=1)
        user_embeddings2 = F.normalize(user_embeddings2, dim=1)
        item_embeddings2 = F.normalize(item_embeddings2, dim=1)

        user_embs = F.embedding(users, user_embeddings)
        item_embs = F.embedding(items, item_embeddings)
        neg_item_embs = F.embedding(neg_items, item_embeddings)
        user_embs1 = F.embedding(users, user_embeddings1)
        item_embs1 = F.embedding(items, item_embeddings1)
        user_embs2 = F.embedding(users, user_embeddings2)
        item_embs2 = F.embedding(items, item_embeddings2)

        sup_pos_ratings = inner_product(user_embs, item_embs)       # [batch_size]
        sup_neg_ratings = inner_product(user_embs, neg_item_embs)   # [batch_size]
        sup_logits = sup_pos_ratings - sup_neg_ratings              # [batch_size]

        pos_ratings_user = inner_product(user_embs1, user_embs2)    # [batch_size]
        pos_ratings_item = inner_product(item_embs1, item_embs2)    # [batch_size]
        tot_ratings_user = torch.matmul(user_embs1, 
                                        torch.transpose(user_embeddings2, 0, 1))        # [batch_size, num_users]
        tot_ratings_item = torch.matmul(item_embs1, 
                                        torch.transpose(item_embeddings2, 0, 1))        # [batch_size, num_items]

        ssl_logits_user = tot_ratings_user - pos_ratings_user[:, None]                  # [batch_size, num_users]
        ssl_logits_item = tot_ratings_item - pos_ratings_item[:, None]                  # [batch_size, num_users]

        return sup_logits, ssl_logits_user, ssl_logits_item
    def _forward_gcn(self, norm_adj):
        ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            if isinstance(norm_adj, list):
                ego_embeddings = torch_sp.mm(norm_adj[k], ego_embeddings)
            else:
                ego_embeddings = torch_sp.mm(norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)
        
        return user_embeddings, item_embeddings


    def predict(self, users):
        if self._user_embeddings_final is None or self._item_embeddings_final is None:
            raise ValueError("Please first switch to 'eval' mode.")
        user_embs = F.embedding(users, self._user_embeddings_final)
        temp_item_embs = self._item_embeddings_final
        ratings = torch.matmul(user_embs, temp_item_embs.T)
        return ratings

    def eval(self):
        super(_LightGCN, self).eval()
        user_embs, item_embs = self._forward_gcn(self.norm_adj)
        self._user_embeddings_final = self.denoise_with_diffusion(user_embs, self.unet_user, steps=10)
        self._item_embeddings_final = self.denoise_with_diffusion(item_embs, self.unet_item, steps=10)





class SGL(AbstractRecommender):
    def __init__(self, config):
        super(SGL, self).__init__(config)

        self.config = config
        self.model_name = config["recommender"]
        self.dataset_name = config["dataset"]

        # General hyper-parameters
        self.reg = config['reg']
        self.emb_size = config['embed_size']
        self.batch_size = config['batch_size']
        self.test_batch_size = config['test_batch_size']
        self.epochs = config["epochs"]
        self.verbose = config["verbose"]
        self.stop_cnt = config["stop_cnt"]
        self.learner = config["learner"]
        self.lr = config['lr']
        self.param_init = config["param_init"]

        # Hyper-parameters for GCN
        self.n_layers = config['n_layers']

        # Hyper-parameters for SSL
        self.ssl_aug_type = config["aug_type"].lower()
        assert self.ssl_aug_type in ['nd','ed', 'rw']
        self.ssl_reg = config["ssl_reg"]
        self.ssl_ratio = config["ssl_ratio"]
        self.ssl_mode = config["ssl_mode"]
        self.ssl_temp = config["ssl_temp"]

        # Other hyper-parameters
        self.best_epoch = 0
        self.best_result = np.zeros([2], dtype=float)

        self.model_str = '#layers=%d-reg=%.0e' % (
            self.n_layers,
            self.reg
        )
        self.model_str += '/ratio=%.1f-mode=%s-temp=%.2f-reg=%.0e' % (
            self.ssl_ratio,
            self.ssl_mode,
            self.ssl_temp,
            self.ssl_reg
        )
        self.pretrain_flag = config["pretrain_flag"]
        if self.pretrain_flag:
            self.epochs = 0
        self.save_flag = config["save_flag"]
        self.save_dir, self.tmp_model_dir = None, None
        if self.pretrain_flag or self.save_flag:
            self.tmp_model_dir = config.data_dir + '%s/model_tmp/%s/%s/' % (
                self.dataset_name, 
                self.model_name,
                self.model_str)
            self.save_dir = config.data_dir + '%s/pretrain-embeddings/%s/n_layers=%d/' % (
                self.dataset_name, 
                self.model_name,
                self.n_layers,)
            ensureDir(self.tmp_model_dir)
            ensureDir(self.save_dir)

        self.num_users, self.num_items, self.num_ratings = self.dataset.num_users, self.dataset.num_items, self.dataset.num_train_ratings

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        adj_matrix = self.create_adj_mat()
        adj_matrix = sp_mat_to_sp_tensor(adj_matrix).to(self.device)

        self.lightgcn = _LightGCN(self.num_users, self.num_items, self.emb_size,
                                  adj_matrix, self.n_layers).to(self.device)
        if self.pretrain_flag:
            self.lightgcn.reset_parameters(pretrain=self.pretrain_flag, dir=self.save_dir)
        else:
            self.lightgcn.reset_parameters(init_method=self.param_init)
        self.optimizer = torch.optim.Adam(
            list(self.lightgcn.parameters()) +
            list(self.lightgcn.unet_user.parameters()) +
            list(self.lightgcn.unet_item.parameters()),
            lr=self.lr
        )


 
    def create_adj_mat(self, is_subgraph=False, aug_type='ed', prune=True):
        n_nodes = self.num_users + self.num_items
        users_items = self.dataset.train_data.to_user_item_pairs()
        users_np, items_np = users_items[:, 0], users_items[:, 1]
        prune = False
        n_clusters=10
        outlier_threshold=2
        cluster_pruning = False
        if prune:
            print("Prune öncesi toplam etkileşim sayısı:", len(users_np))

            # Kullanıcı başına etkileşim sayılarını hesapla
            unique_users, user_interaction_counts = np.unique(users_np, return_counts=True)

            # Dinamik alpha hesaplama (Ortalama - 2 * Standart Sapma)
            mean_interactions = np.mean(user_interaction_counts)
            std_interactions = np.std(user_interaction_counts)
            alpha = max(1, int(mean_interactions - 1 * 3*std_interactions/8))  # Negatif olmaması için min 1 sınırı
            alpha = 15
            print(f"📊 Kullanıcı başına ortalama etkileşim: {mean_interactions:.2f}")
            print(f"📊 Kullanıcı başına etkileşim standart sapması: {std_interactions:.2f}")
            print(f"Dinamik prune için belirlenen alpha değeri: {alpha}")
            short_tail = False
            long_tail = False
            if short_tail == True:
                # Alpha'dan düşük etkileşimi olan kullanıcıları belirle
                users_to_prune = unique_users[user_interaction_counts < alpha]

                # Bu kullanıcıların etkileşimlerini kaldır
                prune_mask = np.isin(users_np, users_to_prune, invert=True)
                users_np = users_np[prune_mask]
                items_np = items_np[prune_mask]
            if long_tail == True :
                unique_users, user_interaction_counts = np.unique(users_np, return_counts=True)

                # Etkileşimi alpha'dan büyük olan kullanıcıları belirle
                users_to_boost = unique_users[user_interaction_counts > alpha]
                users_to_boost = unique_users[user_interaction_counts < alpha]

                # Alpha’dan küçük etkileşimi olan kullanıcıların etkileşimlerini ikiyle çarp (aynı etkileşimi tekrar ekleyerek)
                boost_mask = np.isin(users_np, users_to_boost)

                # Bu kullanıcıların etkileşimlerini iki kez ekleyerek etkisini artırıyoruz
                users_np = np.concatenate([users_np, users_np[boost_mask]])
                items_np = np.concatenate([items_np, items_np[boost_mask]])

        if cluster_pruning:
            print("🔍 Kümeleme tabanlı pruning başlatılıyor...")
            
            # Kullanıcı-Öğe Sparse Matrisi
            user_item_matrix = sp.csr_matrix(
                (np.ones_like(users_np, dtype=np.float32), (users_np, items_np)),
                shape=(self.num_users, self.num_items)
            )

            # **Öğeleri (Items) Kümelere Ayır**
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            item_clusters = kmeans.fit_predict(user_item_matrix.T.toarray())  # Transpose, çünkü öğeleri kümeliyoruz

            # **Her Kümedeki Öğelerin Bağlantı Sayısını Hesapla**
            item_cluster_map = {item: cluster for item, cluster in enumerate(item_clusters)}
            cluster_interactions = {i: [] for i in range(n_clusters)}

            for item in range(self.num_items):
                cluster_id = item_cluster_map[item]
                total_connections = user_item_matrix[:, item].sum()
                cluster_interactions[cluster_id].append((item, total_connections))

            # **Her Kümede Gürültü Olan Öğeleri Belirle**
            noise_edges = set()
            for cluster_id, item_list in cluster_interactions.items():
                if len(item_list) < 2:  # Tek bir öğe varsa kıyaslama yapamayız
                    continue
                
                total_connections = np.array([count for _, count in item_list])
                mean_connections = np.mean(total_connections)
                threshold = mean_connections * outlier_threshold  # Ortalama bağlantı sayısının %20'si alt sınır

                for item, count in item_list:
                    if count < threshold:
                        # Gürültü olarak işaretlenmiş öğenin, sadece bu kümedeki bağlantılarını silmeliyiz
                        affected_users = np.where(items_np == item)[0]  # Bu öğeye bağlanan kullanıcıları bul
                        for user_idx in affected_users:
                            if item_cluster_map[items_np[user_idx]] == cluster_id:  # Sadece bu kümede gürültü ise sil
                                noise_edges.add(user_idx)

            # Gürültü olan **bağlantıları** kaldır (öğelerin tamamını değil)
            prune_mask = np.ones(len(users_np), dtype=bool)
            prune_mask[list(noise_edges)] = False  # Gürültü olan bağlantıları kaldır
            users_np = users_np[prune_mask]
            items_np = items_np[prune_mask]

            print(f"🔹 Gürültü olarak belirlenen bağlantı sayısı: {len(noise_edges)}")
            print("🔹 Prune sonrası toplam etkileşim sayısı:", len(users_np))
            print("Prune sonrası toplam etkileşim sayısı:", len(users_np))

        if is_subgraph and self.ssl_ratio > 0:
            if aug_type == 'nd':
                drop_user_idx = randint_choice(self.num_users, size=self.num_users * self.ssl_ratio, replace=False)
                drop_item_idx = randint_choice(self.num_items, size=self.num_items * self.ssl_ratio, replace=False)
                indicator_user = np.ones(self.num_users, dtype=np.float32)
                indicator_item = np.ones(self.num_items, dtype=np.float32)
                indicator_user[drop_user_idx] = 0.
                indicator_item[drop_item_idx] = 0.
                diag_indicator_user = sp.diags(indicator_user)
                diag_indicator_item = sp.diags(indicator_item)
                R = sp.csr_matrix(
                    (np.ones_like(users_np, dtype=np.float32), (users_np, items_np)), 
                    shape=(self.num_users, self.num_items))
                R_prime = diag_indicator_user.dot(R).dot(diag_indicator_item)
                (user_np_keep, item_np_keep) = R_prime.nonzero()
                ratings_keep = R_prime.data
                tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep+self.num_users)), shape=(n_nodes, n_nodes))
            if aug_type in ['ed', 'rw']:
                keep_idx = randint_choice(len(users_np), size=int(len(users_np) * (1 - self.ssl_ratio)), replace=False)
                user_np = np.array(users_np)[keep_idx]
                item_np = np.array(items_np)[keep_idx]
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.num_users)), shape=(n_nodes, n_nodes))
        else:
            ratings = np.ones_like(users_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (users_np, items_np+self.num_users)), shape=(n_nodes, n_nodes))

        adj_mat = tmp_adj + tmp_adj.T

        # normalize adjacency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        return adj_matrix


    def train_model(self):
        data_iter = PairwiseSamplerV2(self.dataset.train_data, num_neg=1, batch_size=self.batch_size, shuffle=True)                    
        self.logger.info(self.evaluator.metrics_info())
        stopping_step = 0
        for epoch in range(1, self.epochs + 1):
            total_loss, total_bpr_loss, total_reg_loss = 0.0, 0.0, 0.0
            training_start_time = time()
            if self.ssl_aug_type in ['nd', 'ed']:
                sub_graph1 = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                sub_graph1 = sp_mat_to_sp_tensor(sub_graph1).to(self.device)
                sub_graph2 = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                sub_graph2 = sp_mat_to_sp_tensor(sub_graph2).to(self.device)
            else:
                sub_graph1, sub_graph2 = [], []
                for _ in range(0, self.n_layers):
                    tmp_graph = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                    sub_graph1.append(sp_mat_to_sp_tensor(tmp_graph).to(self.device))
                    tmp_graph = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                    sub_graph2.append(sp_mat_to_sp_tensor(tmp_graph).to(self.device))
            self.lightgcn.train()
            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)
                sup_logits, ssl_logits_user, ssl_logits_item = self.lightgcn(
                    sub_graph1, sub_graph2, bat_users, bat_pos_items, bat_neg_items)
                
                # BPR Loss
                bpr_loss = -torch.sum(F.logsigmoid(sup_logits))

                # Reg Loss
                reg_loss = l2_loss(
                    self.lightgcn.user_embeddings(bat_users),
                    self.lightgcn.item_embeddings(bat_pos_items),
                    self.lightgcn.item_embeddings(bat_neg_items),
                )

                # InfoNCE Loss
                clogits_user = torch.logsumexp(ssl_logits_user / self.ssl_temp, dim=1)
                clogits_item = torch.logsumexp(ssl_logits_item / self.ssl_temp, dim=1)
                infonce_loss = torch.sum(clogits_user + clogits_item)
                
                loss = bpr_loss + self.ssl_reg * infonce_loss + self.reg * reg_loss
                total_loss += loss
                total_bpr_loss += bpr_loss
                total_reg_loss += self.reg * reg_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.logger.info("[iter %d : loss : %.4f = %.4f + %.4f + %.4f, time: %f]" % (
                epoch, 
                total_loss/self.num_ratings,
                total_bpr_loss / self.num_ratings,
                (total_loss - total_bpr_loss - total_reg_loss) / self.num_ratings,
                total_reg_loss / self.num_ratings,
                time()-training_start_time,))

            if epoch % self.verbose == 0 and epoch > self.config['start_testing_epoch']:
                result, flag = self.evaluate_model()
                self.logger.info("epoch %d:\t%s" % (epoch, result))
                if flag:
                    self.best_epoch = epoch
                    stopping_step = 0
                    self.logger.info("Find a better model.")
                    if self.save_flag:
                        self.logger.info("Save model to file as pretrain.")
                        torch.save(self.lightgcn.state_dict(), self.tmp_model_dir)
                        self.saver.save(self.sess, self.tmp_model_dir)
                else:
                    stopping_step += 1
                    if stopping_step >= self.stop_cnt:
                        self.logger.info("Early stopping is trigger at epoch: {}".format(epoch))
                        break

        self.logger.info("best_result@epoch %d:\n" % self.best_epoch)
        if self.save_flag:
            self.logger.info('Loading from the saved best model during the training process.')
            self.lightgcn.load_state_dict(torch.load(self.tmp_model_dir))
            uebd = self.lightgcn.user_embeddings.weight.cpu().detach().numpy()
            iebd = self.lightgcn.item_embeddings.weight.cpu().detach().numpy()
            np.save(self.save_dir + 'user_embeddings.npy', uebd)
            np.save(self.save_dir + 'item_embeddings.npy', iebd)
            buf, _ = self.evaluate_model()
        elif self.pretrain_flag:
            buf, _ = self.evaluate_model()
        else:
            buf = '\t'.join([("%.4f" % x).ljust(12) for x in self.best_result])
        self.logger.info("\t\t%s" % buf)

    # @timer
    def evaluate_model(self):
        flag = False
        self.lightgcn.eval()
        current_result, buf = self.evaluator.evaluate(self)
        if self.best_result[1] < current_result[1]:
            self.best_result = current_result
            flag = True
        return buf, flag

    def predict(self, users):
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        return self.lightgcn.predict(users).cpu().detach().numpy()
