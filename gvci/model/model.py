import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, RelaxedBernoulli

import torch_geometric as pyg
from torch_geometric.utils import add_remaining_self_loops

from .module import GVCIEncoder, GVCIDecoder

from gvci.utils.graph_utils import to_dense_adj

from vci.model import VCI
from vci.model.module import (
    MLP, NegativeBinomial, ZeroInflatedNegativeBinomial
)

#####################################################
#                    MODEL SAVING                   #
#####################################################

def load_GVCI(args, state_dict=None):
    device = (
        "cuda:" + str(args["gpu"])
            if (not args["cpu"]) 
                and torch.cuda.is_available() 
            else 
        "cpu"
    )

    model = GVCI(
        args["graph_path"],
        args["num_outcomes"],
        args["num_treatments"],
        args["num_covariates"],
        omega0=args["omega0"],
        omega1=args["omega1"],
        omega2=args["omega2"],
        outcome_dist=args["outcome_dist"],
        dist_mode=args["dist_mode"],
        graph_mode=args["graph_mode"],
        encode_aggr=args["encode_aggr"],
        decode_aggr=args["decode_aggr"],
        patience=args["patience"],
        device=device,
        hparams=args["hparams"]
    )
    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model

#####################################################
#                     MAIN MODEL                    #
#####################################################

class GVCI(VCI):
    def __init__(
        self,
        base_graph,
        num_outcomes,
        num_treatments,
        num_covariates,
        embed_outcomes=True,
        embed_treatments=False,
        embed_covariates=True,
        omega0=1.0,
        omega1=2.0,
        omega2=0.1,
        mc_sample_size=3,
        outcome_dist="normal",
        dist_mode="match",
        graph_mode="sparse",
        encode_aggr="sum",
        decode_aggr="dot",
        feature_grad=True,
        edge_grad=True,
        best_score=-1e3,
        patience=5,
        device="cuda",
        hparams=""
    ):
        # graph parameters
        if type(base_graph) == str:
            base_graph = torch.load(base_graph)
        self.num_nodes, self.num_features = base_graph.x.size()

        self.graph_mode = graph_mode
        self.encode_aggr = encode_aggr
        self.decode_aggr = decode_aggr
        self.feature_grad = feature_grad
        self.edge_grad = edge_grad

        # set hyperparameters
        self._init_hparams()

        super().__init__(
            num_outcomes,
            num_treatments,
            num_covariates,
            embed_outcomes,
            embed_treatments,
            embed_covariates,
            omega0,
            omega1,
            omega2,
            mc_sample_size,
            outcome_dist,
            dist_mode,
            best_score,
            patience,
            "cpu",
            hparams
        )

        self._init_graph(base_graph)

        self.to_device(device)

    def _init_hparams(self):
        self.g_hparams = {
            "graph_latent_dim": 128,
            "graph_encoder_width": 128,
            "graph_encoder_depth": 1,
            "graph_discriminator_width": 64,
            "graph_discriminator_depth": 1,
            "graph_esimator_width": 64,
            "graph_esimator_depth": 1,
            "feature_emb_lr": 3e-4,
            "edge_weight_lr": 3e-4,
            "reg_edge_weight": 20.0,
        }

    def _init_graph(self, graph):

        self.features_embeddings = nn.Parameter(
            graph.x, requires_grad=self.feature_grad
        )
        if self.feature_grad:
            self.optimizer_autoencoder.add_param_group(
                {'params': self.features_embeddings, 'lr': self.g_hparams["feature_emb_lr"]}
            )

        assert graph.edge_index is not None
        edge_index = graph.edge_index
        edge_index, _ = add_remaining_self_loops(
            edge_index, num_nodes=self.num_nodes
        )
        if self.graph_mode == "dense":
            edge_weight_logits = to_dense_adj(edge_index,
                edge_attr=2.*torch.ones(edge_index.size(1)), fill_value=-2.
            )[0].t()
            edge_index = to_dense_adj(edge_index)[0].t()
        elif self.graph_mode == "sparse":
            edge_weight_logits = torch.ones(edge_index.size(1))
        else:
            ValueError("graph_mode not recognized")

        self.edge_index = nn.Parameter(
            edge_index, requires_grad=False
        )
        self.edge_weight_logits = nn.Parameter(
            edge_weight_logits, requires_grad=self.edge_grad
        )

        if self.edge_grad:
            self.optimizer_autoencoder.add_param_group(
                {'params': self.edge_weight_logits, 'lr': self.g_hparams["edge_weight_lr"]}
            )

        return self.features_embeddings, self.edge_weight_logits

    def _init_indiv_model(self):

        params = []

        # embeddings
        if self.embed_outcomes:
            self.outcomes_embeddings = MLP(
                [self.num_outcomes, self.hparams["outcome_emb_dim"]], final_act="relu"
            )
            outcome_dim = self.hparams["outcome_emb_dim"]
            params.extend(list(self.outcomes_embeddings.parameters()))
        else:
            outcome_dim = self.num_outcomes

        if self.embed_treatments:
            self.treatments_embeddings = nn.Embedding(
                self.num_treatments, self.hparams["treatment_emb_dim"]
            )
            treatment_dim = self.hparams["treatment_emb_dim"]
            params.extend(list(self.treatments_embeddings.parameters()))
        else:
            treatment_dim = self.num_treatments

        if self.embed_covariates:
            self.covariates_embeddings = []
            for num_covariate in self.num_covariates:
                self.covariates_embeddings.append(
                    nn.Embedding(num_covariate, 
                        self.hparams["covariate_emb_dim"]
                    )
                )
            self.covariates_embeddings = nn.Sequential(
                *self.covariates_embeddings
            )
            covariate_dim = self.hparams["covariate_emb_dim"]*len(self.num_covariates)
            for emb in self.covariates_embeddings:
                params.extend(list(emb.parameters()))
        else:
            covariate_dim = sum(self.num_covariates)

        # encoder
        self.encoder = GVCIEncoder(
            mlp_sizes=[outcome_dim+treatment_dim+covariate_dim]
                + [self.hparams["encoder_width"]] * (self.hparams["encoder_depth"] - 1)
                + [self.hparams["latent_dim"]],
            gnn_sizes=[self.num_features]
                + [self.g_hparams["graph_encoder_width"]] * (self.g_hparams["graph_encoder_depth"] - 1)
                + [self.g_hparams["graph_latent_dim"]],
            num_nodes=self.num_nodes,
            aggr_heads=2,
            graph_mode=self.graph_mode,
            aggr_mode=self.encode_aggr,
            final_act="relu"
        )
        params.extend(list(self.encoder.parameters()))

        self.encoder_eval = copy.deepcopy(self.encoder)

        # decoder
        self.decoder = GVCIDecoder(
            mlp_sizes=[self.hparams["latent_dim"]+treatment_dim]
                + [self.hparams["decoder_width"]] * (self.hparams["decoder_depth"] - 1),
            num_features=self.g_hparams["graph_latent_dim"],
            aggr_heads=self.num_dist_params,
            aggr_mode=self.decode_aggr
        )
        params.extend(list(self.decoder.parameters()))

        self.optimizer_autoencoder = torch.optim.Adam(
            params,
            lr=self.hparams["autoencoder_lr"],
            weight_decay=self.hparams["autoencoder_wd"],
        )
        self.scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
            self.optimizer_autoencoder, step_size=self.hparams["step_size_lr"]
        )

        return self.encoder, self.decoder

    def _init_covar_model(self):

        if self.dist_mode == "discriminate":
            params = []

            # embeddings
            if self.embed_outcomes:
                self.adv_outcomes_emb = MLP(
                    [self.num_outcomes, self.hparams["outcome_emb_dim"]], final_act="relu"
                )
                outcome_dim = self.hparams["outcome_emb_dim"]
                params.extend(list(self.adv_outcomes_emb.parameters()))
            else:
                outcome_dim = self.num_outcomes

            if self.embed_treatments:
                self.adv_treatments_emb = nn.Embedding(
                    self.num_treatments, self.hparams["treatment_emb_dim"]
                )
                treatment_dim = self.hparams["treatment_emb_dim"]
                params.extend(list(self.adv_treatments_emb.parameters()))
            else:
                treatment_dim = self.num_treatments

            if self.embed_covariates:
                self.adv_covariates_emb = []
                for num_covariate in self.num_covariates:
                    self.adv_covariates_emb.append(
                        nn.Embedding(num_covariate, 
                            self.hparams["covariate_emb_dim"]
                        )
                    )
                self.adv_covariates_emb = nn.Sequential(
                    *self.adv_covariates_emb
                )
                covariate_dim = self.hparams["covariate_emb_dim"]*len(self.num_covariates)
                for emb in self.adv_covariates_emb:
                    params.extend(list(emb.parameters()))
            else:
                covariate_dim = sum(self.num_covariates)

            # model
            if self.encode_aggr == "sum":
                assert self.hparams["discriminator_width"] == self.g_hparams["graph_discriminator_width"]

            self.discriminator = nn.Sequential(
                GVCIEncoder(
                    mlp_sizes=[outcome_dim+treatment_dim+covariate_dim]
                        + [self.hparams["discriminator_width"]] 
                            * (self.hparams["discriminator_depth"] - 1),
                    gnn_sizes=[self.num_features]
                        + [self.g_hparams["graph_discriminator_width"]] 
                            * (self.g_hparams["graph_discriminator_depth"] - 1),
                    num_nodes=self.num_nodes,
                    aggr_heads=1,
                    graph_mode=self.graph_mode,
                    aggr_mode=self.encode_aggr,
                    final_act="relu"
                ),
                MLP(self.hparams["discriminator_width"], 1)
            )
            self.loss_discriminator = nn.BCEWithLogitsLoss()
            params.extend(list(self.discriminator.parameters()))

            self.optimizer_discriminator = torch.optim.Adam(
                params,
                lr=self.hparams["discriminator_lr"],
                weight_decay=self.hparams["discriminator_wd"],
            )
            self.scheduler_discriminator = torch.optim.lr_scheduler.StepLR(
                self.optimizer_discriminator, step_size=self.hparams["step_size_lr"]
            )

            return self.discriminator

        elif self.dist_mode == "fit":
            params = []

            # embeddings
            if self.embed_treatments:
                self.adv_treatments_emb = nn.Embedding(
                    self.num_treatments, self.hparams["treatment_emb_dim"]
                )
                treatment_dim = self.hparams["treatment_emb_dim"]
                params.extend(list(self.adv_treatments_emb.parameters()))
            else:
                treatment_dim = self.num_treatments

            if self.embed_covariates:
                self.adv_covariates_emb = []
                for num_covariate in self.num_covariates:
                    self.adv_covariates_emb.append(
                        nn.Embedding(num_covariate, 
                            self.hparams["covariate_emb_dim"]
                        )
                    )
                self.adv_covariates_emb = nn.Sequential(
                    *self.adv_covariates_emb
                )
                covariate_dim = self.hparams["covariate_emb_dim"]*len(self.num_covariates)
                for emb in self.adv_covariates_emb:
                    params.extend(list(emb.parameters()))
            else:
                covariate_dim = sum(self.num_covariates)

            # model
            if self.encode_aggr == "sum":
                assert self.hparams["estimator_width"] == self.g_hparams["graph_estimator_width"]

            self.outcome_estimator = nn.Sequential(
                GVCIEncoder(
                    mlp_sizes=[treatment_dim+covariate_dim]
                        + [self.hparams["estimator_width"]] 
                            * (self.hparams["estimator_depth"] - 1),
                    gnn_sizes=[self.num_features]
                        + [self.g_hparams["graph_estimator_width"]] 
                            * (self.g_hparams["graph_estimator_depth"] - 1),
                    num_nodes=self.num_nodes,
                    aggr_heads=1,
                    graph_mode=self.graph_mode,
                    aggr_mode=self.encode_aggr,
                    final_act="relu"
                ),
                MLP(self.hparams["estimator_width"], self.num_outcomes)
            )
            self.loss_outcome_estimator = nn.MSELoss()
            params.extend(list(self.outcome_estimator.parameters()))

            self.optimizer_estimator = torch.optim.Adam(
                params,
                lr=self.hparams["estimator_lr"],
                weight_decay=self.hparams["estimator_wd"],
            )
            self.scheduler_estimator = torch.optim.lr_scheduler.StepLR(
                self.optimizer_estimator, step_size=self.hparams["step_size_lr"]
            )

            return self.outcome_estimator

        elif self.dist_mode == "match":
            return None

        else:
            raise ValueError("dist_mode not recognized")

    def encode(self, outcomes, treatments, covariates, eval=False):
        if self.embed_outcomes:
            outcomes = self.outcomes_embeddings(outcomes)
        if self.embed_treatments:
            treatments = self.treatments_embeddings(treatments.argmax(1))
        if self.embed_covariates:
            covariates = [emb(covar.argmax(1)) 
                for covar, emb in zip(covariates, self.covariates_embeddings)
            ]

        inputs = torch.cat([outcomes, treatments] + covariates, -1)
        features = self.features_embeddings

        if eval:
            return self.encoder_eval(
                inputs, features, self.edge_index, self.edge_weight_logits, return_graph=True
            )
        else:
            return self.encoder(
                inputs, features, self.edge_index, self.edge_weight_logits, return_graph=True
            )

    def decode(self, latents, graph_latents, treatments):
        if self.embed_treatments:
            treatments = self.treatments_embeddings(treatments.argmax(1))

        inputs = torch.cat([latents, treatments], -1)
        features = graph_latents

        return self.decoder(inputs, features)

    def discriminate(self, outcomes, treatments, covariates):
        if self.embed_outcomes:
            outcomes = self.outcomes_embeddings(outcomes)
        if self.embed_treatments:
            treatments = self.treatments_embeddings(treatments.argmax(1))
        if self.embed_covariates:
            covariates = [emb(covar.argmax(1)) 
                for covar, emb in zip(covariates, self.covariates_embeddings)
            ]

        inputs = torch.cat([outcomes, treatments] + covariates, -1)
        features = self.features_embeddings

        return self.discriminator(
            inputs, features, self.edge_index, self.edge_weight_logits
        ).squeeze()

    def distributionize(self, constructions, dist=None, eps=1e-3):
        if dist is None:
            dist = self.outcome_dist

        if dist == "nb":
            mus = F.softplus(constructions[:, :, 0]).add(eps)
            thetas = F.softplus(constructions[:, :, 1]).add(eps)
            dist = NegativeBinomial(
                mu=mus, theta=thetas
            )
        elif dist == "zinb":
            mus = F.softplus(constructions[:, :, 0]).add(eps)
            thetas = F.softplus(constructions[:, :, 1]).add(eps)
            zi_logits = constructions[:, :, 2].add(eps)
            dist = ZeroInflatedNegativeBinomial(
                mu=mus, theta=thetas, zi_logits=zi_logits
            )
        elif dist == "normal":
            locs = constructions[:, :, 0]
            scales = F.softplus(constructions[:, :, 1]).add(eps)
            dist = Normal(
                loc=locs, scale=scales
            )

        return dist

    def sample(self, latent_mu, latent_sigma, graph_latent, treatments, size=1):
        latent_mu = latent_mu.repeat(size, 1)
        latent_sigma = latent_sigma.repeat(size, 1)
        treatments = treatments.repeat(size, 1)

        latents = self.reparameterize(latent_mu, latent_sigma)

        return self.decode(latents, graph_latent, treatments)

    def predict(
        self,
        outcomes,
        treatments,
        cf_treatments,
        covariates,
        return_dist=False
    ):
        """
        Predict "what would have the gene expression `outcomes` been, had the
        cells in `outcomes` with cell types `cell_types` been treated with
        combination of treatments `treatments`.
        """
        outcomes, treatments, cf_treatments, covariates = self.move_inputs(
            outcomes, treatments, cf_treatments, covariates
        )
        if cf_treatments is None:
            cf_treatments = treatments

        with torch.autograd.no_grad():
            latents_constr, graph_latent = self.encode(
                outcomes, treatments, covariates
            )
            latents_dist = self.distributionize(latents_constr, dist="normal")

            outcomes_constr = self.decode(latents_dist.mean, graph_latent, cf_treatments)
            outcomes_dist = self.distributionize(outcomes_constr)

        if return_dist:
            return outcomes_dist
        else:
            return outcomes_dist.mean

    def generate(
        self,
        outcomes,
        treatments,
        cf_treatments,
        covariates,
        return_dist=False
    ):
        outcomes, treatments, cf_treatments, covariates = self.move_inputs(
            outcomes, treatments, cf_treatments, covariates
        )
        if cf_treatments is None:
            cf_treatments = treatments

        with torch.autograd.no_grad():
            latents_constr, graph_latent = self.encode(
                outcomes, treatments, covariates
            )
            latents_dist = self.distributionize(latents_constr, dist="normal")

            outcomes_constr_samp = self.sample(
                latents_dist.mean, latents_dist.stddev, graph_latent, treatments
            )
            outcomes_dist_samp = self.distributionize(outcomes_constr_samp)

        if return_dist:
            return outcomes_dist_samp
        else:
            return outcomes_dist_samp.mean

    def update(self, outcomes, treatments, cf_outcomes, cf_treatments, covariates,
                rsample=True, detach_encode=False, detach_eval=True):
        """
        Update GVCI's parameters given a minibatch of outcomes, treatments, and
        cell types.
        """
        outcomes, treatments, cf_outcomes, cf_treatments, covariates = self.move_inputs(
            outcomes, treatments, cf_outcomes, cf_treatments, covariates
        )

        # q(z | y, g, x, t)
        latents_constr, graph_latent = self.encode(outcomes, treatments, covariates)
        latents_dist = self.distributionize(latents_constr, dist="normal")

        # p(y | z, t)
        outcomes_constr_samp = self.sample(
            latents_dist.mean, latents_dist.stddev, graph_latent,
            treatments, size=self.mc_sample_size
        )
        outcomes_dist_samp = self.distributionize(outcomes_constr_samp)

        # p(y' | z, t')
        if rsample:
            cf_outcomes_constr = self.decode(
                latents_dist.rsample(), graph_latent, cf_treatments
            )
            cf_outcomes_out = self.distributionize(cf_outcomes_constr).rsample()
        else:
            cf_outcomes_constr = self.decode(
                latents_dist.mean, graph_latent, cf_treatments
            )
            cf_outcomes_out = self.distributionize(cf_outcomes_constr).mean

        # q(z | y', g, x, t')
        if detach_encode:
            if rsample:
                cf_outcomes_in = self.distributionize(
                    self.decode(latents_dist.sample(), cf_treatments)
                ).rsample()
            else:
                cf_outcomes_in = self.distributionize(
                    self.decode(latents_dist.mean.detach(), cf_treatments)
                ).mean
        else:
            cf_outcomes_in = cf_outcomes_out

        cf_latents_constr, _ = self.encode(
            cf_outcomes_in, cf_treatments, covariates, eval=detach_eval
        )
        cf_latents_dist = self.distributionize(cf_latents_constr, dist="normal")

        indiv_spec_nllh, covar_spec_nllh, kl_divergence = self.loss(
            outcomes, outcomes_dist_samp,
            cf_outcomes, cf_outcomes_out,
            latents_dist, cf_latents_dist,
            treatments, covariates
        )

        loss = (self.omega0 * indiv_spec_nllh
            + self.omega1 * covar_spec_nllh
            + self.omega2 * kl_divergence
        )

        if self.graph_mode=="dense" and self.edge_grad:
            loss += (
                self.hparams["reg_edge_weight"] * 
                torch.mean(torch.sigmoid(self.edge_weight_logits))
            )

        self.optimizer_autoencoder.zero_grad()
        loss.backward()
        self.optimizer_autoencoder.step()
        self.iteration += 1

        return {
            "Indiv-spec NLLH": indiv_spec_nllh.item(),
            "Covar-spec NLLH": covar_spec_nllh.item(),
            "KL Divergence": kl_divergence.item()
        }

    def update_discriminator(self, inputs_tru, inputs_fls):
        loss_tru = self.loss_discriminator(
            self.discriminator(inputs_tru).squeeze(),
            torch.ones(inputs_tru.size(0), device=inputs_tru.device)
        )

        loss_fls = self.loss_discriminator(
            self.discriminator(inputs_fls).squeeze(),
            torch.zeros(inputs_fls.size(0), device=inputs_fls.device)
        )

        loss = (loss_tru+loss_fls)/2.
        self.optimizer_discriminator.zero_grad()
        loss.backward()
        self.optimizer_discriminator.step()

        return loss.item()
