import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .module import Enc_graphVCI, Dec_graphVCI

from gvci.utils.graph_utils import get_graph

from vci.model import VCI
from vci.model.module import (
    MLP, NegativeBinomial, ZeroInflatedNegativeBinomial
)

#####################################################
#                    MODEL SAVING                   #
#####################################################

def load_graphVCI(args, state_dict=None):
    device = (
        "cuda:" + str(args["gpu"])
            if (not args["cpu"]) 
                and torch.cuda.is_available() 
            else 
        "cpu"
    )

    model = graphVCI(
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

class graphVCI(VCI):
    def __init__(
        self,
        graph_data,
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
        node_grad=True,
        edge_grad=True,
        best_score=-1e3,
        patience=5,
        device="cuda",
        hparams=""
    ):
        # set hyperparameters
        self._set_g_hparams()

        self.graph_mode = graph_mode
        self.encode_aggr = encode_aggr
        self.decode_aggr = decode_aggr
        self.node_grad = node_grad
        self.edge_grad = edge_grad

        # make graph
        if self.graph_mode == "dense": # row target, col source
            output_adj_mode = "target_to_source"
        elif self.graph_mode == "sparse": # first row souce, second row target
            output_adj_mode = "source_to_target"
        else:
            ValueError("graph_mode not recognized")

        if graph_data is None:
            node_features, adjacency, edge_features = get_graph(
                n_nodes=num_outcomes, n_features=self.g_hparams["graph_latent_dim"],
                graph_mode=graph_mode, output_adj_mode=output_adj_mode,
                add_self_loops=True)
        elif type(graph_data) == str:
            node_features, adjacency, edge_features = get_graph(graph=torch.load(graph_data),
                n_nodes=num_outcomes, n_features=self.g_hparams["graph_latent_dim"],
                graph_mode=graph_mode, output_adj_mode=output_adj_mode,
                add_self_loops=True)
        else:
            node_features, adjacency, edge_features = get_graph(graph_data,
                n_nodes=num_outcomes, n_features=self.g_hparams["graph_latent_dim"],
                graph_mode=graph_mode, output_adj_mode=output_adj_mode,
                add_self_loops=True)
        self.num_nodes, self.num_features = node_features.size()
        self.edge_dim = 1 if edge_features.dim() == 1 else edge_features.size(-1)

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

        self._init_graph(node_features, adjacency, edge_features)

        self.to_device(device)

    def _set_g_hparams(self):
        self.g_hparams = {
            "graph_latent_dim": 128,
            "graph_encoder_width": 128,
            "graph_encoder_depth": 1,
            "graph_discriminator_width": 64,
            "graph_discriminator_depth": 1,
            "graph_esimator_width": 64,
            "graph_esimator_depth": 1,
            "attention_heads": 2,
            "node_emb_lr": 3e-4,
            "edge_emb_lr": 3e-4
        }
        assert self.g_hparams["graph_latent_dim"] % self.g_hparams["attention_heads"] == 0
        assert self.g_hparams["graph_encoder_width"] % self.g_hparams["attention_heads"] == 0
        assert self.g_hparams["graph_discriminator_width"] % self.g_hparams["attention_heads"] == 0
        assert self.g_hparams["graph_esimator_width"] % self.g_hparams["attention_heads"] == 0

    def _init_graph(self, node_features, adjacency, edge_features):
        # node
        self.node_features = nn.Parameter(
            node_features, requires_grad=self.node_grad
        )
        if self.node_grad:
            self.optimizer_autoencoder.add_param_group(
                {'params': self.node_features, 'lr': self.g_hparams["node_emb_lr"]}
            )
        # edge
        self.adjacency = nn.Parameter(
            adjacency, requires_grad=False
        )
        self.edge_features = nn.Parameter(
            edge_features, requires_grad=self.edge_grad
        )
        if self.edge_grad:
            self.optimizer_autoencoder.add_param_group(
                {'params': self.edge_features, 'lr': self.g_hparams["edge_emb_lr"]}
            )

        return self.node_features, self.adjacency, self.edge_features

    def encode(self, outcomes, treatments, covariates, eval=False):
        if self.embed_outcomes:
            outcomes = self.outcomes_embeddings(outcomes)
        if self.embed_treatments:
            treatments = self.treatments_embeddings(
                treatments if treatments.shape[-1] == 1 else treatments.argmax(1))
        if self.embed_covariates:
            covariates = [emb(covars if covars.shape[-1] == 1 else covars.argmax(1)) 
                for covars, emb in zip(covariates, self.covariates_embeddings)
            ]

        inputs = torch.cat([outcomes, treatments] + covariates, -1)

        if eval:
            return self.encoder_eval(inputs,
                self.node_features, self.adjacency, self.edge_features, return_graph=True)
        else:
            return self.encoder(inputs,
                self.node_features, self.adjacency, self.edge_features, return_graph=True)

    def decode(self, latents, graph_latents, treatments):
        if self.embed_treatments:
            treatments = self.treatments_embeddings(
                treatments if treatments.shape[-1] == 1 else treatments.argmax(1))

        inputs = torch.cat([latents, treatments], -1)

        return self.decoder(inputs, graph_latents)

    def discriminate(self, outcomes, treatments, covariates):
        if self.embed_outcomes:
            outcomes = self.adv_outcomes_emb(outcomes)
        if self.embed_treatments:
            treatments = self.adv_treatments_emb(
                treatments if treatments.shape[-1] == 1 else treatments.argmax(1))
        if self.embed_covariates:
            covariates = [emb(covars if covars.shape[-1] == 1 else covars.argmax(1)) 
                for covars, emb in zip(covariates, self.adv_covariates_emb)
            ]

        inputs = torch.cat([outcomes, treatments] + covariates, -1)

        return self.discriminator(
            inputs, self.node_features, self.adjacency, self.edge_features
        ).squeeze()

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
                latents_dist.mean, latents_dist.stddev, graph_latent, cf_treatments
            )
            outcomes_dist_samp = self.distributionize(outcomes_constr_samp)

        if return_dist:
            return outcomes_dist_samp
        else:
            return outcomes_dist_samp.mean

    def forward(self, outcomes, treatments, cf_treatments, covariates,
                sample_latent=True, sample_outcome=False, detach_encode=False, detach_eval=True):
        """
        Execute the workflow.
        """

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
        if sample_latent:
            cf_outcomes_constr_out = self.decode(
                latents_dist.rsample(), graph_latent, cf_treatments
            )
        else:
            cf_outcomes_constr_out = self.decode(
                latents_dist.mean, graph_latent, cf_treatments
            )
        if sample_outcome:
            cf_outcomes_out = self.distributionize(cf_outcomes_constr_out).rsample()
        else:
            cf_outcomes_out = self.distributionize(cf_outcomes_constr_out).mean

        # q(z | y', g, x, t')
        if detach_encode:
            if sample_latent:
                cf_outcomes_constr_in = self.decode(latents_dist.sample(), cf_treatments)
            else:
                cf_outcomes_constr_in = self.decode(latents_dist.mean.detach(), cf_treatments)
            if sample_outcome:
                cf_outcomes_in = self.distributionize(cf_outcomes_constr_in).rsample()
            else:
                cf_outcomes_in = self.distributionize(cf_outcomes_constr_in).mean
        else:
            cf_outcomes_in = cf_outcomes_out

        cf_latents_constr, _ = self.encode(
            cf_outcomes_in, cf_treatments, covariates, eval=detach_eval
        )
        cf_latents_dist = self.distributionize(cf_latents_constr, dist="normal")

        return (outcomes_dist_samp, cf_outcomes_out,latents_dist, cf_latents_dist)

    def init_encoder(self):
        return Enc_graphVCI(
            mlp_sizes=[self.outcome_dim+self.treatment_dim+self.covariate_dim]
                + [self.hparams["encoder_width"]] * (self.hparams["encoder_depth"] - 1)
                + [self.hparams["latent_dim"]],
            gnn_sizes=[self.num_features]
                + [self.g_hparams["graph_encoder_width"]] * (self.g_hparams["graph_encoder_depth"] - 1)
                + [self.g_hparams["graph_latent_dim"]],
            attention_heads=self.g_hparams["attention_heads"],
            edge_dim=self.edge_dim,
            aggr_heads=2,
            graph_mode=self.graph_mode,
            aggr_mode=self.encode_aggr,
            final_act="relu"
        )
    
    def init_decoder(self):
        return Dec_graphVCI(
            mlp_sizes=[self.hparams["latent_dim"]+self.treatment_dim]
                + [self.hparams["decoder_width"]] * (self.hparams["decoder_depth"] - 1),
            num_features=self.g_hparams["graph_latent_dim"],
            aggr_heads=self.num_dist_params,
            aggr_mode=self.decode_aggr
        )

    def init_discriminator(self):
        return nn.Sequential(
            Enc_graphVCI(
                mlp_sizes=[self.outcome_dim+self.treatment_dim+self.covariate_dim]
                    + [self.hparams["discriminator_width"]] 
                        * (self.hparams["discriminator_depth"] - 1),
                gnn_sizes=[self.num_features]
                    + [self.g_hparams["graph_discriminator_width"]] 
                        * (self.g_hparams["graph_discriminator_depth"] - 1),
                attention_heads=self.g_hparams["attention_heads"],
                edge_dim=self.edge_dim,
                aggr_heads=1,
                graph_mode=self.graph_mode,
                aggr_mode=self.encode_aggr,
                final_act="relu"
            ),
            MLP(self.hparams["discriminator_width"], 1)
        )
