import json
import numpy as np
import networkx as nx

import torch
import torch.nn.functional as F
from torch.distributions import Normal, RelaxedBernoulli

import torch_geometric as pyg
from torch_geometric.utils import add_remaining_self_loops

from .module import (
    MLP, MyDenseGCN, MyGCN, DenseGCNDecoder, GCNDecoder,
    NegativeBinomial, ZeroInflatedNegativeBinomial
)

from utils.math_utils import (
    logprob_normal,
    logprob_nb_positive,
    logprob_zinb_positive
)
from utils.graph_utils import to_dense_adj

#####################################################
#                    MODEL SAVING                   #
#####################################################

def init_DPI(state):
    # TODO
    net = DPI(state['input_dim'],
                state['nb_hidden'],
                state['nb_layers'],
                state['output_dim'])
    net.load_state_dict(state['state_dict'])
    return net

def load_DPI(state_path):
    state = torch.load(state_path)
    return init_DPI(state)

def save_DPI(net, state_path=None):
    torch.save(net.get_dict(), state_path)

#####################################################
#                     MAIN MODEL                    #
#####################################################

class DPI(torch.nn.Module):
    def __init__(
        self,
        base_graph,
        num_nodes,
        num_treatments,
        num_covariates,
        embed_outcomes=False,
        embed_treatments=False,
        embed_covariates=False,
        outcome_dist="normal",
        dist_mode='match',
        device="cuda",
        seed=0,
        patience=5,
        dropout=True,
        encoder_model=MyDenseGCN,
        decoder_model=MyDenseGCN,
        feature_grad=False,
        edge_grad=True,
        hparams=""
    ):
        super(DPI, self).__init__()
        ############################## graph ##############################
        if type(base_graph) == str:
            base_graph = torch.load(base_graph)

        self.set_hparams_(hparams)
        num_outcomes = 1
        num_treatments = 0

        assert embed_outcomes==False
        assert embed_covariates==False
        self.num_nodes = num_nodes
        self.node_aggr = True if decoder_model in [DenseGCNDecoder, GCNDecoder] else False

        # base graph
        node_attr = base_graph.x
        self.features_embeddings = torch.nn.Parameter(
            node_attr, requires_grad=feature_grad
        )
        if num_covariates is None:
            num_covariates = node_attr.size(1)
        else:
            num_covariates.append(node_attr.size(1))

        self.dropout = dropout
        if dropout:
            assert encoder_model == MyDenseGCN
            assert decoder_model in [MyDenseGCN, DenseGCNDecoder]
            if base_graph.edge_index is None:
                edge_index = torch.stack(
                    [torch.arange(self.num_nodes), torch.arange(self.num_nodes)]
                )
                edge_weight_logits = torch.ones((self.num_nodes, self.num_nodes))
                edge_dropout_logits = to_dense_adj(edge_index,
                    edge_attr=4.*torch.ones(self.num_nodes), fill_value=-4.
                )[0]
            else:
                edge_index = base_graph.edge_index
                edge_index, _ = add_remaining_self_loops(
                    edge_index, num_nodes=self.num_nodes
                )
                edge_weight_logits = torch.ones((self.num_nodes, self.num_nodes))
                '''
                edge_weight_logits = to_dense_adj(edge_index,
                    edge_attr=2.*torch.ones(edge_index.size(1)), fill_value=-2.
                )[0].t()
                '''
                edge_dropout_logits = to_dense_adj(edge_index,
                    edge_attr=4.*torch.ones(edge_index.size(1)), fill_value=-4.
                )[0].t()
            self.edge_weight_logits = torch.nn.Parameter(
                edge_weight_logits, requires_grad=edge_grad
            )
            self.edge_dropout_logits = torch.nn.Parameter(
                edge_dropout_logits, requires_grad=False
            )
        else:
            assert base_graph.edge_index is not None
            edge_index = base_graph.edge_index
            edge_index, _ = add_remaining_self_loops(
                edge_index, num_nodes=self.num_nodes
            )
            if encoder_model == MyDenseGCN:
                assert decoder_model in [MyDenseGCN, DenseGCNDecoder]
                edge_weight_logits = to_dense_adj(edge_index,
                    edge_attr=2.*torch.ones(edge_index.size(1)), fill_value=-2.
                )[0].t()
                edge_index = to_dense_adj(edge_index)[0].t()
            elif encoder_model == MyGCN:
                assert decoder_model in [MyGCN, GCNDecoder]
                edge_weight_logits = torch.ones(edge_index.size(1))

            self.edge_index = torch.nn.Parameter(
                edge_index, requires_grad=False
            )
            self.edge_weight_logits = torch.nn.Parameter(
                edge_weight_logits, requires_grad=edge_grad
            )

        ###################################################################

        # set generic attributes
        self.num_outcomes = num_outcomes
        self.num_treatments = num_treatments
        self.num_covariates = num_covariates
        self.embed_outcomes = embed_outcomes
        self.embed_treatments = embed_treatments
        self.embed_covariates = embed_covariates
        self.seed = seed
        self.outcome_dist = outcome_dist
        self.dist_mode = dist_mode
        # early-stopping
        self.patience = patience
        self.best_score = -1e3
        self.patience_trials = 0

        # set hyperparameters
        self.set_hparams_(hparams)

        if self.outcome_dist == 'nb':
            self.num_dist_params = 2
        elif self.outcome_dist == 'zinb':
            self.num_dist_params = 3
        elif self.outcome_dist == 'normal':
            self.num_dist_params = 2
        else:
            raise ValueError("outcome_dist not recognized")

        params = []

        if self.embed_outcomes:
            self.outcomes_embeddings = MLP(
                [num_outcomes, self.hparams["outcome_emb_dim"]], final_act='relu'
            )
            outcome_dim = self.hparams["outcome_emb_dim"]
            params.extend(list(self.outcomes_embeddings.parameters()))
        else:
            outcome_dim = num_outcomes

        if self.embed_treatments:
            self.treatments_embeddings = torch.nn.Embedding(
                self.num_treatments, self.hparams["treatment_emb_dim"]
            )
            treatment_dim = self.hparams["treatment_emb_dim"]
            params.extend(list(self.treatments_embeddings.parameters()))
        else:
            treatment_dim = num_treatments

        if self.embed_covariates:
            self.covariates_embeddings = []
            for num_covariate in self.num_covariates:
                self.covariates_embeddings.append(
                    torch.nn.Embedding(num_covariate, 
                        self.hparams["covariate_emb_dim"]
                    )
                )
            self.covariates_embeddings = torch.nn.Sequential(
                *self.covariates_embeddings
            )
            covariate_dim = self.hparams["covariate_emb_dim"]*len(self.num_covariates)
            for emb in self.covariates_embeddings:
                params.extend(list(emb.parameters()))
        else:
            covariate_dim = sum(num_covariates)

        # set models
        self.encoder = encoder_model(
            [outcome_dim+treatment_dim+covariate_dim]
            + [self.hparams["encoder_width"]] * self.hparams["encoder_depth"]
            + [self.hparams["latent_dim"]],
            final_act='relu'
        )
        params.extend(list(self.encoder.parameters()))

        self.decoder = decoder_model(
            [self.hparams["latent_dim"]+treatment_dim]
            + [self.hparams["decoder_width"]] * self.hparams["decoder_depth"]
            + [num_outcomes * self.num_dist_params]
        )
        params.extend(list(self.decoder.parameters()))

        # optimizer
        self.optimizer_autoencoder = torch.optim.Adam(
            params,
            lr=self.hparams["autoencoder_lr"],
            weight_decay=self.hparams["autoencoder_wd"],
        )
        self.scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
            self.optimizer_autoencoder, step_size=self.hparams["step_size_lr"]
        )

        # distribution mode
        if self.dist_mode == 'classify':
            self.treatment_classifier = MLP(
                [num_outcomes]
                + [self.hparams["classifier_width"]] * self.hparams["classifier_depth"]
                + [num_treatments]
            )
            self.loss_treatment_classifier = torch.nn.CrossEntropyLoss()
            params = list(self.treatment_classifier.parameters())

            self.covariate_classifier = []
            self.loss_covariate_classifier = []
            for num_covariate in self.num_covariates:
                classifier = MLP(
                    [num_outcomes]
                    + [self.hparams["classifier_width"]]
                        * self.hparams["classifier_depth"]
                    + [num_covariate],
                    final_act=(None if num_covariate==1 else 'softmax')
                )
                self.covariate_classifier.append(classifier)
                self.loss_covariate_classifier.append(torch.nn.CrossEntropyLoss())
                params.extend(list(classifier.parameters()))

            self.optimizer_classifier = torch.optim.Adam(
                params,
                lr=self.hparams["classifier_lr"],
                weight_decay=self.hparams["classifier_wd"],
            )
            self.scheduler_classifier = torch.optim.lr_scheduler.StepLR(
                self.optimizer_classifier, step_size=self.hparams["step_size_lr"]
            )
        elif self.dist_mode == 'discriminate':
            self.discriminator = MLP(
                [num_outcomes+num_treatments+sum(num_covariates)]
                + [self.hparams["discriminator_width"]] * self.hparams["discriminator_depth"]
                + [1]
            )
            self.loss_discriminator = torch.nn.BCEWithLogitsLoss()
            params = list(self.discriminator.parameters())

            self.optimizer_discriminator = torch.optim.Adam(
                params,
                lr=self.hparams["discriminator_lr"],
                weight_decay=self.hparams["discriminator_wd"],
            )
            self.scheduler_discriminator = torch.optim.lr_scheduler.StepLR(
                self.optimizer_discriminator, step_size=self.hparams["step_size_lr"]
            )
        elif self.dist_mode == 'fit':
            self.outcome_estimator = MLP(
                [treatment_dim+covariate_dim]
                + [self.hparams["estimator_width"]] * self.hparams["estimator_depth"]
                + [num_outcomes * self.num_dist_params]
            )
            self.loss_outcome_estimator = torch.nn.MSELoss()
            params = list(self.outcome_estimator.parameters())

            self.optimizer_estimator = torch.optim.Adam(
                params,
                lr=self.hparams["estimator_lr"],
                weight_decay=self.hparams["estimator_wd"],
            )
            self.scheduler_estimator = torch.optim.lr_scheduler.StepLR(
                self.optimizer_estimator, step_size=self.hparams["step_size_lr"]
            )
        elif self.dist_mode == 'match':
            pass
        else:
            raise ValueError("dist_mode not recognized")

        self.iteration = 0

        self.history = {"epoch": [], "stats_epoch": []}

        self.to_device(device)

        ############################## graph ##############################
        self.optimizer_autoencoder.add_param_group(
            {'params': self.features_embeddings, 'lr': self.hparams["feature_emb_lr"]}
        )
        self.optimizer_autoencoder.add_param_group(
            {'params': self.edge_weight_logits, 'lr': self.hparams["edge_weight_lr"]}
        )

        if self.node_aggr:
            self.final_layer = MLP(
                [self.hparams["decoder_width"], self.num_nodes * self.num_dist_params]
            )
            self.optimizer_autoencoder.add_param_group(
                {'params': self.final_layer.parameters(), 'lr': self.hparams["autoencoder_lr"]}
            )

            self.final_layer.to(self.device)
        ###################################################################

    def set_hparams_(self, hparams):
        """
        Set hyper-parameters to default values or values fixed by user for those
        hyper-parameters specified in the JSON string `hparams`.
        """
        ############################## graph ##############################
        ###################################################################
        self.hparams = {
            "latent_dim": 16,
            "outcome_emb_dim": 4,
            "treatment_emb_dim": 8,
            "covariate_emb_dim": 4,
            "feature_emb_dim": 8,
            "encoder_width": 16,
            "encoder_depth": 1,
            "decoder_width": 64,
            "decoder_depth": 4,
            "classifier_width": 64,
            "classifier_depth": 2,
            "discriminator_width": 64,
            "discriminator_depth": 2,
            "estimator_width": 64,
            "estimator_depth": 2,
            "reg_recon": 2.0,
            "reg_edge_weight": 20.0,
            "kde_kernel_std": 1.,
            "autoencoder_lr": 3e-4,
            "classifier_lr": 3e-4,
            "discriminator_lr": 3e-4,
            "estimator_lr": 3e-4,
            "feature_emb_lr": 3e-4,
            "edge_weight_lr": 5e-3,
            "autoencoder_wd": 4e-7,
            "classifier_wd": 4e-7,
            "discriminator_wd": 4e-7,
            "estimator_wd": 4e-7,
            "batch_size": 64,
            "adversary_steps": 3,
            "step_size_lr": 45,
        }

        # the user may fix some hparams
        if hparams != "":
            if isinstance(hparams, str):
                self.hparams.update(json.loads(hparams))
            else:
                self.hparams.update(hparams)

        return self.hparams

    def early_stopping(self, score):
        """
        Decays the learning rate, and possibly early-stops training.
        """
        self.scheduler_autoencoder.step()

        if score > self.best_score:
            self.best_score = score
            self.patience_trials = 0
        else:
            self.patience_trials += 1

        return self.patience_trials > self.patience

    def encode(self, outcomes, covariates,
                edge_index, edge_weight_logits):
        ############################## graph ##############################
        ###################################################################
        if self.embed_outcomes:
            outcomes = self.outcomes_embeddings(outcomes)
        if self.embed_covariates:
            covariates = [emb(covar.argmax(1)) 
                for covar, emb in zip(covariates, self.covariates_embeddings)
            ]
        
        features = self.features_embeddings

        # expand dims
        outcomes = outcomes.unsqueeze(-1)
        features = features.repeat(outcomes.size(0), 1, 1)
        covariates = torch.cat(covariates, -1)
        if covariates.dim() < outcomes.dim():
            covariates = torch.repeat_interleave(
                covariates.unsqueeze(1), outcomes.size(1), dim=1
            )

        inputs = torch.cat([outcomes, features, covariates], -1)

        return self.encoder(inputs, edge_index, edge_weight_logits)

    def decode(self, latents,
                edge_index, edge_weight_logits):
        ############################## graph ##############################
        ###################################################################
        outputs = self.decoder(latents, edge_index, edge_weight_logits)
        return self.final_layer(outputs) if self.node_aggr else outputs

    def predict(
        self,
        outcomes,
        covariates,
        return_dist=False
    ):
        """
        Predict "what would have the gene expression `outcomes` been, had the
        cells in `outcomes` with cell types `cell_types` been treated with
        combination of treatments `treatments`.
        """ 
        ############################## graph ##############################
        ###################################################################
        outcomes, covariates = self.move_inputs(
            outcomes, covariates
        )

        with torch.autograd.no_grad():
            edge_index, edge_weight_logits = self.get_edge()

            latent = self.encode(
                outcomes, covariates, 
                edge_index, edge_weight_logits
            )
            outcomes_cf = self.decode(
                latent, 
                edge_index, edge_weight_logits
            )

            outcomes_dist = self.distributionize(outcomes_cf)

        if return_dist:
            return outcomes_dist
        else:
            return outcomes_dist.mean

    def distributionize(self, constructions, dim=None, dist=None, eps=1e-3):
        ############################## graph ##############################
        ###################################################################
        if dist is None:
            dist = self.outcome_dist

        if self.node_aggr:
            if dim is None:
                dim = self.num_nodes

            if dist == 'nb':
                mus = F.softplus(constructions[:, :dim]).add(eps)
                thetas = F.softplus(constructions[:, dim:]).add(eps)
                dist = NegativeBinomial(
                    mu=mus, theta=thetas
                )
            elif dist == 'zinb':
                mus = F.softplus(constructions[:, :dim]).add(eps)
                thetas = F.softplus(constructions[:, dim:(2*dim)]).add(eps)
                zi_logits = constructions[:, (2*dim):].add(eps)
                dist = ZeroInflatedNegativeBinomial(
                    mu=mus, theta=thetas, zi_logits=zi_logits
                )
            elif dist == 'normal':
                locs = constructions[:, :dim]
                scales = F.softplus(constructions[:, dim:]).add(eps)
                dist = Normal(
                    loc=locs, scale=scales
                )
        else:
            if dim is None:
                dim = self.num_outcomes

            if dist == 'nb':
                mus = F.softplus(constructions[:, :, :dim]).squeeze(-1).add(eps)
                thetas = F.softplus(constructions[:, :, dim:]).squeeze(-1).add(eps)
                dist = NegativeBinomial(
                    mu=mus, theta=thetas
                )
            elif dist == 'zinb':
                mus = F.softplus(constructions[:, :, :dim]).squeeze(-1).add(eps)
                thetas = F.softplus(constructions[:, :, dim:(2*dim)]).squeeze(-1).add(eps)
                zi_logits = constructions[:, :, (2*dim):].squeeze(-1).add(eps)
                dist = ZeroInflatedNegativeBinomial(
                    mu=mus, theta=thetas, zi_logits=zi_logits
                )
            elif dist == 'normal':
                locs = constructions[:, :, :dim].squeeze(-1)
                scales = F.softplus(constructions[:, :, dim:]).squeeze(-1).add(eps)
                dist = Normal(
                    loc=locs, scale=scales
                )

        return dist

    def logprob(self, outcomes, outcomes_param, dist=None):
        """
        Compute log likelihood.
        """
        if dist is None:
            dist = self.outcome_dist

        num = len(outcomes)
        if isinstance(outcomes, list):
            notNone = [o != None for o in outcomes]
            outcomes = [o for (o, n) in zip(outcomes, notNone) if n]
            outcomes_re = [out[notNone] for out in outcomes_re]

            num = len(outcomes)
            sizes = torch.tensor(
                [out.size(0) for out in outcomes], device=self.device
            )
            weights = torch.repeat_interleave(1./sizes, sizes, dim=0)
            outcomes_param = [
                torch.repeat_interleave(out, sizes, dim=0) 
                for out in outcomes_param
            ]
            outcomes = torch.cat(outcomes, 0)
        elif isinstance(outcomes_param[0], list):
            sizes = torch.tensor(
                [out.size(0) for out in outcomes_param[0]], device=self.device
            )
            weights = torch.repeat_interleave(1./sizes, sizes, dim=0)
            outcomes = torch.repeat_interleave(outcomes, sizes, dim=0)
            outcomes_param = [
                torch.cat(out, 0)
                for out in outcomes_param
            ]
        else:
            weights = None

        if dist == 'nb':
            logprob = logprob_nb_positive(outcomes,
                mu=outcomes_param[0],
                theta=outcomes_param[1],
                weight=weights
            )
        elif dist == 'zinb':
            logprob = logprob_zinb_positive(outcomes,
                mu=outcomes_param[0],
                theta=outcomes_param[1],
                zi_logits=outcomes_param[2],
                weight=weights
            )
        elif dist == 'normal':
            logprob = logprob_normal(outcomes,
                loc=outcomes_param[0],
                scale=outcomes_param[1],
                weight=weights
            )

        return (logprob.sum(0)/num).mean()

    def update(self, outcomes, covariates):
        """
        Update DPI's parameters given a minibatch of outcomes, treatments, and
        cell types.
        """
        ############################## graph ##############################
        ###################################################################
        outcomes, covariates = self.move_inputs(
            outcomes, covariates
        )

        edge_index, edge_weight_logits = self.get_edge()

        latent = self.encode(
            outcomes, covariates,
            edge_index, edge_weight_logits)

        outcomes_re = self.decode(latent,
            edge_index, edge_weight_logits)

        outcomes_dist = self.distributionize(outcomes_re)
        reconstruction_loss = -outcomes_dist.log_prob(outcomes).mean()

        loss = reconstruction_loss
        ############################## graph ##############################
        loss += self.hparams["reg_edge_weight"]*torch.mean(torch.sigmoid(self.edge_weight_logits))
        ###################################################################
        self.optimizer_autoencoder.zero_grad()
        loss.backward()
        self.optimizer_autoencoder.step()
        self.iteration += 1

        return {
            "loss_reconstruction": reconstruction_loss.item()
        }

    def get_edge(self):
        ############################## graph ##############################
        ###################################################################
        if self.dropout:
            mask_dist = RelaxedBernoulli(
                temperature=0.1, logits=self.edge_dropout_logits
            )
            return mask_dist.sample(), self.edge_weight_logits
        else:
            return self.edge_index, self.edge_weight_logits

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

    def move_input(self, input):
        """
        Move minibatch tensors to CPU/GPU.
        """
        if isinstance(input, list):
            return [i.to(self.device) if i is not None else None for i in input]
        else:
            return input.to(self.device)

    def move_inputs(self, *inputs: torch.Tensor):
        """
        Move minibatch tensors to CPU/GPU.
        """
        return [self.move_input(i) if i is not None else None for i in inputs]

    def to_device(self, device):
        self.device = device
        self.to(self.device)

    @classmethod
    def defaults(self):
        """
        Returns the list of default hyper-parameters for DPI
        """

        return self.set_hparams_(self, "")
