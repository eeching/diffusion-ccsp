import ipdb
from os.path import join, abspath
import torch
import torch.nn as nn
import torch.nn.functional as F
import jactorch
import numpy as np
import math
import matplotlib.pyplot as plt
import jactorch.nn as jacnn

from collections import defaultdict
from inspect import isfunction
import pdb


puzzle_constraints = ['in', 'cfree']
robot_constraints = ['gin', 'gfree']
stability_constraints = ['within', 'supportedby', 'cfree']
qualitative_constraints = [
    'in', 'center-in', 'left-in', 'right-in', 'top-in', 'bottom-in',
    'cfree', 'left-of', 'top-of',
    'close-to', 'away-from', 'h-aligned', 'v-aligned'
]
robot_qualitative_constraints = robot_constraints + qualitative_constraints
ignored_constraints = ['right-of', 'bottom-of']
# tidy_constraints = ['aligned_bottom', 'aligned_top', 'aligned_left', 'aligned_right' 'left_of', 'right_of', 'centered', 'avoid_edge',
#                      'not_obstructed', 'in_container', 'on_top_of', 'regular_grid', 'stacked', 'ordered']
# v0
# tidy_constraints = ['aligned_bottom', 'aligned_top', 'in', 'center-in', 'left-in', 'right-in', 'top-in', 'bottom-in',
#     'cfree', 'h-aligned', 'v-aligned']
# tidy_constraints = ['aligned_bottom', 'in', 'cfree']
tidy_constraints = ['aligned_bottom', 'next_to', 'centered_table', 'centered', 'next_to_edge', 'on_top_of']
# tidy_constraints_complement = ['ccollide', 'ccollidse-complement', 'next_to_edge_complement']

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


#################################################################################


# Define how to multiply two different EBM distributions together
class ComposedEBMDenoiseFn(nn.Module):
    """ wrapper around ConstraintDiffuser as a composition of diffusion models """

    def __init__(self, input_mode="tidy", dims=((2, 0, 2), (2, 2, 4)), hidden_dim=256, device='cuda', relation_sets=None, 
                 EBM="MALA", pretrained=False, normalize=True, energy_wrapper=True, verbose=True, ebm_per_steps=1, eval_only=False):
        super().__init__()
        self.dims = dims
        self.device = device
        self.input_mode = input_mode
        self.ebm_per_steps = ebm_per_steps
        self.relation_sets = relation_sets

        self.EBM = EBM
        self.pretrained = pretrained
        self.normalize = normalize
        self.verbose = verbose
        self.hidden_dim = hidden_dim
        
        self.energy_wrapper = energy_wrapper
        self.eval_only = eval_only
        self.use_image = False
        

        self.geom_encoder = nn.Sequential(
            nn.Linear(dims[0][0], hidden_dim//2),
            nn.SiLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.SiLU(),
        ).to(self.device)
        if self.verbose: print_network(self.geom_encoder[0], 'geom_encoder')
        
        ## for encoding object pose, e.g. (x, y)
        self.pose_encoder = nn.Sequential(
            nn.Linear(dims[-1][0], hidden_dim//2),
            nn.SiLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.SiLU(),
        ).to(self.device)

        self.pose_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.SiLU(),
            nn.Linear(hidden_dim//2, dims[-1][0]),
        ).to(self.device)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.Mish(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.models = self.initiate_denoise_fn()

    def neg_logp_unnorm(self, poses_in, batch, t, **kwargs):
        # poses_in.requires_grad_(True)
    
        gradients, energy = self.forward(poses_in, batch, t, **kwargs)
        return energy

    
    def initiate_denoise_fn(self):

        # if self.verbose: print(f'denoise_fns({len(self.constraint_sets)})', self.constraint_sets)
        if self.verbose: print(f'denoise_fns({len(self.relation_sets)})', self.relation_sets)

        models = {}
        for relation in self.relation_sets:
            """ o, o, p, p, t """
            model = ConstraintDiffuser(dims=self.dims, pretrained=self.pretrained, device=self.device, 
                                       hidden_dim=self.hidden_dim, verbose=self.verbose, 
                                       relation=relation, pose_decoder=self.pose_decoder
                                       )
            if self.energy_wrapper:
                model = EBMDiffusionModel(model, self.ebm_per_steps)
            models[relation] = model # list of ebms
            
        if self.verbose: print('-' * 50)
        return models

    def _get_constraint_inputs(self, relation, batch, t, emb_dict, edge_index):
        
        ## find the nodes used by all constraints of this type
        ### edge_attr encodes the constraint type
        idx = tidy_constraints.index(relation)
        edges = torch.where(batch.edge_attr == idx)[0]
        edges = edges.detach().cpu().numpy()

        args_1 = edge_index[edges][:, 0]
        args_2 = edge_index[edges][:, 1]

        args = torch.stack([args_1, args_2], dim=1)
        input_dict = {'args': args}
        
        input_dict.update({
                'geoms_emb': emb_dict['geoms_emb'][args],
                'poses_emb': emb_dict['poses_emb'][args],
                'time_embedding': self.time_mlp(jactorch.add_dim(t, 0, edges.shape[0]))[:, 0],
        })
       
        return input_dict

    # check if you are computing the energy for each entry/relation
    def _compute_energy(self, input_dict, outputs, all_energy_out, all_counts_out, pos_relation):  # noqa
        # input_poses_in = poses_in[input_dict['args']]
        # return ((outputs - input_poses_in) ** 2).sum()

        # batch_size
        n_features = all_energy_out.shape[0]
        
        # all nodes
        args = input_dict['args'].reshape(-1)

        # batch_size*2, dim
        # outputs = outputs.reshape(-1, outputs.shape[-1])
        outputs = outputs.reshape(-1)
        if pos_relation is False:
            outputs = -outputs
        all_energy_out.scatter_add_(0, args, outputs)

        ## take the average of the output pose features of each object
        if all_counts_out is not None:
            all_counts_out += torch.bincount(args, minlength=n_features).to(self.device)

        return all_energy_out, all_counts_out

    def _add_constraints_outputs(self, input_dict, outputs, all_poses_out, all_counts_out, pos_relation):
        
        # batch_size
        n_features = all_poses_out.shape[0]
        
        # all nodes
        args = input_dict['args'].reshape(-1)

        # batch_size*2, dim
        outputs = outputs.reshape(-1, outputs.shape[-1])

        if pos_relation is False:
            outputs = -outputs

        all_poses_out.scatter_add_(0, args.unsqueeze(-1).expand(outputs.shape), outputs)

        ## take the average of the output pose features of each object
        if all_counts_out is not None:
            all_counts_out += torch.bincount(args, minlength=n_features).to(self.device)

        return all_poses_out, all_counts_out
    
    def _get_EBM_gradients(self, poses_in, all_energy_out, **kwargs):

        total_energy = all_energy_out.sum()
       
        if self.eval_only:
            gradients = torch.autograd.grad(total_energy, poses_in, **kwargs)[0]
        else:
            gradients = torch.autograd.grad(total_energy, poses_in, create_graph=True, retain_graph=True, **kwargs)[0]            

        return gradients, total_energy


    def forward(self, poses_in, batch, t, verbose=False, debug=False, tag='EBM', eval=False, **kwargs):
        """ denoising a batch of ConstraintGraphData
        Args:
            poses_in:       torch.Tensor, which are noisy pose features to be denoised
            batch:         DataBatch
                x:              torch.Tensor, which are original geom and pose features
                edge_index:     torch.Tensor, pairs of indices of features, each is a constraint
                edge_attr:      torch.Tensor, indices of constraints
            t:              torch.Size([1]), e.g. tensor([938], device='cuda:0')
        Returns:
            updated_features:    torch.Tensor, which are denoised pose features
        """

        if isinstance(poses_in, np.ndarray):
            poses_in = torch.tensor(poses_in, device=self.model.device)
            t = torch.tensor(t, device=self.model.device)

        kwargs['tag'] = 'EBM'

        x = batch.x.clone().to(self.device)

        ## read geoms_emb from x
        if eval and self.use_image and x.shape[1] != sum([d[0] for d in self.dims]):
            geoms_emb = x[:, self.dims[0][2]:-self.dims[-1][0]]

        ## compute geoms_emb
        else:
            geoms_in = x[:, self.dims[1][1]:self.dims[1][2]] if self.use_image else x[:, :self.dims[0][2]]
            geoms_emb = self.geom_encoder(geoms_in)
            ## save the processed image features to save sampling time
            if eval and self.use_image:
                batch.x = torch.cat([x[:, :self.dims[0][2]], geoms_emb, x[:, self.dims[-1][1]:]], dim=-1)

        poses_in = poses_in.to(self.device)
        poses_in.requires_grad_(True)

        emb_dict = {'geoms_emb': geoms_emb, 'poses_emb': self.pose_encoder(poses_in)}

        edge_index = batch.edge_index.T.to(self.device)
        all_poses_out = torch.zeros_like(poses_in)
        all_counts_out = torch.zeros_like(poses_in[:, 0])
        all_energy_out = torch.zeros_like(poses_in[:, 0])

        for relation in tidy_constraints:

            input_dict = self._get_constraint_inputs(relation, batch, t, emb_dict, edge_index)
          
            if len(input_dict['args']) == 0:
                continue
            
            if relation in self.relation_sets:
                model = self.models[relation]
                pos_relation = True
            else:
                relation, _ = relation.split('-')
                model = self.models[relation]
                pos_relation = False
            if tag == 'EBM' and self.energy_wrapper:
                outputs = model.neg_logp_unnorm(poses_in, input_dict)
                self._compute_energy(input_dict, outputs, all_energy_out, all_counts_out, pos_relation)
            else:
                outputs = model(poses_in, input_dict)
                self._add_constraints_outputs(input_dict, outputs, all_poses_out, all_counts_out, pos_relation) ##

        if tag == 'EBM' and self.energy_wrapper:

            if self.normalize:
                all_energy_out /= torch.sqrt(all_counts_out).clip(1, 1e6)
            gradients, total_energy = self._get_EBM_gradients(poses_in, all_energy_out)

            return gradients, total_energy
        else:
            """ lastly, assign the original pose features to the static objects """
            if self.normalize:
                all_poses_out /= torch.sqrt(all_counts_out.unsqueeze(-1)).clip(1, 1e6)

            mask = batch.mask.bool().to(self.device)
            all_poses_out[mask] = x[:, -self.dims[-1][0]:][mask]

            if debug:
                self.print_debug_info(batch, x, poses_in, x[:, :self.dims[0][2]], all_poses_out, tag=tag)
            
            all_poses_out = all_poses_out
            return all_poses_out

    def print_debug_info(self, batch, x, poses_in, geoms_in, poses_out, tag='train'):
        graph_indices = batch.x_extract.unique().numpy().tolist()
        for j in graph_indices:
            if j != 0:
                continue
            indices = torch.where(batch.x_extract == j)[0].numpy().tolist()
            print('-' * 50 + f"\n[{tag}]\t graph {int(j)} ({len(indices) - 1} objects)")
            for i in indices:
                print('\tshape =', nice(geoms_in[i]),
                      '\t actual =', nice(x[:, self.dims[-1][1]:self.dims[-1][2]][i]),
                      '\t | noisy =', nice(poses_in[i]),
                      '\t -> predicted =', nice(poses_out[i]))

#################################################################################


class GeomEncoderImage(torch.nn.Module):

    log_dir = abspath(join(__file__, '..', 'encoder_checkpoints', 'GeomEncoderImage'))

    def __init__(self, in_features=64, hidden_dim=256, num_channel=32):
        super(GeomEncoderImage, self).__init__()
        conv2d = nn.Conv2d  ## jacnn.CoordConv2D ##
        self.in_features = in_features
        self.num_channel = num_channel

        self.conv1 = conv2d(in_channels=1, out_channels=num_channel, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = conv2d(in_channels=num_channel, out_channels=num_channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv2d(in_channels=num_channel, out_channels=num_channel, kernel_size=3, stride=1, padding=1)
        self.feature_dim = in_features // (2 ** 3)
        self.fc = nn.Linear(in_features=self.feature_dim ** 2 * num_channel, out_features=hidden_dim)

    def forward(self, x):
        ## reshape x from [b, 2, 4096] to [b x 2, 1, 64, 64]
        if len(x.shape) == 3:
            b, p = x.shape[0], x.shape[1]
            x = x.reshape([b * p, 1, self.in_features, self.in_features])
        ## reshape x from [b, 4096] to [b x 2, 1, 64, 64]
        else:
            b = x.shape[0]
            p = 1
            x = x.reshape([b, 1, self.in_features, self.in_features])

        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))
        x = F.dropout(x, training=self.training)
        if len(x.shape) == 3:
            x = x.reshape([b, p, self.feature_dim ** 2 * self.num_channel])
        else:
            x = x.reshape([b, self.feature_dim ** 2 * self.num_channel])
        return self.fc(x)

    def load_pretrained_weights(self):
        model_dict = torch.load(join(self.log_dir, 'best_model.pt'))
        self.load_state_dict(model_dict)
        for param in self.parameters():
            param.requires_grad = False


class GeomDecoderImage(torch.nn.Module):
    def __init__(self, out_features=64, hidden_dim=256, num_channel=32):
        super(GeomDecoderImage, self).__init__()
        self.out_features = out_features
        self.num_channel = num_channel
        self.feature_dim = out_features // (2 ** 3)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=self.feature_dim ** 2 * num_channel)
        self.t_conv1 = nn.ConvTranspose2d(num_channel, num_channel, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(num_channel, num_channel, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(num_channel, 1, 2, stride=2)

    def forward(self, x):
        x = self.fc(x)
        x = x.reshape([x.shape[0], self.num_channel, self.feature_dim, self.feature_dim])
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = torch.sigmoid(self.t_conv3(x))
        x = x.reshape([x.shape[0], self.out_features * self.out_features])
        return x


class GeomAutoEncoder(torch.nn.Module):
    def __init__(self, in_features=64, hidden_dim=256, num_channel=32):
        super(GeomAutoEncoder, self).__init__()
        self.in_features = in_features
        self.encoder = GeomEncoderImage(in_features, hidden_dim, num_channel)
        self.decoder = GeomDecoderImage(in_features, hidden_dim, num_channel)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def visualize_image(self, before, after, png_name):
        num_images = before.shape[0]
        plt.figure(figsize=(10, 5*num_images))
        for i, (x, title) in enumerate(zip([before, after], ['Before', 'After'])):
            x = x.reshape([num_images, self.in_features, self.in_features]).detach().cpu().numpy()
            for j in range(num_images):
                y = x[j] * 255
                plt.subplot(num_images, 2, j*2+i+1)
                plt.imshow(y, interpolation='nearest')
                plt.axis('off')
                plt.title(title)
        plt.savefig(png_name, bbox_inches='tight')
        plt.close()


def print_network(net, name):
    print(name, '\t', net)

class EBMDiffusionModel(torch.nn.Module):

    def __init__(self, model, ebm_per_steps=1):
        super().__init__()
        self.model = model
        self.device = model.device
        self.dims = model.dims
        self.ebm_per_steps = ebm_per_steps

        self.energy_wrapper = True
    
    
    def neg_logp_unnorm(self, poses_in, input_dict, **kwargs):
        """
        computing the energy for each entry/relation
        """
        kwargs['tag'] = 'EBM'
        output = self.model.forward(poses_in, input_dict, **kwargs)
        input_poses_in = poses_in[input_dict['args']]

        energy = ((input_poses_in - output) ** 2).sum(-1)
       
        return energy
    
    def forward(self, poses_in, input_dict, **kwargs):
        
        neg_logp_unnorm = lambda _x:self.neg_logp_unnorm(_x, input_dict, **kwargs)
        return torch.autograd.grad(neg_logp_unnorm, create_graph=True, retain_graph=True)(poses_in)[0]

class ProductEBMDiffusionModel(torch.nn.Module):
        
    def __init__(self, models, ebm_per_steps=1):
        super().__init__()
        self.models = models
        self.device = models[0].device
        self.dims = models[0].dims
        self.input_mode = models[0].input_mode
        self.ebm_per_steps = ebm_per_steps
    
        self.energy_wrapper = True
        

    def neg_logp_unnorm(self, poses_in, batch, t, **kwargs):
            
        kwargs['tag'] = 'EBM'
        unorms = torch.array([model.neg_logp_unnorm(poses_in, batch, t, **kwargs) for model in self.models])
        unorms = unorms.sum()        
        return unorms
        
    def forward(self, poses_in, batch, t, **kwargs):
        if isinstance(poses_in, np.ndarray):
            poses_in = torch.tensor(poses_in, device=self.models[0].device)
            t = torch.tensor(t, device=self.models[0].device)
            
        poses_in.requires_grad_(True)
        kwargs['tag'] = 'EBM'
        scores = torch.array([model(poses_in, batch, t, **kwargs) for model in self.models])
        scores = scores.sum()

        return scores    

class MixtureEBMDiffusionModel(torch.nn.Module):
            
    def __init__(self, models, ebm_per_steps=1):
        super().__init__()
        self.models = models
        self.device = models[0].device
        self.dims = models[0].dims
        self.input_mode = models[0].input_mode
        self.ebm_per_steps = ebm_per_steps
    
        self.energy_wrapper = True
            
    
    # for p_{sum}, we sum the logp_norm
    def neg_logp_unnorm(self, poses_in, batch, t, **kwargs):
                
        kwargs['tag'] = 'EBM'
        concat_energy = torch.stack([model.neg_logp_unnorm(poses_in, batch, t, **kwargs) for model in self.models], axis=-1)
        energy = - torch.logsumexp(-concat_energy*3.5, dim=-1)
            
        return energy
            
    def forward(self, poses_in, batch, t, **kwargs):
        if isinstance(poses_in, np.ndarray):
            poses_in = torch.tensor(poses_in, device=self.models[0].device)
            t = torch.tensor(t, device=self.models[0].device)
                
        poses_in.requires_grad_(True)
        kwargs['tag'] = 'EBM'
        neg_logp_unnorm = lambda _x:self.neg_logp_unnorm(_x, batch, t, **kwargs)
        return torch.autograd.grad(neg_logp_unnorm)(poses_in)[0]
    
class NegationEBMDiffusionModel(torch.nn.Module):
            
    def __init__(self, model, ebm_per_steps=1):
        super().__init__()
        self.model = model
        self.device = model.device
        self.dims = model.dims
        self.input_mode = model.input_mode
        self.ebm_per_steps = ebm_per_steps
    
        self.energy_wrapper = True
            
    # for p_{sum}, we sum the logp_norm
    def neg_logp_unnorm(self, poses_in, batch, t, **kwargs):
                
        kwargs['tag'] = 'EBM'
        energy = - self.model.neg_logp_unnorm(poses_in, batch, t, **kwargs)
            
        return energy
            
    def forward(self, poses_in, batch, t, **kwargs):
        if isinstance(poses_in, np.ndarray):
            poses_in = torch.tensor(poses_in, device=self.model.device)
            t = torch.tensor(t, device=self.model.device)
                
        poses_in.requires_grad_(True)
        kwargs['tag'] = 'EBM'
        neg_logp_unnorm = lambda _x:self.neg_logp_unnorm(_x, batch, t, **kwargs)
        return torch.autograd.grad(neg_logp_unnorm)(poses_in)[0]
    
'''
Diffusion model that encodes a single type of relations
'''
class ConstraintDiffuser(torch.nn.Module):
    def __init__(self, dims=((2, 0, 2), (2, 2, 4)), hidden_dim=256, max_num_obj=12, pretrained=False,
                device='cuda', verbose=True, pose_decoder=None, relation=None):
        """ in_features: list of input feature dimensions for each variable type (geometry, pose)
            e.g. for puzzle constraints ((6, 0, 6), (4, 6, 10)) = {(length, begin, end)}
                means 6 geometry features for pose, 4 for pose features
            e.g. for robot constraints ((6, 0, 6), (5, 8, 13), (5, 14, 18)) = {(length, begin, end)}
                means 6 geometry features for pose, taking position 0-5
        """
        super(ConstraintDiffuser, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_num_obj = max_num_obj
        self.device = device
        self.dims = dims
        self.use_image = False
        self.verbose = verbose
        self.relation = relation
        self.pose_decoder = pose_decoder

        # register the active tidy constraint
        self.active_constraint_idx = tidy_constraints.index(relation)
       
        # self.geom_encoder = nn.Sequential(
        #     nn.Linear(dims[0][0], hidden_dim//2),
        #     nn.SiLU(),
        #     nn.Linear(hidden_dim//2, hidden_dim),
        #     nn.SiLU(),
        # ).to(self.device)
        # if self.verbose: print_network(self.geom_encoder[0], 'geom_encoder')
        
        # ## for encoding object pose, e.g. (x, y)
        # self.pose_encoder = nn.Sequential(
        #     nn.Linear(dims[-1][0], hidden_dim//2),
        #     nn.SiLU(),
        #     nn.Linear(hidden_dim//2, hidden_dim),
        #     nn.SiLU(),
        # ).to(self.device)

        # self.pose_decoder = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim//2),
        #     nn.SiLU(),
        #     nn.Linear(hidden_dim//2, dims[-1][0]),
        # ).to(self.device)

        # self.time_mlp = nn.Sequential(
        #     SinusoidalPosEmb(hidden_dim),
        #     nn.Linear(hidden_dim, hidden_dim * 4),
        #     nn.Mish(),
        #     nn.Linear(hidden_dim * 4, hidden_dim),
        # )

        ## for each type of constraints
       
        self.mlp = self.initiate_denoise_fn()

        self.ebm_per_steps = 1

    def initiate_denoise_fn(self):
        out_feature = 2 * self.hidden_dim  ## change two poses for now

        if self.verbose: print(f'denoise_fns', self.relation)

        if self.relation in robot_constraints:
            """ g, o, o, p, p, t """
            linear = nn.Linear(self.hidden_dim * 6, out_feature)
        else:
            """ o, o, p, p, t """
            linear = nn.Linear(self.hidden_dim * 5, out_feature) ## * 2 because of o and p
        mlp = nn.Sequential(linear, nn.SiLU()).to(self.device) 
            
        if self.verbose: print_network(mlp, '\t'+self.relation)
        if self.verbose: print('-' * 50)
        return mlp

    def forward(self, poses_in, input_dict, verbose=False, debug=False, tag='EBM'):

        geom_emb = input_dict['geoms_emb']  ## torch.Size([4, 2, 256])
        pose_emb = input_dict['poses_emb']  ## torch.Size([4, 2, 256])
        
        embeddings = [
            geom_emb.reshape(geom_emb.shape[0], -1),  ## torch.Size([4, 512])
            pose_emb.reshape(pose_emb.shape[0], -1),  ## torch.Size([4, 512])
            input_dict['time_embedding']  ## torch.Size([4, 256])
        ]

        inputs = torch.cat(embeddings, dim=-1)  ## [b, 5 * hidden_dim]

        outputs = self.mlp(inputs)
        outs_1 = outputs[:, :self.hidden_dim]
        outs_2 = outputs[:, self.hidden_dim:]
        ## decode the output pose features to objects
        outputs = torch.stack([outs_1, outs_2], dim=1)  # [B, 2, dim]
       
        outputs = self.pose_decoder(outputs)
        # if self.composing_weight[0] != 1:
        #     outputs *= self.composing_weight[0]
        
        return outputs

    def print_debug_info(self, batch, x, poses_in, geoms_in, poses_out, tag='train'):
        graph_indices = batch.x_extract.unique().numpy().tolist()
        for j in graph_indices:
            if j != 0:
                continue
            indices = torch.where(batch.x_extract == j)[0].numpy().tolist()
            print('-' * 50 + f"\n[{tag}]\t graph {int(j)} ({len(indices) - 1} objects)")
            for i in indices:
                print('\tshape =', nice(geoms_in[i]),
                      '\t actual =', nice(x[:, self.dims[-1][1]:self.dims[-1][2]][i]),
                      '\t | noisy =', nice(poses_in[i]),
                      '\t -> predicted =', nice(poses_out[i]))


def nice(x):
    return [round(n, 2) for n in x.cpu().detach().numpy()]
