import math
from pprint import pprint
import numpy as np
import torch
import copy
import json
import os
from os.path import dirname, abspath, join, isdir
from collections import defaultdict
import pdb
from networks.denoise_fns import tidy_constraints_dict

torch.set_printoptions(linewidth=1000, precision=2, sci_mode=False)
np.set_printoptions(linewidth=1000, precision=2, floatmode="fixed")


def print_tensor(name: str, tensor: torch.Tensor):
    shape = tensor.shape if isinstance(tensor, torch.Tensor) else len(tensor)
    print(f'\n{name} {shape}\n{tensor}')


def print_line(label):
    line = "-"*len(label)
    print(f'{line}\n{label}\n{line}')


def get_one_hot(inputs, vocab, max_index):
    one_hot = []
    for item in inputs:
        one_hot.append(np.eye(max_index)[vocab.index(item)])
    return np.stack(one_hot)


def get_grid_index(pose, width, length, grid_size=1):
    """ get the index of the grid that the pose is in """
    x, y, _ = pose
    x = int((width/2 + x) / grid_size)
    y = int((length/2 - y) / grid_size)
    return tuple([y, x])


def get_grid_offset(pose, width, length, grid_size=1):
    """ get the offset of the pose from the center of the grid """
    row, col = get_grid_index(pose, width, length, grid_size)
    label = row * int(width / grid_size) + col
    x, y, _ = pose
    dx = x - (col * grid_size - width/2 + grid_size/2)
    dy = y - (length/2 - row * grid_size - grid_size/2)
    return [label, dx, dy]


def get_grid_offset_from_pred(c_pred, topk=1):
    y_pred_softmax = torch.softmax(c_pred, dim=1)  ## log_softmax
    prob, grids = torch.topk(y_pred_softmax, dim=1, k=topk)
    # if topk > 1:
    #     prob = prob.T
    #     grids = grids.T
    return prob, grids
    # return prob.T, grids.T


def get_grid_indices(pose, width, length, grid_size=1):
    """ get the index of the grid that the pose is in """
    from math import ceil as c, floor as f
    col_max, row_max = int(width / grid_size), int(length / grid_size)
    x, y, _ = pose
    m = (width/2 + x) / grid_size
    n = (length/2 - y) / grid_size
    points = [ [f(m), f(n)], [f(m), c(n)], [c(m), f(n)], [c(m), c(n)]]
    points = [tuple([row, col]) for col, row in points if row_max > row >= 0 and col_max > col >= 0]
    return set(points)


def get_grids_offsets(pose, width, length, grid_size=1):
    """ get the offset of the pose from the center of the grid """
    points = get_grid_indices(pose, width, length, grid_size)
    x, y, _ = pose
    labels = []
    for row, col in points:
        grid = row * int(width / grid_size) + col
        dx = x - (col * grid_size - width/2 + grid_size/2)
        dy = y - (length/2 - row * grid_size - grid_size/2)
        labels.append([grid, dx, dy])
    labels = {tuple(p): p[1] ** 2 + p[2] ** 2 for p in labels}
    labels = [list(p) for p in dict(sorted(labels.items(), key=lambda item: item[1])).keys()]
    return labels


def grid_offset_to_pose(grid, offset, width, length, grid_size=1.0):
    """ get the pose of the grid and offset """
    num_cols = int(width / grid_size)
    col = grid % num_cols
    row = (grid - col) / num_cols
    x = col * grid_size - width/2 + grid_size/2 + offset[0]
    y = length/2 - row * grid_size - grid_size/2 + offset[1]
    return (x.item(), y.item())


def get_graph_indices(types):
    gid = []
    last_sum = 0
    y = 1
    while y < len(types):
        y = int(y)
        if types[y] == 0:
            if last_sum != 0:
                gid.append(int(last_sum-1))
                y += last_sum * (last_sum - 2) + 1
                last_sum = 0
        else:
            last_sum += types[y]
            y += 1

    indices = []
    for i in range(len(gid)):
        indices.extend([i]+[' ']*(gid[i]-1))  ## [i]*gid[i]
    return indices


def save_graph_data(nodes, edge_index, labels, data_path):
    import torch
    from torch_geometric.data import Data
    if not isinstance(edge_index[0][0], str):
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    if labels[0] is None:
        labels = [0] * len(nodes)
    data = Data(x=torch.tensor(nodes, dtype=torch.float), edge_index=edge_index,
                y=torch.tensor(np.asarray(labels), dtype=torch.float))
    
    torch.save(data, data_path)


def r(item, roundto=2):
    if isinstance(item, torch.Tensor):
        item = item.numpy()
    elif isinstance(item, np.ndarray):
        item = item.round(roundto)
    elif isinstance(item, list) or isinstance(item, tuple):
        item = [r(i, roundto) for i in item]
    elif isinstance(item, float):
        item = round(item, roundto)
    return item


def apply_grid_mask(grid, offsets):
    """ only use the one pair for the correct grid"""
    left = grid * 2
    right = left + 1
    if len(left.shape) == 0:
        return offsets[left: right + 1]
    return torch.cat([offsets.gather(1, left.view(-1, 1)),
                      offsets.gather(1, right.view(-1, 1))], dim=1)


def apply_grid_masks(grids, offsets):
    """ use pairs corresponding to top four grids """
    grids = [g for g in grids.detach().cpu().tolist() if g != -1]
    offsets_copy = offsets.clone().detach().cpu()
    output = []
    for grid in grids:
        g = torch.tensor([grid], dtype=torch.int64).view(1, 1)
        output.append(apply_grid_mask(g, offsets_copy))
    return torch.cat(output, dim=0)


def world_from_pt(pt_path, world_name='ShapeSettingWorld'):
    """ node = [type, width, length]
        label = [grid, dx, dy]
    """
    import torch
    data = torch.load(pt_path)
    nodes = labels = torch.cat([data.x, data.y], dim=1).numpy()
    return world_from_graph(nodes, labels, world_name=world_name)


def constraint_from_edge_attr(edge_attr, edge_index, composed_inference=False):
    from networks.denoise_fn import qualitative_constraints
    if composed_inference:
        from networks.denoise_fn import robot_qualitative_constraints as qualitative_constraints
    constraints = []
    for i in range(len(edge_attr)):
        typ = int(edge_attr[i].detach().cpu().numpy().item())
        if typ >= len(qualitative_constraints):
            continue
        if composed_inference and typ < 2:
            continue
        constraint = [qualitative_constraints[typ]] + edge_index.T[i].detach().cpu().numpy().tolist()
        constraints.append(constraint)
    return constraints

def tidy_constraint_from_edge_attr(edge_attr, edge_index, model):
    from networks.denoise_fns import tidy_constraints, table_settings
    constraints = []
    if model == "StructDiffusion":
        all_constraints = tidy_constraints + table_settings
    elif model == "Diffusion-CCSP":
        all_constraints = tidy_constraints
    for i in range(len(edge_attr)):
        typ = int(edge_attr[i].detach().cpu().numpy().item())
        constraint = tuple([all_constraints[typ]] + edge_index.T[i].detach().cpu().numpy().tolist()) # still use the index from the tidy constraints
        constraints.append(constraint)
    return constraints

def clustered_type_from_edge_attr(edge_attr, edge_index, model):
    from networks.denoise_fns import clustered_types
    clustered_relations = []
    for i in range(len(edge_attr)):
        typ = int(edge_attr[i].detach().cpu().numpy().item())
        clustered_relation = tuple([clustered_types[typ]] + edge_index.T[i].detach().cpu().numpy().tolist()) # still use the index from the tidy constraints
        clustered_relations.append(clustered_relation)
    return clustered_relations


def world_from_graph(nodes, world_name='ShapeSettingWorld', **kwargs):
    """ input_mode = 'grid_offset_mp4'
            node = [type, width, length, grid, dx, dy]
            label = [type, width, length, grid, dx, dy]
        input_mode = 'grid_offset_oh4'
            node = [(type)x3, width, length, x, y]
            y = [(grid)x25, dx, dy] x 4
        input_mode = 'diffuse_pairwise'
            node = [type_flag_1, type_flag_1, width, length, _, _] & [_, _, _, _, x, y]
            y = [(type)x3, width, length, x, y]
    """
    from envs.worlds import get_world_class

    if isinstance(nodes, torch.Tensor):
        nodes = nodes.detach().cpu().numpy()
    w, l = nodes[0, 1:3]
    world_class = get_world_class(world_name)
    world = world_class(w=w, l=l, h=0.5, grid_size=0.5)
    world.construct_scene_from_graph_data(nodes, **kwargs)
    return world


def render_world(nodes, png_name='png_name', show=False, save=True, array=False,
                 show_grid=True, **kwargs):
    world = world_from_graph(nodes, **kwargs)
    if world.__class__.__name__ == 'TableToBoxWorld':
        show_grid = False
    if save:
        world.render(show=show, show_grid=show_grid, img_name=png_name, array=array)
    return world


def render_world_from_graph(features, world_dims=(3, 2), png_name='diffusion_batch.png', array=False,
                            log=False, verbose=False, **kwargs):
    w_tray, l_tray = world_dims

    def get_node(f, typ):
        """ reconstruct from normalized features """
        if not isinstance(f, list):
            f = f.tolist()

        if len(f) == 4:
            w, l, x, y = f
            geom = [w * w_tray, l * l_tray]
            pose = [x * w_tray/2, y * l_tray/2]

        elif len(f) == 6:

            ## triangle P1 encoding with theta
            if w_tray == 3 and l_tray == 3:
                if typ == 0:
                    w, l, _, x, y, _ = f[-6:]
                    geom = [w * w_tray, l * l_tray, 0.1]
                    pose = [x * w_tray / 2, y * l_tray / 2, 0]
                else:
                    l, x3, y3, x1, y1, r1 = f[-6:]
                    geom = [l * w_tray, x3 * w_tray, y3 * l_tray]
                    pose = [x1 * w_tray / 2, y1 * l_tray / 2, r1 * np.pi]

            ## box encoding with sin/cos (stability)
            else:
                if typ == 0:
                    w, l, x, y, _, _ = f
                    geom = [w * w_tray, l * l_tray]
                    pose = [x * w_tray/2, y * l_tray/2, 0]
                else:
                    w, l, x, y, sn, cs = f
                    roll = yaw_from_sn_cs(sn, cs)
                    geom = [w * w_tray, l * l_tray]
                    pose = [x * w_tray/2, y * l_tray/2, roll]

        ## triangle P1 encoding with sin/cos
        elif len(f) == 7 or len(f) == 7 + 64 ** 2:
            if typ == 0:
                w, l, _, x, y, _, _ = f[:7]
                geom = [w * w_tray, l * l_tray, 0.1]
                pose = [x * w_tray/2, y * l_tray/2, 0, 0]
            else:
                l, x3, y3, x1, y1, cs, sn = f[:7]
                geom = [l * w_tray, x3 * w_tray, y3 * l_tray]
                pose = [x1 * w_tray/2, y1 * l_tray/2, cs, sn]

        ## triangle centroid encoding with sin/cos
        elif len(f) == 10 or len(f) == 10 + 64 ** 2:
            if typ == 0:
                w, l, _, _, _, _ = f[:6]
                x, y, _, _ = f[-4:]
                geom = [w * w_tray, l * l_tray, 0.1, 0, 0, 0]
                pose = [x * w_tray/2, y * l_tray/2, 0, 0]
            else:
                x1, y1, x2, y2, x3, y3 = f[:6]
                x, y, cs, sn = f[-4:]
                geom = [x1 * w_tray, y1 * l_tray, x2 * w_tray, y2 * l_tray, x3 * w_tray, y3 * l_tray]
                pose = [x * w_tray, y * l_tray, cs, sn]

        elif len(f) in [21, 21+7, 21+7+7]:
            if typ == 0:
                geom = [w_tray, l_tray, 0.1, 0, 0]
                pose = [0, 0, 0, 0] + [0]
            else:
                # w, l, h, w0, l0, h0, mobility_id, scale, x, y, z, sn, cs, x0, y0, g1, g2, g3, g4, g5, grasp_id = f[:21]
                w, l, h, w0, l0, h0, x0, y0, mobility_id, scale, g1, g2, g3, g4, g5, grasp_id, x, y, z, sn, cs = f[:21]
                yaw = yaw_from_sn_cs(sn, cs)
                geom = [w*w0, l*l0, h*h0, mobility_id, scale]
                pose = [x*w0/2, y*l0/2, z*h0, yaw] + [grasp_id]

            ## passing hand_pose and grasp_pose for debugging
            if len(f) != 21:
                pose += f[21:]

        return [typ] + geom + pose, pose

    if verbose: print()
    nodes = []
    poses = []

    for i in range(len(features)):
        typ = int(i != 0)
        node, pose = get_node(features[i], typ)
        if verbose: print(f'node {i}: {r(node, 3)}')
        nodes.append(node)
        poses.append(pose)
    nodes = np.asarray(nodes)

    world = render_world(nodes, png_name=png_name, array=array, **kwargs)
  
    evaluations = world.check_constraints_satisfied(verbose=False)
    object_states = []
    if world.__class__.__name__ == 'TableToBoxWorld':

        def get_collisions(eval, i):
            all_collisions = []
            for e in eval:
                if e[0] == 'floating_gripper' and 'tile_' in e[1]:
                    tile_index = int(e[1][e[1].index('tile_')+5: e[1].index('[')])
                    if i == tile_index + 1:
                        continue
                all_collisions.append(tuple([i] + list(e)))
            return all_collisions

        # i = len(nodes) - 1
        # all_evaluations = get_collisions(evaluations, i)
        # while i > 1:
        #     i -= 1
        #     nodes = nodes[:-1]
        #     world = render_world(nodes, png_name=png_name, array=array, **kwargs)
        #     evaluations = world.check_constraints_satisfied(verbose=False)
        #     all_evaluations += get_collisions(evaluations, i)
        # evaluations = all_evaluations
        evaluations = []

        ## only save the poses of the tiles
        dx, dy, _ = poses[0][:3]
        poses = [(pose[:3], quat_from_yaw(pose[4])) for pose in poses[1:]]
        labels = [l for l in world.labels.values() if 'tile_' in l]

        ## as well as pose of the gripper
        for i in range(len(poses)):
            label = labels[i].split('[')[1].split(']')[0]
            object_states.append([label, poses[i], {}])
            object_states.append([f'gripper_{label}', world.gripper_poses[i], {}])

    else:
        if log and len(evaluations) == 0:
            json_name = png_name.replace('.png', '.json')
            world.generate_json(input_mode='diffuse_pairwise', json_name=json_name)
            if 'Triangle' in world.__class__.__name__:
                world.generate_meshes(nodes, mesh_dir_name=png_name.replace('.png', ''))
    if array:
        return world.images[-1], object_states, evaluations
    return evaluations


def yaw_from_sn_cs(sn, cs):
    total = np.sqrt(sn ** 2 + cs ** 2)
    sn /= total
    cs /= total
    return np.arctan2(sn, cs)


def quat_from_yaw(yaw):
    import pybullet_planning as pp
    return pp.quat_from_euler([0, 0, yaw])


def test_cfree(nodes, labels, predictions=None, world_name='ShapeSettingWorld', debug=False):
    nodes = nodes.clone().detach().cpu()
    world = world_from_graph(nodes, labels, predictions=predictions, world_name=world_name, phase='prediction')
    collisions = world.check_collisions_in_scene(verbose=False)
    if debug:
        world.summarize_objects()
        print('\ncollisions', collisions)
        world.render(show=True, show_grid=True)
    return collisions


def compute_pairwise_collisions(world):
    collisions_by_obj = {}
    for elems in world['collisions']:
        for elem in elems:
            if elem not in collisions_by_obj:
                collisions_by_obj[elem] = []
            collisions_by_obj[elem].append([e for e in elems if e != elem][0])
    for k, data in world['objects'].items():
        if data['label'] in collisions_by_obj:
            data['collisions'] = collisions_by_obj[data['label']]
        else:
            data['collisions'] = []
    return world


def compute_world_constraints(world, same_order=False, **kwargs):
    objects = copy.deepcopy(world['objects'])
    objects = {v['label']: v for k, v in objects.items()}  ##  if v['label'] not in ['bottom']
    constraints = compute_qualitative_constraints(objects, **kwargs)
    if not same_order:
        constraints = randomize_unordered_constraints(constraints)
    world['constraints'] += constraints
    return world

def compute_tidy_constraints(world, same_order=False, **kwargs):
    objects = copy.deepcopy(world['objects'])
    objects = {v['label']: v for k, v in objects.items()}  ##  if v['label'] not in ['bottom']
    constraints, verbose_constraints = compute_tidy_atomic_constraints(objects, **kwargs)
    if not same_order:
        constraints = randomize_unordered_constraints(constraints)
    world['constraints'] += constraints
    world['verbose_constraints'] += verbose_constraints
    return world

def compute_clustered_constraints(world, relation, **kwargs):
    objects = copy.deepcopy(world['objects'])
    objects = {v['label']: v for k, v in objects.items()}  ##  if v['label'] not in ['bottom']
    X_0, X_1, Y_0, Y_1, total_area = compute_clustered_atomic_constraints(objects, world['w'], world['l'], world['h'], relation, **kwargs)
 
    world['regions'] += [(X_0, X_1, Y_0, Y_1)]
    world['areas'] += [total_area]
    return world

def randomize_unordered_constraints(constraints):
    new_constraints = []
    for c in constraints:
        if c[0] in ['close-to', 'away-from', 'h-aligned', 'v-aligned'] and np.random.rand() < 0.5:
            new_constraints.append(tuple([c[0], c[2], c[1]]))
        else:
            new_constraints.append(c)
    return new_constraints


def expand_unordered_constraints(constraints):
    new_constraints = []
    for c in constraints:
        if c[0] in ['close-to', 'away-from', 'h-aligned', 'v-aligned', 'cfree', 'horizontally_aligned']:
            new_constraints.append(tuple([c[0], c[2], c[1]]))
        new_constraints.append(c)
    return new_constraints

def get_ordered_constraints(constraints):
    new_constraints = dict()
    
    from networks.denoise_fns import tidy_constraints, ignored_constraints
    constraint_list = [c for c in tidy_constraints if c not in ignored_constraints]
    for c in constraint_list:
        if tidy_constraints_dict[c] == 1:
            new_constraints[c] = [(c, con[1]) for con in constraints if con[0] == c]
        elif c == "vertical_line_symmetry" or c == "horizontal_line_symmetry":
            new_constraints[c] = set()
            c_set = [(c, con[1], con[2], con[3]) for con in constraints if con[0] == c]
            for con in c_set:
                if con[2] > con[3]:
                    new_constraints[c].add((c, con[1], con[3], con[2]))
                else:
                    new_constraints[c].add((c, con[1], con[2], con[3]))
            new_constraints[c] = list(new_constraints[c])
        elif c == "left_of_front" or c == "right_of_front" or c == "left_of_back" or c == "right_of_back" or c == "on_top_of" or c == "centered":
            new_constraints[c] = [(c, con[1], con[2]) for con in constraints if con[0] == c]
        else:
            c_set = [(c, con[1], con[2]) for con in constraints if con[0] == c]
            new_constraints[c] = set()

            for con in c_set:
                if con[1] > con[2]:
                    new_constraints[c].add((c, con[2], con[1]))
                else:
                    new_constraints[c].add((c, con[1], con[2]))
            new_constraints[c] = list(new_constraints[c])
    return new_constraints

def compute_qualitative_constraints(objects, rotations=None, debug=False, scale=1):
    """ objects is a dictionary of tile_name: {center, extents} """
    sides = ['east', 'west', 'north', 'south']
    if debug:
        print('scale', scale)
        print('rotations', rotations)
        pprint({k: {kk: r(vv) for kk, vv in v.items() if kk in ['extents', 'center']}
                for k, v in objects.items() if k not in sides})
        print()

    constraints = []
    names = list(objects.keys())
    tiles = ['bottom'] + [n for n in names if 'tile_' in n]
    neighbors = {n: defaultdict(list) for n in names}  ## the neighbors of each object
    # print(rotations)

    """ left, right, top, bottom """
    alignment = 0.05 * scale
    farness = 0.5 * scale
    closeness = 0.3 * scale
    touching = 0.1 * scale
    overlap_threshold = 0.6 * scale
    for i in range(len(names)):
        m = objects[names[i]]
        if names[i] == 'bottom':
            continue
        name1 = names[i] if 'tile_' in names[i] else 'bottom'

        x1, y1, z1 = m['center']
        lx1, ly1, lz1 = m['extents']
        if rotations is not None and name1 in rotations:
            rot = rotations[name1]
            if abs(abs(rot) - np.pi /2) < 0.1:
                ly1, lx1, lz1 = m['extents']

        x1_left = x1 - lx1 / 2
        x1_right = x1 + lx1 / 2
        y1_top = y1 + ly1 / 2
        y1_bottom = y1 - ly1 / 2

        if math.sqrt(x1**2 + y1**2) < closeness:
            constraints.append(('center-in', tiles.index(name1), tiles.index('bottom')))
        if x1_right < 0:
            constraints.append(('left_half', tiles.index(name1), tiles.index('bottom')))
        if x1_left > 0:
            constraints.append(('right_half', tiles.index(name1), tiles.index('bottom')))
        if y1_top < 0:
            constraints.append(('front_half', tiles.index(name1), tiles.index('bottom')))
        if y1_bottom > 0:
            constraints.append(('back_half', tiles.index(name1), tiles.index('bottom')))

        for j in range(i + 1, len(names)):
            n = objects[names[j]]
            if names[j] == 'bottom':
                continue
            name2 = names[j] if 'tile_' in names[j] else 'bottom'
            if name1 == name2:
                continue

            # if name1 not in sides and name2 not in sides:
            #     debug = True

            x2, y2, z2 = n['center']
            lx2, ly2, lz2 = n['extents']
            if rotations is not None and name2 in rotations:
                rot = rotations[name2]
                if abs(abs(rot) - np.pi /2) < 0.1:
                    ly2, lx2, lz2 = n['extents']
            x2_left = x2 - lx2 / 2
            x2_right = x2 + lx2 / 2
            y2_top = y2 + ly2 / 2
            y2_bottom = y2 - ly2 / 2
            if debug: print(name1, name2)
            if debug: print('\t', name1, r(objects[names[i]]['center'][:2]), r(objects[names[i]]['extents'][:2]))
            if debug: print('\t', name2, r(n['center'][:2]), r(n['extents'][:2]))

            """ aligned """
            if name1 != 'bottom' and name2 != 'bottom':
                if abs(x1 - x2) < alignment:
                    constraints.append(('v-aligned', tiles.index(name1), tiles.index(name2)))
                if abs(y1 - y2) < alignment:
                    constraints.append(('h-aligned', tiles.index(name1), tiles.index(name2)))

            """ top / bottom """

            ## check if they overlap enough on x-axis
            if debug: print(x2_left, x1_left, x1_right, x2_right)
            in_x_range = (x2_left <= x1_left < x1_right <= x2_right) or (x1_left <= x2_left < x2_right <= x1_right)
            if debug: print('\tin_x_range =', in_x_range)

            if not in_x_range:
                overlap = 0
                min_w = min(lx1, lx2)
                if x2_left <= x1_left <= x2_right <= x1_right:
                    overlap = x2_right - x1_left
                elif x1_left <= x2_left <= x1_right <= x2_right:
                    overlap = x1_right - x2_left
                in_x_range = in_x_range or (overlap > min_w * overlap_threshold)
                if debug: print('\tin_x_range overlap =', overlap)

            if in_x_range:

                ref_dist = max(ly1, ly2)

                d = y2_bottom - y1_top
                if debug: print('\td1 =', d)
                if -0.05 <= d < farness:
                    if debug: print('\t0 <= d1 < farness')
                    neighbors[name1]['top'].append((name2, d, ref_dist))
                    neighbors[name2]['bottom'].append((name1, d, ref_dist))

                d = y1_bottom - y2_top
                if debug: print('\td2 =', d)
                if -0.05 <= d < farness:
                    if debug: print('\t0 <= d2 < farness')
                    neighbors[name1]['bottom'].append((name2, d, ref_dist))
                    neighbors[name2]['top'].append((name1, d, ref_dist))

            """ left / right """

            ## check if they overlap enough on y-axis
            in_y_range = (y2_bottom <= y1_bottom < y1_top <= y2_top) or (y1_bottom <= y2_bottom < y2_top <= y1_top)
            if debug: print('\tin_y_range =', in_x_range)

            if not in_y_range:
                overlap = 0
                min_h = min(ly1, ly2)
                if y2_bottom <= y1_bottom <= y2_top <= y1_top:
                    overlap = y2_top - y1_bottom
                elif y1_bottom <= y2_bottom <= y1_top <= y2_top:
                    overlap = y1_top - y2_bottom
                in_y_range = in_y_range or (overlap > min_h * overlap_threshold)
                if debug: print('\tin_y_range overlap =', overlap)

            if in_y_range:

                ref_dist = max(lx1, lx2)

                d = x1_left - x2_right
                if debug: print('\td3 =', d)
                if -0.05 <= d < farness:
                    if debug: print('\t0 <= d3 < farness')
                    neighbors[name1]['left'].append((name2, d, ref_dist))
                    neighbors[name2]['right'].append((name1, d, ref_dist))

                d = x2_left - x1_right
                if debug: print('\td4 =', d)
                if -0.05 <= d < farness:
                    if debug: print('\t0 <= d4 < farness')
                    neighbors[name1]['right'].append((name2, d, ref_dist))
                    neighbors[name2]['left'].append((name1, d, ref_dist))

    for name, relations in neighbors.items():
        if name not in tiles:
            continue
        if debug:
            print(name, '\t', {k: [(v[0], round(v[1], 3)) for v in vv] for k, vv in relations.items()})
        a = tiles.index(name)
        neighbor = [name, 'bottom']
        for k, vv in relations.items():
            for v in vv:
                b = tiles.index(v[0])
                if a == b:
                    continue
                if v[1] < closeness:
                    if k in ['left', 'top']:
                        k2 = {'left': 'right', 'top': 'bottom'}[k]
                        if a != 0 and b != 0:
                            constraints.append((f"{k}-of", b, a))
                            constraints.append((f"{k2}-of", a, b))
                if v[1] < touching and (f"close-to", b, a) not in constraints and \
                        (f"close-to", a, b) not in constraints and a != 0 and b != 0:
                    constraints.append((f"close-to", b, a))
            neighbor += [v[0] for v in vv]
        if name == 'bottom':
            continue
        constraints += [(f"away-from", tiles.index(m), a) for m in tiles if m not in neighbor
                        if (f"away-from", a, tiles.index(m)) not in constraints]
    constraints.sort()

    for x in [c[1] for c in constraints if c[0] == 'right_half']:
        if x in [c[1] for c in constraints if c[0] == 'left_half']:
            constraints.remove(('right_half', x, 0))
            constraints.remove(('left_half', x, 0))
    for x in [c[1] for c in constraints if c[0] == 'front_half']:
        if x in [c[1] for c in constraints if c[0] == 'back_half']:
            constraints.remove(('front_half', x, 0))
            constraints.remove(('back_half', x, 0))

    from networks.denoise_fn import ignored_constraints
    constraints = [c for c in constraints if c[0] not in ignored_constraints]
    # print()
    if debug:
        summarize_constraints(constraints)
    return constraints

def compute_tidy_atomic_constraints(objects, rotations=None, debug=False, scale=1, test_only=False, model_relation=None, composed_relation=None, generating_data=False): # if the relation is None, compute all relations, if not, only compute the specified relations
    """ objects is a dictionary of tile_name: {center, extents} """

    sides = ['east', 'west', 'north', 'south']
    if debug:
        print('scale', scale)
        print('rotations', rotations)
        pprint({k: {kk: r(vv) for kk, vv in v.items() if kk in ['extents', 'center']}
                for k, v in objects.items() if k not in sides})
        print()

    constraints = []
    verbose_constraints = []
    names = list(objects.keys())
    tiles = ['bottom'] + [n for n in names if 'tile_' in n]
    neighbors = {n: defaultdict(list) for n in names}  ## the neighbors of each object
    # print(rotations)

    # obj_names = [objects[names[idx]]["name"] for idx in range(5, len(names))]

    if model_relation is not None and ("regular_grid" in model_relation or "aligned_horizontal_line" in model_relation or "aligned_vertical_line" in model_relation) and composed_relation == "partial":
        composed_partial = True
    else:
        composed_partial = False

    def pad_constraints(tup):
        length = len(tup)
        
        if length == 25:
            return tup
        else:
            padding = (0,) * (25 - length)
            return tup + padding

    # adding constraints for structdiffusion    
    # if model_relation == "study_table" or model_relation == "dining_table" or model_relation == "coffee_table":
    #     constraints.append(tuple([model_relation] + [i for i in range(1, len(tiles))]))
    #     constraints = [pad_constraints(c) for c in constraints]
    #     return list(set(constraints)), []
        
    if model_relation is not None and model_relation == "survey_table":
        return [["central_column", 1]], [["central_column", 1]]
        
    if model_relation is not None and "train" not in model_relation and ("study_table_" in model_relation or "dining_table_" in model_relation or "coffee_table_" in model_relation):
        
        table_type, _, mode = model_relation.split("_")

        if mode == "StructDiffusion":
            obj_names = ['bottom'] + [objects[names[idx]]["name"] for idx in range(5, len(names))]
            
            constraints = [tuple([f"{table_type}_table"] + [i for i in range(1, len(obj_names))])]
            verbose_constraints = [tuple([f"{table_type}_table"] + [obj_names[i] for i in range(1, len(obj_names))])]
        else:
            filename = f'tmp/{table_type}_table/{mode}.json'
            with open(filename) as json_file:
                input_relations = json.load(json_file)
                print("relation_idx: ")
                x = input()
                input_relations = input_relations[x]
            obj_names = ['bottom'] + [objects[names[idx]]["name"] for idx in range(5, len(names))]
            for c in input_relations:
                
                if c[0] == "right_of":
                    constraints.append(tuple(["left_of_front", obj_names.index(c[2]), obj_names.index(c[1])]))
                else:
                    if c[0] == "left_of":
                        typ = "left_of_front"
                    else:
                        typ = c[0]
                    constraints.append(tuple([typ] + [obj_names.index(c[i]) for i in range(1, len(c))]))
            verbose_constraints = [tuple(c) for c in input_relations]

        constraints = [pad_constraints(c) for c in constraints]
        
        return list(set(constraints)), list(set(verbose_constraints))
            
                
    """in regular grid"""
    if generating_data:
        if "regular_grid" in model_relation:
            n_to_set_mapping = {4: [[4]], 6: [[6]], 8: [[8], [4, 4]], 9: [[9]], 10: [[10], [4, 6]],
                                    12: [[6, 6], [4, 8], [4, 4, 4]], 13: [[4, 9]], 14: [[6, 8], [6, 4, 4], [4, 10]], 15: [[6, 9]], 
                                    16: [[16], [6, 10], [4, 6, 6], [8, 8], [8, 4, 4]]}
            _, _, n_obj, n_combi_idx = model_relation.split("_")
            n_obj = int(n_obj)
            n_combi_idx = int(n_combi_idx)
            n_combi = n_to_set_mapping[n_obj][n_combi_idx]
            idx = 1
            for k in n_combi:
                start_obj = objects[names[idx]]
                x1, y1, z1 = start_obj['center']
                lx1, ly1, lz1 = start_obj['extents']
                if rotations is not None and names[idx] in rotations:
                    if abs(abs(rot1) - np.pi /2) < 0.2:
                        ly1, lx1, lz1 = start_obj['extents']
            
                x1_left = x1 - lx1 / 2
                y1_bottom = y1 - ly1 / 2

                end_obj = objects[names[idx + k - 1]]
                x2, y2, z2 = end_obj['center']
                lx2, ly2, lz2 = end_obj['extents']
                if rotations is not None and names[idx + k - 1] in rotations:
                    if abs(abs(rot2) - np.pi /2) < 0.2:
                        ly2, lx2, lz2 = end_obj['extents']
                
                x2_right = x2 + lx2 / 2
                y2_top = y2 + ly2 / 2

                region_w = x2_right - x1_left
                region_h = y2_top - y1_bottom
                
                if region_w > region_h:
                    con = ["horizontal_regular_grid"] + [i for i in range(idx, idx + k)]
                else:
                    con = ["vertical_regular_grid"] + [i for i in range(idx, idx + k)]
                
                constraints.append(tuple(con))
                idx += k
        if model_relation == "aligned_horizontal_line":
            constraints.append(tuple(["aligned_in_horizontal_line"] + [i for i in range(1, len(tiles))]))

        if model_relation == "aligned_vertical_line":
            constraints.append(tuple(["aligned_in_vertical_line"] + [i for i in range(1, len(tiles))]))
        
        # obj_names = [objects[names[idx]]["name"] for idx in range(5, len(names))]
        # if "study_table" in model_relation and "book_1" in obj_names:
        #     books_idx = [idx for idx in range(5, len(names)) if "book" in objects[names[idx]]["name"]]
        #     constraints.append(tuple(["aligned_in_horizontal_line"] + [i-4 for i in books_idx]))
            
    """ left, right, top, bottom """
    # alignment = 0.05 * scale
    alignment = 0.1 * scale
    farness = 0.5 * scale
    closeness = 0.3 * scale
    touching = 0.15 * scale
    overlap_threshold = 0.6 * scale
    for i in range(len(names)):
        m = objects[names[i]]
        if names[i] == 'bottom' or names[i] in sides:
            continue
        
        name1 = names[i]
        x1, y1, z1 = m['center']
        lx1, ly1, lz1 = m['extents']
        if rotations is not None and name1 in rotations:
            rot1 = rotations[name1]
            if abs(abs(rot1) - np.pi /2) < 0.2:
                ly1, lx1, lz1 = m['extents']
        else: 
            rot1 = 0 

        x1_left = x1 - lx1 / 2
        x1_right = x1 + lx1 / 2
        y1_top = y1 + ly1 / 2
        y1_bottom = y1 - ly1 / 2

        if model_relation == "in" or composed_relation == "all" or composed_partial == True or generating_data == False:

            if x1 > -0.75 and x1 < 0.75:
                constraints.append(('central_column', tiles.index(name1)))
            if y1 > -0.5 and y1 < 0.5:
                constraints.append(('central_row', tiles.index(name1)))
            if x1_right < 0:
                constraints.append(('left_half', tiles.index(name1)))
            if x1_left > 0:
                constraints.append(('right_half', tiles.index(name1)))
            if y1_top < 0:
                constraints.append(('front_half', tiles.index(name1)))
            if y1_bottom > 0:
                constraints.append(('back_half', tiles.index(name1)))
        
        if model_relation == "next_to_edge" or composed_relation == "all" or composed_partial == True or generating_data == False:
    
            north_surface = objects['north']['center'][1] + objects['north']['extents'][1]/2
            south_surface = objects['south']['center'][1] - objects['south']['extents'][1]/2
            east_surface = objects['east']['center'][0] + objects['east']['extents'][0]/2
            west_surface = objects['west']['center'][0] - objects['west']['extents'][0]/2

            if abs(y1_top - north_surface) < 0.3:
                constraints.append(('near_back_edge', tiles.index(name1)))
            elif abs(y1_bottom - south_surface) < 0.3:
                constraints.append(('near_front_edge', tiles.index(name1)))
            elif abs(x1_left - west_surface) < 0.3:
                constraints.append(('near_left_edge', tiles.index(name1)))
            elif abs(x1_right - east_surface) < 0.3:
                constraints.append(('near_right_edge', tiles.index(name1)))

        if model_relation == "centered" or composed_relation == "all" or composed_partial == True or generating_data == False:
            if math.sqrt(x1**2 + y1**2) < touching:
                constraints.append(('centered_table', tiles.index(name1)))
       
        for j in range(i + 1, len(names)):
            n = objects[names[j]]

            if names[j] == 'bottom' or names[j] in sides:
                continue
            
            name2 = names[j]

            if name1 == name2:
                continue

            x2, y2, z2 = n['center']
            lx2, ly2, lz2 = n['extents']
            if rotations is not None and name2 in rotations:
                rot2 = rotations[name2]
                if abs(abs(rot2) - np.pi /2) < 0.2:
                    ly2, lx2, lz2 = n['extents']
            else:
                rot2 = 0
            x2_left = x2 - lx2 / 2
            x2_right = x2 + lx2 / 2
            y2_top = y2 + ly2 / 2
            y2_bottom = y2 - ly2 / 2

            if debug: print(name1, name2)
            if debug: print('\t', name1, r(objects[names[i]]['center'][:2]), r(objects[names[i]]['extents'][:2]))
            if debug: print('\t', name2, r(n['center'][:2]), r(n['extents'][:2]))   

            x_1_cover = x1_left <= x2_left < x2_right <= x1_right
            x_2_cover = x2_left <= x1_left < x1_right <= x2_right
            y_1_cover = y1_bottom <= y2_bottom < y2_top <= y1_top
            y_2_cover = y2_bottom <= y1_bottom < y1_top <= y2_top

            """ on_top_of """
            if model_relation == "on_top_of" or composed_relation == "all" or generating_data == False:
                x1_on_top = x_2_cover and y_2_cover       
                x2_on_top = x_1_cover and y_1_cover

                if x1_on_top:
                    constraints.append(('on_top_of', tiles.index(name1), tiles.index(name2)))
                if x2_on_top:
                    constraints.append(('on_top_of', tiles.index(name2), tiles.index(name1)))

            
            x_overlap = 0
            min_w = min(lx1, lx2)
            if x2_left <= x1_left <= x2_right <= x1_right:
                x_overlap = x2_right - x1_left
            elif x1_left <= x2_left <= x1_right <= x2_right:
                x_overlap = x1_right - x2_left
            elif x_1_cover:
                x_overlap = x2_right - x2_left
            elif x_2_cover:
                x_overlap = x1_right - x1_left

            x_overlap = x_overlap > min_w * overlap_threshold

            y_overlap = 0
            min_h = min(ly1, ly2)
            if y1_bottom <= y2_bottom <= y1_top <= y2_top:
                y_overlap = y1_top - y2_bottom
            elif y2_bottom <= y1_bottom <= y2_top <= y1_top:
                y_overlap = y2_top - y1_bottom
            elif y_1_cover:
                y_overlap = y2_top - y2_bottom
            elif y_2_cover:
                y_overlap = y1_top - y1_bottom

            y_overlap = y_overlap > min_h * overlap_threshold

            x_no_overlap = x2_right < x1_left or x1_right < x2_left
            y_no_overlap = y2_top < y1_bottom or y1_top < y2_bottom

            no_overlap = x_no_overlap or y_no_overlap

            x_closeness = abs(x1_left - x2_right) < closeness or abs(x1_right - x2_left) < closeness
            y_closeness = abs(y1_top - y2_bottom) < closeness or abs(y1_bottom - y2_top) < closeness

            
            if name1 != 'bottom' and name2 != 'bottom':
                """ aligned """
                if model_relation == "horizontally_aligned" or composed_relation == "all" or generating_data == False:
                    if abs(y1_bottom - y2_bottom) < alignment and abs(rot1-rot2)%(np.pi/2) < alignment:
                        constraints.append(('horizontally_aligned', tiles.index(name1), tiles.index(name2))) 
                        if generating_data:
                            constraints.append(('horizontally_aligned', tiles.index(name2), tiles.index(name1))) 

                if model_relation == "vertically_aligned" or composed_relation == "all" or generating_data == False:
                    if abs(x1 - x2) < alignment:
                        constraints.append(('vertically_aligned', tiles.index(name1), tiles.index(name2)))
                        if generating_data:
                            constraints.append(('vertically_aligned', tiles.index(name2), tiles.index(name1)))

                """left_of/right_of (obj, ref_obj)"""
                if model_relation == "next_to" or composed_relation == "all" or generating_data == False:
                    if 0 < x1_left - x2_right < closeness and y_overlap:
                        constraints.append(('left_of_front', tiles.index(name2), tiles.index(name1)))
                        if generating_data:
                            constraints.append(('right_of_front', tiles.index(name1), tiles.index(name2)))
                        # constraints.append(('left_of_back', tiles.index(name1), tiles.index(name2)))
                        # constraints.append(('right_of_back', tiles.index(name2), tiles.index(name1)))

                    if 0 < x2_left - x1_right < closeness and y_overlap:
                        constraints.append(('left_of_front', tiles.index(name1), tiles.index(name2)))
                        if generating_data:
                            constraints.append(('right_of_front', tiles.index(name2), tiles.index(name1)))
                        # constraints.append(('left_of_back', tiles.index(name2), tiles.index(name1)))
                        # constraints.append(('right_of_back', tiles.index(name1), tiles.index(name2)))
                    

                """ centered"""
                if model_relation == "centered" or composed_relation == "all" or generating_data == False:
                    if math.sqrt((x1 - x2)**2 + (y1 - y2)**2) < touching:
                        constraints.append(('centered', tiles.index(name1), tiles.index(name2)))

                """ symmetry """
                if model_relation == "symmetry" or composed_relation == "all" or generating_data == False:
                    if abs(x1 + x2) < alignment and abs(y1-y2) < alignment:
                        constraints.append(('vertical_symmetry_on_table', tiles.index(name1), tiles.index(name2)))
                        if generating_data:
                            constraints.append(('vertical_symmetry_on_table', tiles.index(name2), tiles.index(name1)))
                    if abs(y1 + y2) < alignment and abs(x1-x2) < alignment:
                        constraints.append(('horizontal_symmetry_on_table', tiles.index(name1), tiles.index(name2)))
                        if generating_data:
                            constraints.append(('horizontal_symmetry_on_table', tiles.index(name2), tiles.index(name1)))

            for k in range(j + 1, len(names)):
                p = objects[names[k]]

                if names[k] == 'bottom' or names[k] in sides:
                    continue
                name3 = names[k]
                if name3 == name2 or name3 == name1:
                    continue

                x3, y3, z3 = p['center']
                lx3, ly3, lz3 = p['extents']
                if rotations is not None and name3 in rotations:
                    rot3 = rotations[name3]
                    if abs(abs(rot3) - np.pi /2) < 0.2:
                        ly3, lx3, lz3 = n['extents']
                else:
                    rot3 = 0
                x3_left = x3 - lx3 / 2
                x3_right = x3 + lx3 / 2
                y3_top = y3 + ly3 / 2
                y3_bottom = y3 - ly3 / 2

                """ symmetry """
                if model_relation == "symmetry" or composed_relation == "all" or generating_data == False:
                    if abs(y1 - y2) < alignment and abs(y1 - y3) < alignment and abs(y2 - y3) < alignment:
                        indexed_xs = [(name, x) for name, x in zip([name1, name2, name3],[x1, x2, x3])]    
                        sorted_xs = sorted(indexed_xs, key=lambda x: x[1])
                        sorted_names = [x[0] for x in sorted_xs]
                        sorted_xs = [x[1] for x in sorted_xs]

                        if abs(abs(sorted_xs[0] - sorted_xs[1]) - abs(sorted_xs[2] - sorted_xs[1])) < alignment:
                            constraints.append(('vertical_line_symmetry', tiles.index(sorted_names[1]), tiles.index(sorted_names[2]), 
                                                tiles.index(sorted_names[0])))
                            if generating_data:
                                constraints.append(('vertical_line_symmetry', tiles.index(sorted_names[1]), tiles.index(sorted_names[0]), 
                                                tiles.index(sorted_names[2])))
                            
                    if abs(x1 - x2) < alignment and abs(x1 - x3) < alignment and abs(x2 - x3) < alignment:
                        indexed_ys = [(name, y) for name, y in zip([name1, name2, name3],[y1, y2, y3])]    
                        sorted_ys = sorted(indexed_ys, key=lambda y: y[1])
                        sorted_names = [y[0] for y in sorted_ys]
                        sorted_ys = [y[1] for y in sorted_ys]

                        if abs(abs(sorted_ys[0] - sorted_ys[1]) - abs(sorted_ys[2] - sorted_ys[1])) < alignment:
                            constraints.append(('horizontal_line_symmetry', tiles.index(sorted_names[1]), tiles.index(sorted_names[2]), 
                                                tiles.index(sorted_names[0])))
                            if generating_data:
                                constraints.append(('horizontal_line_symmetry', tiles.index(sorted_names[1]), tiles.index(sorted_names[0]), 
                                                tiles.index(sorted_names[2])))

            
    constraints.sort()

    
    if "name" in objects[names[5]].keys():
        obj_names = ["bottom"] + [objects[n]['name'] for n in names[5:]]
        for c in constraints:
            if c[0] == "left_of_front":
                typ = "left_of"
            elif c[0] == "right_of_front":
                typ = "right_of"
            else:
                typ = c[0]
            verbose_constraints.append(tuple([typ] + [obj_names[c[i]] for i in range(1, len(c))]))
        
    constraints = [pad_constraints(c) for c in constraints]

    if debug:
        summarize_constraints(constraints)
    return list(set(constraints)), list(set(verbose_constraints))

def compute_clustered_atomic_constraints(objects, W, L, H, relation, rotations=None, scale=None, debug=False):

    sides = ['east', 'west', 'north', 'south']
    if debug:
        print('scale', scale)
        print('rotations', rotations)
        pprint({k: {kk: r(vv) for kk, vv in v.items() if kk in ['extents', 'center']}
                for k, v in objects.items() if k not in sides})
        print()

    
    names = list(objects.keys())
    tiles = ['bottom'] + [n for n in names if 'tile_' in n]
    # print(rotations)

    """ left, right, top, bottom """
    X_0, Y_0, X_1, Y_1 = W, L, 0, 0
    total_area = 0
    for i in range(len(names)):
        m = objects[names[i]]
        if names[i] in ['bottom'] + sides:
            continue
        name = names[i] if 'tile_' in names[i] else 'bottom'

        x1, y1, z1 = m['center']
        lx1, ly1, lz1 = m['extents']
        if rotations is not None and name in rotations:
            rot = rotations[name]
            if abs(abs(rot) - np.pi /2) < 0.1:
                ly1, lx1, lz1 = m['extents']

        x1_left = x1 - lx1 / 2
        x1_right = x1 + lx1 / 2
        y1_top = y1 + ly1 / 2
        y1_bottom = y1 - ly1 / 2

        X_0 = min(X_0, x1_left)
        X_1 = max(X_1, x1_right)
        Y_0 = min(Y_0, y1_bottom)
        Y_1 = max(Y_1, y1_top)
        total_area += lx1 * ly1

    return X_0, Y_0, X_1, Y_1, total_area
    

def summarize_constraints(lst):
    data = defaultdict(list)
    for c in lst:
        data[c[0]].append(c[1:])
    print('-'*50)
    for k, v in data.items():
        print(f'{k} ({len(v)})\t{v}')
    print('-'*50)


def translate_cfree_evaluations(evaluations):
    """ usedto visualize unsatisfied constraints in each time step """
    new_evaluations = []
    for eva in evaluations:
        new_elems = ['cfree']
        for elem in eva:
            if elem in ['east', 'west', 'north', 'south', 'bottom']:
                new_elems += [0]
            else:
                new_elems += [eval(elem[-1]) + 1]
        if tuple(new_elems) not in new_evaluations:
            new_evaluations.append(tuple(new_elems))
    return new_evaluations


#######################################################################################


def verify_triangles_encoding_decoding():
    render_dir = abspath(join(dirname(__file__), '..', 'renders', 'triangles_encoding_decoding_2'))
    if not isdir(render_dir):
        os.makedirs(render_dir)
    num = 12
    for i in range(0, num):
        theta = np.deg2rad(i * 360 / num)
        cs = np.cos(theta)
        sn = np.sin(theta)
        # features = [
        #     [1, 1, 0, 0, 0, 0, 0],
        #     [0.25, 0, 0.25, 0, 0, cs, sn],
        # ]

        ## can't be some random set of vectors, need to be from centroid to three vertices
        features = [
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0.2, -0.3, -0.1, 0.3, -0.1, 0, 0, cs, sn],
        ]
        render_world_from_graph(features, (3, 3), png_name=join(render_dir, f'{i}.png'), array=False,
                                log=False, verbose=False, world_name='TriangularRandomSplitWorld')


def get_ont_hot_grasp_side(grasp_side):
    all_sides = {k: 0 for k in ["x+", "x-", "y+", "y-", "z+"]}
    all_sides.update({k[0]: abs(k[1]) for k in grasp_side})
    return list(all_sides.values())


def get_grasp_db():
    grasp_file = join(dirname(__file__), '..', 'packing_models', 'grasps', 'hand_grasps_PandaRobot.json')
    return json.load(open(grasp_file, 'r'))


def get_grasp_side_from_grasp(name, grasp_quat):
    grasp_db = get_grasp_db()
    grasp_info = [v for v in grasp_db.values() if v['name'] == name][0]
    grasp_eulers = [p[3:] for p in grasp_info['grasps']]
    for i, euler in enumerate(grasp_eulers):
        if np.allclose(pybullet_planning.quat_from_euler(euler), grasp_quat):
            return grasp_info['grasp_sides'][i]
    return None


def cat_from_model_id(model_id):
    grasp_db = get_grasp_db()
    for v in grasp_db.values():
        cat, idx = v['name'].split('_')
        if idx == str(model_id):
            return cat


def grasps_from_model_id(model_id, grasp_id):
    grasp_db = get_grasp_db()
    for v in grasp_db.values():
        if v['name'].split('_')[-1] == str(model_id):
            return v['grasps'][grasp_id], v['scale']


def grasp_from_id_scale(model_id, grasp_id, scale):
    import pybullet_planning as pp
    grasp, grasp_scale = grasps_from_model_id(model_id, int(grasp_id))
    return (np.array(grasp[:3]) / grasp_scale * scale).tolist(), pp.quat_from_euler(grasp[3:])


if __name__ == '__main__':
    verify_triangles_encoding_decoding()
