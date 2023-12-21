from numpy.random import rand
from typing import Iterable, Tuple
import numpy as np
from math import sqrt
import random
import numpy.random as rn
import math
from envs.data_utils import compute_qualitative_constraints, summarize_constraints, r as rd
import pdb
from networks.denoise_fns import tidy_constraints_dict


def get_tray_splitting_gen(num_samples=40, min_num_regions=2, max_num_regions=6, max_depth=3, default_min_size=0.3):

    Region = Tuple[float, float, float, float]  # l, t, w, h

    def partition(box: Region, depth: int = 3) -> Iterable[Region]:
        if rand() < 0.3 or depth == 0:
            yield box

        else:
            if rand() < 0.5:
                axis = 0
            else:
                axis = 1

            
            split_point = int(rand() * box[axis + 2] * 5)/5
             
            if axis == 0:
                yield from partition((box[0], box[1], split_point, box[3]), depth - 1)
                yield from partition((box[0] + split_point, box[1], box[2] - split_point, box[3]), depth - 1)
            else:
                yield from partition((box[0], box[1], box[2], split_point), depth - 1)
                yield from partition((box[0], box[1] + split_point, box[2], box[3] - split_point), depth - 1)

    def filter_regions(regions: Iterable[Region], min_size: float) -> Iterable[Region]:
        return [r for r in regions if r[2] > min_size and r[3] > min_size]

    def gen(w, l):
        min_size = min([w, l]) / 2 * default_min_size
        def get_regions():
            regions = []
            for region in partition((0, 0, w, l), max_depth):
                regions.append(region)
            return regions
        count = num_samples
        while True:
            regions = get_regions()
            regions = filter_regions(regions, min_size)
            if min_num_regions <= len(regions) <= max_num_regions:
                count -= 1
                yield regions
            if count == 0:
                break
        yield None
    return gen

def get_tidy_data_gen(num_samples=40, min_num_regions=2, max_num_regions=6, max_depth=3, default_min_size=0.075, relation="mixed"):

    Region = Tuple[float, float, float, float] # x, y, w, l

    def partition(box: Region, depth: int = 3) -> Iterable[Region]:

        if rand() < 0.15 or depth == 0:
            yield box

        else:
            if rand() < 0.6:
                axis = 0
            else:
                axis = 1

            split_point = rand() * box[axis + 2]
            if axis == 0:
                yield from partition((box[0], box[1], split_point, box[3]), depth - 1)
                yield from partition((box[0] + split_point, box[1], box[2] - split_point, box[3]), depth - 1)
            else:
                yield from partition((box[0], box[1], box[2], split_point), depth - 1)
                yield from partition((box[0], box[1] + split_point, box[2], box[3] - split_point), depth - 1)


    def filter_regions(regions: Iterable[Region], min_size: float) -> Iterable[Region]:
        
        if len(regions) == 0:
            return regions
        else:
            return [r for r in regions if r[2] > min_size and r[3] > min_size]

    def gen(W, L, relation, offset=0.05):

        min_size = min([W, L]) / 2 * default_min_size
        n_mapping = {3: [0, 1], 4: [2, 0], 5: [1, 1], 6: [0, 2], 7: [2, 1], 8: [1, 2], 9: [0, 3], 10: [2, 2]}

        def get_aligned_regions():

            regions = []
            n = rn.randint(min_num_regions, max_num_regions+1)

            y_bottom = rn.uniform(0.2, 0.95)*(L - 2* offset) 
            xs = np.sort(rn.uniform(0+offset, W - 2*offset, size = n * 2))
            ys = rn.uniform(0, L - y_bottom, size = n)

            for i in range(n):
                x1 = xs[i * 2]
                w1 = xs[i * 2 + 1] - x1
                l1 = ys[i] 
                regions.append((x1, y_bottom, w1, l1))

            return regions, 'aligned_bottom'
        
        def get_aligned_vertical_regions():

            regions = []
            n = rn.randint(min_num_regions, max_num_regions+1)

            x_center = rn.uniform(0.2, 0.8)*W 
            ys = np.sort(rn.uniform(0+offset, L - 2*offset, size = n * 2))
            xs = rn.uniform(0.1, min(W - x_center, x_center), size = n)
            
            for i in range(n):
                x1 = x_center - xs[i]
                w1 = xs[i] * 2
                y1 = ys[i * 2]
                l1 = ys[i * 2 + 1] - y1

                if x1 > offset and y1 > offset and w1 > min_size and l1 > min_size and x1 + w1 < W - offset and y1 + l1 < L - offset:
                    regions.append((x1, y1, w1, l1))
    
            return regions, 'aligned_vertical'
             
        def get_pairwise_aligned_cfree():

            regions = []
            n = rn.randint(2, 11)

            y_bottom = rn.uniform(0.2, 0.95)*(L - 2* offset) 
            xs = np.sort(rn.uniform(0+offset, W - 2*offset, size = n * 2))
            ys = rn.uniform(0, L - y_bottom, size = n)

            for i in rn.choice(n, size = 2, replace = False):
                x1 = xs[i * 2]
                w1 = xs[i * 2 + 1] - x1
                l1 = ys[i] 
                regions.append((x1, y_bottom, w1, l1))

            return regions, 'aligned_bottom&cfree'
        
        def get_pairwise_aligned_ccollide():

            import math

            p = np.array([1, 3, 6, 10, 15, 21, 28, 36, 45])/165

            sqrt_n_1 = math.sqrt(np.random.choice(np.arange(2, 11), p = p))
            sqrt_n_2 = math.sqrt(np.random.choice(np.arange(2, 11), p = p))

            # sample the poses of the first object            
            w, l = np.clip(rn.uniform(0.45, 1)*W/sqrt_n_1, 0.2*W, 0.8*W), np.clip(rn.uniform(0.45, 1)*L/sqrt_n_1, 0.2*L, 0.8*L)

            x1 = rn.uniform(offset, W - w - offset)
            y_bottom = rn.uniform(offset, L - l - offset)

            regions = [(x1, y_bottom, w, l)]

            l2 = rn.uniform(0, L - y_bottom - offset)

            c_x2 = rn.uniform(x1, x1 + w)

            if rand() < 0.5: # right
                w2 = rn.uniform(0, (W - offset-c_x2)/sqrt_n_2)
                x2 = c_x2
            else: # left       
                w2 = rn.uniform(0, (c_x2 - offset)/sqrt_n_2)            
                x2 = c_x2 - w2

            regions.append((x2, y_bottom, w2, l2))

            return regions, 'aligned_bottom&ccollide'

        def get_cfree_regions(max_depth, X, Y, W, L, offset, offset_grid=True):

            regions = []
            for region in partition((X + offset, Y + offset, W - 2*offset, L - 2*offset), max_depth):

                if offset_grid:
                    x, y, w, l = region
                    ps = [0, 0, 0, 0]
                    if w < 0.2: # is the rectangle is already very thin
                        w_ratio = [0.05, 0.01]
                    else:
                        w_ratio = [0.1, 0.25]
                
                    if l < 0.2:
                        l_ratio = [0.05, 0.01]
                    else:
                        l_ratio = [0.1, 0.25]

                    y_offset = np.random.uniform(l_ratio[0], l_ratio[1], size=2) * l
                    x_offset = np.random.uniform(w_ratio[0], w_ratio[1], size = 2) * w

                    region = (x + x_offset[0], y + y_offset[0], w - x_offset[0] - x_offset[1], l - y_offset[0] - y_offset[1]) 
                regions.append(region)

            return regions, 'cfree'

        def get_ccollide_regions():

            import math

            p = np.array([1, 3, 6, 10, 15, 21, 28, 36, 45])/165

            sqrt_n_1 = math.sqrt(np.random.choice(np.arange(2, 11), p = p))
            sqrt_n_2 = math.sqrt(np.random.choice(np.arange(2, 11), p = p))

            # sample the poses of the first object            
            w, l = np.clip(rn.uniform(0.45, 1)*W/sqrt_n_1, 0.2*W, 0.8*W), np.clip(rn.uniform(0.45, 1)*L/sqrt_n_1, 0.2*L, 0.8*L)

            x1 = rn.uniform(offset, W - w - offset)
            y1 = rn.uniform(offset, L - l - offset)

            regions = [(x1, y1, w, l)]

            # sample the poses of the second object

            c_x2, c_y2 = rn.uniform(x1, x1 + w), rn.uniform(y1, y1 + l)

            prob = rand()

            if prob < 0.25: # right-bottom
                w2 = rn.uniform(0, (W - offset-c_x2)/sqrt_n_2)
                l2 = rn.uniform(0, (L - offset-c_y2)/sqrt_n_2)
                x2, y2 = c_x2, c_y2
            elif prob < 0.5: # right-top
                w2 = rn.uniform(0, (W - offset-c_x2)/sqrt_n_2)
                l2 = rn.uniform(0, (c_y2 - offset)/sqrt_n_2)
                x2, y2 = c_x2, c_y2 - l2
            elif prob < 0.75: # left-top    
                w2 = rn.uniform(0, (c_x2 - offset)/sqrt_n_2)
                l2 = rn.uniform(0, (c_y2 - offset)/sqrt_n_2)
                x2, y2 = c_x2 - w2, c_y2 - l2   
            else: # left-bottom 
                w2 = rn.uniform(0, (c_x2-offset)/sqrt_n_2)
                l2 = rn.uniform(0, (L - offset-c_y2)/sqrt_n_2)
                x2, y2 = c_x2 - w2, c_y2

            regions.append((x2, y2, w2, l2))

            return regions, 'ccollide'

        def get_next_to_regions(W, L, offset):

            regions = []
            mac_regions = []
            if min_num_regions == max_num_regions:
                n_1, n_2 = n_mapping[min_num_regions]
            else:
                n = rn.randint(min_num_regions, max_num_regions+1)
                n_1, n_2 = n_mapping[n]
            
            mac_regions, _ = get_cfree_regions(max_depth = max(3, max_depth), X=0, Y=0, W=W, L=L, offset=offset)
            mac_regions = filter_regions(mac_regions, 0.2)
           
            while n_1 > 0:
                # while len(mac_regions) == 0:
                    # mac_regions, _ = get_cfree_regions(max_depth = 3, X=0, Y=0, W=W, L=L, offset=offset)
                    # mac_regions = filter_regions(mac_regions, 0.2)
                if mac_regions == []:
                    return [], 'next_to'
                
                x_0, y_0, w_0, l_0 = mac_regions.pop()

                _sub_reg = []

                x_center, y_center = x_0 + w_0/2, y_0 + l_0/2
                xs = rn.uniform(x_0 + min_size, x_0 + w_0 - min_size)
                w1 = xs - x_0
                w2 = x_0 + w_0 - xs
                x1_padding = rn.uniform(0.1, 0.35) * w1
                x2_padding = rn.uniform(0.1, 0.35) * w2
                x_interval_padding = rn.uniform(0.1, 0.15, size = 2)

                w1 = w1 - x1_padding - x_interval_padding[0]
                w2 = w2 - x2_padding - x_interval_padding[1]

                y_padding = rn.uniform(0.1, 0.3, size = 2)* l_0

                x1 = x_center - w1 - x_interval_padding[0]
                x2 = x_center + x_interval_padding[1]

                if l_0 - y_padding[0] > min_size:
                    l_1 = l_0 - y_padding[0]
                    _sub_reg.append((x1, y_center - l_1/2 , w1, l_1))
                else:
                    _sub_reg.append((x1, y_0, w1, l_0))

                if l_0 - y_padding[1] > min_size:
                    l_2 = l_0 - y_padding[1]
                    _sub_reg.append((x2, y_center - l_2/2, w2, l_2))
                else:
                    _sub_reg.append((x2, y_0, w2, l_0))

                _sub_reg = filter_regions(_sub_reg, min_size)
                if len(_sub_reg) == 2:
                    regions.extend(_sub_reg)
                    n_1 -= 1

            while n_2 > 0:

                # while len(mac_regions) == 0:
                #     mac_regions, _ = get_cfree_regions(max_depth = 3, X=0, Y=0, W=W, L=L, offset=offset)
                #     mac_regions = filter_regions(mac_regions, 0.3)

                if mac_regions == []:
                    return [], 'next_to'

                x_0, y_0, w_0, l_0 = mac_regions.pop()

                _sub_reg = []

                y_center = y_0 + l_0/2
                xs = rn.uniform(x_0 + min_size, x_0 + w_0 - min_size, size = 2)
                w1 = xs[0] - x_0
                w2 = xs[1] - xs[0]
                w3 = x_0 + w_0 - xs[1]
                x1_padding = rn.uniform(0.1, 0.3) * w1
                x2_padding = rn.uniform(0.05, 0.15, size=2) * w2  
                x3_padding = rn.uniform(0.1, 0.3) * w3

                w1 = w1 - x1_padding 
                w2 = w2 - x2_padding[0] - x2_padding[1]
                w3 = w3 - x3_padding 

                y_padding = rn.uniform(0.1, 0.3, size = 3)* l_0

                x1 = x_0 
                x2 = x1 + w1 + x1_padding + x2_padding[0]
                x3 = x2 + w2 + x2_padding[1] + x3_padding

                if l_0 - y_padding[0] > min_size:
                    l_1 = l_0 - y_padding[0]
                    _sub_reg.append((x1, y_center - l_1/2 , w1, l_1))
                else:
                    _sub_reg.append((x1, y_0, w1, l_0))

                if l_0 - y_padding[1] > min_size:
                    l_2 = l_0 - y_padding[1]
                    _sub_reg.append((x2, y_center - l_2/2, w2, l_2))
                else:
                    _sub_reg.append((x2, y_0, w2, l_0))

                if l_0 - y_padding[2] > min_size:
                    l_3 = l_0 - y_padding[2]
                    _sub_reg.append((x3, y_center - l_3/2, w3, l_3))
                else:
                    _sub_reg.append((x3, y_0, w3, l_0))

                _sub_reg = filter_regions(_sub_reg, min_size)

                if len(_sub_reg) == 3:
                    regions.extend(_sub_reg)
                    n_2 -= 1

            return regions, 'next_to'
        
        def get_on_top_of_regions(W, L, offset):

            _max_depth = math.ceil(math.log2(max_num_regions)) + 1 

            mac_regions, _ = get_cfree_regions(max_depth = _max_depth, X=0, Y=0, W=W, L=L, offset=offset)
            mac_regions = filter_regions(mac_regions, 0.4)
            regions = []

            if len(mac_regions) > max_num_regions//2:
                mac_regions_idx = rn.choice(np.arange(len(mac_regions)), size = max_num_regions//2, replace = False)
            else:
                mac_regions_idx = np.arange(len(mac_regions))

            for idx in mac_regions_idx:
                    
                x, y, w, l = mac_regions[idx]
                sub_regions, _ = get_cfree_regions(max_depth = 2, X=x, Y=y, W=w, L=l, offset=0)    
                sub_regions = filter_regions(sub_regions, min_size)

                if len(sub_regions) > 0:

                    if len(regions) + len(sub_regions) + 1 <= max_num_regions:
                        regions.append(mac_regions[idx])
                        regions.extend(sub_regions)
                    elif len(regions) + 2 <= max_num_regions:
                        regions.append(mac_regions[idx])
                        regions.extend(sub_regions[:max_num_regions - len(regions) - 1])
                        return regions, 'on_top_of'
                    else:
                        return regions, 'on_top_of'
            return regions, 'on_top_of'

        def get_centered_regions(W, L, offset):

            regions = []

            if rand() < 0.4:

                w, l = rn.uniform(0.15, 0.4)*W, rn.uniform(0.15, 0.4)*L

                if w > min_size and l > min_size:
                    regions.append((W/2 - w/2, L/2 - l/2,  w, l))
            
            base_regions = []
            if len(regions) > 0:

                x_0, y_0, w_0, l_0 = regions[0]

                sub_regions, _ = get_cfree_regions(max_depth = 2, X=0, Y=0, W=x_0 + w_0, 
                                                   L=y_0, offset=offset)
                
                base_regions.extend(sub_regions)
                
                sub_regions, _ = get_cfree_regions(max_depth = 2, X=0, Y=y_0, W=x_0, 
                                                   L=L-y_0, offset=offset)
                
                base_regions.extend(sub_regions)

                sub_regions, _ = get_cfree_regions(max_depth = 2, X=x_0, Y=y_0+l_0, W=W-x_0, 
                                                   L=L-y_0-l_0, offset=offset)
                
                base_regions.extend(sub_regions)

                sub_regions, _ = get_cfree_regions(max_depth = 2, X=x_0+w_0, Y=y_0, W=W-x_0-w_0, 
                                                   L=y_0+l_0, offset=offset)
                
                base_regions.extend(sub_regions)

            else:
                if max_num_regions > 6:
                    _max_depth = 4
                else:
                    _max_depth = 3
                sub_regions, _ = get_cfree_regions(max_depth = _max_depth, X=0, Y=0, W=W, 
                                                   L=L, offset=offset)
                
                base_regions.extend(sub_regions)
            
            if len(base_regions) > max_num_regions:
                base_regions_idx = rn.choice(np.arange(len(base_regions)), size = max_num_regions, replace = False)
            else:
                base_regions_idx = np.arange(len(base_regions))
            for idx in base_regions_idx:

                if len(regions) >= max_num_regions:
                    break

                x, y, w, l = base_regions[idx]

                c_x = x + w/2
                c_y = y + l/2

                w_, l_ = rn.uniform(0.2, 0.5)*w, rn.uniform(0.2, 0.5)*l
                    
                if w_ > min_size and l_ > min_size:
                    regions.append(base_regions[idx])
                    regions.append((c_x - w_/2, c_y - l_/2, w_, l_))
                    
            
            return regions, 'centered'

        def get_edge_regions():
            # when the edge is between 0.5 to the edges
            regions = []

            n = rn.randint(3, 10)

            y_bottoms = rn.uniform(0.05, 0.1, size=n) 
            xs = np.sort(rn.uniform(0+offset, W - 2*offset, size = n * 2))
            ys = rn.uniform(0, 0.3*L-0.15, size = n)

            for i in range(n):
                x1 = xs[i * 2]
                w1 = xs[i * 2 + 1] - x1
                l1 = ys[i] 

                if rand() < 0.5:
                    y1 = y_bottoms[i]
                else:
                    y1 = L - y_bottoms[i] - l1
                regions.append((x1, y1, w1, l1))

            m = rn.randint(3, 7)

            x_bottoms = rn.uniform(0.05, 0.1, size=m) 
            ys = np.sort(rn.uniform(0+offset, L - 2*offset, size = m * 2))
            xs = rn.uniform(0, 0.3*W-0.15, size = m)

            for i in range(m):
                y1 = ys[i * 2]
                l1 = ys[i * 2 + 1] - y1
                w1 = xs[i] 

                if rand() < 0.5:
                    x1 = x_bottoms[i]
                else:
                    x1 = W - x_bottoms[i] - w1
                regions.append((x1, y1, w1, l1))

            return regions, 'next_to_edge'        
        
        def get_in_regions(W, L, offset):

            regions = []

            rand_num = rand()

            _max_depth = math.ceil(math.log2(max_num_regions)) + 1 

            if rand_num < 0.15: #center-h

                _regions, _ = get_cfree_regions(max_depth = _max_depth, X=0, Y=L/4, W=W, 
                                                   L=L/2, offset=offset)
                regions.extend(_regions)
            if rand_num < 0.15: #center

                _regions, _ = get_cfree_regions(max_depth = _max_depth, X=W/4, Y=0, W=W/2, 
                                                   L=L, offset=offset)
                regions.extend(_regions)

            elif rand_num < 0.65: # left/right
                _max_depth = max(_max_depth//2, 1)

                _regions, _ = get_cfree_regions(max_depth = _max_depth, X=0, Y=0, W=W/2, 
                                                   L=L, offset=offset)
                regions.extend(_regions)

                _regions, _ = get_cfree_regions(max_depth = _max_depth, X=W/2, Y=0, W=W/2,
                                                    L=L, offset=offset)
                regions.extend(_regions)
            else: # top/bottom
                _max_depth = max(_max_depth//2, 1)
                _regions, _ = get_cfree_regions(max_depth = _max_depth, X=0, Y=0, W=W, 
                                                   L=L/2, offset=offset)
                regions.extend(_regions)

                _regions, _ = get_cfree_regions(max_depth = _max_depth, X=0, Y=L/2, W=W,
                                                    L=L/2, offset=offset)
                regions.extend(_regions)

            regions = filter_regions(regions, min_size)
            if len(regions) > max_num_regions:
                random.shuffle(regions)
                regions = regions[:max_num_regions]
            return regions, 'in'

        def get_symmetry_regions(W, L):
            regions = []

            if min_num_regions == max_num_regions:
                n = min_num_regions
            else:
                n = rn.randint(min_num_regions, max_num_regions+1)

            n_1, n_2 = n_mapping[n]
            if n >= 8:
                _max_depth = 4
            else:
                _max_depth = 3

            hori = rand() < 0.5
            # symmetry_table
            
            if hori: # horizontal
                _n = rn.randint(2, 5)
                y_s = np.sort(rn.uniform(0.05, 0.95, size=max(_n, n_1) * 2)*L)
                ind_n_1 = np.sort(rn.choice(np.arange(max(_n, n_1)), size = n_1, replace = False))
                for i in ind_n_1:
                    y1 = y_s[i * 2]
                    l1 = y_s[i * 2 + 1] - y1
                    y_cent = y1 + l1/2
                    dist = rn.uniform(0.1, 0.4)*W
                    width = rn.uniform(0.1, min(dist, 0.5*W-dist))
                    widths = rn.uniform(width - 0.05, width, size = 2) 
                    w1 = widths[0]*2
                    w2 = widths[1]*2
                    x1 = W/2 - dist - w1/2
                    x2 = W/2 + dist - w2/2
                    y_padding = rn.uniform(0.025, 0.05, size=2)*L
                    if l1 - y_padding[0]*2 > min_size:
                        regions.append((x1, y1 + y_padding[0], w1, l1 - y_padding[0]*2))
                    else:
                        regions.append((x1, y1, w1, l1))
                    if l1 - y_padding[1]*2 > min_size:
                        regions.append((x2, y1 + y_padding[1], w2, l1 - y_padding[1]*2))
                    else:
                        regions.append((x2, y1, w2, l1))
            else: # vertical
                _n = rn.randint(2, 5)
                x_s = np.sort(rn.uniform(0.05, 0.95, size=max(_n, n_1) * 2)*W)
                ind_n_1 = np.sort(rn.choice(np.arange(max(_n, n_1)), size = n_1, replace = False))

                for i in ind_n_1:
                    x1 = x_s[i * 2]
                    w1 = x_s[i * 2 + 1] - x1
                    x_cent = x1 + w1/2
                    dist = rn.uniform(0.1, 0.4)*L
                    width = rn.uniform(0.1, min(dist, 0.5*L-dist))
                    widths = rn.uniform(width - 0.05, width, size = 2) 
                    l1 = widths[0]*2
                    l2 = widths[1]*2
                    y1 = L/2 - dist - l1/2
                    y2 = L/2 + dist - l2/2
                    x_padding = rn.uniform(0.025, 0.05, size=2)*W
                    if w1 - x_padding[0]*2 > min_size:
                        regions.append((x1 + x_padding[0], y1, w1 - x_padding[0]*2, l1))
                    else:
                        regions.append((x1, y1, w1, l1))
                    if w1 - x_padding[1]*2 > min_size:
                        regions.append((x1 + x_padding[1], y2, w1 - x_padding[1]*2, l2))
                    else:
                        regions.append((x1, y2, w1, l2))

            
            # symmetry_object
            mac_regions = []
            mac_regions, _ = get_cfree_regions(max_depth = _max_depth, X=0, Y=0, W=W, L=L, offset=offset)
            mac_regions = filter_regions(mac_regions, 0.5)

            while n_2 > 0:
                if len(mac_regions) == 0:
                    return regions, 'symmetry' 
                x_0, y_0, w_0, l_0 = mac_regions.pop()

                _sub_reg = []
                if rand() < 0.5: # horizontal
                    x_cent, y_cent = x_0 + w_0/2, y_0 + l_0/2
                    wid = rn.uniform(max(min_size, 0.1*w_0), max(min_size, 0.3*w_0))

                    dist = rn.uniform(min(wid*1.2, 0.42*w_0), 0.45*w_0)
                    width = rn.uniform(0.075, min(dist-min(wid*1.2, 0.42*w_0), 0.5*w_0-dist))
                    x_padding = rn.uniform(0.1, 0.4, size=2) * width
    
                    w1 = width - x_padding[0]
                    w2 = width - x_padding[1]
                    x1 = x_cent - dist - w1/2
                    x2 = x_cent + dist - w2/2
                    y_padding = rn.uniform(0.05, 0.1, size=3)*l_0
                        
                    if w1 - y_padding[1]*2 > min_size:
                        _sub_reg.append((x1, y_0 + y_padding[1], w1, l_0 - y_padding[1]*2))
                    else:
                        _sub_reg.append((x1, y_0, w1, l_0))
                        
                    if w2 - y_padding[2]*2 > min_size:
                        _sub_reg.append((x2, y_0 + y_padding[2], w2, l_0 - y_padding[2]*2))
                    else:   
                        _sub_reg.append((x2, y_0, w2, l_0))

                    if l_0 - y_padding[0]*2 > min_size:
                        _sub_reg.append((x_cent - wid, y_0 + y_padding[0], wid * 2, l_0 - y_padding[0]*2))
                    else:
                        _sub_reg.append((x_cent - wid, y_0, wid * 2, l_0))
          
                else: # vertical

                    x_cent, y_cent = x_0 + w_0/2, y_0 + l_0/2
                    wid = rn.uniform(max(min_size, 0.1*l_0), max(min_size, 0.4*l_0))

                    dist = rn.uniform(min(wid*1.2, 0.42*l_0), 0.45*l_0)
                    width = rn.uniform(0.075, min(dist-min(wid*1.2, 0.42*l_0), 0.5*l_0-dist))
                    y_padding = rn.uniform(0.1, 0.4, size=2) * width
                    l1 = width - y_padding[0]
                    l2 = width - y_padding[1]
                    y1 = y_cent - dist - l1/2
                    y2 = y_cent + dist - l2/2

            
                    x_padding = rn.uniform(0.025, 0.05, size=3)*w_0
                            
                    if l1 - x_padding[1]*2 > min_size:
                        _sub_reg.append((x_0 + x_padding[1], y1, w_0 - x_padding[1]*2, l1))
                    else:
                        _sub_reg.append((x_0, y1, w_0, l1))
                        
                    if l2 - x_padding[2]*2 > min_size:
                        _sub_reg.append((x_0 + x_padding[2], y2, w_0 - x_padding[2]*2, l2))
                    else:
                        _sub_reg.append((x_0, y2, w_0, l2))

                    if w_0 - x_padding[0]*2 > min_size:
                        _sub_reg.append((x_0 + x_padding[0], y_cent - wid, w_0 - x_padding[0]*2, wid * 2))
                    else:
                        _sub_reg.append((x_0, y_cent - wid, w_0, wid * 2))


                
                _sub_reg = filter_regions(_sub_reg, min_size)
                if len(_sub_reg) == 3:
                    regions.extend(_sub_reg)
                    n_2 -= 1


            return regions, 'symmetry' 

        def get_in_regular_grid(W, L, offset):

            # possible number of regions: 4, 6, 8, 9

            n_to_set_mapping = {4: [[4]], 6: [[6]], 8: [[8], [4, 4]], 9: [[9]], 10: [[10], [4, 6]],
                                12: [[6, 6], [4, 8], [4, 4, 4]], 13: [[4, 9]], 14: [[6, 8], [6, 4, 4], [4, 10]], 15: [[6, 9]], 
                                16: [[16], [6, 10], [4, 6, 6], [8, 8], [8, 4, 4]]}
            if min_num_regions == max_num_regions:
                if min_num_regions not in n_to_set_mapping.keys():
                    return [], 'regular_grid'
                else:
                    n = min_num_regions
            else:
                n = rn.randint(min_num_regions, max_num_regions+1)
                while n not in n_to_set_mapping.keys():
                    n = rn.randint(min_num_regions, max_num_regions+1)
            
            # pdb.set_trace()
            n_combi_idx = rn.randint(0, len(n_to_set_mapping[n]))
            n_combi = n_to_set_mapping[n][n_combi_idx]

            mac_regions = []
            while len(mac_regions) < len(n_combi):
                mac_regions, _ = get_cfree_regions(max_depth = 3, X=0, Y=0, W=W, L=L, offset=offset)
                mac_regions = filter_regions(mac_regions, 0.63)

            regions = []

            padding = 0.025

            def _get_unit_sub_regions(x_cut, y_cut, mac_region):
                x_0, y_0, w_0, l_0 = mac_region
                sub_regions = []
                for i in range(x_cut):
                        for j in range(y_cut):
                            sub_regions.append((x_0 + i*w_0/x_cut + padding, y_0 + j*l_0/y_cut + padding, 
                                               w_0/x_cut - 2*padding, l_0/y_cut-2*padding))

                sub_regions = filter_regions(sub_regions, min_size)

                return  sub_regions, len(sub_regions) == x_cut * y_cut
            
            flag = False
            for idx, m in enumerate(n_combi):
                
                mac_region = mac_regions[idx]
                sub_regions = []
               
                _, _, w_0, l_0 = mac_region
                if m == 4: # 2,2 
                    sub_regions, flag = _get_unit_sub_regions(2, 2, mac_region)

                elif m == 6: # 2,3
                    if w_0 > l_0:
                        sub_regions, flag = _get_unit_sub_regions(3, 2, mac_region)
                    else:
                        sub_regions, flag = _get_unit_sub_regions(2, 3, mac_region)
                        
                elif m == 8: # 2,4
                    if w_0 > l_0:
                        sub_regions, flag = _get_unit_sub_regions(4, 2, mac_region)
                    else:
                        sub_regions, flag = _get_unit_sub_regions(2, 4, mac_region)
                elif m == 9:
                    sub_regions, flag = _get_unit_sub_regions(3, 3, mac_region)
                elif m == 10:
                    if w_0 > l_0:
                        sub_regions, flag = _get_unit_sub_regions(5, 2, mac_region)
                    else:
                        sub_regions, flag = _get_unit_sub_regions(2, 5, mac_region)
                elif m == 16:
                    sub_regions, flag = _get_unit_sub_regions(4, 4, mac_region)

                if flag:
                    regions.extend(sub_regions)

            return regions, f'regular_grid_{n}_{n_combi_idx}'
            
        def get_customized_1(W, L):
            tissue_box = (0.2*W, 0.45*L, 0.1*W, 0.15*L)
            cleaning_can = (0.45*W, 0.5*L, 0.05*W, 0.3*L)
            starbucks_cup = (0.75*W, 0.55*L, 0.05*W, 0.25*L)
            plate_1 = (0.3*W, 0.2*L, 0.14*W, 0.21*L)
            plate_2 = (0.55*W, 0.3*L, 0.14*W, 0.21*L)

            return [tissue_box, cleaning_can, starbucks_cup, plate_1, plate_2], 'customized_1'
        
        def get_customized_2(W, L, relation):
            plate_1 = (0.1*W, 0.55*L, 0.12*W, 0.18*L)
            plate_2 = (0.35*W, 0.55*L, 0.12*W, 0.18*L)
            fork_1 = (0.6*W, 0.55*L, 0.04*W, 0.17*L)
            fork_2 = (0.7*W, 0.55*L, 0.04*W, 0.17*L)
            knife_1 = (0.8*W, 0.55*L, 0.04*W, 0.17*L)
            knife_2 = (0.9*W, 0.55*L, 0.04*W, 0.17*L)
            candel = (0.4*W, 0.3*L, 0.1*W, 0.15*L)

            return [plate_1, plate_2, fork_1, fork_2, knife_1, knife_2, candel], relation
        
        def get_customized_3(W, L):
            plate_1 = (0.1*W, 0.55*L, 0.12*W, 0.18*L)
            plate_2 = (0.35*W, 0.55*L, 0.12*W, 0.18*L)
            fork_1 = (0.6*W, 0.55*L, 0.04*W, 0.17*L)
            fork_2 = (0.7*W, 0.55*L, 0.04*W, 0.17*L)
            knife_1 = (0.8*W, 0.55*L, 0.04*W, 0.17*L)
            knife_2 = (0.9*W, 0.55*L, 0.04*W, 0.17*L)
            candel = (0.4*W, 0.3*L, 0.1*W, 0.15*L)

            return [plate_1, plate_2, fork_1, fork_2, knife_1, knife_2, candel], 'customized_3'

        def get_customized_5(W, L, relation):
            plate_1 = (0.1*W, 0.55*L, 0.12*W, 0.18*L)
            plate_2 = (0.35*W, 0.55*L, 0.12*W, 0.18*L)
            fork_1 = (0.6*W, 0.55*L, 0.04*W, 0.17*L)
            fork_2 = (0.7*W, 0.55*L, 0.04*W, 0.17*L)
            knife_1 = (0.8*W, 0.55*L, 0.04*W, 0.17*L)
            knife_2 = (0.9*W, 0.55*L, 0.04*W, 0.17*L)
            candel = (0.4*W, 0.3*L, 0.1*W, 0.15*L)

            return [plate_1, fork_1, knife_1, plate_2, fork_2, knife_2, candel], relation
        
        count = num_samples

        if "mixed" in relation:
            _, relation_2 = relation.split("_")
            relation_1 = "aligned_bottom"

            if rand() < 0.5:
                relation = relation_1
            else:
                relation = relation_2
        elif "integrated" in relation:
            relation = relation.split("_")[1]
            relation_list = relation.split("&")
            if len(relation_list) == 2:
                if rand() < 0.5:
                    relation = "integrated_cfree"
                else:
                    relation = "integrated_ccollide"
            else:
                relation = f"integrated_{relation_list[0]}"
        elif "all" in relation:
            if min_num_regions == max_num_regions:
                n = min_num_regions
            else:
                n = rn.choice(np.arange(min_num_regions, max_num_regions+1))
                while n == 11:
                    n = rn.choice(np.arange(min_num_regions, max_num_regions+1))
            if min_num_regions in [4, 6, 8, 9, 10]:
                relation = rn.choice(["aligned_bottom", "aligned_vertical", "on_top_of", "centered", "next_to_edge", "in", "symmetry", "next_to", "regular_grid"])
            elif min_num_regions in [3, 5, 7]:
                relation = rn.choice(["aligned_bottom", "aligned_vertical", "on_top_of", "centered", "next_to_edge", "in", "symmetry", "next_to"])
            elif min_num_regions in [12, 13, 14, 15, 16]:
                relation = "regular_grid"
        while True:
            if relation == "aligned_bottom":
                regions, relation_mode = get_aligned_regions()
            elif relation == "cfree":
                regions, relation_mode = get_cfree_regions(max_depth, 0, 0, W, L, offset)
            elif relation == "ccollide":
                regions, relation_mode = get_ccollide_regions()
            elif relation == "next_to":
                regions, relation_mode = get_next_to_regions(W, L, offset)
            elif relation == "on_top_of":
                regions, relation_mode = get_on_top_of_regions(W, L, offset)
            elif relation == "centered":
                regions, relation_mode = get_centered_regions(W, L, offset)
            elif relation == "next_to_edge":
                regions, relation_mode = get_edge_regions()
            elif relation == "in":
                regions, relation_mode = get_in_regions(W, L, offset)
            elif relation == "aligned_vertical":
                regions, relation_mode = get_aligned_vertical_regions()
            elif relation == "symmetry":
                regions, relation_mode = get_symmetry_regions(W, L)
            elif relation == "integrated_cfree":
                 regions, relation_mode = get_pairwise_aligned_cfree()
            elif relation == "integrated_ccollide":
                regions, relation_mode = get_pairwise_aligned_ccollide()
            elif relation == "customized_1":
                regions, relation_mode = get_customized_1(W, L)
            elif relation == "customized_2" or relation == "customized_4":
                regions, relation_mode = get_customized_2(W, L, relation)
            elif relation == "customized_3":
                regions, relation_mode = get_customized_3(W, L)
            elif relation == "customized_5":
                regions, relation_mode = get_customized_5(W, L, relation)
            elif "regular_grid" in relation:
                regions, relation_mode = get_in_regular_grid(W, L, offset)

            regions = filter_regions(regions, min_size)
        
            # (("ccollide" in relation or "integrated" in relation) and len(regions) == 2) or
            if min_num_regions <= len(regions) <= max_num_regions:
                count -= 1
                print(len(regions), "added!")
                # pdb.set_trace()
                yield regions, relation_mode
            if count == 0:
                break
        yield None
        
    return gen

def get_sub_region_tray_splitting_gen(num_samples=40, min_num_regions=2, max_num_regions=6, max_depth=3, default_min_size=0.075):

    Region = Tuple[float, float, float, float]  # l, t, w, h

    def partition(box: Region, depth: int = 3) -> Iterable[Region]:
        if rand() < 0.3 or depth == 0:
            yield box

        else:
            # if rand() < 0.5:
            if rand() < 1:
                axis = 0
            else:
                axis = 1

            split_point = random.uniform(0.25, 0.75) * box[axis + 2]
            if axis == 0:
                yield from partition((box[0], box[1], split_point, box[3]), depth - 1)
                yield from partition((box[0] + split_point, box[1], box[2] - split_point, box[3]), depth - 1)
            else:
                yield from partition((box[0], box[1], box[2], split_point), depth - 1)
                yield from partition((box[0], box[1] + split_point, box[2], box[3] - split_point), depth - 1)

    def filter_regions(regions: Iterable[Region], min_size: float) -> Iterable[Region]:
        return [r for r in regions if r[2] > min_size and r[3] > min_size]

    def gen(w, l):

        w_offset, l_offset = random.uniform(0.2, 0.95)*w, random.uniform(0.2, 0.95)*l 

        x_0, y_0 = random.uniform(0, w - w_offset), random.uniform(0, l - l_offset)

        min_size = min([w, l]) / 2 * default_min_size
        def get_regions():
            regions = []
            for region in partition((x_0, y_0, w_offset, l_offset), max_depth):
                regions.append(region)
            return regions
        count = num_samples
        while True:
            regions = get_regions()
            # print("before: ", len(regions))
            regions = filter_regions(regions, min_size)
            # print("after filtering: ", len(regions))
            if min_num_regions <= len(regions) <= max_num_regions:
                count -= 1
                print(len(regions), "added!")
                yield regions
            if count == 0:
                break
        yield None
        
    return gen


def test_tray_splitting():
    gen = get_tray_splitting_gen(num_samples=2)
    for boxes in gen(4, 3):
        print(boxes)


##########################################################################


def construct_objects(regions, w, l, h, z):
    objects = {
        'bottom': {
            'extents': (w, l, 0.1),
            'center': (0, 0, -0.05)
        }
    }
    for i, region in enumerate(regions):
        objects[f"tile_{i}"] = {
            'extents': (region[2], region[3], h),
            'center': (-w/2+region[0]+region[2]/2, -l/2+region[1]+region[3]/2, z + h / 2)
        }
    return objects


def get_3d_box_splitting_gen(num_samples=40, min_num_regions=6, max_num_regions=10, **kwargs):

    bottom_gen = get_tray_splitting_gen(num_samples=num_samples, min_num_regions=min_num_regions-3,
                                        max_num_regions=max_num_regions-2, **kwargs)
    top_gen = get_tray_splitting_gen(num_samples=num_samples, min_num_regions=1,
                                     max_num_regions=2, **kwargs)

    def point_outside_of_box(point, box):
        if box[0] < point[0] < box[0] + box[2] and box[1] < point[1] < box[1] + box[3]:
            return False
        return True

    def point_in_boxes(point, boxes):
        for box in boxes:
            if box[0] <= point[0] <= box[0] + box[2] and box[1] <= point[1] <= box[1] + box[3]:
                return True
        return False

    def get_sides(boxes):
        lefts = sorted([box[0] for box in boxes])
        tops = sorted([box[1] for box in boxes], reverse=True)
        rights = sorted([box[0] + box[2] for box in boxes], reverse=True)
        bottoms = sorted([box[1] + box[3] for box in boxes])
        return lefts, tops, rights, bottoms

    def compute_secondary_support_region(boxes, region):
        """ given a set of boxes, compute the largest region that is supported
            by one of them but not the primary region """
        areas = []
        for box in boxes:
            left = max(region[0], box[0])
            top = max(region[1], box[1])
            right = min(region[0] + region[2], box[0] + box[2])
            bottom = min(region[1] + region[3], box[1] + box[3])
            w = right - left
            h = bottom - top
            areas.append(box[2] * box[3] - w * h)
        box = boxes[areas.index(max(areas))]

        ## find the region that's inside box but outside of region
        boxes = [box, region]
        lefts, tops, rights, bottoms = get_sides(boxes)
        xx = sorted(lefts + rights)
        yy = sorted(bottoms + tops)
        areas = {}
        for x1 in xx:
            for x2 in reversed(xx):
                if x1 >= x2:
                    continue
                for y1 in yy:
                    for y2 in reversed(yy):
                        if y1 >= y2:
                            continue
                        points = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
                        failed = False
                        for point in points:
                            if not point_outside_of_box(point, region) or not point_in_boxes(point, [box]):
                                failed = True
                                break
                        if failed:
                            continue
                        w = x2 - x1
                        h = y2 - y1
                        areas[(x1, y1, w, h)] = w * h
        if len(areas) == 0:
            return None
        return max(areas, key=areas.get)

    def compute_support_region(boxes):
        """ given a set of boxes, compute the largest region that is supported by all of them """
        lefts, tops, rights, bottoms = get_sides(boxes)
        areas = {}
        for l in range(2):
            for t in range(2):
                for r in range(2):
                    for b in range(2):
                        left = lefts[l]
                        top = tops[t]
                        right = rights[r]
                        bottom = bottoms[b]
                        points = [(left, top), (right, top), (left, bottom), (right, bottom)]
                        if not all([point_in_boxes(point, boxes) for point in points]):
                            continue
                        w = right - left
                        h = bottom - top
                        areas[(left, top, w, h)] = w * h
        return max(areas, key=areas.get)

    def sample_support_boxes(constraints):
        """ select two or three boxes that are adjacent to each other to be the support of the next split """
        finding = []
        found = []
        used_pairs = []
        repetitions = []
        pool = [c[1:] for c in constraints if c[0] == 'close-to']
        random.shuffle(pool)
        for pair in pool:
            if (pair[1], pair[0]) in repetitions or pair in used_pairs:
                continue
            repetitions.append(pair)

            for k in finding:
                for X in k:
                    if X in pair:
                        Y = [p for p in k if p != X][0]
                        Z = [p for p in pair if p != X][0]
                        if (Y, Z) in pool or (Z, Y) in pool:
                            found.append((X, Y, Z))
                            used_pairs.extend([(X, Y), (Y, Z), (Z, X), (Y, X), (Z, Y), (X, Z)])
                            finding.remove(k)
                            continue

            if pair not in used_pairs:
                finding.append(pair)

        if len(found) > 0:
            random.shuffle(found)
            return [f-1 for f in found[0]]
        return None

    def add_top_regions(r, h2, h4):
        x, y, w, l = r
        new_regions = []
        cc = num_samples
        while True:
            for small_regions in top_gen(w, l):
                cc -= 1
                if cc == 0:
                    return None
                if small_regions is None or len(small_regions) < 2:
                    continue
                for sr in small_regions:
                    xx, yy, ww, ll = sr
                    new_regions.append((x+xx, y+yy, h2, ww, ll, h4))
                return new_regions

    def add_3d_regions(regions, w, l, h):
        """ let's call the 2d split algorithm -> 2D-SPLIT
            A. use 2D-SPLIT to generate 3d regions in the bottom layer
            B. select three of the regions (h1) to support a top layer (h3) generated by 2D-SPLIT,
                the largest remaining area supports another box (h5)
            C. for each region not selected (h2), it will support a top layer (h4) generated by 2D-SPLIT
        """
        h1 = np.random.uniform(0, h * 0.66)
        h2 = np.random.uniform(h1, h * 0.8)
        h3 = np.random.uniform(0.2*(h-h1), h-h1)
        h4 = np.random.uniform(0.2*(h-h2), h-h2)
        h5 = np.random.uniform(0.2*(h-h1), h-h1)

        def dh():
            return np.random.uniform(0, h * 0.1)

        ## generate middle layer
        objects = construct_objects(regions, w, l, h1, 0)
        constraints = compute_qualitative_constraints(objects)
        boxes = sample_support_boxes(constraints)
        if boxes is None:
            return None
        selected_regions = [regions[b] for b in boxes]
        region = compute_support_region(selected_regions)
        region_secondary = compute_secondary_support_region(selected_regions, region)

        ## add heights to all regions
        new_regions = []
        for i, r in enumerate(regions):
            x, y, w, l = r
            if i not in boxes:
                new_regions.append((x, y, 0, w, l, h2))
                top_regions = add_top_regions(r, h2, h4)
                if top_regions is None:
                    return None
                new_regions.extend(top_regions)
            else:
                new_regions.append((x, y, 0, w, l, h1))
        x, y, w, l = region
        new_regions.append((x, y, h1, w, l, h3))
        if region_secondary is not None:
            x, y, w, l = region_secondary
            new_regions.append((x, y, h1, w, l, h5))

        ## minus for stability concerns
        new_regions = [tuple(list(r[:5]) + [r[5]-dh()]) for r in new_regions]

        return new_regions

    def fn(w, l, h):
        count = num_samples
        for regions in bottom_gen(w, l):
            if regions is None:
                continue
            if min_num_regions-3 > len(regions) or len(regions) > max_num_regions-2:
                continue
            regions = add_3d_regions(regions, w, l, h)
            if regions is None:
                continue
            if min_num_regions <= len(regions) <= max_num_regions:
                count -= 1
                yield regions
            if count == 0:
                break
    return fn


def test_3d_box_splitting():
    gen = get_3d_box_splitting_gen()
    for triangle in gen(3, 2, 1):
        print(triangle)


###########################################################################

class Delaunay2D(object):
    """
    Class to compute a Delaunay triangulation in 2D
    ref: https://github.com/jmespadero/pyDelaunay2D/blob/master/delaunay2D.py
    """

    def __init__(self, center=(0, 0), radius=0.5):
        """ Init and create a new frame to contain the triangulation
        center -- Optional position for the center of the frame. Default (0,0)
        radius -- Optional distance from corners to the center. """
        center = np.asarray(center)
        # Create coordinates for the corners of the frame
        self.coords = [center + radius * np.array((-1, -1)),
                       center + radius * np.array((+1, -1)),
                       center + radius * np.array((+1, +1)),
                       center + radius * np.array((-1, +1))]

        # Create two dicts to store triangle neighbours and circumcircles.
        self.triangles = {}
        self.circles = {}

        # Create two CCW triangles for the frame
        T1 = (0, 1, 3)
        T2 = (2, 3, 1)
        self.triangles[T1] = [T2, None, None]
        self.triangles[T2] = [T1, None, None]

        # Compute circumcenters and circumradius for each triangle
        for t in self.triangles:
            self.circles[t] = self.circumcenter(t)

    def circumcenter(self, tri):
        """ Compute circumcenter and circumradius of a triangle in 2D.
        Uses an extension of the method described here:
        http://www.ics.uci.edu/~eppstein/junkyard/circumcenter.html """
        pts = np.asarray([self.coords[v] for v in tri])
        pts2 = np.dot(pts, pts.T)
        A = np.bmat([[2 * pts2, [[1],
                                 [1],
                                 [1]]],
                     [[[1, 1, 1, 0]]]])

        b = np.hstack((np.sum(pts * pts, axis=1), [1]))
        x = np.linalg.solve(A, b)
        bary_coords = x[:-1]
        center = np.dot(bary_coords, pts)

        # radius = np.linalg.norm(pts[0] - center) # euclidean distance
        radius = np.sum(np.square(pts[0] - center))  # squared distance
        return (center, radius)

    def in_circle_fast(self, tri, p):
        """ Check if point p is inside of precomputed circumcircle of tri. """
        center, radius = self.circles[tri]
        return np.sum(np.square(center - p)) <= radius

    def in_circle_robust(self, tri, p):
        """ Check if point p is inside of circumcircle around the triangle tri.
        This is a robust predicate, slower than compare distance to centers
        ref: http://www.cs.cmu.edu/~quake/robust.html """
        m1 = np.asarray([self.coords[v] - p for v in tri])
        m2 = np.sum(np.square(m1), axis=1).reshape((3, 1))
        m = np.hstack((m1, m2))  # The 3x3 matrix to check
        return np.linalg.det(m) <= 0

    def add_point(self, p):
        """ Add a point to the current DT, and refine it using Bowyer-Watson. """
        p = np.asarray(p)
        idx = len(self.coords)
        # print("coords[", idx,"] ->",p)
        self.coords.append(p)

        # Search the triangle(s) whose circumcircle contains p
        bad_triangles = []
        for T in self.triangles:
            # Choose one method: inCircleRobust(T, p) or inCircleFast(T, p)
            if self.in_circle_fast(T, p):
                bad_triangles.append(T)

        # Find the CCW boundary (star shape) of the bad triangles,
        # expressed as a list of edges (point pairs) and the opposite
        # triangle to each edge.
        boundary = []
        # Choose a "random" triangle and edge
        T = bad_triangles[0]
        edge = 0
        # get the opposite triangle of this edge
        while True:
            # Check if edge of triangle T is on the boundary...
            # if opposite triangle of this edge is external to the list
            tri_op = self.triangles[T][edge]
            if tri_op not in bad_triangles:
                # Insert edge and external triangle into boundary list
                boundary.append((T[(edge + 1) % 3], T[(edge - 1) % 3], tri_op))

                # Move to next CCW edge in this triangle
                edge = (edge + 1) % 3

                # Check if boundary is a closed loop
                if boundary[0][0] == boundary[-1][1]:
                    break
            else:
                # Move to next CCW edge in opposite triangle
                edge = (self.triangles[tri_op].index(T) + 1) % 3
                T = tri_op

        # Remove triangles too near of point p of our solution
        for T in bad_triangles:
            del self.triangles[T]
            del self.circles[T]

        # Retriangle the hole left by bad_triangles
        new_triangles = []
        for (e0, e1, tri_op) in boundary:
            # Create a new triangle using point p and edge extremes
            T = (idx, e0, e1)

            # Store circumcenter and circumradius of the triangle
            self.circles[T] = self.circumcenter(T)

            # Set opposite triangle of the edge as neighbour of T
            self.triangles[T] = [tri_op, None, None]

            # Try to set T as neighbour of the opposite triangle
            if tri_op:
                # search the neighbour of tri_op that use edge (e1, e0)
                for i, neigh in enumerate(self.triangles[tri_op]):
                    if neigh:
                        if e1 in neigh and e0 in neigh:
                            # change link to use our new triangle
                            self.triangles[tri_op][i] = T

            # Add triangle to a temporal list
            new_triangles.append(T)

        # Link the new triangles each another
        N = len(new_triangles)
        for i, T in enumerate(new_triangles):
            self.triangles[T][1] = new_triangles[(i + 1) % N]  # next
            self.triangles[T][2] = new_triangles[(i - 1) % N]  # previous

    def export_triangles(self):
        """ Export the current list of Delaunay triangles """
        # Filter out triangles with any vertex in the extended BBox
        return self.triangles
        # return [(a - 4, b - 4, c - 4)
        #         for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]

    def export_circles(self):
        """ Export the circumcircles as a list of (center, radius) """
        # Remember to compute circumcircles if not done before
        # for t in self.triangles:
        #     self.circles[t] = self.circumcenter(t)

        # Filter out triangles with any vertex in the extended BBox
        # Do sqrt of radius before of return
        return [(self.circles[(a, b, c)][0], sqrt(self.circles[(a, b, c)][1]))
                for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]


def get_triangles_splitting_gen():
    """ randomly sample some 2D points in a rectangle and connects them to form triangles """

    def get_sides_area(points):
        # x, y = zip(*points)
        # area = 0.5 * (x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2] * (y[0] - y[1]))
        x1, y1 = points[0]
        x2, y2 = points[1]
        x3, y3 = points[2]
        l3 = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        l1 = sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
        l2 = sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
        p = (l1 + l2 + l3) / 2
        area = sqrt(p * (p - l1) * (p - l2) * (p - l3))
        return (l1, l2, l3), area

    def move_points_closer(point, points):
        middle = np.mean([p for p in points if tuple(p) != tuple(point)], axis=0)
        return middle + (point-middle) * (1 - 0.3 * np.random.random())

    def gen(w, h, num_points=4):
        # Create a random set of points
        bbox = np.asarray([w, h])
        seeds = np.random.random((num_points, 2))

        # Create Delaunay Triangulation and insert points one by one
        dt = Delaunay2D(radius=0.5)
        for s in seeds:
            dt.add_point(s - 0.5)

        triangles = dt.export_triangles()
        tri_points = []
        for triangle in triangles:
            points = [dt.coords[t] * bbox for t in triangle]
            modified_points = [move_points_closer(p, points) for p in points]
            lengths, area = get_sides_area(modified_points)
            if area < 0.01 * w * h or min([area / l for l in lengths]) < 0.1:
                continue
            tri_points.append([[p for p in modified_points], lengths])
        yield tri_points
    return gen


def test_triangle_splitting():
    gen = get_triangles_splitting_gen()
    for triangle in gen(4, 3):
        print(triangle)


###########################################################################


if __name__ == "__main__":
    # test_tray_splitting()
    # test_triangle_splitting()
    test_3d_box_splitting()
