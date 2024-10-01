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
from typing import Tuple, List, Dict
import random as rn

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

def get_tidy_data_gen(num_samples=40, min_num_regions=2, max_num_regions=6, max_depth=3, default_min_size=0.05, relation="mixed"):

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

            return regions, 'horizontally_aligned'
        
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
    
            return regions, 'vertically_aligned'
        
        def get_aligned_bottom_line():

            locations = ["top", "center", "bottom", "right_top", "left_top", "right_bottom", "left_bottom", 
                         "right_center", "left_center", "center_top", "center_bottom", "center_center"]

            regions = []
            n = rn.randint(min_num_regions, max_num_regions+1)

            if n > 6:
                loc = rn.choice(locations[:3], size = 1)
            else:
                loc = rn.choice(locations, size = 1)

            if loc == "top":
                _X, _Y, _W, _L = 0, 0.5*L, W, 0.5*L
            elif loc == "center":
                _X, _Y, _W, _L = 0, 0.25*L, W, 0.5*L
            elif loc == "bottom":
                _X, _Y, _W, _L = 0, 0, W, 0.5*L
            elif loc == "right_top":
                _X, _Y, _W, _L = 0.5*W, 0.5*L, 0.5*W, 0.5*L
            elif loc == "left_top":
                _X, _Y, _W, _L = 0, 0.5*L, 0.5*W, 0.5*L
            elif loc == "right_bottom":
                _X, _Y, _W, _L = 0.5*W, 0, 0.5*W, 0.5*L
            elif loc == "left_bottom":
                _X, _Y, _W, _L = 0, 0, 0.5*W, 0.5*L
            elif loc == "right_center":
                _X, _Y, _W, _L = 0.5*W, 0.25*L, 0.5*W, 0.5*L
            elif loc == "left_center":
                _X, _Y, _W, _L = 0, 0.25*L, 0.5*W, 0.5*L
            elif loc == "center_top":
                _X, _Y, _W, _L = 0.25*W, 0.5*L, 0.5*W, 0.5*L
            elif loc == "center_bottom":
                _X, _Y, _W, _L = 0.25*W, 0, 0.5*W, 0.5*L
            elif loc == "center_center":
                _X, _Y, _W, _L = 0.25*W, 0.25*L, 0.5*W, 0.5*L


            while True:
            
                start_padding = rn.uniform(0.02, 0.1)*_W
                __W = rn.uniform(0.6, 0.9)*(_W - start_padding)
            
                x_padding = rn.uniform(0.02, 0.08)
                _W_tmp = __W - (x_padding+default_min_size)*n
                if _W_tmp > 0:
                    break

            xs = [0] + np.sort(rn.uniform(0, _W_tmp, size = n)).tolist()
            __L = rn.uniform(0.3, 0.8)*(_L - 2*offset)
            y_bottom = rn.uniform(_Y + 0.02*L, 0.98*L - __L)
            ys = rn.uniform(default_min_size, __L, size = n)
            

            running_x = start_padding
            for i in range(n):
                x1 = _X + running_x 
                w1 = xs[i+1]-xs[i] + default_min_size
                l1 = ys[i] 
                running_x += (w1 + x_padding)
        
                regions.append((x1, y_bottom, w1, l1))

            return regions, 'aligned_horizontal_line'
        
        def get_aligned_vertical_line():

            locations = ["left", "center", "right", "right_top", "left_top", "right_bottom", "left_bottom", 
                         "right_center", "left_center", "center_top", "center_bottom", "center_center"]

            regions = []
            n = rn.randint(min_num_regions, max_num_regions+1)

            if n > 6:
                loc = rn.choice(locations[:3], size = 1)
            else:
                loc = rn.choice(locations, size = 1)

            if loc == "left":
                _X, _Y, _W, _L = 0, 0, 0.5*W, L
            elif loc == "center":
                _X, _Y, _W, _L = 0.25*W, 0, 0.5*W, L
            elif loc == "right":
                _X, _Y, _W, _L = 0.5*W, 0, 0.5*W, L
            elif loc == "right_top":
                _X, _Y, _W, _L = 0.5*W, 0.5*L, 0.5*W, 0.5*L
            elif loc == "left_top":
                _X, _Y, _W, _L = 0, 0.5*L, 0.5*W, 0.5*L
            elif loc == "right_bottom":
                _X, _Y, _W, _L = 0.5*W, 0, 0.5*W, 0.5*L
            elif loc == "left_bottom":
                _X, _Y, _W, _L = 0, 0, 0.5*W, 0.5*L
            elif loc == "right_center":
                _X, _Y, _W, _L = 0.5*W, 0.25*L, 0.5*W, 0.5*L
            elif loc == "left_center":
                _X, _Y, _W, _L = 0, 0.25*L, 0.5*W, 0.5*L
            elif loc == "center_top":
                _X, _Y, _W, _L = 0.25*W, 0.5*L, 0.5*W, 0.5*L
            elif loc == "center_bottom":
                _X, _Y, _W, _L = 0.25*W, 0, 0.5*W, 0.5*L
            elif loc == "center_center":
                _X, _Y, _W, _L = 0.25*W, 0.25*L, 0.5*W, 0.5*L


            while True:
                start_padding = rn.uniform(0.02, 0.1)*_L
                __L = rn.uniform(0.6, 0.9)*(_L - start_padding)
                y_padding = rn.uniform(0.02, 0.08)
                _L_tmp = __L - (y_padding+default_min_size)*n
                if _L_tmp > 0:
                    break
            
            x_center = rn.uniform(0.3, 0.7)*_W
            xs = rn.uniform(0.05, min(_W - x_center, x_center), size = n)
            ys = [0] + np.sort(rn.uniform(0, _L_tmp, size = n)).tolist()

            running_y = start_padding
            for i in range(n):
                x1 = _X + x_center - xs[i] 
                w1 = xs[i] * 2
                y1 = _Y + running_y 
                l1 = ys[i+1] - ys[i] + default_min_size
                running_y += l1 + y_padding
                regions.append((x1, y1, w1, l1))

            return regions, 'aligned_vertical_line'
             
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
            mac_regions, _ = get_cfree_regions(max_depth = 5, X=0, Y=0, W=W, L=L, offset=offset, offset_grid=False)
            mac_regions = filter_regions(mac_regions, 0.40)
            random.shuffle(mac_regions)

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
            
        
            possible_splits = ["center", "vertical", "horizontal"]

            splits = rn.choice(possible_splits, size = 2, replace = False)
            locs = []
            for split in splits:
                if split == "center":
                    loc = rn.choice(["center", "center_col", "center_row"], size = 1, p=[0.6, 0.2, 0.2])
                    locs.append(loc)
                elif split == "vertical":
                    locs.extend(["left", "right"])
                else:
                    locs.extend(["front", "back"])
            locs = locs[:len(n_combi)]

            for loc in locs:
                if loc == "center":
                    half_w = rn.uniform(0.1, 0.25)*W
                    half_l = rn.uniform(0.1, 0.25)*L
                    x_0 = W/2 - half_w
                    y_0 = L/2 - half_l
                    w_0 = half_w*2
                    l_0 = half_l*2
                    mac_regions.append((x_0, y_0, w_0, l_0))
                elif loc == "center_col":
                    half_w = rn.uniform(0.1, 0.25)*W
                    x_0 = W/2 - half_w
                    w_0 = half_w*2
                    l_0 = rn.uniform(0.2, 0.5)*L
                    if rand() < 0.5:
                        y_0 = rn.uniform(0.02, 0.2)*L
                    else:
                        y_0 = rn.uniform(0.8, 0.98)*L - l_0
                    mac_regions.append((x_0, y_0, w_0, l_0))
                elif loc == "center_row":
                    half_l = rn.uniform(0.1, 0.25)*L
                    y_0 = L/2 - half_l
                    l_0 = half_l*2
                    w_0 = rn.uniform(0.2, 0.5)*W
                    if rand() < 0.5:
                        x_0 = rn.uniform(0.02, 0.2)*W
                    else:
                        x_0 = rn.uniform(0.8, 0.98)*W - w_0
                    mac_regions.append((x_0, y_0, w_0, l_0))
                elif loc == "left":
                    half_w = rn.uniform(0.1, 0.2)*W
                    w_0 = half_w*2
                    x_0 = rn.uniform(0.02*W, 0.5*W-w_0)
                    
                    l_0 = rn.uniform(0.2, 0.4)*L
                    y_0 = rn.uniform(0.05*L, 0.95*L-l_0)
                    mac_regions.append((x_0, y_0, w_0, l_0))
                elif loc == "right":
                    half_w = rn.uniform(0.1, 0.2)*W
                    w_0 = half_w*2
                    x_0 = rn.uniform(0.5*W, 0.98*W-w_0)
                    
                    l_0 = rn.uniform(0.2, 0.4)*L
                    y_0 = rn.uniform(0.05*L, 0.95*L-l_0)
                    mac_regions.append((x_0, y_0, w_0, l_0))
                
                elif loc == "front":
                    half_l = rn.uniform(0.1, 0.2)*L
                    l_0 = half_l*2
                    y_0 = rn.uniform(0.05*L, 0.5*L-l_0)
                    
                    w_0 = rn.uniform(0.2, 0.4)*W
                    x_0 = rn.uniform(0.05*W, 0.95*W-w_0)
                    mac_regions.append((x_0, y_0, w_0, l_0))
                elif loc == "back":
                    half_l = rn.uniform(0.1, 0.2)*L
                    l_0 = half_l*2
                    y_0 = rn.uniform(0.5*L, 0.95*L-l_0)
                    
                    w_0 = rn.uniform(0.2, 0.4)*W
                    x_0 = rn.uniform(0.05*W, 0.95*W-w_0)
                    mac_regions.append((x_0, y_0, w_0, l_0)) 
            # while len(mac_regions) < len(n_combi):
            #     mac_regions, _ = get_cfree_regions(max_depth = 3, X=0, Y=0, W=W, L=L, offset=offset)
            #     mac_regions = filter_regions(mac_regions, 0.63)

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
        
        def study_table(W, L, x, relation):

            laptop_w, laptop_l = 0.34*W, 0.3*L
            book_w, book_l = 0.04*W, 0.2*L
            mug_w, mug_l = W/12, L/8
            paper_w, paper_l = 0.22*W, 0.45*L
            pen_w, pen_l = 0.025*W, 0.2*L
            tissue_box_w, tissue_box_l = 0.084*W, 0.125*L
            lamp_w, lamp_l = 0.067*W, 0.15*L
            monitor_w, monitor_l = 0.5*W, 0.2*L
            keyboard_w, keyboard_l = 0.5*W, 0.2*L
            mouse_w, mouse_l = 0.05*W, 0.1*L
            toy_w, toy_l = 0.067*W, 0.1*L
            glasses_w, glasses_l = 0.15*W, 0.05*L
            
            if x == "1":
                laptop = (0.33*W, 0.1*L, 0.34*W, 0.3*L)
                book_1 = (0.05*W, 0.75*L, 0.04*W, 0.2*L)
                book_2 = (0.09*W, 0.75*L, 0.04*W, 0.2*L)
                book_3 = (0.13*W, 0.75*L, 0.04*W, 0.2*L)
                book_4 = (0.17*W, 0.75*L, 0.04*W, 0.2*L)
                mug = (0.85*W, 0.25*L, W/12, L/8)
                obj_list = [laptop, book_1, book_2, book_3, book_4, mug]
                names = ["laptop", "book_1", "book_2", "book_3", "book_4", "mug"]
            elif x == "2":
                paper = (0.39*W, 0.15*L, 0.22*W, 0.45*L)
                pen = (0.63*W, 0.3*L, 0.025*W, 0.2*L)
                tissue_box = (0.1*W, 0.8*L, W/12, L/8)
                mug = (0.8*W, 0.8*L, W/12, L/8)
                lamp = (0.9*W, 0.8*L, 0.2*W/3, 0.15*L)
                obj_list = [paper, pen, tissue_box, mug, lamp]
                names = ["paper", "pen", "tissue_box", "mug", "lamp"]
            elif x == "3":
                monitor = (0.25*W, 0.75*L, 0.5*W, 0.2*L)
                keyboard = (0.25*W, 0.2*L, 0.5*W, 0.2*L)
                mouse = (0.8*W, 0.2*L, 0.05*W, 0.1*L)
                lamp = (0.9*W, 0.8*L, 0.2*W/3, 0.15*L)
                mug = (0.8*W, 0.6*L, W/12, L/8)
                obj_list = [monitor, keyboard, mouse, lamp, mug]
                names = ["monitor", "keyboard", "mouse", "lamp", "mug"]
            elif x == "4":
                book_1 = (0.05*W, 0.75*L, 0.04*W, 0.2*L)
                book_2 = (0.09*W, 0.75*L, 0.04*W, 0.2*L)
                book_3 = (0.13*W, 0.75*L, 0.04*W, 0.2*L)
                book_4 = (0.17*W, 0.75*L, 0.04*W, 0.2*L)
                lamp = (0.9*W, 0.8*L, 0.2*W/3, 0.15*L)
                paper_1 = (0.27*W, 0.15*L, 0.22*W, 0.45*L)
                paper_2 = (0.51*W, 0.15*L, 0.22*W, 0.45*L)
                pen = (0.78*W, 0.3*L, 0.025*W, 0.2*L)
                obj_list = [book_1, book_2, book_3, book_4, lamp, paper_1, paper_2, pen]
                names = ["book_1", "book_2", "book_3", "book_4", "lamp", "paper_1", "paper_2", "pen"]
            elif x == "5":
                monitor = (0.25*W, 0.75*L, 0.5*W, 0.2*L)
                laptop = (0.33*W, 0.425*L, 0.34*W, 0.3*L)
                keyboard = (0.25*W, 0.2*L, 0.5*W, 0.2*L)
                mouse = (0.8*W, 0.2*L, 0.05*W, 0.1*L)
                obj_list = [monitor, laptop, keyboard, mouse]
                names = ["monitor", "laptop", "keyboard", "mouse"]
            elif x == "6":
                laptop = (0.33*W, 0.25*L, 0.34*W, 0.3*L)
                mouse = (0.79*W, 0.25*L, 0.05*W, 0.1*L)
                lamp = (0.9*W, 0.8*L, 0.2*W/3, 0.15*L)
                mug = (0.8*W, 0.6*L, W/12, L/8)
                tissue_box = (0.1*W, 0.8*L, W/12, L/8)
                obj_list = [laptop, mouse, lamp, mug, tissue_box]
                names = ["laptop", "mouse", "lamp", "mug", "tissue_box"]
            elif x == "7":
                monitor = (0.25*W, 0.75*L, 0.5*W, 0.2*L)
                paper = (0.39*W, 0.15*L, 0.22*W, 0.45*L)
                pen = (0.63*W, 0.3*L, 0.025*W, 0.2*L)
                mug = (0.8*W, 0.5*L, W/12, L/8)
                lamp = (0.9*W, 0.8*L, 0.2*W/3, 0.15*L)
                obj_list = [monitor, paper, pen, mug, lamp]
                names = ["monitor", "paper", "pen", "mug", "lamp"] 
            elif x == "8":
                monitor = (0.25*W, 0.75*L, 0.5*W, 0.2*L)
                laptop = (0.19*W, 0.2*L, 0.3*W, 0.32*L)
                paper = (0.51*W, 0.2*L, 0.22*W, 0.46*L)
                pen = (0.75*W, 0.3*L, 0.025*W, 0.2*L)
                mug = (0.8*W, 0.6*L, W/12, L/8)
                tissue_box = (0.1*W, 0.75*L, W/12, L/8)
                toy = (0.8*W, 0.75*L, 0.2*W/3, 0.1*L)
                lamp = (0.9*W, 0.8*L, 0.2*W/3, 0.15*L)
                obj_list = [monitor, laptop, paper, pen, mug, tissue_box, toy, lamp]
                names = ["monitor", "laptop", "paper", "pen", "mug", "tissue_box", "toy", "lamp"]
            elif x == "9":
                monitor = (0.25*W, 0.75*L, 0.5*W, 0.2*L)
                laptop = (0.33*W, 0.425*L, 0.34*W, 0.3*L)
                keyboard = (0.25*W, 0.2*L, 0.5*W, 0.2*L)
                mouse = (0.8*W, 0.2*L, 0.05*W, 0.1*L)
                tissue_box = (0.1*W, 0.75*L, 0.2*W/3, 0.1*L)
                mug = (0.85*W, 0.5*L, W/12, L/8)
                glasses = (0.8*W, 0.65*L, 0.15*W, 0.05*L)
                obj_list = [monitor, laptop, keyboard, mouse, tissue_box, mug, glasses]
                names = ["monitor", "laptop", "keyboard", "mouse", "tissue_box", "mug", "glasses"]
            elif x == "10":
                monitor = (0.25*W, 0.75*L, 0.5*W, 0.2*L)
                laptop = (0.15*W, 0.1*L, 0.34*W, 0.3*L)
                book_1 = (0.05*W, 0.75*L, 0.04*W, 0.2*L)
                book_2 = (0.09*W, 0.75*L, 0.04*W, 0.2*L)
                book_3 = (0.13*W, 0.75*L, 0.04*W, 0.2*L)
                book_4 = (0.17*W, 0.75*L, 0.04*W, 0.2*L)
                tissue_box = (0.1*W, 0.45*L, W/12, L/8)
                mug = (0.8*W, 0.45*L, W/12, L/8)
                paper = (0.51*W, 0.1*L,0.22*W, 0.46*L)
                pen = (0.75*W, 0.2*L, 0.025*W, 0.2*L)
                obj_list = [monitor, laptop, book_1, book_2, book_3, book_4, tissue_box, mug, paper, pen]
                names = ["monitor", "laptop", "book_1", "book_2", "book_3", "book_4", "tissue_box", "mug", "paper", "pen"]
            
        
            return obj_list, relation, names

        def dining_table(W, L, x, relation): 

            serving_plate_w, serving_plate_l = 0.12*W, 0.18*L
            napkin_w, napkin_l = 0.05*W, 0.24*L
            fork_w, fork_l = 0.03*W, 0.22*L
            knife_w, knife_l = 0.03*W, 0.22*L
            spoon_w, spoon_l = 0.03*W, 0.2*L
            chopsticks_w, chopsticks_l = 0.03*W, 0.22*L
            glass_w, glass_l = 0.06*W, 0.09*L
            medium_plate_w, medium_plate_l = 0.12*W, 0.18*L
            small_plate_w, small_plate_l = 0.1*W, 0.15*L
            rice_bowl_w, rice_bowl_l = 0.09*W, 0.12*L
            ramen_bowl_w, ramen_bowl_l = 0.12*W, 0.18*L
            seasoning_w, seasoning_l = 0.12*W, 0.1*L
            baby_bowl_w, baby_bowl_l = 0.08*W, 0.12*L
            baby_plate_w, baby_plate_l = 0.1*W, 0.15*L
            baby_spoon_w, baby_spoon_l = 0.02*W, 0.15*L
            baby_cup_w, baby_cup_l = 0.05*W, 0.07*L

            front_base = 0.05*L
            back_base = 0.95*L


            if x == "1":
                # dinner table for 2
                serving_plate_1= (0.44*W, front_base, serving_plate_w, serving_plate_l)
                napkin_1 = (0.37*W, front_base, napkin_w, napkin_l)
                fork_1 = (0.38*W, front_base, fork_w, fork_l)
                knife_1 = (0.57*W, front_base, knife_w, knife_l)
                spoon_1 = (0.63*W, front_base, spoon_w, spoon_l)
                glass_1 = (0.68*W, 0.1*L, glass_w, glass_l)
                serving_plate_2 = (0.44*W, back_base - serving_plate_l, serving_plate_w, serving_plate_l)
                napkin_2 = (0.58*W, back_base - napkin_l, napkin_w, napkin_l)
                fork_2 = (0.59*W, back_base - fork_l, fork_w, fork_l)
                knife_2 = (0.40*W, back_base - knife_l, knife_w, knife_l)
                spoon_2 = (0.36*W, back_base - spoon_l, spoon_w, spoon_l)
                glass_2 = (0.28*W, 0.81*L, glass_w, glass_l)
                obj_list = [serving_plate_1, napkin_1, fork_1, knife_1, spoon_1, glass_1, serving_plate_2, napkin_2, fork_2, knife_2, spoon_2, glass_2]
                names = ["serving_plate_1", "napkin_1", "fork_1", "knife_1", "spoon_1", "glass_1", 
                         "serving_plate_2", "napkin_2", "fork_2", "knife_2", "spoon_2", "glass_2"]
            
            if x == "2":
                # chinese dinner table for 2
                medium_plate_1 = (0.375*W, 0.315*L, medium_plate_w, medium_plate_l)
                medium_plate_2 = (0.505*W, 0.315*L, medium_plate_w, medium_plate_l)
                medium_plate_3 = (0.375*W, 0.505*L, medium_plate_w, medium_plate_l)
                medium_plate_4 = (0.505*W, 0.505*L, medium_plate_w, medium_plate_l)
                small_plate_1 = (0.45*W, front_base, small_plate_w, small_plate_l)
                small_plate_2 = (0.45*W, back_base - small_plate_l, small_plate_w, small_plate_l)
                rice_bowl_1 = (0.455*W, front_base + 0.015*L, rice_bowl_w, rice_bowl_l)
                rice_bowl_2 = (0.455*W, back_base - 0.015*L - rice_bowl_l, rice_bowl_w, rice_bowl_l)
                chopsticks_1 = (0.57*W, front_base, chopsticks_w, chopsticks_l)
                chopsticks_2 = (0.40*W, back_base - chopsticks_l, chopsticks_w, chopsticks_l)
                spoon_1 = (0.62*W, front_base, spoon_w, spoon_l)
                spoon_2 = (0.35*W, back_base - spoon_l, spoon_w, spoon_l)
                obj_list = [medium_plate_1, medium_plate_2, medium_plate_3, medium_plate_4, small_plate_1, small_plate_2, rice_bowl_1, rice_bowl_2, chopsticks_1, chopsticks_2, spoon_1, spoon_2]
                names = ["medium_plate_1", "medium_plate_2", "medium_plate_3", "medium_plate_4", "small_plate_1", 
                         "small_plate_2", "rice_bowl_1", "rice_bowl_2", "chopsticks_1", "chopsticks_2", "spoon_1", 
                         "spoon_2"]
            
            if x == "3":
                # dinner table for 2 same side
                serving_plate_1= (0.17*W, front_base, serving_plate_w, serving_plate_l)
                napkin_1 = (0.09*W, front_base, napkin_w, napkin_l)
                fork_1 = (0.10*W, front_base, fork_w, fork_l)
                knife_1 = (0.31*W, front_base, knife_w, knife_l)
                spoon_1 = (0.35*W, front_base, spoon_w, spoon_l)
                glass_1 = (0.39*W, front_base, glass_w, glass_l)
                serving_plate_2= (0.68*W, front_base, serving_plate_w, serving_plate_l)
                napkin_2 = (0.60*W, front_base, napkin_w, napkin_l)
                fork_2 = (0.61*W, front_base, fork_w, fork_l)
                knife_2 = (0.82*W, front_base, knife_w, knife_l)
                spoon_2 = (0.86*W, front_base, spoon_w, spoon_l)
                glass_2 = (0.90*W, front_base, glass_w, glass_l)
                obj_list = [serving_plate_1, napkin_1, fork_1, knife_1, spoon_1, glass_1, serving_plate_2, napkin_2, fork_2, knife_2, spoon_2, glass_2]
                names = ["serving_plate_1", "napkin_1", "fork_1", "knife_1", "spoon_1", "glass_1",
                         "serving_plate_2", "napkin_2", "fork_2", "knife_2", "spoon_2", "glass_2"]
            
            if x == "4":
                # ramen table for 1
                ramen_bowl = (0.44*W, front_base, ramen_bowl_w, ramen_bowl_l)
                chopsticks = (0.59*W, front_base, chopsticks_w, chopsticks_l)
                spoon = (0.64*W, front_base, spoon_w, spoon_l)
                medium_plate_1 = (0.35*W, 0.41*L, medium_plate_w, medium_plate_l)
                medium_plate_2 = (0.53*W, 0.41*L, medium_plate_w, medium_plate_l)
                seasoning_1 = (0.85*W, 0.64*L, seasoning_w, seasoning_l)
                seasoning_2 = (0.85*W, 0.52*L, seasoning_w, seasoning_l)
                seasoning_3 = (0.85*W, 0.4*L, seasoning_w, seasoning_l)
                seasoning_4 = (0.85*W, 0.28*L, seasoning_w, seasoning_l)
                obj_list = [ramen_bowl, chopsticks, spoon, medium_plate_1, medium_plate_2, seasoning_1, seasoning_2, seasoning_3, seasoning_4]
                names = ["ramen_bowl", "chopsticks", "spoon", "medium_plate_1", "medium_plate_2", "seasoning_1",
                          "seasoning_2", "seasoning_3", "seasoning_4"]

            if x == "5":
                # with baby
                baby_plate = (0.18*W, front_base, baby_plate_w, baby_plate_l)
                baby_bowl = (0.19*W, front_base + 0.015*L, baby_bowl_w, baby_bowl_l)
                baby_spoon = (0.30*W, front_base, baby_spoon_w, baby_spoon_l)
                baby_cup = (0.34*W, front_base, baby_cup_w, baby_cup_l)
                serving_plate = (0.63*W, front_base, serving_plate_w, serving_plate_l)
                napkin = (0.56*W, front_base, napkin_w, napkin_l)
                fork = (0.57*W, front_base, fork_w, fork_l)
                knife = (0.77*W, front_base, knife_w, knife_l)
                spoon = (0.83*W, front_base, spoon_w, spoon_l)
                glass = (0.88*W, 0.1*L, glass_w, glass_l)
                seasoning_1 = (0.85*W, 0.62*L, seasoning_w, seasoning_l)
                seasoning_2 = (0.85*W, 0.5*L, seasoning_w, seasoning_l)
                seasoning_3 = (0.85*W, 0.38*L, seasoning_w, seasoning_l)
                obj_list = [baby_plate, baby_bowl, baby_spoon, baby_cup, serving_plate, napkin, fork, knife, spoon, glass, seasoning_1, seasoning_2, seasoning_3]
                names = ["baby_plate", "baby_bowl", "baby_spoon", "baby_cup", "serving_plate", "napkin", "fork", 
                         "knife", "spoon", "glass", "seasoning_1", "seasoning_2", "seasoning_3"]

            if x == "6":
                # left handed diner
                serving_plate_1= (0.44*W, front_base, serving_plate_w, serving_plate_l)
                napkin_1 = (0.37*W, front_base, napkin_w, napkin_l)
                fork_1 = (0.38*W, front_base, fork_w, fork_l)
                knife_1 = (0.57*W, front_base, knife_w, knife_l)
                spoon_1 = (0.63*W, front_base, spoon_w, spoon_l)
                glass_1 = (0.68*W, 0.1*L, glass_w, glass_l)
                serving_plate_2 = (0.44*W, back_base - serving_plate_l, serving_plate_w, serving_plate_l)
                napkin_2 = (0.37*W, back_base - napkin_l, napkin_w, napkin_l)
                fork_2 = (0.38*W, back_base - fork_l, fork_w, fork_l)
                knife_2 = (0.57*W, back_base - knife_l, knife_w, knife_l)
                spoon_2 = (0.63*W, back_base - spoon_l, spoon_w, spoon_l)
                glass_2 = (0.68*W, 0.81*L, glass_w, glass_l)
                obj_list = [serving_plate_1, napkin_1, fork_1, knife_1, spoon_1, glass_1, serving_plate_2, napkin_2, fork_2, knife_2, spoon_2, glass_2]
                names = ["serving_plate_1", "napkin_1", "fork_1", "knife_1", "spoon_1", "glass_1", 
                         "serving_plate_2", "napkin_2", "fork_2", "knife_2", "spoon_2", "glass_2"]

            if x == "7":
                # ramen set with baby
                baby_bowl = (0.21*W, front_base, baby_bowl_w, baby_bowl_l)
                baby_spoon = (0.31*W, front_base, baby_spoon_w, baby_spoon_l)
                baby_cup = (0.35*W, front_base, baby_cup_w, baby_cup_l)

                ramen_bowl = (0.6*W, front_base, ramen_bowl_w, ramen_bowl_l)
                chopsticks = (0.74*W, front_base, chopsticks_w, chopsticks_l)
                spoon = (0.79*W, front_base, spoon_w, spoon_l)
                glass = (0.85*W, front_base, glass_w, glass_l)

                medium_plate_1 = (0.35*W, 0.41*L, medium_plate_w, medium_plate_l)
                medium_plate_2 = (0.53*W, 0.41*L, medium_plate_w, medium_plate_l)
                
                seasoning_1 = (0.85*W, 0.64*L, seasoning_w, seasoning_l)
                seasoning_2 = (0.85*W, 0.52*L, seasoning_w, seasoning_l)
                seasoning_3 = (0.85*W, 0.4*L, seasoning_w, seasoning_l)
                seasoning_4 = (0.85*W, 0.28*L, seasoning_w, seasoning_l)

                obj_list = [baby_bowl, baby_spoon, baby_cup, 
                            ramen_bowl, chopsticks, spoon, 
                            medium_plate_1, medium_plate_2, 
                            glass, seasoning_1, seasoning_2, seasoning_3, seasoning_4]
                names = ["baby_bowl", "baby_spoon", "baby_cup", "ramen_bowl", "chopsticks", "spoon", 
                         "medium_plate_1", "medium_plate_2", "glass", "seasoning_1", 
                         "seasoning_2", "seasoning_3", "seasoning_4"]
            
            if x == "8":
                # 6 main dishes for sharing
                medium_plate_1 = (0.31*W, 0.315*L, medium_plate_w, medium_plate_l)
                medium_plate_2 = (0.44*W, 0.315*L, medium_plate_w, medium_plate_l)
                medium_plate_3 = (0.57*W, 0.315*L, medium_plate_w, medium_plate_l)
                medium_plate_4 = (0.31*W, 0.505*L, medium_plate_w, medium_plate_l)
                medium_plate_5 = (0.44*W, 0.505*L, medium_plate_w, medium_plate_l)
                medium_plate_6 = (0.57*W, 0.505*L, medium_plate_w, medium_plate_l)
        
                small_plate_1 = (0.45*W, front_base, small_plate_w, small_plate_l)
                small_plate_2 = (0.45*W, back_base - small_plate_l, small_plate_w, small_plate_l)
                rice_bowl_1 = (0.455*W, front_base + 0.015*L, rice_bowl_w, rice_bowl_l)
                rice_bowl_2 = (0.455*W, back_base - 0.015*L - rice_bowl_l, rice_bowl_w, rice_bowl_l)
                chopsticks_1 = (0.58*W, front_base, chopsticks_w, chopsticks_l)
                chopsticks_2 = (0.40*W, back_base - chopsticks_l, chopsticks_w, chopsticks_l)
                spoon_1 = (0.63*W, front_base, spoon_w, spoon_l)
                spoon_2 = (0.35*W, back_base - spoon_l, spoon_w, spoon_l)

                obj_list = [medium_plate_1, medium_plate_2, medium_plate_3, medium_plate_4, 
                            medium_plate_5, medium_plate_6,
                            small_plate_1, small_plate_2, rice_bowl_1, 
                            rice_bowl_2, chopsticks_1, chopsticks_2, spoon_1, spoon_2]
                names = ["medium_plate_1", "medium_plate_2", "medium_plate_3", "medium_plate_4", 
                         "medium_plate_5", "medium_plate_6",
                         "small_plate_1", "small_plate_2", "rice_bowl_1", "rice_bowl_2", "chopsticks_1", 
                         "chopsticks_2", "spoon_1", "spoon_2"]

            if x == "9":
                # ramen set for 2
                ramen_bowl_1 = (0.44*W, front_base, ramen_bowl_w, ramen_bowl_l)
                chopsticks_1 = (0.59*W, front_base, chopsticks_w, chopsticks_l)
                spoon_1 = (0.64*W, front_base, spoon_w, spoon_l)
                glass_1 = (0.70*W, front_base, glass_w, glass_l)

                medium_plate_1 = (0.375*W, 0.315*L, medium_plate_w, medium_plate_l)
                medium_plate_2 = (0.505*W, 0.315*L, medium_plate_w, medium_plate_l)
                medium_plate_3 = (0.375*W, 0.505*L, medium_plate_w, medium_plate_l)
                medium_plate_4 = (0.505*W, 0.505*L, medium_plate_w, medium_plate_l)

                ramen_bowl_2 = (0.44*W, back_base - ramen_bowl_l, ramen_bowl_w, ramen_bowl_l)
                chopsticks_2 = (0.39*W, back_base - chopsticks_l, chopsticks_w, chopsticks_l)
                spoon_2 = (0.34*W, back_base - spoon_l, spoon_w, spoon_l)
                glass_2 = (0.26*W, back_base - glass_l, glass_w, glass_l)

                seasoning_1 = (0.85*W, 0.62*L, seasoning_w, seasoning_l)
                seasoning_2 = (0.85*W, 0.5*L, seasoning_w, seasoning_l)
                seasoning_3 = (0.85*W, 0.38*L, seasoning_w, seasoning_l)

                obj_list = [ramen_bowl_1, chopsticks_1, spoon_1, glass_1, 
                            medium_plate_1, medium_plate_2, medium_plate_3, medium_plate_4, 
                            ramen_bowl_2, chopsticks_2, spoon_2, glass_2, 
                            seasoning_1, seasoning_2, seasoning_3]
                names = ["ramen_bowl_1", "chopsticks_1", "spoon_1", "glass_1", 
                         "medium_plate_1", "medium_plate_2", "medium_plate_3", "medium_plate_4", 
                         "ramen_bowl_2", "chopsticks_2", "spoon_2", "glass_2", "seasoning_1", 
                         "seasoning_2", "seasoning_3"]
                
            if x == "10":
                # 4_plates
                serving_plate_1= (0.19*W, front_base, serving_plate_w, serving_plate_l)
                napkin_1 = (0.11*W, front_base, napkin_w, napkin_l)
                fork_1 = (0.12*W, front_base, fork_w, fork_l)
                knife_1 = (0.33*W, front_base, knife_w, knife_l)
                spoon_1 = (0.37*W, front_base, spoon_w, spoon_l)
                glass_1 = (0.41*W, front_base, glass_w, glass_l)
                serving_plate_2= (0.69*W, front_base, serving_plate_w, serving_plate_l)
                napkin_2 = (0.61*W, front_base, napkin_w, napkin_l)
                fork_2 = (0.62*W, front_base, fork_w, fork_l)
                knife_2 = (0.83*W, front_base, knife_w, knife_l)
                spoon_2 = (0.87*W, front_base, spoon_w, spoon_l)
                glass_2 = (0.91*W, front_base, glass_w, glass_l)

                serving_plate_3= (0.19*W, back_base - serving_plate_l, serving_plate_w, serving_plate_l)
                napkin_3 = (0.33*W, back_base - napkin_l, napkin_w, napkin_l)
                fork_3 = (0.34*W, back_base - fork_l, fork_w, fork_l)
                knife_3 = (0.14*W, back_base - knife_l, knife_w, knife_l)
                spoon_3 = (0.1*W, back_base - spoon_l, spoon_w, spoon_l)
                glass_3 = (0.03*W, back_base - glass_l, glass_w, glass_l)
                serving_plate_4= (0.69*W, back_base - serving_plate_l, serving_plate_w, serving_plate_l)
                napkin_4 = (0.83*W, back_base - napkin_l, napkin_w, napkin_l)
                fork_4 = (0.84*W, back_base - fork_l, fork_w, fork_l)
                knife_4 = (0.64*W, back_base - knife_l, knife_w, knife_l)
                spoon_4 = (0.6*W, back_base - spoon_l, spoon_w, spoon_l)
                glass_4 = (0.53*W, back_base - glass_l, glass_w, glass_l)
                obj_list = [serving_plate_1, napkin_1, fork_1, knife_1, spoon_1, glass_1, 
                            serving_plate_2, napkin_2, fork_2, knife_2, spoon_2, glass_2,
                            serving_plate_3, napkin_3, fork_3, knife_3, spoon_3, glass_3,
                            serving_plate_4, napkin_4, fork_4, knife_4, spoon_4, glass_4
                            ]
                names =  ["serving_plate_1", "napkin_1", "fork_1", "knife_1", "spoon_1", "glass_1",
                         "serving_plate_2", "napkin_2", "fork_2", "knife_2", "spoon_2", "glass_2",
                            "serving_plate_3", "napkin_3", "fork_3", "knife_3", "spoon_3", "glass_3",
                            "serving_plate_4", "napkin_4", "fork_4", "knife_4", "spoon_4", "glass_4"]
                
            return obj_list, relation, names

        def coffee_table(W, L, x, relation):

            candle_w, candle_l = 0.08*W, 0.12*L
            diffuser_w, diffuser_l = 0.06*W, 0.09*L
            vase_w, vase_l = 0.10*W, 0.15*L
            book_w, book_l = 0.18*W, 0.37*L
            plant_w, plant_l = 0.12*W, 0.18*L

            laptop_w, laptop_l = 0.34*W, 0.3*L

            tray_w, tray_l = 0.3*W, 0.3*L
            keys_w, keys_l = 0.07*W, 0.07*L
            remote_controller_w, remote_controller_l = 0.05*W, 0.12*L
            glasses_w, glasses_l = 0.12*W, 0.07*L

            ashtray_w, ashtray_l = 0.08*W, 0.12*L

            beverage_w, beverage_l = 0.06*W, 0.09*L
            snack_bowl_w, snack_bowl_l = 0.16*W, 0.24*L

            coffee_pot_w, coffee_pot_l = 0.24*W, 0.36*L
            tea_pot_w, tea_pot_l = 0.2*W, 0.3*L

            coffee_cup_w, coffee_cup_l = 0.12*W, 0.18*L
            tea_cup_w, tea_cup_l = 0.1*W, 0.15*L
            cake_plate_w, cake_plate_l = 0.10*W, 0.15*L
            board_game_w, board_game_l = 0.4*W, 0.6*L

            ipad_w, ipad_l = 0.15*W, 0.3*L

            if x == "1":
                # standard coffee table
                vase = (0.45*W, 0.425*L, vase_w, vase_l) 
                book_1 = (0.21*W, 0.425*L, book_w, book_l)
                book_2 = (0.21*W, 0.425*L, book_w, book_l)
                book_3 = (0.21*W, 0.425*L, book_w, book_l)
                book_4 = (0.21*W, 0.425*L, book_w, book_l)
                diffuser = (0.56*W, 0.455*L, diffuser_w, diffuser_l)
                tray = (0.35*W, 0.05*L, tray_w, tray_l)
                keys = (0.42*W, 0.12*L, keys_w, keys_l)
                remote_controller = (0.52*W, 0.12*L, remote_controller_w, remote_controller_l)
                object_list = [vase, book_1, book_2, book_3, book_4, diffuser, tray, keys, remote_controller]
                names = ["vase", "book_1", "book_2", "book_3", "book_4", "diffuser", "tray", "keys", "remote_controller"]
            elif x == "2":
                # coffee table with symmetry candles
                vase = (0.45*W, 0.425*L, vase_w, vase_l)
                candle_1 = (0.36*W, 0.44*L, candle_w, candle_l)
                candle_2 = (0.56*W, 0.44*L, candle_w, candle_l)
                tray = (0.35*W, 0.05*L, tray_w, tray_l)
                keys = (0.41*W, 0.15*L, keys_w, keys_l)
                glass = (0.41*W, 0.2*L, glasses_w, glasses_l)
                remote_controller = (0.52*W, 0.15*L, remote_controller_w, remote_controller_l)
                object_list = [vase, candle_1, candle_2, tray, keys, glass, remote_controller]
                names = ["vase", "candle_1", "candle_2", "tray", "keys", "glass", "remote_controller"]
            elif x == "3":
                # coffee table for party
                book_1 = (0.02*W, 0.62*L, book_w, book_l)
                book_2 = (0.02*W, 0.62*L, book_w, book_l)
                book_3 = (0.02*W, 0.62*L, book_w, book_l)
                book_4 = (0.02*W, 0.62*L, book_w, book_l)
                beverage_1 = (0.4 * W, 0.51*L, beverage_w, beverage_l)
                beverage_2 = (0.47 * W, 0.51*L, beverage_w, beverage_l)
                beverage_3 = (0.54 * W, 0.51*L, beverage_w, beverage_l)
                beverage_4 = (0.4 * W, 0.4*L, beverage_w, beverage_l)
                beverage_5 = (0.47 * W, 0.4*L, beverage_w, beverage_l)
                beverage_6 = (0.54 * W, 0.4*L, beverage_w, beverage_l)

                snack_bowl_1 = (0.15 * W, 0.35*L, snack_bowl_w, snack_bowl_l)
                snack_bowl_2 = (0.65 * W, 0.35*L, snack_bowl_w, snack_bowl_l)

                ashtray = (0.46*W, 0.125*L, ashtray_w, ashtray_l)
                object_list = [book_1, book_2, book_3, book_4, beverage_1, beverage_2, beverage_3, beverage_4, beverage_5, beverage_6, snack_bowl_1, snack_bowl_2, ashtray]
                names = ["book_1", "book_2", "book_3", "book_4", "beverage_1", "beverage_2", "beverage_3", "beverage_4", "beverage_5", "beverage_6", "snack_bowl_1", "snack_bowl_2", "ashtray"]
                
            elif x == "4":
                # coffee table for study
                vase = (0.45*W, 0.425*L, vase_w, vase_l)
                laptop = (0.33*W, 0.1*L, laptop_w, laptop_l)
                book_1 = (0.02*W, 0.62*L, book_w, book_l)
                book_2 = (0.02*W, 0.62*L, book_w, book_l)
                book_3 = (0.02*W, 0.62*L, book_w, book_l)
                book_4 = (0.02*W, 0.62*L, book_w, book_l)
                tray = (0.65*W, 0.65*L, tray_w, tray_l)
                keys = (0.66*W, 0.75*L, keys_w, keys_l)
                glass = (0.66*W, 0.66*L, glasses_w, glasses_l)
                remote_controller = (0.77*W, 0.66*L, remote_controller_w, remote_controller_l)
                object_list = [vase, laptop, book_1, book_2, book_3, book_4, tray, keys, glass, remote_controller]
                names = ["vase", "laptop", "book_1", "book_2", "book_3", "book_4", "tray", "keys", "glass", "remote_controller"]

            elif x == "5":
                # coffee table for 2
                coffee_pot = (0.38*W, 0.36*L, coffee_pot_w, coffee_pot_l)
                coffee_cup_1 = (0.2*W, 0.1*L, coffee_cup_w, coffee_cup_l)
                coffee_cup_2 = (0.76*W, 0.1*L, coffee_cup_w, coffee_cup_l)
                book_1 = (0.02*W, 0.62*L, book_w, book_l)
                book_2 = (0.02*W, 0.62*L, book_w, book_l)
                book_3 = (0.02*W, 0.62*L, book_w, book_l)
                book_4 = (0.02*W, 0.62*L, book_w, book_l)
                tray = (0.65*W, 0.65*L, tray_w, tray_l)
                keys = (0.66*W, 0.75*L, keys_w, keys_l)
                glass = (0.66*W, 0.66*L, glasses_w, glasses_l)
                remote_controller = (0.77*W, 0.66*L, remote_controller_w, remote_controller_l)
                object_list = [coffee_pot, coffee_cup_1, coffee_cup_2, book_1, book_2, book_3, book_4, tray, keys, glass, remote_controller]
                names = ["coffee_pot", "coffee_cup_1", "coffee_cup_2", "book_1", "book_2", "book_3", "book_4", "tray", "keys", "glass", "remote_controller"]


            elif x == "6":
                # standard coffee table
                plant = (0.44*W, 0.41*L, plant_w, plant_l)
                candle_1 = (0.26*W, 0.44*L, candle_w, candle_l)
                candle_2 = (0.35*W, 0.44*L, candle_w, candle_l)
                candle_3 = (0.57*W, 0.44*L, candle_w, candle_l)
                candle_4 = (0.66*W, 0.44*L, candle_w, candle_l)
                book_1 = (0.02*W, 0.62*L, book_w, book_l)
                book_2 = (0.02*W, 0.62*L, book_w, book_l)
                book_3 = (0.02*W, 0.62*L, book_w, book_l)
                book_4 = (0.02*W, 0.62*L, book_w, book_l)
                object_list = [plant, candle_1, candle_2, candle_3, candle_4, book_1, book_2, book_3, book_4]
                names = ["plant", "candle_1", "candle_2", "candle_3", "candle_4", "book_1", "book_2", "book_3", "book_4"]


            elif x == "7":
                diffuser = (0.47*W, 0.8*L, diffuser_w, diffuser_l)
                candle_1 = (0.36*W, 0.8*L, candle_w, candle_l)
                candle_2 = (0.56*W, 0.8*L, candle_w, candle_l)
                board_game = (0.3*W, 0.2*L, board_game_w, board_game_l)
                coffee_cup_1 = (0.2*W, 0.1*L, coffee_cup_w, coffee_cup_l)
                coffee_cup_2 = (0.76*W, 0.1*L, coffee_cup_w, coffee_cup_l)
                object_list = [diffuser, candle_1, candle_2, board_game, coffee_cup_1, coffee_cup_2]
                names = ["diffuser", "candle_1", "candle_2", "board_game", "coffee_cup_1", "coffee_cup_2"]
            
            elif x == "8":  
                snack_bowl_1 = (0.33 * W, 0.25*L, snack_bowl_w, snack_bowl_l)
                snack_bowl_2 = (0.51 * W, 0.25*L, snack_bowl_w, snack_bowl_l)
                snack_bowl_3 = (0.33 * W, 0.51*L, snack_bowl_w, snack_bowl_l)
                snack_bowl_4 = (0.51 * W, 0.51*L, snack_bowl_w, snack_bowl_l)

                tea_cup_1 = (0.2*W, 0.1*L, tea_cup_w, tea_cup_l)
                tea_cup_2 = (0.76*W, 0.1*L, tea_cup_w, tea_cup_l)

                tray = (0.65*W, 0.65*L, tray_w, tray_l)
                keys = (0.66*W, 0.75*L, keys_w, keys_l)
                glass = (0.66*W, 0.66*L, glasses_w, glasses_l)
                remote_controller = (0.77*W, 0.66*L, remote_controller_w, remote_controller_l)
                object_list = [snack_bowl_1, snack_bowl_2, snack_bowl_3, snack_bowl_4, tea_cup_1, tea_cup_2, tray, keys, glass, remote_controller]
                names = ["snack_bowl_1", "snack_bowl_2", "snack_bowl_3", "snack_bowl_4", "tea_cup_1", "tea_cup_2", "tray", "keys", "glass", "remote_controller"]

            elif x == "9":
                plant = (0.44*W, 0.41*L, plant_w, plant_l)
                book_1 = (0.02*W, 0.62*L, book_w, book_l)
                book_2 = (0.02*W, 0.62*L, book_w, book_l)
                book_3 = (0.02*W, 0.62*L, book_w, book_l)
                book_4 = (0.02*W, 0.62*L, book_w, book_l)
                ipad = (0.425*W, 0.05*L, ipad_w, ipad_l)
                coffee_cup = (0.63*W, 0.15*L, coffee_cup_w, coffee_cup_l)
                object_list = [plant, book_1, book_2, book_3, book_4, ipad, coffee_cup]
                names = ["plant", "book_1", "book_2", "book_3", "book_4", "ipad", "coffee_cup"]
            
            elif x == "10":
                # coffee table for serving cakes and tea fir 2
                tea_pot = (0.4*W, 0.35*L, tea_pot_w, tea_pot_l)
                tea_cup_1 = (0.3*W, 0.1*L, tea_cup_w, tea_cup_l)
                tea_cup_2 = (0.7*W, 0.1*L, tea_cup_w, tea_cup_l)
                tray = (0.65*W, 0.65*L, tray_w, tray_l)
                keys = (0.66*W, 0.75*L, keys_w, keys_l)
                glass = (0.66*W, 0.66*L, glasses_w, glasses_l)
                remote_controller = (0.77*W, 0.66*L, remote_controller_w, remote_controller_l)
                object_list = [tea_pot, tea_cup_1, tea_cup_2, tray, keys, glass, remote_controller]
                names = ["tea_pot", "tea_cup_1", "tea_cup_2", "tray", "keys", "glass", "remote_controller"]

            return object_list, relation, names  

        def study_table_train(W, L, x, relation):

            laptop_w, laptop_l = 0.34*W, 0.3*L
            book_w, book_l = 0.04*W, 0.2*L
            mug_w, mug_l = W/12, L/8
            paper_w, paper_l = 0.22*W, 0.45*L
            pen_w, pen_l = 0.025*W, 0.2*L
            tissue_box_w, tissue_box_l = 0.084*W, 0.125*L
            lamp_w, lamp_l = 0.067*W, 0.15*L
            monitor_w, monitor_l = 0.5*W, 0.2*L
            keyboard_w, keyboard_l = 0.5*W, 0.2*L
            mouse_w, mouse_l = 0.05*W, 0.1*L
            toy_w, toy_l = 0.067*W, 0.1*L
            glasses_w, glasses_l = 0.15*W, 0.05*L
            
            # laptop & books
            if x == "1":
                laptop = (0.33*W, 0.3*L, 0.34*W, 0.3*L)
                book_1 = (0.05*W, 0.6*L, 0.04*W, 0.2*L)
                book_2 = (0.11*W, 0.6*L, 0.04*W, 0.2*L)
                book_3 = (0.17*W, 0.6*L, 0.04*W, 0.2*L)
                book_4 = (0.23*W, 0.6*L, 0.04*W, 0.2*L)
                mouse = (0.82*W, 0.4*L, 0.05*W, 0.1*L)
                lamp = (0.8*W, 0.7*L, 0.2*W/3, 0.15*L)
                obj_list = [laptop, book_1, book_2, book_3, book_4, mouse, lamp]
                names = ["laptop", "book_1", "book_2", "book_3", "book_4", "mouse", "lamp"]

            # external monitor 
            elif x == "2":
                monitor = (0.25*W, 0.7*L, 0.5*W, 0.2*L)
                laptop = (0.33*W, 0.4*L, 0.34*W, 0.3*L)
                mouse = (0.8*W, 0.45*L, 0.05*W, 0.1*L)
                lamp = (0.8*W, 0.7*L, 0.2*W/3, 0.15*L)
                mug = (0.9*W, 0.45*L, W/12, L/8)
                obj_list = [monitor, laptop, mouse, lamp, mug]
                names = ["monitor", "laptop", "mouse", "lamp", "mug"]
            
            #external keyboard
            elif x == "3":
                laptop = (0.33*W, 0.4*L, 0.34*W, 0.3*L)
                keyboard = (0.25*W, 0.15*L, 0.5*W, 0.2*L)
                mouse = (0.8*W, 0.17*L, 0.05*W, 0.1*L)
                lamp = (0.8*W, 0.7*L, 0.2*W/3, 0.15*L)
                mug = (0.9*W, 0.45*L, W/12, L/8)
                obj_list = [laptop, keyboard, mouse, lamp, mug]
                names = ["laptop", "keyboard", "mouse", "lamp", "mug"]

            # 2 laptops
            elif x == "4":
                book_1 = (0.05*W, 0.6*L, 0.04*W, 0.2*L)
                book_2 = (0.11*W, 0.6*L, 0.04*W, 0.2*L)
                book_3 = (0.17*W, 0.6*L, 0.04*W, 0.2*L)
                book_4 = (0.23*W, 0.6*L, 0.04*W, 0.2*L)
                lamp = (0.8*W, 0.7*L, 0.2*W/3, 0.15*L)
                laptop_1 = (0.15*W, 0.1*L, 0.3*W, 0.32*L)
                laptop_2 = (0.55*W, 0.1*L, 0.3*W, 0.32*L)
                obj_list = [book_1, book_2, book_3, book_4, lamp, laptop_1, laptop_2]
                names = ["book_1", "book_2", "book_3", "book_4", "lamp", "laptop_1", "laptop_2"]

            # notepad and books   
            elif x == "5":
                book_1 = (0.05*W, 0.6*L, 0.04*W, 0.2*L)
                book_2 = (0.11*W, 0.6*L, 0.04*W, 0.2*L)
                book_3 = (0.17*W, 0.6*L, 0.04*W, 0.2*L)
                book_4 = (0.23*W, 0.6*L, 0.04*W, 0.2*L)
                lamp = (0.8*W, 0.7*L, 0.2*W/3, 0.15*L)
                paper = (0.39*W, 0.10*L, 0.22*W, 0.45*L)
                pen = (0.63*W, 0.15*L, 0.025*W, 0.2*L)
                mug = (0.9*W, 0.45*L, W/12, L/8)
                obj_list = [book_1, book_2, book_3, book_4, lamp, mug, pen, paper]
                names = ["book_1", "book_2", "book_3", "book_4", "lamp", "mug", "pen", "paper"]

            return obj_list, relation, names
            
        def coffee_table_train(W, L, x, relation):

            candle_w, candle_l = 0.08*W, 0.12*L
            diffuser_w, diffuser_l = 0.06*W, 0.09*L
            vase_w, vase_l = 0.10*W, 0.15*L
            book_w, book_l = 0.18*W, 0.37*L
            plant_w, plant_l = 0.12*W, 0.18*L

            laptop_w, laptop_l = 0.34*W, 0.3*L

            tray_w, tray_l = 0.3*W, 0.3*L
            keys_w, keys_l = 0.07*W, 0.07*L
            remote_controller_w, remote_controller_l = 0.05*W, 0.12*L
            glasses_w, glasses_l = 0.12*W, 0.07*L

            ashtray_w, ashtray_l = 0.08*W, 0.12*L

            beverage_w, beverage_l = 0.06*W, 0.09*L
            snack_bowl_w, snack_bowl_l = 0.16*W, 0.24*L

            coffee_pot_w, coffee_pot_l = 0.24*W, 0.36*L
            tea_pot_w, tea_pot_l = 0.2*W, 0.3*L

            coffee_cup_w, coffee_cup_l = 0.12*W, 0.18*L
            tea_cup_w, tea_cup_l = 0.1*W, 0.15*L
            small_plate_w, small_plate_l = 0.1*W, 0.15*L
            board_game_w, board_game_l = 0.4*W, 0.6*L

            ipad_w, ipad_l = 0.15*W, 0.3*L

            if x == "1":
                # standard coffee table
                vase = (0.45*W, 0.425*L, vase_w, vase_l)
                candle_1 = (0.36*W, 0.44*L, candle_w, candle_l)
                candle_2 = (0.56*W, 0.44*L, candle_w, candle_l)
                book_1 = (0.1*W, 0.6*L, book_w, book_l)
                book_2 = (0.1*W, 0.6*L, book_w, book_l)
                book_3 = (0.1*W, 0.6*L, book_w, book_l)
                book_4 = (0.1*W, 0.6*L, book_w, book_l)
                tray = (0.35*W, 0.05*L, tray_w, tray_l)
                keys = (0.45*W, 0.12*L, keys_w, keys_l)
                remote_controller = (0.55*W, 0.15*L, remote_controller_w, remote_controller_l)
                object_list = [vase, candle_1, candle_2, book_1, book_2, book_3, book_4, tray, keys, remote_controller]
                names = ["vase", "candle_1", "candle_2", "book_1", "book_2", "book_3", "book_4", "tray", "keys", "remote_controller"]
            elif x == "2":  
                # snack for sharing
                snack_bowl_1 = (0.33 * W, 0.35*L, snack_bowl_w, snack_bowl_l)
                snack_bowl_2 = (0.51 * W, 0.35*L, snack_bowl_w, snack_bowl_l)
               
                tray = (0.6*W, 0.6*L, tray_w, tray_l)
                keys = (0.65*W, 0.65*L, keys_w, keys_l)
                remote_controller = (0.77*W, 0.66*L, remote_controller_w, remote_controller_l)
                object_list = [snack_bowl_1, snack_bowl_2, tray, keys, remote_controller]
                names = ["snack_bowl_1", "snack_bowl_2", "tray", "keys","remote_controller"]

            elif x == "3":
                # coffee table for cakes and coffee
                coffee_pot = (0.38*W, 0.36*L, coffee_pot_w, coffee_pot_l)
                small_plate_1 = (0.15*W, 0.1*L, small_plate_w, small_plate_l)
                coffee_cup_1 = (0.27*W, 0.1*L, coffee_cup_w, coffee_cup_l)
                small_plate_2 = (0.71*W, 0.1*L, small_plate_w, small_plate_l)
                coffee_cup_2 = (0.83*W, 0.1*L, coffee_cup_w, coffee_cup_l)
                book_1 = (0.02*W, 0.62*L, book_w, book_l)
                book_2 = (0.02*W, 0.62*L, book_w, book_l)
                book_3 = (0.02*W, 0.62*L, book_w, book_l)
                book_4 = (0.02*W, 0.62*L, book_w, book_l)
                
                object_list = [coffee_pot, small_plate_1, coffee_cup_1, small_plate_2, coffee_cup_2, book_1, book_2, book_3, book_4]
                names = ["coffee_pot", "small_plate_1", "coffee_cup_1", "small_plate_2", "coffee_cup_2", "book_1", "book_2", "book_3", "book_4"]

            elif x == "4":
                # coffee table for one book
                vase = (0.45*W, 0.425*L, vase_w, vase_l)
                candle_1 = (0.36*W, 0.44*L, candle_w, candle_l)
                candle_2 = (0.56*W, 0.44*L, candle_w, candle_l)
                book = (0.41*W, 0.03*L, book_w, book_l)
                tray = (0.6*W, 0.6*L, tray_w, tray_l)
                keys = (0.65*W, 0.65*L, keys_w, keys_l)
                remote_controller = (0.77*W, 0.66*L, remote_controller_w, remote_controller_l)
                object_list = [vase, candle_1, candle_2, book, tray, keys, remote_controller]
                names = ["vase", "candle_1", "candle_2", "book", "tray", "keys", "remote_controller"]

            elif x == "5":
                # coffee table for key
                vase_1 = (0.33*W, 0.425*L, vase_w, vase_l)
                vase_2 = (0.45*W, 0.425*L, vase_w, vase_l)
                vase_3 = (0.57*W, 0.425*L, vase_w, vase_l)
                book_1 = (0.02*W, 0.62*L, book_w, book_l)
                book_2 = (0.02*W, 0.62*L, book_w, book_l)
                book_3 = (0.02*W, 0.62*L, book_w, book_l)
                book_4 = (0.02*W, 0.62*L, book_w, book_l)
                tray = (0.35*W, 0.05*L, tray_w, tray_l)
                keys = (0.45*W, 0.12*L, keys_w, keys_l)
                remote_controller = (0.55*W, 0.15*L, remote_controller_w, remote_controller_l)
                
                object_list = [vase_1, vase_2, vase_3, book_1, book_2, book_3, book_4, tray, keys, remote_controller]
                names = ["vase_1", "vase_2", "vase_3", "book_1", "book_2", "book_3", "book_4", "tray", "keys", "remote_controller"]

            return object_list, relation, names  
        
        def survey_table(W, L, x, relation):

            serving_plate_w, serving_plate_l = 0.12*W, 0.18*L
            napkin_w, napkin_l = 0.05*W, 0.24*L
            fork_w, fork_l = 0.03*W, 0.22*L
            knife_w, knife_l = 0.03*W, 0.22*L
            spoon_w, spoon_l = 0.03*W, 0.2*L
            seasoning_w, seasoning_l = 0.12*W, 0.1*L
            candle_w, candle_l = 0.08*W, 0.12*L
            
            medium_plate_w, medium_plate_l = 0.12*W, 0.18*L
            small_plate_w, small_plate_l = 0.1*W, 0.15*L
            rice_bowl_w, rice_bowl_l = 0.09*W, 0.12*L
            chopsticks_w, chopsticks_l = 0.03*W, 0.22*L
            
            
            # case 1: 5pts

            if x == "1":
                front_base = 0.05*L
                serving_plate_1= (0.44*W, front_base, serving_plate_w, serving_plate_l)
                napkin_1 = (0.37*W, front_base, napkin_w, napkin_l)
                fork_1 = (0.38*W, front_base, fork_w, fork_l)
                knife_1 = (0.57*W, front_base, knife_w, knife_l)
                spoon_1 = (0.63*W, front_base, spoon_w, spoon_l)
                seasoning_1 = (0.85*W, 0.64*L, seasoning_w, seasoning_l)
                seasoning_2 = (0.85*W, 0.52*L, seasoning_w, seasoning_l)
                seasoning_3 = (0.85*W, 0.4*L, seasoning_w, seasoning_l)
                seasoning_4 = (0.85*W, 0.28*L, seasoning_w, seasoning_l)
                candle_1 = (0.2*W, 0.36*L, candle_w, candle_l)
                candle_2 = (0.2*W, 0.52*L, candle_w, candle_l)

                object_list = [serving_plate_1, napkin_1, fork_1, knife_1, spoon_1, seasoning_1, seasoning_2, seasoning_3, seasoning_4, candle_1, candle_2]
                names = ["serving_plate_1", "napkin_1", "fork_1", "knife_1", "spoon_1", "seasoning_1", "seasoning_2", "seasoning_3", "seasoning_4", "candle_1", "candle_2"]

                return object_list, relation, names
            
            # case 2: 4pts, objects not aligned
            elif x == "2":
                front_base = 0.15*L

                random_padding = rn.uniform(-0.075, 0.075, size=5)*L
                serving_plate_1= (0.44*W, front_base + random_padding[0], serving_plate_w, serving_plate_l)
                napkin_1 = (0.37*W, front_base + random_padding[1], napkin_w, napkin_l)
                fork_1 = (0.38*W, front_base + random_padding[2], fork_w, fork_l)
                knife_1 = (0.57*W, front_base + random_padding[3], knife_w, knife_l)
                spoon_1 = (0.63*W, front_base + random_padding[4], spoon_w, spoon_l)
                seasoning_1 = (0.82*W, 0.72*L, seasoning_w, seasoning_l)
                seasoning_2 = (0.81*W, 0.68*L, seasoning_w, seasoning_l)
                seasoning_3 = (0.82*W, 0.65*L, seasoning_w, seasoning_l)
                seasoning_4 = (0.80*W, 0.7*L, seasoning_w, seasoning_l)
                candle_1 = (0.2*W, 0.45*L, candle_w, candle_l)
                candle_2 = (0.23*W, 0.52*L, candle_w, candle_l)
                

                object_list = [serving_plate_1, napkin_1, fork_1, knife_1, spoon_1, seasoning_1, seasoning_2, seasoning_3, seasoning_4, candle_1, candle_2]
                names = ["serving_plate_1", "napkin_1", "fork_1", "knife_1", "spoon_1", "seasoning_1", "seasoning_2", "seasoning_3", "seasoning_4", "candle_1", "candle_2"]

                return object_list, relation, names

            # case 3: 3pts, objects crowded
            elif x == "3":
                front_base = 0.15*L

                random_padding = rn.uniform(-0.075, 0.075, size=5)*L
                serving_plate_1= (0.44*W, front_base + random_padding[0], serving_plate_w, serving_plate_l)
                napkin_1 = (0.37*W, front_base + random_padding[1], napkin_w, napkin_l)
                fork_1 = (0.38*W, front_base + random_padding[2], fork_w, fork_l)
                knife_1 = (0.57*W, front_base + random_padding[3], knife_w, knife_l)
                spoon_1 = (0.63*W, front_base + random_padding[4], spoon_w, spoon_l)
                seasoning_1 = (0.82*W, 0.72*L, seasoning_w, seasoning_l)
                seasoning_2 = (0.81*W, 0.68*L, seasoning_w, seasoning_l)
                seasoning_3 = (0.82*W, 0.65*L, seasoning_w, seasoning_l)
                seasoning_4 = (0.80*W, 0.7*L, seasoning_w, seasoning_l)
                candle_1 = (0.78*W, 0.68*L, candle_w, candle_l)
                candle_2 = (0.78*W, 0.60*L, candle_w, candle_l)

                object_list = [serving_plate_1, napkin_1, fork_1, knife_1, spoon_1, seasoning_1, seasoning_2, seasoning_3, seasoning_4, candle_1, candle_2]
                names = ["serving_plate_1", "napkin_1", "fork_1", "knife_1", "spoon_1", "seasoning_1", "seasoning_2", "seasoning_3", "seasoning_4", "candle_1", "candle_2"]

                return object_list, relation, names
            
            # case 4: 2pts, utensils misplaced
            elif x == "4":
                front_base = 0.4*L

                random_padding = rn.uniform(-0.075, 0.075, size=5)*L
                serving_plate_1= (0.44*W, front_base + random_padding[0], serving_plate_w, serving_plate_l)
                napkin_1 = (0.37*W, front_base + random_padding[1], napkin_w, napkin_l)
                fork_1 = (0.38*W, front_base + random_padding[2], fork_w, fork_l)
                knife_1 = (0.31*W, front_base + random_padding[3], knife_w, knife_l)
                spoon_1 = (0.25*W, front_base + random_padding[4], spoon_w, spoon_l)
                seasoning_1 = (0.82*W, 0.72*L, seasoning_w, seasoning_l)
                seasoning_2 = (0.81*W, 0.68*L, seasoning_w, seasoning_l)
                seasoning_3 = (0.82*W, 0.65*L, seasoning_w, seasoning_l)
                seasoning_4 = (0.80*W, 0.7*L, seasoning_w, seasoning_l)
                candle_1 = (0.78*W, 0.68*L, candle_w, candle_l)
                candle_2 = (0.78*W, 0.60*L, candle_w, candle_l)

                object_list = [serving_plate_1, napkin_1, fork_1, knife_1, spoon_1, seasoning_1, seasoning_2, seasoning_3, seasoning_4, candle_1, candle_2]
                names = ["serving_plate_1", "napkin_1", "fork_1", "knife_1", "spoon_1", "seasoning_1", "seasoning_2", "seasoning_3", "seasoning_4", "candle_1", "candle_2"]

                return object_list, relation, names
            
            # case 5,1 pt
            if x == "5":
                front_base = 0.15*L

                random_padding = rn.uniform(-0.075, 0.075, size=5)*L
                serving_plate_1= (0.67*W, 0.64*L, serving_plate_w, serving_plate_l)
                napkin_1 = (0.73*W, 0.65*L, napkin_w, napkin_l)
                fork_1 = (0.74*W, 0.76*L, fork_w, fork_l)
                knife_1 = (0.64*W, 0.59*L, knife_w, knife_l)
                spoon_1 = (0.64*W, 0.79*L, spoon_w, spoon_l)
                seasoning_1 = (0.82*W, 0.72*L, seasoning_w, seasoning_l)
                seasoning_2 = (0.81*W, 0.68*L, seasoning_w, seasoning_l)
                seasoning_3 = (0.82*W, 0.65*L, seasoning_w, seasoning_l)
                seasoning_4 = (0.80*W, 0.7*L, seasoning_w, seasoning_l)
                candle_1 = (0.78*W, 0.68*L, candle_w, candle_l)
                candle_2 = (0.78*W, 0.60*L, candle_w, candle_l)

                object_list = [serving_plate_1, napkin_1, fork_1, knife_1, spoon_1, seasoning_1, seasoning_2, seasoning_3, seasoning_4, candle_1, candle_2]
                names = ["serving_plate_1", "napkin_1", "fork_1", "knife_1", "spoon_1", "seasoning_1", "seasoning_2", "seasoning_3", "seasoning_4", "candle_1", "candle_2"]

                return object_list, relation, names
            
            elif x == "6":
                front_base = 0.05*L
                back_base = 0.95*L

                rice_bowl = (0.455*W+0.2*W, front_base + 0.015*L, rice_bowl_w, rice_bowl_l)
    
                small_plate = (0.45*W+0.2*W, back_base - small_plate_l, small_plate_w, small_plate_l)

                chopsticks = (0.48*W, front_base, chopsticks_w, chopsticks_l)
            
                spoon = (0.48*W, back_base - spoon_l, spoon_w, spoon_l)

                medium_plate_1 = (0.24*W,  front_base, medium_plate_w, medium_plate_l)
                medium_plate_2 = (0.24*W,  back_base - medium_plate_l, medium_plate_w, medium_plate_l)

                object_list = [rice_bowl, small_plate, chopsticks, spoon, medium_plate_1, medium_plate_2]
                names = ["rice_bowl", "small_plate", "chopsticks", "spoon", "medium_plate_1", "medium_plate_2"]

                return object_list, relation, names
        
        def dining_table_train(W, L, x, relation): 

            serving_plate_w, serving_plate_l = 0.12*W, 0.18*L
            napkin_w, napkin_l = 0.05*W, 0.24*L
            fork_w, fork_l = 0.03*W, 0.22*L
            knife_w, knife_l = 0.03*W, 0.22*L
            spoon_w, spoon_l = 0.03*W, 0.2*L
            chopsticks_w, chopsticks_l = 0.03*W, 0.22*L
            glass_w, glass_l = 0.06*W, 0.09*L
            medium_plate_w, medium_plate_l = 0.12*W, 0.18*L
            small_plate_w, small_plate_l = 0.1*W, 0.15*L
            rice_bowl_w, rice_bowl_l = 0.09*W, 0.12*L
            ramen_bowl_w, ramen_bowl_l = 0.12*W, 0.18*L
            seasoning_w, seasoning_l = 0.12*W, 0.1*L
            baby_bowl_w, baby_bowl_l = 0.08*W, 0.12*L
            baby_plate_w, baby_plate_l = 0.1*W, 0.15*L
            baby_spoon_w, baby_spoon_l = 0.02*W, 0.15*L
            baby_cup_w, baby_cup_l = 0.05*W, 0.07*L

            front_base = 0.1*L
            back_base = 0.9*L


            if x == "1":
                # dinner table for 1
                serving_plate_1= (0.44*W, front_base, serving_plate_w, serving_plate_l)
                napkin_1 = (0.35*W, front_base, napkin_w, napkin_l)
                fork_1 = (0.36*W, front_base, fork_w, fork_l)
                knife_1 = (0.59*W, front_base, knife_w, knife_l)
                spoon_1 = (0.67*W, front_base, spoon_w, spoon_l)
               
                serving_plate_2 = (0.44*W, back_base - serving_plate_l, serving_plate_w, serving_plate_l)
                napkin_2 = (0.60*W, back_base - napkin_l, napkin_w, napkin_l)
                fork_2 = (0.61*W, back_base - fork_l, fork_w, fork_l)
                knife_2 = (0.38*W, back_base - knife_l, knife_w, knife_l)
                spoon_2 = (0.32*W, back_base - spoon_l, spoon_w, spoon_l)
                
                obj_list = [serving_plate_1, napkin_1, fork_1, knife_1, spoon_1, serving_plate_2, napkin_2, fork_2, knife_2, spoon_2]
                names = ["serving_plate_1", "napkin_1", "fork_1", "knife_1", "spoon_1",
                         "serving_plate_2", "napkin_2", "fork_2", "knife_2", "spoon_2"]
            
            if x == "2":
                # chinese dinner table for 2
                medium_plate_1 = (0.375*W, 0.41*L, medium_plate_w, medium_plate_l)
                medium_plate_2 = (0.505*W, 0.41*L, medium_plate_w, medium_plate_l)
                small_plate_1 = (0.45*W, front_base, small_plate_w, small_plate_l)
                small_plate_2 = (0.45*W, back_base - small_plate_l, small_plate_w, small_plate_l)
                rice_bowl_1 = (0.455*W, front_base + 0.015*L, rice_bowl_w, rice_bowl_l)
                rice_bowl_2 = (0.455*W, back_base - 0.015*L - rice_bowl_l, rice_bowl_w, rice_bowl_l)
                chopsticks_1 = (0.59*W, front_base, chopsticks_w, chopsticks_l)
                chopsticks_2 = (0.38*W, back_base - chopsticks_l, chopsticks_w, chopsticks_l)
                spoon_1 = (0.66*W, front_base, spoon_w, spoon_l)
                spoon_2 = (0.31*W, back_base - spoon_l, spoon_w, spoon_l)
                obj_list = [medium_plate_1, medium_plate_2, small_plate_1, small_plate_2, rice_bowl_1, rice_bowl_2, chopsticks_1, chopsticks_2, spoon_1, spoon_2]
                names = ["medium_plate_1", "medium_plate_2", "small_plate_1", 
                         "small_plate_2", "rice_bowl_1", "rice_bowl_2", "chopsticks_1", "chopsticks_2", "spoon_1", 
                         "spoon_2"]
            
            if x == "3":
                # dinner table for 2 same side
                serving_plate_1= (0.17*W, front_base, serving_plate_w, serving_plate_l)
                napkin_1 = (0.09*W, front_base, napkin_w, napkin_l)
                fork_1 = (0.10*W, front_base, fork_w, fork_l)
                knife_1 = (0.31*W, front_base, knife_w, knife_l)
                spoon_1 = (0.35*W, front_base, spoon_w, spoon_l)
               
                serving_plate_2= (0.68*W, front_base, serving_plate_w, serving_plate_l)
                napkin_2 = (0.60*W, front_base, napkin_w, napkin_l)
                fork_2 = (0.61*W, front_base, fork_w, fork_l)
                knife_2 = (0.82*W, front_base, knife_w, knife_l)
                spoon_2 = (0.86*W, front_base, spoon_w, spoon_l)
                seasoning_1 = (0.85*W, 0.62*L, seasoning_w, seasoning_l)
                seasoning_2 = (0.85*W, 0.5*L, seasoning_w, seasoning_l)
                seasoning_3 = (0.85*W, 0.38*L, seasoning_w, seasoning_l)
                
                obj_list = [serving_plate_1, napkin_1, fork_1, knife_1, spoon_1, serving_plate_2, napkin_2, fork_2, knife_2, spoon_2,
                seasoning_1, seasoning_2, seasoning_3]
                names = ["serving_plate_1", "napkin_1", "fork_1", "knife_1", "spoon_1", 
                         "serving_plate_2", "napkin_2", "fork_2", "knife_2", "spoon_2",
                          "seasoning_1",  "seasoning_2", "seasoning_3"]
            
            if x == "4":
                # ramen set for 2, left-handed
                ramen_bowl_1 = (0.44*W, front_base, ramen_bowl_w, ramen_bowl_l)
                chopsticks_1 = (0.59*W, front_base, chopsticks_w, chopsticks_l)
                spoon_1 = (0.64*W, front_base, spoon_w, spoon_l)
                glass_1 = (0.70*W, front_base, glass_w, glass_l)

                ramen_bowl_2 = (0.44*W, back_base - ramen_bowl_l, ramen_bowl_w, ramen_bowl_l)
                chopsticks_2 = (0.59*W, back_base - chopsticks_l, chopsticks_w, chopsticks_l)
                spoon_2 = (0.64*W, back_base - spoon_l, spoon_w, spoon_l)
                glass_2 = (0.70*W, back_base - glass_l, glass_w, glass_l)

                seasoning_1 = (0.85*W, 0.62*L, seasoning_w, seasoning_l)
                seasoning_2 = (0.85*W, 0.5*L, seasoning_w, seasoning_l)
                seasoning_3 = (0.85*W, 0.38*L, seasoning_w, seasoning_l)

                obj_list = [ramen_bowl_1, chopsticks_1, spoon_1, glass_1, 
                            ramen_bowl_2, chopsticks_2, spoon_2, glass_2, 
                            seasoning_1, seasoning_2, seasoning_3]
                names = ["ramen_bowl_1", "chopsticks_1", "spoon_1", "glass_1", 
                         "ramen_bowl_2", "chopsticks_2", "spoon_2", "glass_2", "seasoning_1", 
                         "seasoning_2", "seasoning_3"]
                
            if x == "5":
                # dinner table for two with sharing dishes
                serving_plate_1= (0.44*W, front_base, serving_plate_w, serving_plate_l)
                napkin_1 = (0.35*W, front_base, napkin_w, napkin_l)
                fork_1 = (0.36*W, front_base, fork_w, fork_l)
                knife_1 = (0.59*W, front_base, knife_w, knife_l)
                spoon_1 = (0.67*W, front_base, spoon_w, spoon_l)
                medium_plate_1 = (0.375*W, 0.41*L, medium_plate_w, medium_plate_l)
                medium_plate_2 = (0.505*W, 0.41*L, medium_plate_w, medium_plate_l)
               
                serving_plate_2 = (0.44*W, back_base - serving_plate_l, serving_plate_w, serving_plate_l)
                napkin_2 = (0.60*W, back_base - napkin_l, napkin_w, napkin_l)
                fork_2 = (0.61*W, back_base - fork_l, fork_w, fork_l)
                knife_2 = (0.38*W, back_base - knife_l, knife_w, knife_l)
                spoon_2 = (0.32*W, back_base - spoon_l, spoon_w, spoon_l)
                seasoning_1 = (0.85*W, 0.62*L, seasoning_w, seasoning_l)
                seasoning_2 = (0.85*W, 0.5*L, seasoning_w, seasoning_l)
                seasoning_3 = (0.85*W, 0.38*L, seasoning_w, seasoning_l)
                obj_list = [serving_plate_1, napkin_1, fork_1, knife_1, spoon_1, serving_plate_2, napkin_2, fork_2, knife_2, spoon_2, medium_plate_1, medium_plate_2, 
                seasoning_1, seasoning_2, seasoning_3]
                names = ["serving_plate_1", "napkin_1", "fork_1", "knife_1", "spoon_1",
                         "serving_plate_2", "napkin_2", "fork_2", "knife_2", "spoon_2",
                         "medium_plate_1", "medium_plate_2", "seasoning_1", "seasoning_2",
                         "seasoning_3"]
                
            return obj_list, relation, names

        def evaluate(method_name, table_type, x):
            import json
            with open(f"./tmp/result/{method_name}/{table_type}_table_obj_list.json", "r") as f:
                obj_list = json.load(f)[f"{x}"]
                
                objs = []
                names = []

                for obj_name, obj_info in obj_list.items():
                    centroid = obj_info["centroid"]
                    size = obj_info["size"]

                    obj = (centroid[0]-size[0]/2+1.5, centroid[1]-size[1]/2+1, size[0], size[1])
                    objs.append(obj)
                    names.append(obj_name)
                return objs, "evaluate", names
            
        # def dining_table(W, L, x, relation): 
        count = num_samples

        if "all" in relation:
            if min_num_regions == max_num_regions:
                n = min_num_regions
            else:
                n = rn.choice(np.arange(min_num_regions, max_num_regions+1))
                while n == 11:
                    n = rn.choice(np.arange(min_num_regions, max_num_regions+1))
            if min_num_regions in [4, 6, 8, 9, 10]:
                relation = rn.choice(["horizontally_aligned", "vertically_aligned", "aligned_bottom_line", "aligned_vertical_line", 
                                      "on_top_of", "centered", "next_to_edge", "in", "symmetry", "next_to", "regular_grid"])
            elif min_num_regions in [3, 5, 7]:
                relation = rn.choice(["horizontally_aligned", "vertically_aligned", "aligned_bottom_line", "aligned_vertical_line", 
                                      "on_top_of", "centered", "next_to_edge", "in", "symmetry", "next_to"])
            elif min_num_regions in [12, 13, 14, 15, 16]:
                relation = "regular_grid"
        if "n_arity" in relation:
            if min_num_regions == max_num_regions:
                n = min_num_regions
            else:
                n = rn.choice(np.arange(min_num_regions, max_num_regions+1))
                while n == 11:
                    n = rn.choice(np.arange(min_num_regions, max_num_regions+1))
            if min_num_regions in [4, 6, 8, 9, 10]:
                relation = rn.choice(["aligned_bottom_line", "aligned_vertical_line", "regular_grid"])
            elif min_num_regions in [3, 5, 7]:
                relation = rn.choice(["aligned_bottom_line", "aligned_vertical_line"])
            elif min_num_regions in [12, 13, 14, 15, 16]:
                relation = "regular_grid"

        names = []
        while True:
            if relation == "horizontally_aligned":
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
            elif relation == "vertically_aligned":
                regions, relation_mode = get_aligned_vertical_regions()
            elif relation == "symmetry":
                regions, relation_mode = get_symmetry_regions(W, L)
            elif "regular_grid" in relation:
                regions, relation_mode = get_in_regular_grid(W, L, offset)
            elif relation == "aligned_bottom_line":
                regions, relation_mode = get_aligned_bottom_line()
            elif relation == "aligned_vertical_line":   
                regions, relation_mode = get_aligned_vertical_line()
            elif "dining_table_train" in relation:
                print("which idx")
                x = input()
                regions, relation_mode, names = dining_table_train(W, L, x, relation)
            elif "study_table_train" in relation:
                print("which idx")
                x = input()
                regions, relation_mode, names = study_table_train(W, L, x, relation)
            elif "coffee_table_train" in relation:
                print("which idx")
                x = input()
                regions, relation_mode, names = coffee_table_train(W, L, x, relation)
            elif "dining_table" in relation:
                print("which idx")
                x = input()
                regions, relation_mode, names = dining_table(W, L, x, relation)
            elif "study_table" in relation:
                print("which idx")
                x = input()
                regions, relation_mode, names = study_table(W, L, x, relation)
            elif "coffee_table" in relation:
                print("which idx")
                x = input()
                regions, relation_mode, names = coffee_table(W, L, x, relation)
            elif "survey_table" in relation:
                print("which idx")
                x = input()
                regions, relation_mode, names = survey_table(W, L, x, relation)
            elif "evaluate" in relation:
                print("which idx")
                x = input()
                method_name = relation.split("_")[1]
                table_type = relation.split("_")[2]
                regions, relation_mode, names = evaluate(method_name, table_type, x)
            try:
                regions = filter_regions(regions, min_size)
            except:
                pdb.set_trace()
            # (("ccollide" in relation or "integrated" in relation) and len(regions) == 2) or
            if min_num_regions <= len(regions) <= max_num_regions or "study_table" in relation or "dining_table" in relation or "coffee_table" in relation or "survey_table" in relation or "evaluate" in relation:
                count -= 1
                print(len(regions), "added!")
                yield regions, relation_mode, names
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

def get_cluster_data_gen(num_samples=40, min_num_regions=3, max_num_regions=16, default_min_size=0.05):
    
    Region = Tuple[float, float, float, float] # x, y, w, l

    def partition(box: Region, depth: int = 3) -> Iterable[Region]:
        if rand() < 0.15 or depth == 0 or box[2] < 0.15 or box[3] < 0.15:
            yield box

        else:
            if rand() < 0.5:
                axis = 0
            else:
                axis = 1

            split_point = rand() * (box[axis + 2] * 0.6)
            padding = 0.2 * box[axis + 2]
            if axis == 0:
                yield from partition((box[0], box[1], split_point + padding, box[3]), depth - 1)
                yield from partition((box[0] + split_point + padding, box[1], box[2] - split_point - padding, box[3]), depth - 1)
            else:
                yield from partition((box[0], box[1], box[2], split_point + padding), depth - 1)
                yield from partition((box[0], box[1] + split_point + padding, box[2], box[3] - split_point - padding), depth - 1)


    def filter_regions(regions: Iterable[Region], min_size: float) -> Iterable[Region]:
        
        if len(regions) == 0:
            return regions
        else:
            return [r for r in regions if r[2] > min_size and r[3] > min_size]

    def gen(W, L, relation, offset=0.05):

        min_size = min([W, L]) / 2 * default_min_size

        def get_base_region():

            w, l = rn.uniform(0.3, 0.5)*W, rn.uniform(0.3, 0.5)*L

            x_0, y_0 = rn.uniform(0, W - w), rn.uniform(0.3*L, L - l)

            return x_0, y_0, w, l

        def get_2D_regular():

            x_0, y_0, w_0, l_0 = get_base_region()

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

            while not flag:
            
                m = rn.choice([3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16])
                
                mac_region = (x_0, y_0, w_0, l_0)
            
                if m == 3:
                    if w_0 > l_0:
                        x_cut, y_cut = 3, 1
                    else:
                        x_cut, y_cut = 1, 3

                elif m == 4: # 2,2 
                    x_cut, y_cut = 2, 2

                elif m == 5:
                    if w_0 > l_0:
                        x_cut, y_cut = 5, 1
                    else:
                        y_cut, x_cut = 5, 1
                    
                elif m == 6: # 2,3
                    if w_0 > l_0:
                        x_cut, y_cut = 3, 2
                    else:
                        x_cut, y_cut = 2, 3

                elif m == 7:
                    if w_0 > l_0:
                        x_cut, y_cut = 7, 1
                    else:
                        x_cut, y_cut = 1, 7
                            
                elif m == 8: # 2,4
                    if w_0 > l_0:
                        x_cut, y_cut = 4, 2
                    else:
                        x_cut, y_cut = 2, 4

                elif m == 9:
                    x_cut, y_cut = 3, 3

                elif m == 10:
                    if w_0 > l_0:
                        x_cut, y_cut = 5, 2
                    else:
                        x_cut, y_cut = 2, 5
                elif m == 12:
                    if w_0 > l_0:
                        if w_0 > 1.5*l_0:
                            x_cut, y_cut = 6, 2
                        else:
                            x_cut, y_cut = 3, 4
                    else:
                        if l_0 > 1.5*w_0:
                            x_cut, y_cut = 2, 6
                        else:
                            x_cut, y_cut = 4, 3
                elif m == 14: 
                    if w_0 > l_0:
                        x_cut, y_cut = 7, 2
                    else:
                        x_cut, y_cut = 2, 7   
                elif m == 15:
                    if w_0 > l_0:
                        x_cut, y_cut = 5, 3
                    else:
                        x_cut, y_cut = 3, 5

                elif m == 16:
                    x_cut, y_cut = 4, 4
                
                
                sub_regions, flag = _get_unit_sub_regions(x_cut, y_cut, mac_region)

    
            return sub_regions, f"2D_regular_{x_cut}_{y_cut}"
        
        def get_2D_irregular():

            x_0, y_0, w_0, l_0 = get_base_region()

            regions = []

            max_depth = rn.choice([2, 3, 4])

            for region in partition((x_0, y_0, w_0, l_0), max_depth):
                regions.append(region)

            print(len(regions))
            # pdb.set_trace()

            return regions, f"2D_irregular_{len(regions)}"
        
        def get_3D_stacking():
            return [], []
        
        def get_3D_stacking_2D_regular():
            return [], []
        
        def get_3D_stacking_2D_irregular():
            return [], []

        while True:
            if relation == "2D_regular":
                regions, relation_mode = get_2D_regular()
            elif relation == "2D_irregular":
                regions, relation_mode = get_2D_irregular()
            elif relation == "3D_stacking":
                regions, relation_mode = get_3D_stacking()
            elif relation == "3D_stacking_2D_regular":
                regions, relation_mode = get_3D_stacking_2D_regular()
            elif relation == "3D_stacking_2D_irregular":
                regions, relation_mode = get_3D_stacking_2D_irregular()
           
            try:
                regions = filter_regions(regions, min_size)
            except:
                pdb.set_trace()
            
            count = num_samples

            if min_num_regions <= len(regions) <= max_num_regions:
                count -= 1
                print(len(regions), "added!")
                yield regions, relation_mode
            if count == 0:
                break
        yield None
        
    return gen

def test_tray_splitting():
    gen = get_tray_splitting_gen(num_samples=2)
    for boxes in gen(4, 3):
        print(boxes)

##########################################################################
             
def get_bedroom_data_gen(num_samples, window_loc=None, door_loc=None, furniture_num=None, bedroom_dim=(3, 4), relation="against-wall"):
    
    # A bedroom is a rectangle with width W and length L that can be read from bedroom_dim
    # It has four walls: "back-wall", "right-wall", "front-wall", "left-wall" 
    # It has four corners: "back-left-corner", "back-right-corner", "front-left-corner", "front-right-corner"
    # For each wall, it has three relative sides: "left", "right", "center"
    # The relative side is viewed from the center of the room facing that wall, therefore the left for the back-wall and front-wall is reversed, similarly for the left-wall and right-wall
    # The front-left-corner is with coordinates (0, 0), the back-right-corner is with coordinates (W, L)

    # Furniture_region is a tuple of 5 floats: (x, y, w, l, orientation)
    # (x, y) is the coordinate of the lower left corner of the furniture, w is the width, l is the length, orientation is the angle of the furniture, 0 means the furniture is facing the back wall, Pi/2 means the furniture is facing the right wall.
   
    Furniture_region = Tuple[float, float, float, float, float]  

    def get_furniture_dim(furniture_type = None, furniture_num = 1, along_same_wall = False, wall_index=None, room_dim=(3, 4)):
        # Return the dims of the furniture
        furniture_dims = []
        if furniture_type == "window":
            # Create window dim (width only)
            window_width = rn.uniform(0.5, 1.5)
            return window_width
        elif furniture_type == "door":
            # Create door dim (width only)
            door_width = rn.uniform(0.8, 1.2)
            return door_width
        elif along_same_wall:
            # Create furniture_num of furnitures that fit along the wall_index
            wall_length = room_dim[0] if wall_index in ["back-wall", "front-wall"] else room_dim[1]
            max_total_width = wall_length * 0.9
            min_furniture_width = wall_length * 0.1
            max_furniture_width = wall_length * 0.5

            widths = []
            total_width = 0
            for _ in range(furniture_num):
                remaining_width = max_total_width - total_width
                if remaining_width <= 0:
                    break
                width = rn.uniform(min_furniture_width, min(max_furniture_width, remaining_width))
                total_width += width
                widths.append(width)
            furniture_num = len(widths)  # Update furniture_num in case we couldn't fit all

            # Now, generate lengths and orientations
            for width in widths:
                max_furniture_length = min(room_dim[1], room_dim[0]) * 0.4
                min_furniture_length = max_furniture_length * 0.5
                length = rn.uniform(min_furniture_length, max_furniture_length)
                if wall_index in ["back-wall", "front-wall"]:
                    orientation = 0 if wall_index == "back-wall" else math.pi
                else:
                    orientation = math.pi / 2 if wall_index == "left-wall" else -math.pi/2
                furniture_dims.append((width, length, orientation))
            return furniture_dims
        else:
            for _ in range(furniture_num):
                max_furniture_width = min(room_dim) * 0.35
                max_furniture_length = max(room_dim) * 0.35
                min_furniture_width = max_furniture_width * 0.2
                min_furniture_length = max_furniture_length * 0.2
                width = rn.uniform(min_furniture_width, max_furniture_width)
                length = rn.uniform(min_furniture_length, max_furniture_length)
                if wall_index in ["back-wall", "front-wall"]:
                    orientation = 0 if wall_index == "back-wall" else math.pi
                else:
                    orientation = math.pi / 2 if wall_index == "left-wall" else -math.pi/2
                furniture_dims.append((width, length, orientation))
            return furniture_dims

    def init_bedroom_layout(bedroom_dim, window_loc, door_loc):
        W, L = bedroom_dim

        if window_loc is None:
            window_wall_index = rn.choice(["back-wall", "front-wall", "left-wall", "right-wall"])
        else:
            window_wall_index = window_loc

        # window_loc is a string indicating the wall index of the window, the window center would be on the middle of the wall
        window_region, window_wall_index = get_window_location(bedroom_dim, window_wall_index)
        regions = [(window_region, ["window", f"againts-{window_wall_index}"])]
        # door_loc is a tuple of string, the first indicates the wall index of the door, the second indicates if the door center is on the left or right side of the wall, viewed from the room center.
        if door_loc is not None:
            door_region, door_wall_index, door_side = get_door_location(bedroom_dim, door_loc)
            regions.append((door_region, ["door", f"against-{door_wall_index}", f"{door_side}_of_wall"]))
            door_info = (door_region, door_wall_index, door_side)
        else:
            door_info = None

        return {"window": (window_region, window_wall_index), "door": None}, regions
      
    def get_window_location(bedroom_dim, window_loc):
        W, L = bedroom_dim
        window_width = get_furniture_dim(furniture_type="window", room_dim=bedroom_dim)
        window_length = 0.1
        wall_index = window_loc
        # Calculate the region tuple for the window based on its dim, the wall index, and the wall_length
        if wall_index == "back-wall":
            x_center = W / 2
            orientation = 0
            x = x_center - window_width / 2
            y = L 
        elif wall_index == "front-wall":
            x_center = W / 2
            orientation = math.pi
            x = x_center - window_width / 2
            y = 0 - window_length
        elif wall_index == "left-wall":
            y_center = L / 2
            orientation = math.pi / 2
            x = 0 - window_length
            y = y_center - window_width / 2
        elif wall_index == "right-wall":
            y_center = L / 2
            orientation = -math.pi / 2
            x = W 
            y = y_center - window_width / 2
        else:
            raise ValueError("Invalid wall index for window location")
        window_region = (x, y, window_width, window_length, orientation)
        return window_region, wall_index

    def get_door_location(bedroom_dim, door_loc):
        W, L = bedroom_dim
        wall_index, door_side = door_loc
        door_width = get_furniture_dim(furniture_type="door", room_dim=bedroom_dim)
        if wall_index == "back-wall":
            y = L
            if door_side == "left":
                x = 0
            elif door_side == "right":
                x = W - door_width
            elif door_side == "center":
                x = (W - door_width) / 2
            else:
                raise ValueError("Invalid door side")
            orientation = 0
        elif wall_index == "front-wall":
            y = 0
            if door_side == "left":
                x = W - door_width
            elif door_side == "right":
                x = 0
            elif door_side == "center":
                x = (W - door_width) / 2
            else:
                raise ValueError("Invalid door side")
            orientation = math.pi
        elif wall_index == "left-wall":
            x = 0
            if door_side == "left":
                y = 0
            elif door_side == "right":
                y = L - door_width
            elif door_side == "center":
                y = (L - door_width) / 2
            else:
                raise ValueError("Invalid door side")
            orientation = math.pi / 2
        elif wall_index == "right-wall":
            x = W
            if door_side == "left":
                y = L - door_width
            elif door_side == "right":
                y = 0
            elif door_side == "center":
                y = (L - door_width) / 2
            else:
                raise ValueError("Invalid door side")
            orientation = -math.pi / 2
        else:
            raise ValueError("Invalid wall index for door location")
        door_region = (x, y, door_width, 0.1, orientation)
        return door_region, wall_index, door_side

    def get_test_case_furniture_locations(test_idx, room_dim=(3, 4)):

        def get_furniture_dim(name, orientation):

            furniture_dims = {

                "bed": (0.9, 1.8),
                "wardrobe": (2.0, 0.5),
                "study_desk": (1.1, 0.5),
                "study_chair": (0.3, 0.3),
                "bookshelf": (0.6, 0.3),
                "bedside_table": (0.25, 0.25),
                "piano": (1.0, 0.4),
                "guitar": (0.3, 0.2),
                "carpet": (0.5, 0.5),
                "cello": (0.4, 0.25),
                "painting_s": (0.45, 0.1),
                "painting_m": (0.5, 0.1),
                "painting_l": (0.55, 0.1),
                "painting_xl": (0.6, 0.1),
                "couch": (0.4, 0.4),
                "mirror": (0.3, 0.1),
                "dressing_table": (1.0, 0.4),
                "dressing_chair": (0.3, 0.3),
                "storage_cabinet": (0.35, 0.35)
            }

            w, l = furniture_dims[name]
            
            return (w, l, orientation)
            

        # furniture dim (W, L, rotation)

        W, L = room_dim

        if test_idx == 1:

            furniture_names = ["bed", "bedside_table_1", "bedside_table_2", "wardrobe", "study_desk", "study_chair", "bookshelf", "couch"]

            door_window_loc, regions = init_bedroom_layout(room_dim, window_loc="right-wall", door_loc=("left-wall", "left"))

            bed_dim = get_furniture_dim("bed", 0)
            bedside_table_1_dim = get_furniture_dim("bedside_table", 0)
            bedside_table_2_dim = get_furniture_dim("bedside_table", 0)
            wardrobe_dim = get_furniture_dim("wardrobe", np.pi/2)
            study_desk_dim = get_furniture_dim("study_desk", np.pi)
            study_chair_dim = get_furniture_dim("study_chair", np.pi)
            bookshelf_dim = get_furniture_dim("bookshelf", np.pi)
            couch_dim = get_furniture_dim("couch", 0)

            bed_location = ((W-bed_dim[0])/2, L-bed_dim[1], bed_dim[0], bed_dim[1], bed_dim[2])
            bedside_table_1_location = (bed_location[0] - bedside_table_1_dim[0], L - bedside_table_1_dim[1], bedside_table_1_dim[0], bedside_table_1_dim[1], bedside_table_1_dim[2])
            bedside_table_2_location = (bed_location[0] + bed_dim[0], L - bedside_table_2_dim[1], bedside_table_2_dim[0], bedside_table_2_dim[1], bedside_table_2_dim[2])
            wardrobe_location = (0, L-wardrobe[0], wardrobe[0], wardrobe[1], wardrobe[2])
            study_desk_location = ((W-study_desk[0])/2, 0, study_desk[0], study_desk[1], study_desk[2])
            study_chair_location = ((W-study_chair[0])/2, study_desk[1] + 0.1, study_chair[0], study_chair[1], study_chair[2])
            bookshelf_location = (W - bookshelf[0]-0.3, 0, bookshelf[0], bookshelf[1], bookshelf[2])
            couch_location = (W - couch_dim[0], L - couch_dim[0], couch[0], couch[1], couch[2])

            region_list = regions + [(bed_location, ["bed"]), (bedside_table_1_location, ["bedside_table"]), (bedside_table_2_location, ["bedside_table"]), (wardrobe_location, ["wardrobe"]), (study_desk_location, ["study_desk"]), (study_chair_location, ["study_chair"]), (bookshelf_location, ["bookshelf"]), (couch_location, ["couch"])]

        if test_idx == 2:

            furniture_names = ["bed", "bedside_table_1", "bedside_table_2", "wardrobe", "study_desk", "study_chair", "painting_s_1", "painting_s_2"]

            door_window_loc, regions = init_bedroom_layout(room_dim, window_loc="front-wall", door_loc=("right-wall", "right"))

            bed_dim = get_furniture_dim("bed", 0)
            bedside_table_1_dim = get_furniture_dim("bedside_table", 0)
            bedside_table_2_dim = get_furniture_dim("bedside_table", 0)
            wardrobe_dim = get_furniture_dim("wardrobe", -np.pi/2)
            study_desk_dim = get_furniture_dim("study_desk", np.pi)
            study_chair_dim = get_furniture_dim("study_chair", np.pi)
            painting_s_1_dim = get_furniture_dim("painting_s", np.pi/2)
            painting_s_2_dim = get_furniture_dim("painting_s", np.pi/2)

            bed_location = ((W-bed_dim[0])/2, L-bed_dim[1], bed_dim[0], bed_dim[1], bed_dim[2])
            bedside_table_1_location = (bed_location[0] - bedside_table_1_dim[0], L - bedside_table_1_dim[1], bedside_table_1_dim[0], bedside_table_1_dim[1], bedside_table_1_dim[2])
            bedside_table_2_location = (bed_location[0] + bed_dim[0], L - bedside_table_2_dim[1], bedside_table_2_dim[0], bedside_table_2_dim[1], bedside_table_2_dim[2])
            wardrobe_location = (W-wardrobe[1], L-wardrobe[0], wardrobe[0], wardrobe[1], wardrobe[2])
            study_desk_location = ((W-study_desk[0])/2, 0, study_desk[0], study_desk[1], study_desk[2])
            study_chair_location = ((W-study_chair[0])/2, study_desk[1] + 0.1, study_chair[0], study_chair[1], study_chair[2])
            painting_s_1_location = (0, L/2-painting_s_1_dim[0]-0.3, painting_s_1_dim[0], painting_s_1_dim[1], painting_s_1_dim[2])
            painting_s_2_location = (0, L/2+0.3, painting_s_2_dim[0], painting_s_2_dim[1], painting_s_2_dim[2])

            region_list = regions + [(bed_location, ["bed"]), (bedside_table_1_location, ["bedside_table"]), (bedside_table_2_location, ["bedside_table"]), (wardrobe_location, ["wardrobe"]), (study_desk_location, ["study_desk"]), (study_chair_location, ["study_chair"]), (painting_s_1_location, ["painting_s"]), (painting_s_2_location, ["painting_s"])]

            return region_list

        if test_idx == 3:

            furniture_names = ["bed", "bedside_table_1", "bedside_table_2", "wardrobe", "dressing_table", "dressing_chair", "couch", "mirror"]
            door_window_loc, regions = init_bedroom_layout(room_dim, window_loc="back-wall", door_loc=("front-wall", "left"))

            bed_dim = get_furniture_dim("bed", np.pi/2)
            bedside_table_1_dim = get_furniture_dim("bedside_table", np.pi/2)
            bedside_table_2_dim = get_furniture_dim("bedside_table", np.pi/2)
            wardrobe_dim = get_furniture_dim("wardrobe", np.pi)
            dressing_table_dim = get_furniture_dim("dressing_table", 0)
            dressing_chair_dim = get_furniture_dim("dressing_chair", 0)
            couch_dim = get_furniture_dim("couch", 0)
            mirror_dim = get_furniture_dim("mirror", 0)

            bed_location = (0, (L-bed_dim[0])/2, bed_dim[0], bed_dim[1], bed_dim[2])
            bedside_table_1_location = (0, bed_location[1] - bedside_table_1_dim[0], bedside_table_1_dim[0], bedside_table_1_dim[1], bedside_table_1_dim[2])
            bedside_table_2_location = (0, bed_location[1] + bed_dim[0], bedside_table_2_dim[0], bedside_table_2_dim[1], bedside_table_2_dim[2])
            wardrobe_location = (0, 0, wardrobe_dim[0], wardrobe_dim[1], wardrobe_dim[2])
            dressing_table_location = ((W - dressing_table_dim[0])/2, L - dressing_table_dim[1], dressing_table_dim[0], dressing_table_dim[1], dressing_table_dim[2])
            dressing_chair_location = ((W - dressing_chair_dim[0])/2, dressing_table_location[1] - dressing_chair_dim[1] - 0.1, dressing_chair_dim[0], dressing_chair_dim[1], dressing_chair_dim[2])
            couch_location = (W - couch_dim[0], L - couch_dim[1], couch_dim[0], couch_dim[1], couch_dim[2])
            mirror_location = (0, L - mirror_dim[1], mirror_dim[0], mirror_dim[1], mirror_dim[2])

            region_list = regions + [(bed_location, ["bed"]), (bedside_table_1_location, ["bedside_table"]), (bedside_table_2_location, ["bedside_table"]), (wardrobe_location, ["wardrobe"]), (dressing_table_location, ["dressing_table"]), (dressing_chair_location, ["dressing_chair"]), (couch_location, ["couch"]), (mirror_location, ["mirror"])]

            return region_list
        
        if test_idx == 4:

            furniture_names = ["bed", "wardrobe", "study_desk", "study_chair", "bookshelf", "storage_unit_1", "storage_unit_2", "carpet"]

            door_window_loc, regions = init_bedroom_layout(room_dim, window_loc="right-wall", door_loc=("left-wall", "left"))

            bed_dim = get_furniture_dim("bed", -np.pi/2)
            wardrobe_dim = get_furniture_dim("wardrobe", np.pi/2)
            study_desk_dim = get_furniture_dim("study_desk", np.pi/2)
            study_chair_dim = get_furniture_dim("study_chair", np.pi/2)
            bookshelf_dim = get_furniture_dim("bookshelf", np.pi/2)
            storage_unit_1_dim = get_furniture_dim("storage_cabinet", np.pi/2)
            storage_unit_2_dim = get_furniture_dim("storage_cabinet", np.pi/2)
            carpet_dim = get_furniture_dim("carpet", 0)

            bed_location = (W - bed_dim[1], L - bed_dim[0], bed_dim[0], bed_dim[1], bed_dim[2])
            wardrobe_location = (0, L - wardrobe_dim[0], wardrobe_dim[0], wardrobe_dim[1], wardrobe_dim[2])
            study_desk_location = (W - study_desk[1], (L - study_desk[0])/2, study_desk_dim[0], study_desk_dim[1], study_desk_dim[2])
            study_chair_location = (study_desk_location[0] - study_chair_dim[0] - 0.1, (L - study_chair_dim[0])/2, study_chair_dim[0], study_chair_dim[1], study_chair_dim[2])
            bookshelf_location = (W - bookshelf_dim[1], study_desk_location[1] - bookshelf_dim[0]-0.2, bookshelf_dim[0], bookshelf_dim[1], bookshelf_dim[2])
            storage_unit_1_location = (0, wardrobe_location[1] - storage_unit_1_dim[1], storage_unit_1_dim[0], storage_unit_1_dim[1], storage_unit_1_dim[2])
            storage_unit_2_location = (0, storage_unit_1_location[1] - storage_unit_2_dim[1] , storage_unit_2_dim[0], storage_unit_2_dim[1], storage_unit_2_dim[2])
            carpet_location = ((W - carpet_dim[0])/2, (L - carpet_dim[1])/2, carpet_dim[0], carpet_dim[1], carpet_dim[2])

            region_list = regions + [(bed_location, ["bed"]), (wardrobe_location, ["wardrobe"]), (study_desk_location, ["study_desk"]), (study_chair_location, ["study_chair"]), (bookshelf_location, ["bookshelf"]), (storage_unit_1_location, ["storage_unit"]), (storage_unit_2_location, ["storage_unit"]), (carpet_location, ["carpet"])]

            return region_list
        
        if test_idx == 5:

            furniture_names = ["bed", "wardrobe", "study_desk", "study_chair", "bookshelf", "storage_unit_1", "storage_unit_2", "carpet"]

            door_window_loc, regions = init_bedroom_layout(room_dim, window_loc="right-wall", door_loc=("left-wall", "left"))

            bed_dim = get_furniture_dim("bed", -np.pi/2)
            wardrobe_dim = get_furniture_dim("wardrobe", np.pi/2)
            study_desk_dim = get_furniture_dim("study_desk", np.pi/2)
            study_chair_dim = get_furniture_dim("study_chair", np.pi/2)
            bookshelf_dim = get_furniture_dim("bookshelf", np.pi/2)
            storage_unit_1_dim = get_furniture_dim("storage_cabinet", np.pi/2)
            storage_unit_2_dim = get_furniture_dim("storage_cabinet", np.pi/2)
            carpet_dim = get_furniture_dim("carpet", 0)

            bed_location = (W - bed_dim[1], L - bed_dim[0], bed_dim[0], bed_dim[1], bed_dim[2])
            wardrobe_location = (0, L - wardrobe_dim[0], wardrobe_dim[0], wardrobe_dim[1], wardrobe_dim[2])
            study_desk_location = (W - study_desk[1], (L - study_desk[0])/2, study_desk_dim[0], study_desk_dim[1], study_desk_dim[2])
            study_chair_location = (study_desk_location[0] - study_chair_dim[0] - 0.1, (L - study_chair_dim[0])/2, study_chair_dim[0], study_chair_dim[1], study_chair_dim[2])
            bookshelf_location = (W - bookshelf_dim[1], study_desk_location[1] - bookshelf_dim[0]-0.2, bookshelf_dim[0], bookshelf_dim[1], bookshelf_dim[2])
            storage_unit_1_location = (0, wardrobe_location[1] - storage_unit_1_dim[1], storage_unit_1_dim[0], storage_unit_1_dim[1], storage_unit_1_dim[2])
            storage_unit_2_location = (0, storage_unit_1_location[1] - storage_unit_2_dim[1] , storage_unit_2_dim[0], storage_unit_2_dim[1], storage_unit_2_dim[2])
            carpet_location = ((W - carpet_dim[0])/2, (L - carpet_dim[1])/2, carpet_dim[0], carpet_dim[1], carpet_dim[2])

            region_list = regions + [(bed_location, ["bed"]), (wardrobe_location, ["wardrobe"]), (study_desk_location, ["study_desk"]), (study_chair_location, ["study_chair"]), (bookshelf_location, ["bookshelf"]), (storage_unit_1_location, ["storage_unit"]), (storage_unit_2_location, ["storage_unit"]), (carpet_location, ["carpet"])]
        
        return region_list

    def get_furniture_locations_against_wall(furniture_num, wall_index=None, rel_side=None, room_dim=(3, 4)):
        W, L = room_dim
        regions = []
        # Step 1: determine the wall_index if not provided
        if wall_index is None:
            wall_index = rn.choice(["back-wall", "right-wall", "front-wall", "left-wall"])
        
        # Step 2: create the furniture dims that can fit into the same wall
        furniture_dims = get_furniture_dim(furniture_num=furniture_num, along_same_wall=True, wall_index=wall_index, room_dim=room_dim)
        furniture_num = len(furniture_dims)

        if rel_side:
            assert len(rel_side) == furniture_num and furniture_num <= 3
            # For each side ("left", "center", "right"), collect the furniture assigned to that side
            side_furniture_map = {"center": [], "left": [], "right": []}
            for i, side in enumerate(rel_side):
                side_furniture_map[side].append(i)
            
            # Start by placing the 'center' furniture
            placed_regions = {}
            if side_furniture_map["center"]:
                idx = side_furniture_map["center"][0]  # Assuming only one center furniture
                width, length, _ = furniture_dims[idx]
                if wall_index in ["back-wall", "front-wall"]:
                    x = (W - width) / 2 + np.random.uniform(-0.1, 0.1)
                    y = L - length if wall_index == "back-wall" else 0
                    orientation = 0 if wall_index == "back-wall" else math.pi
                else:
                    y = (L - width) / 2 + np.random.uniform(-0.1, 0.1)
                    x = 0 if wall_index == "left-wall" else W - length
                    orientation = math.pi / 2 if wall_index == "left-wall" else -math.pi / 2
                region = (x, y, width, length, orientation)
                regions.append((region, [f"against-{wall_index}", "center_of_wall"]))
                placed_regions[idx] = region
                center_furniture_dims = (x, y, width, length)
            else:
                center_furniture_dims = None
            
            def place_side_furniture(side):
                indices = side_furniture_map[side]
                if not indices:
                    return
                if wall_index in ["back-wall", "front-wall"]:
                    wall_length = W
                    if side == "left":
                        x_start = 0
                        if center_furniture_dims:
                            x_end = center_furniture_dims[0] - 0.01
                        else:
                            x_end = (W / 2) - 0.01
                    else:  # side == "right"
                        if center_furniture_dims:
                            x_start = center_furniture_dims[0] + center_furniture_dims[2] + 0.01
                        else:
                            x_start = (W / 2) + 0.01  # Slight buffer
                        x_end = W
                    orientation = 0 if wall_index == "back-wall" else math.pi
                    y = L - max([furniture_dims[i][1] for i in indices]) if wall_index == "back-wall" else 0
                else:
                    wall_length = L
                    if side == "left":
                        y_start = 0
                        if center_furniture_dims:
                            y_end = center_furniture_dims[1] - 0.01
                        else:
                            y_end = (L / 2) - 0.01  # Slight buffer
                    else:  # side == "right"
                        if center_furniture_dims:
                            y_start = center_furniture_dims[1] + center_furniture_dims[2] + 0.01
                        else:
                            y_start = (L / 2) + 0.01  # Slight buffer
                        y_end = L
                    orientation = math.pi / 2 if wall_index == "left-wall" else -math.pi / 2
                    x = 0 if wall_index == "left-wall" else W - max([furniture_dims[i][1] for i in indices])
                
                available_length = (x_end - x_start) if wall_index in ["back-wall", "front-wall"] else (y_end - y_start)
                widths = [furniture_dims[i][0] for i in indices]
                total_width = sum(widths)
                
                overlap = False
                if total_width > available_length:
                    # Adjust sizes proportionally only if necessary
                    scale_factor = available_length / total_width
                    widths = [w * scale_factor for w in widths]
                    for idx, w in zip(indices, widths):
                        furniture_dims[idx] = (w, furniture_dims[idx][1], furniture_dims[idx][2])
                    overlap = True  # Sizes were adjusted
                else:
                    # No size adjustment needed
                    overlap = False
                
                # Place the furniture
                num_furniture = len(indices)
                spacing = (available_length - sum(widths)) / (num_furniture + 1)
                current_pos = x_start + spacing if wall_index in ["back-wall", "front-wall"] else y_start + spacing
                
                for idx, w in zip(indices, widths):
                    width, length, _ = furniture_dims[idx]
                    if wall_index in ["back-wall", "front-wall"]:
                        x = current_pos
                        y = L - length if wall_index == "back-wall" else 0
                        current_pos += width + spacing
                    else:
                        y = current_pos
                        x = 0 if wall_index == "left-wall" else W - length
                        current_pos += width + spacing
                    orientation = 0 if wall_index == "back-wall" else (math.pi if wall_index == "front-wall" else
                                    (math.pi / 2 if wall_index == "left-wall" else -math.pi / 2))
                    region = (x, y, width, length, orientation)
                    regions.append((region, [f"against-{wall_index}", f"{side}_of_wall"]))
                    placed_regions[idx] = region
            
            # Place left and right furniture
            for side in ["left", "right"]:
                place_side_furniture(side)
            
            # Check for overlaps and adjust sizes if necessary
            # We can implement an optional check here, but since we've already adjusted sizes if needed,
            # and included slight buffers, overlaps should be minimal or non-existent.
           
        else:
            # When rel_side is not provided, evenly space the furniture along the wall
            wall_length = W if wall_index in ["back-wall", "front-wall"] else L
            total_width = sum([dim[0] for dim in furniture_dims])
            remaining_space = wall_length - total_width
            if remaining_space < 0:
                raise ValueError("Furniture too big for wall")
            spacing = remaining_space / (furniture_num + 1)
            current_pos = spacing
            for i, dim in enumerate(furniture_dims): # for each furniture
                width, length, _ = dim
                if wall_index in ["back-wall", "front-wall"]:
                    y = L - length if wall_index == "back-wall" else 0
                    x = current_pos
                    orientation = 0 if wall_index == "back-wall" else math.pi
                    current_pos += width + spacing
                    # Determine rel_side based on position
                    if x + width / 2 <= W / 3:
                        side = "left"
                    elif x + width / 2 >= 2 * W / 3:
                        side = "right"
                    else:
                        side = "center"
                else:
                    x = 0 if wall_index == "left-wall" else W - length
                    y = current_pos
                    orientation = math.pi / 2 if wall_index == "left-wall" else -math.pi / 2
                    current_pos += width + spacing
                    if y + width / 2 <= L / 3:
                        side = "left"
                    elif y + width / 2 >= 2 * L / 3:
                        side = "right"
                    else:
                        side = "center"
                region = (x, y, width, length, orientation)
                regions.append((region, [f"against-{wall_index}", f"{side}_of_wall"]))
        return regions
 
    def get_furniture_locations_side_touching(wall_index, room_dim=(3, 4)):
        W, L = room_dim

        # Step 1: Create 2 furniture dims that can fit onto the same wall_index
        furniture_dims = get_furniture_dim(furniture_num=2, along_same_wall=True, wall_index=wall_index, room_dim=room_dim)
        width1, length1, _ = furniture_dims[0]
        width2, length2, _ = furniture_dims[1]
        total_width = width1 + width2
        wall_length = W if wall_index in ["back-wall", "front-wall"] else L

        # If total width exceeds wall_length, scale down the furniture sizes
        if total_width > wall_length * 0.9:
            scale_factor = (wall_length * 0.9) / total_width
            width1 *= scale_factor
            width2 *= scale_factor
            total_width = width1 + width2  # Update total_width after scaling

        # Step 2: Randomly sample starting position along the wall
        max_start = wall_length - total_width
        if max_start < 0:
            raise ValueError("Furniture too big for the wall")
        start_pos = rn.uniform(0, max_start)

        regions = []
        if wall_index == "back-wall":
            # For back-wall, y is fixed at y = L - length (furniture extends into the room)
            y1 = L - length1
            y2 = L - length2  # Both have same y (furniture backs against the wall)
            orientation = 0  # Facing positive y
            # x varies along the wall
            x1 = start_pos
            x2 = x1 + width1  # Right side of furniture1 touching left side of furniture2

            region1 = (x1, y1, width1, length1, orientation)
            region2 = (x2, y2, width2, length2, orientation)
        elif wall_index == "front-wall":
            # For front-wall, y is fixed at y=0
            y1 = 0
            y2 = 0
            orientation = math.pi  # Facing negative y
            # x varies along the wall
            x1 = start_pos
            x2 = x1 + width1

            region1 = (x1, y1, width1, length1, orientation)
            region2 = (x2, y2, width2, length2, orientation)
        elif wall_index == "left-wall":
            # For left-wall, x is fixed at x=0
            x1 = 0
            x2 = 0
            orientation = math.pi / 2  # Facing positive x
            # y varies along the wall
            y1 = start_pos
            y2 = y1 + width1  # Top of furniture1 touching bottom of furniture2

            region1 = (x1, y1, width1, length1, orientation)
            region2 = (x2, y2, width2, length2, orientation)
        elif wall_index == "right-wall":
            # For right-wall, x is fixed at x = W - length (since furniture extends into room)
            x1 = W - length1
            x2 = W - length2
            orientation = -math.pi / 2  # Facing negative x
            # y varies along the wall
            y1 = start_pos
            y2 = y1 + width1

            region1 = (x1, y1, width1, length1, orientation)
            region2 = (x2, y2, width2, length2, orientation)
        else:
            raise ValueError("Invalid wall_index")

        regions = [(region1, [f"against-{wall_index}"]), (region2, [f"against-{wall_index}"])]
        return regions

    def get_furniture_locations_on_left_side(wall_index, room_dim=(3, 4)):
        W, L = room_dim
        regions = []
        
        # Step 1: Create 2 furniture dims that can fit onto the same wall_index
        furniture_dims = get_furniture_dim(furniture_num=2, along_same_wall=True,
                                        wall_index=wall_index, room_dim=room_dim)
        width1, length1, _ = furniture_dims[0]
        width2, length2, _ = furniture_dims[1]
        wall_length = W if wall_index in ["back-wall", "front-wall"] else L
        total_furniture_width = width1 + width2

        # Scale down furniture widths if they are too big
        if total_furniture_width >= wall_length * 0.7:
            scale_factor = (wall_length * 0.7) / total_furniture_width
            width1 *= scale_factor
            width2 *= scale_factor
            total_furniture_width = width1 + width2

        # Set maximum spacing threshold min(20% of wall_length, 35% of the max furniture width)
        if wall_index in ["back-wall", "front-wall"]:
            max_spacing = min(wall_length * 0.2, max(width1, width2) * 0.35)
        else:
            max_spacing = min(wall_length * 0.2, max(length1, length2) * 0.35)

        # Determine maximum possible spacing given furniture sizes and wall_length
        max_possible_spacing = wall_length - total_furniture_width
        max_possible_spacing = min(max_possible_spacing, max_spacing)

        # Minimum spacing (can be zero or a small positive number)
        min_spacing = 0.1

        # Check if there is space to place both furnitures with at least min_spacing
        if total_furniture_width + min_spacing > wall_length:
            raise ValueError("Furniture too big for the wall")

        # Randomly sample spacing between min_spacing and max_possible_spacing
        spacing = rn.uniform(min_spacing, max_possible_spacing)

        # Calculate the maximum starting position for the first furniture
        max_start_pos = wall_length - (total_furniture_width + spacing)
        max_start_pos = max(0, max_start_pos)  # Ensure non-negative

        # Randomly sample the starting position of the first furniture
        start_pos = rn.uniform(0, max_start_pos)

        # Calculate positions of furnitures
        if wall_index in ["back-wall", "front-wall"]:
            ys = [L - length1, L-length2] if wall_index == "back-wall" else [0, 0]
            orientation = 0 if wall_index == "back-wall" else math.pi
            x1 = start_pos
            x2 = x1 + width1 + spacing
            region1 = (x1, ys[0], width1, length1, orientation)
            region2 = (x2, ys[1], width2, length2, orientation)
        elif wall_index == "left-wall":
            x = 0
            orientation = math.pi / 2
            y1 = start_pos
            y2 = y1 + width1 + spacing
            region1 = (x, y1, width1, length1, orientation)
            region2 = (x, y2, width2, length2, orientation)
        elif wall_index == "right-wall":
            x1 = W - length1
            x2 = W - length2
            orientation = - math.pi / 2
            y1 = start_pos
            y2 = y1 + width1 + spacing
            region1 = (x1, y1, width1, length1, orientation)
            region2 = (x2, y2, width2, length2, orientation)
        else:
            raise ValueError("Invalid wall_index")

        regions = [(region1, [f"against-{wall_index}"]), (region2, [f"against-{wall_index}"])]
        return regions

    def get_furniture_locations_in_front_of(wall_index, room_dim=(3, 4)):
        W, L = room_dim
        # Step 1: Create 2 furniture dims
        furniture_dims = get_furniture_dim(furniture_num=2, along_same_wall=False, room_dim=room_dim)
        width1, length1, _ = furniture_dims[0]  # Furniture 1, against wall
        width2, length2, _ = furniture_dims[1]  # Furniture 2, in front of furniture 1

        # Step 2: Calculate the region tuple for the furniture
        if wall_index in ["back-wall", "front-wall"]:
            # We'll align the centroids along the x-axis
            # Calculate the maximum and minimum x positions to ensure both furnitures fit within the room
            max_furniture_width = max(width1, width2)
            x_min = max_furniture_width / 2
            x_max = W - max_furniture_width / 2

            # Randomly sample the centroid x position within the allowable range
            centroid_x = rn.uniform(x_min, x_max)

            # Calculate x positions for each furniture so that their centroids align at centroid_x
            x1 = centroid_x - width1 / 2
            x2 = centroid_x - width2 / 2

            # Ensure furnitures are within room boundaries
            if x1 < 0 or x1 + width1 > W or x2 < 0 or x2 + width2 > W:
                raise ValueError("Furniture does not fit within room boundaries")

            # Place furniture 1 against the wall
            orientation1 = 0 if wall_index == "back-wall" else math.pi
            y1 = L - length1 if wall_index == "back-wall" else 0

            spacing = min(rn.uniform(0.1, 0.3) * length1, 0.2)
            # Place furniture 2 in front of furniture 1
            orientation2 = orientation1
            if wall_index == "back-wall":
                y2 = y1 - length2 - spacing
                if y2 < 0:
                    y2 = 0  # Adjust if out of bounds
            else:
                y2 = y1 + length1 + spacing
                if y2 + length2 > L:
                    y2 = L - length2  # Adjust if out of bounds

            region1 = (x1, y1, width1, length1, orientation1)
            region2 = (x2, y2, width2, length2, orientation2)
        else:  # "left-wall" or "right-wall"
            # Align the centroids along the y-axis
            max_furniture_width = max(width1, width2)
            y_min = max_furniture_width / 2
            y_max = L - max_furniture_width / 2

            # Randomly sample the centroid y position within the allowable range
            centroid_y = rn.uniform(y_min, y_max)

            # Calculate y positions for each furniture so that their centroids align at centroid_y
            y1 = centroid_y - width1 / 2
            y2 = centroid_y - width2 / 2

            # Ensure furnitures are within room boundaries
            if y1 < 0 or y1 + width1 > L or y2 < 0 or y2 + width2 > L:
                raise ValueError("Furniture does not fit within room boundaries")

            # Place furniture 1 against the wall
            orientation1 = math.pi / 2 if wall_index == "left-wall" else - math.pi / 2
            x1 = 0 if wall_index == "left-wall" else W - length1

            spacing = rn.uniform(0.1, 0.3) * length1
            # Place furniture 2 in front of furniture 1
            orientation2 = orientation1
            if wall_index == "left-wall":
                x2 = x1 + length1 + spacing
                if x2 + length2 > W:
                    x2 = W - length2  # Adjust if out of bounds
            else:
                x2 = x1 - length2 - spacing
                if x2 < 0:
                    x2 = 0  # Adjust if out of bounds

            region1 = (x1, y1, width1, length1, orientation1)
            region2 = (x2, y2, width2, length2, orientation2)

        regions = [(region1, [f"against-{wall_index}"]), (region2, [f"against-{wall_index}"])]
        return regions

    def get_furniture_locations_under_window(window_location, room_dim=(3, 4)):
        W, L = room_dim
        window_region, window_wall_index = window_location
        # Step 1: create furniture dim
        furniture_dim = get_furniture_dim(furniture_num=1, along_same_wall=False, room_dim=room_dim)[0]
        width, length, _ = furniture_dim

        # Step 2: calculate the region tuple for the furniture
        if window_wall_index in ["back-wall", "front-wall"]:
            x_window, y_window, w_window, l_window, _ = window_region
            x_furniture = x_window + (w_window - width)/2
            y_furniture = y_window - length if window_wall_index == "back-wall" else y_window + l_window
            orientation = 0 if window_wall_index == "back-wall" else math.pi
            region = (x_furniture, y_furniture, width, length, orientation)
        elif window_wall_index in ["left-wall", "right-wall"]:
            x_window, y_window, w_window, l_window, _ = window_region
            x_furniture = 0 if window_wall_index == "left-wall" else W - length
            y_furniture = y_window + (w_window - width)/2
            orientation = math.pi / 2 if window_wall_index == "left-wall" else -math.pi /2
            region = (x_furniture, y_furniture, width, length, orientation)
        else:
            raise ValueError("Invalid wall index")
        regions = [(region, ["under-window", f"against-{window_wall_index}"])]
        return regions

    def get_furniture_locations_at_center(room_dim=(3, 4)):
        W, L = room_dim
        # Step 1: create furniture dim
        furniture_dim = get_furniture_dim(furniture_num=1, room_dim=room_dim)[0]
        width, length, _ = furniture_dim
        # Random orientation
        orientation = rn.choice([-math.pi/2, 0, math.pi/2, math.pi])
        # Step 2: calculate the region tuple for the furniture
        if orientation in [0, math.pi]:
            x_center = (W - width) /2
            y_center = (L - length)/2
        else:
            x_center = (W - length) /2
            y_center = (L - width) / 2
        region = (x_center, y_center, width, length, orientation)
        regions = [(region, "at-center")]
        return regions

    def get_furniture_locations_at_corners(furniture_num=None, corners=None, facing_wall_indices=None, room_dim=(3, 4)):
        W, L = room_dim
        # Step 1: create furniture dims
        furniture_dims = get_furniture_dim(furniture_num=furniture_num, room_dim=room_dim)
        regions = []
        # Step 2: determine the corners and the facing_wall_indices if not provided
        corner_list = ["front-left-corner", "front-right-corner", "back-left-corner", "back-right-corner"]
        if corners is None:
            corners = rn.sample(corner_list, furniture_num)
        if facing_wall_indices is None:
            facing_wall_indices = []
            for corner in corners:
                if corner == "front-left-corner":
                    facing_wall_indices.append(rn.choice(["front-wall", "left-wall"]))
                elif corner == "front-right-corner":
                    facing_wall_indices.append(rn.choice(["front-wall", "right-wall"]))
                elif corner == "back-left-corner":
                    facing_wall_indices.append(rn.choice(["back-wall", "left-wall"]))
                elif corner == "back-right-corner":
                    facing_wall_indices.append(rn.choice(["back-wall", "right-wall"]))

        # Step 3: calculate the region tuple for the furniture
        for i in range(furniture_num):
            corner = corners[i]
            facing_wall = facing_wall_indices[i]
            width, length, orientation = furniture_dims[i]
            # Orientation based on facing_wall
            if facing_wall == "back-wall":
                orientation = 0
                w, l = width, length
            elif facing_wall == "front-wall":
                orientation = math.pi
                w, l = width, length
            elif facing_wall == "left-wall":
                orientation = math.pi / 2
                w, l = length, width
            elif facing_wall == "right-wall":
                orientation = - math.pi / 2
                w, l = length, width
            else:
                orientation = 0  # default
                w, l = width, length

            if corner == "front-left-corner":
                x = 0
                y = 0
            elif corner == "front-right-corner":
                x = W - w
                y = 0
            elif corner == "back-left-corner":
                x = 0
                y = L - l
            elif corner == "back-right-corner":
                x = W - w
                y = L - l
            else:
                raise ValueError("Invalid corner")
            
            region = (x, y, width, length, orientation)
            regions.append((region, [f"at-{corner}", f"against-{facing_wall}"]))
        return regions
        
    def gen(relation, window_loc=None, door_loc=None, furniture_num=None, bedroom_dim=(3, 4)):

        # should return regions and relations to each region
        count = num_samples

        while True:
            window_door_loc, regions = init_bedroom_layout(bedroom_dim, window_loc, door_loc)

            if relation == "all-composed":
                relation_options = ["against-wall", "side-touching", "on-left-side", "in-front-of", "under-window", "at-center", "at-corners"]
                relation_probability = [0.2, 0.2, 0.2, 0.2, 0.06, 0.07, 0.07]
                relation = rn.choices(relation_options, relation_probability, k=1)[0]
                
            if relation == "against-wall":
                if furniture_num is None:
                    furniture_num = rn.choice([1, 2, 3, 4, 5])
                    
                if furniture_num <=3 and rn.uniform(0, 1) < 0.5:
                    rel_side = rn.sample(["left", "right", "center"], furniture_num)
                else:
                    rel_side = None
                regions += get_furniture_locations_against_wall(furniture_num, wall_index=None, rel_side=rel_side, room_dim=bedroom_dim)
           
            elif relation == "side-touching":
                if furniture_num is None:
                    num_pairs = rn.choice([1, 2])
                else:
                    num_pairs = furniture_num//2
                wall_pairs = rn.sample([["back-wall", "front-wall"], ["right-wall", "left-wall"]], 1)[0]
                wall_indices = rn.sample(wall_pairs, num_pairs)
                
                for wall_index in wall_indices:
                    regions += get_furniture_locations_side_touching(wall_index, bedroom_dim)

            elif relation == "on-left-side":
                if furniture_num is None:
                    num_pairs = rn.choice([1, 2])
                else:
                    num_pairs = furniture_num//2
                wall_pairs = rn.sample([["back-wall", "front-wall"], ["right-wall", "left-wall"]], 1)[0]
                wall_indices = rn.sample(wall_pairs, num_pairs)

                for wall_index in wall_indices:
                    regions += get_furniture_locations_on_left_side(wall_index, bedroom_dim)
                    
            elif relation == "in-front-of":
                wall_index = rn.choice(["back-wall", "front-wall", "left-wall", "right-wall"])
                regions += get_furniture_locations_in_front_of(wall_index, bedroom_dim)

            elif relation == "under-window":
                regions += get_furniture_locations_under_window(window_door_loc["window"], bedroom_dim)
            elif relation == "at-center":
                regions += get_furniture_locations_at_center(bedroom_dim)
            elif relation == "at-corners":
                if furniture_num is None:
                    furniture_num = rn.choice([1, 2, 3, 4])
                regions += get_furniture_locations_at_corners(furniture_num, room_dim=bedroom_dim)

            elif relation == "test-case":
                print("which index")
                x = input()
                regions = get_test_case_furniture_locations(int(x), bedroom_dim)
                
            count -= 1
            print(len(regions), "added!")
            yield relation, regions
            if count == 0:
                break
        yield None
    
    return gen   

def get_shelf_data_gen(num_samples, shelf_dim=None, object_num=None, relation="wall-contact"):
    # A shelf is a rectangle with width W and height H that represent the x-z plane of a single shelf, viewed facing the front of the bookshelf.
    # W, H = shelf_dim.
    # It has two walls: left-wall and right-wall, viewed facing the front of the bookshelf 
    # The bottom of the left-wall is with coordinates (0, 0), the top of the right-wall is with coordinates (W, H)
    # We assume the shelf is deep enough to hold the objects, L > l for all objects
    # Furniture_region is a tuple of 5 floats: (x, z, w, h, rotation)
    # (x, z) is the coordinate of the lower left corner of the furniture, w is the width, h is the height, rotation is the angle of the object on the x-z plane, 0 means the object is standing upright, Pi/2 means the object is sitting on its left surface.
   
    Object_region = Tuple[float, float, float, float, float]  

    def get_object_dim(object_num, shelf_dim):
        # Generate the object dimensions (w, h, rotation) for object_num objects
        W, H = shelf_dim
        object_dims = []

        # Step 1: Define appropriate range for the object dimensions
        min_width = W * 0.05  # 5% of shelf width
        max_width = W * 0.25  # 25% of shelf width
        min_height = H * 0.3  # 30% of shelf height
        max_height = H * 0.9   # 90% of shelf height

        # Step 2: Generate the object dimensions, ensuring they fit within the shelf
        total_width = 0
        for i in range(object_num):
            w = rn.uniform(min_width, max_width)
            h = rn.uniform(min_height, max_height)
            rotation = 0  # Assuming upright objects for simplicity
            object_dims.append((w, h, rotation))
            total_width += w

        # Check if total width exceeds shelf width
        max_W = W * 0.9  # 90% of shelf width
        if total_width > max_W:
            # Calculate the scale factor to adjust widths
            scale_factor = max_W / total_width
            # Adjust the widths of all objects
            adjusted_object_dims = []
            for w, h, rotation in object_dims:
                w_adjusted = w * scale_factor
                adjusted_object_dims.append((w_adjusted, h, rotation))
            object_dims = adjusted_object_dims

        return object_dims

    def divide_total_spacing(x, N):
        # Generate N - 1 random cut points between 0 and x
        cut_points = sorted([random.uniform(0, x) for _ in range(N-1)])
        # Add the start and end points
        points = [0.0] + cut_points + [x]
        # Compute the sizes of each compartment
        compartments = [points[i+1] - points[i] for i in range(N)]
        return compartments

    def get_object_locations_wall_contact(object_num, shelf_dim):
        # Generate object locations that satisfy the wall-contact relation
        W, H = shelf_dim

        assert object_num == 2

        # Step 1: Generate the object relations
        object_relations = ["left-wall-contact", "right-wall-contact"]
        
        # Step 2: Get object dimensions
        object_dims = get_object_dim(object_num, shelf_dim)

        # Step 3: Get object locations based on the object relations
        regions = []
        for i in range(object_num):
            w, h, rotation = object_dims[i]
            if object_relations[i] == "left-wall-contact":
                x = 0  # Left wall
            elif object_relations[i] == "right-wall-contact":
                x = W - w  # Right wall
            z = 0

            regions.append(((x, z, w, h, rotation), [object_relations[i]]))
        return "wall-contact", regions

    def get_object_locations_wall_side(object_num, shelf_dim):
        # Generate object locations that satisfy the wall-side relation
        W, H = shelf_dim

        # Step 1: Generate the object relations
        relations = ["left-side", "right-side", "at-center"]
        object_relations = []
        
        # Ensure 'at-center' is included at most once
        has_center = False
        for _ in range(object_num):
            if not has_center and rn.random() < 0.3:
                object_relations.append("at-center")
                has_center = True
            else:
                object_relations.append(rn.choice(["left-side", "right-side"]))
        # Ensure we have the correct number of relations
        object_relations = object_relations[:object_num]

        # Step 2: Get object dimensions
        object_dims = get_object_dim(object_num, shelf_dim)
        
        # Step 3: Get object locations based on the object relations
        regions = []
        occupied_areas = []

        # Step 3.1: Place the 'at-center' object, if any
        if "at-center" in object_relations:
            center_index = object_relations.index("at-center")
            w_center, h_center, rotation_center = object_dims[center_index]
            x_center = (W - w_center) / 2  # Center horizontally
            z_center = 0  # Bottom of the shelf
            regions.append(((x_center, z_center, w_center, h_center, rotation_center), ["at-center"]))
            occupied_areas.append((x_center, z_center, w_center, h_center))
            # Remove the center object from lists
            del object_relations[center_index]
            del object_dims[center_index]
            # Update left_most and right_most positions
            left_most = x_center
            right_most = x_center + w_center
        else:
            # If there's no center object, the center is at W/2
            left_most = W / 2
            right_most = W / 2

        # Separate remaining objects into left-side and right-side
        left_side_indices = [i for i, rel in enumerate(object_relations) if rel == "left-side"]
        right_side_indices = [i for i, rel in enumerate(object_relations) if rel == "right-side"]

        # Step 3.3: Place the left-side objects
        left_total_width = sum([object_dims[i][0] for i in left_side_indices])

        # Adjust widths if necessary
        if left_total_width > left_most * 0.8:
            scale_factor = left_most * 0.8/ left_total_width
            for i in left_side_indices:
                w, h, rotation = object_dims[i]
                w *= scale_factor
                object_dims[i] = (w, h, rotation)
            left_total_width = sum([object_dims[i][0] for i in left_side_indices])

        # Place the left-side objects without overlap
        left_spacing = left_most - left_total_width
        left_spacings_between_objects = divide_total_spacing(left_spacing, len(left_side_indices) + 1)
        start_x = left_spacings_between_objects[0]

        for i, idx in enumerate(left_side_indices):
            w, h, rotation = object_dims[idx]
            region = (start_x, 0, w, h, rotation)
            regions.append((region, ["left-side"]))
            start_x += w + left_spacings_between_objects[i + 1]

        # Step 3.4: Place the right-side objects
        right_total_width = sum([object_dims[i][0] for i in right_side_indices])

        # Adjust widths if necessary
        available_width_right = W - right_most
        if right_total_width > available_width_right * 0.8:
            scale_factor = available_width_right * 0.8 / right_total_width
            for i in right_side_indices:
                w, h, rotation = object_dims[i]
                w *= scale_factor
                object_dims[i] = (w, h, rotation)
            right_total_width = sum([object_dims[i][0] for i in right_side_indices])

        # Place the right-side objects without overlap

        right_spacing = available_width_right - right_total_width
        right_spacings_between_objects = divide_total_spacing(right_spacing, len(right_side_indices) + 1)
        start_x = right_most + right_spacings_between_objects[0]
       
        for i, idx in enumerate(right_side_indices):
            w, h, rotation = object_dims[idx]
            region = (start_x, 0, w, h, rotation)
            regions.append((region, ["right-side"]))
            start_x += w + right_spacings_between_objects[i + 1]

        return "wall-side", regions

    def get_object_locations_side_of(object_num, shelf_dim):
        # Generate object locations that satisfy the side-of relation
        W, H = shelf_dim
        assert object_num == 2  # This relation involves exactly two objects

        # Step 1: Get object dimensions
        object_dims = get_object_dim(object_num, shelf_dim)
        (w_a, h_a, rotation_a) = object_dims[0]
        (w_b, h_b, rotation_b) = object_dims[1]

        # Decide spacing between objects
        spacing = rn.uniform(max(0.2, min(w_a, w_b) * 0.2), min(min(w_a, w_b) * 0.5, 0.6))

        width_sum = w_a + w_b + spacing
        if width_sum > W * 0.8:
            scale_factor = W * 0.8 / width_sum
            w_a *= scale_factor
            w_b *= scale_factor
            spacing *= scale_factor
        
        x_a_max = W - w_a - w_b - spacing

        x_a, z_a = rn.uniform(0, x_a_max), 0
        x_b, z_b = x_a + w_a + spacing, 0

        regions = []
        regions.append(((x_a, z_a, w_a, h_a, rotation_a), []))
        regions.append(((x_b, z_b, w_b, h_b, rotation_b), []))
        return "side-of", regions

    def get_object_locations_aligned(object_num, shelf_dim):
        # Generate object locations that satisfy alignment relations
        W, H = shelf_dim

        # Step 1: Choose alignment relation
        aligned_relation = rn.choice(["linearly-aligned", "contiguously-aligned"])

        # Step 2: Get object dimensions
        object_dims = get_object_dim(object_num, shelf_dim)

        # Step 3: Calculate total width required including spacing
        if aligned_relation == "linearly-aligned":
            # Include equal spacing between objects
            spacing = rn.uniform(0.03 * W, 0.1 * W)
            total_spacing = spacing * (object_num - 1)
        else:
            # No spacing for contiguously-aligned
            spacing = 0.025
            total_spacing = 0.025 * (object_num - 1)
        total_width = sum([w for w, h, r in object_dims]) + total_spacing

        # Scale down if total width exceeds shelf width
        if total_width > W * 0.9:
            scale = W * 0.9 / total_width
            object_dims = [(w * scale, h, r) for w, h, r in object_dims]
            spacing *= scale
            total_width = sum([w for w, h, r in object_dims]) + spacing * (object_num - 1)

        # Step 4: Position objects
        start_x = (W - total_width) / 2
        x = start_x
        regions = []
        for w, h, rotation in object_dims:
            regions.append(((x, 0, w, h, rotation), []))
            x += w + spacing
        return aligned_relation, regions

    def get_object_locations_sorted(object_num, shelf_dim):
        # Generate object locations that satisfy sorted relations
        W, H = shelf_dim

        # Step 1: Choose sorted relation
        sorted_relation = rn.choice([
            "height-sorted-ascending", "height-sorted-descending",
            "width-sorted-ascending", "width-sorted-descending"
        ])

        # Step 2: Get object dimensions
        object_dims = get_object_dim(object_num, shelf_dim)

        # Step 3: Sort object dimensions based on the relation
        if "height" in sorted_relation:
            reverse = "descending" in sorted_relation
            object_dims.sort(key=lambda x: x[1], reverse=reverse)
        elif "width" in sorted_relation:
            reverse = "descending" in sorted_relation
            object_dims.sort(key=lambda x: x[0], reverse=reverse)

        # Step 4: Position objects linearly

        spacing = rn.uniform(0.03 * W, 0.1 * W)
        total_spacing = spacing * (object_num - 1)
        total_width = sum([w for w, h, r in object_dims]) + total_spacing

        if total_width > W * 0.9:
            # Scale down objects if they don't fit
            scale = W * 0.9 / total_width
            object_dims = [(w * scale, h, r) for w, h, r in object_dims]
            spacing *= scale
            total_width = sum([w for w, h, r in object_dims]) + spacing * (object_num - 1)

        start_x = (W - total_width) / 2
        x = start_x
        regions = []
        for w, h, rotation in object_dims:
            regions.append(((x, 0, w, h, rotation), []))
            x += w + spacing
        return sorted_relation, regions

    def get_test_case_object_locations(test_idx):

        shelf_dim = (4, 2)
        if test_idx == 1:

            num_books = 13

            book_widths = np.random.uniform(0.075, 0.15, 13)
            book_heights = np.random.uniform(0.9, 1.2, 13)
            
            book_dims = list(zip(book_widths, book_heights))    

            regions = []
            start_x = 0
            for i in range(num_books):
                book_region = (start_x, 0, book_dims[i][0], book_dims[i][1], 0)
                regions.append((book_region, []))
                start_x += book_dims[i][0]

            vase = (2.5, 0, 1.0, 1.8, 0)

            regions.append((vase, []))

            return "test-case", regions

        elif test_idx == 2:

            teacup = (0.5, 0, 0.3, 0.5, 0)
            tea_bottle = (1.2, 0, 0.6, 1, 0)
            box = (2.2, 0, 1.6, 1.5, 0)

            regions = [(teacup, []), (tea_bottle, []), (box, [])]

            return "test-case", regions

    def gen(relation, shelf_dim, object_num=None):
        count = num_samples
        while count > 0:
            if relation == "all-composed":
                # Randomly choose a relation
                relation_options = ["wall-contact", "wall-side", "side-of", "aligned", "sorted"]
                probability = [0.2, 0.2, 0.1, 0.2, 0.25]
                current_relation = rn.choices(relation_options, probability, k=1)[0]
            else:
                current_relation = relation

            if current_relation == "wall-contact":
                object_num = 2
                relation, regions = get_object_locations_wall_contact(object_num, shelf_dim)
           
            elif current_relation == "wall-side":
                if object_num is None:
                    object_num = rn.randint(2, 8)
                relation, regions = get_object_locations_wall_side(object_num, shelf_dim)

            elif current_relation == "side-of":
                object_num = 2
                relation, regions = get_object_locations_side_of(object_num, shelf_dim)
                        
            elif current_relation == "aligned":
                if object_num is None:
                    object_num = rn.randint(3, 8)
                relation, regions = get_object_locations_aligned(object_num, shelf_dim)
                
            elif current_relation == "sorted":
                if object_num is None:
                    object_num = rn.randint(3, 8)
                relation, regions = get_object_locations_sorted(object_num, shelf_dim)

            elif current_relation == "test-case":
                print("which index")
                x = input()
                relation, regions = get_test_case_object_locations(int(x))
            
            count -= 1
            print(len(regions), "added!")

            yield relation, regions
            if count == 0:
                break
        yield None
    
    return gen

def get_tabletop_data_gen(num_samples, table_dim=(3, 2), object_num=None, relation="table-edge"):
    # A table is a rectangle with width W and length L that represent the x-y plane of a table.
    # W, L = table_dim.
    # It has four table edges : front-edge, back-edge, left-edge, right-edge. front-edge is the side closest to the camera and back edge is the furthes from the camera.
    # The objects can either face the front-edge or back-edge of the table.
    # The front-left corner of the table is with coordinates (0, 0), the back-right corner is with coordinates (W, L)
    # Object-region is a tuple of 5 floats: (x, y, w, l, rotation)
    # (x, y) is the coordinate of the left-front corner of the object 2D bounding box in x-y plane, w is the width, h is the height, rotation is the angle of the object on the x-y plane, 0 means the object is facing the front, Pi means the object is facing the back.
   
    Object = Tuple[float, float, float, float, float]  

    def get_object_dim(object_num, table_dim=(3, 2), facing="random", relation=None, same_size=False):
        W, L = table_dim
        object_dims = []
        min_width = W * 0.05
        max_width = W * 0.4
        min_length = L * 0.05
        max_length = L * 0.4

        if same_size:
            # All objects have the same size and are arranged in a regular grid, facing the same direction.
            # Ensure total dimensions do not exceed 0.5W and 0.5L.
            n_cols = int(math.ceil(math.sqrt(object_num)))
            n_rows = int(math.ceil(object_num / n_cols))
            obj_w = np.random.uniform(max((0.3 * W) / n_cols, min_width), min((0.6 * W) / n_cols, max_width))
            obj_l = np.random.uniform(max((0.3 * L) / n_rows, min_length), min((0.6 * L) / n_rows, max_length))
            rotation = 0  # All facing the front
            object_dims = [(obj_w, obj_l, rotation)] * object_num

        elif relation and ("horizontal" in relation or "row" in relation) and "symmetry" not in relation:
            # Objects arranged horizontally
            total_width = 0.8 * W
            widths = np.random.uniform(min_width, max_width, size=object_num)
            scale_factor = total_width / widths.sum()
            widths *= scale_factor
            if "row" in relation:
                lengths = np.random.uniform(min_length, 0.3*L, size=object_num)
            else:
                lengths = np.random.uniform(min_length, max_length, size=object_num)
            rotation = rn.choice([0, math.pi])
            rotations = [rotation] * object_num
            object_dims = list(zip(widths, lengths, rotations))

        elif relation and ("vertical" in relation or "column" in relation) and "symmetry" not in relation:
            # Objects arranged vertically
            total_length = 0.8 * L
            lengths = np.random.uniform(min_length, max_length, size=object_num)
            scale_factor = total_length / lengths.sum()
            lengths *= scale_factor
            if "column" in relation:
                widths = np.random.uniform(min_width, 0.3*W, size=object_num)
            else:
                widths = np.random.uniform(min_width, max_width, size=object_num)
            rotation = rn.choice([0, math.pi])
            rotations = [rotation] * object_num
            object_dims = list(zip(widths, lengths, rotations))

        elif relation and "edge" in relation:
            # Objects near an edge
            if "front-edge" in relation or "back-edge" in relation:
                total_width = 0.8 * W
                widths = np.random.uniform(min_width, max_width, size=object_num)
                scale_factor = total_width / widths.sum()
                widths *= scale_factor
                lengths = np.random.uniform(min_length, max_length, size=object_num)
            else:
                total_length = 0.8 * L
                lengths = np.random.uniform(min_length, max_length, size=object_num)
                scale_factor = total_length / lengths.sum()
                lengths *= scale_factor
                widths = np.random.uniform(min_width, max_width, size=object_num)
            rotations = rn.choices([0, math.pi], k=object_num)
            object_dims = list(zip(widths, lengths, rotations))

        elif relation and "symmetry" in relation:
            # Symmetry cases
            if "line-symmetry" in relation:
                group_size = 3
            else:
                group_size = 2
            num_groups = object_num // group_size
            object_dims = []
            for _ in range(num_groups):
                # Generate dimensions for the first object
                w = np.random.uniform(min_width, max_width)
                l = np.random.uniform(min_length, max_length)
                rotation = rn.choice([0, math.pi])
                if group_size == 2:
                    if "vertical" in relation:
                        total_width = w * 2
                        if total_width > 0.5 * W:
                            scale_factor = 0.5 * W / total_width
                            w *= scale_factor
                    elif "horizontal" in relation:
                        total_length = l * 2
                        if total_length > 0.5 * L:
                            scale_factor = 0.5 * L / total_length
                            l *= scale_factor
                    object_dims.extend([(w, l, rotation)] * 2)
                else:
                    axis_obj = (w, l, rotation)
                    w_pair = np.random.uniform(min_width, max_width)
                    l_pair = np.random.uniform(min_length, max_length)
                    if "vertical" in relation:
                        total_width = w + w_pair*2
                        if total_width > 0.6 * W:
                            scale_factor = 0.6 * W / total_width
                            w *= scale_factor
                            w_pair *= scale_factor
                    elif "horizontal" in relation:
                        total_length = l + l_pair*2
                        if total_length > 0.6 * L:
                            scale_factor = 0.6 * L / total_length
                            l *= scale_factor
                            l_pair *= scale_factor
                    object_dims.extend([(w, l, rotation), (w_pair, l_pair, rotation), (w_pair, l_pair, rotation)])     

        else:
            # General case
            if facing == "random":
                rotations = rn.choices([0, math.pi], k=object_num)
            else:
                rotations = [0] * object_num
        
            widths = np.random.uniform(min_width, max_width, size=object_num)
            lengths = np.random.uniform(min_length, max_length, size=object_num)
            object_dims = list(zip(widths, lengths, rotations))

        return object_dims

    def rectangles_overlap(region1, region2):
        x1, y1, w1, l1 = region1
        x2, y2, w2, l2 = region2
        if (x1 + w1 <= x2) or (x2 + w2 <= x1):
            return False
        if (y1 + l1 <= y2) or (y2 + l2 <= y1):
            return False
        return True
    
    def get_object_locations_table_edge(object_num, table_dim):
        """
        Generate object locations that are placed near a specific table edge,
        with the closest distance to that edge being 0.25 units.
        Objects face either the front or back edge.
        There will be 3-8 objects against the table edge.
        The objects should not exceed 0.8W or 0.8L.
        """
        W, L = table_dim

        table_edge_margin = 0.1

        # Step 1: Generate the object relation (pick an edge)
        object_relation = rn.choice(["near-front-edge", "near-back-edge", "near-left-edge", "near-right-edge"])

        # Step 2: Get object dimensions and adjust sizes if necessary
        if object_relation in ["near-front-edge", "near-back-edge"]:
            # Sum of widths should not exceed 0.8W
            total_available_width = 0.8 * W
            object_dims = get_object_dim(object_num, table_dim, facing="random", relation=object_relation)
            widths = [dim[0] for dim in object_dims]
            total_width = sum(widths)

            if total_width > total_available_width:
                # Scale down widths proportionally
                scale_factor = total_available_width / total_width
                object_dims = [(w * scale_factor, l, r) for (w, l, r), w in zip(object_dims, widths)]
                widths = [w * scale_factor for w in widths]
            else:
                widths = widths

            # Step 3: Assign object locations with random gaps and starting position
            total_spacing = 0.9 * W - sum(widths)
            num_gaps = object_num + 1  # Gaps between objects and at the ends

            # Randomly divide the total spacing among the gaps
            random_gaps = np.random.uniform(0.5, 1, size=num_gaps)
            gaps_scaling_factor = total_spacing / random_gaps.sum()
            gap_sizes = random_gaps * gaps_scaling_factor

            # Randomize the starting x position within allowable range
            min_start_x = 0
            max_start_x = W - sum(widths) - sum(gap_sizes)
            start_x = min_start_x + np.random.uniform() * (max_start_x - min_start_x)

            current_x = start_x + gap_sizes[0]  # Initial position including start gap
            regions = []
            for i, (w, l, rotation) in enumerate(object_dims):
                x = current_x
                y = table_edge_margin  # Distance from the front or back edge
                if object_relation == "near-back-edge":
                    y = L - l - table_edge_margin  # Adjust y for back edge
                regions.append((x, y, w, l, rotation))
                current_x += w + gap_sizes[i + 1]  # Move to next position

        else:
            # Sum of lengths should not exceed 0.8L
            total_available_length = 0.8 * L
            object_dims = get_object_dim(object_num, table_dim, facing="random", relation=object_relation)
            lengths = [dim[1] for dim in object_dims]
            total_length = sum(lengths)

            if total_length > total_available_length:
                # Scale down lengths proportionally
                scale_factor = total_available_length / total_length
                object_dims = [(w, l * scale_factor, r) for (w, l, r), l in zip(object_dims, lengths)]
                lengths = [l * scale_factor for l in lengths]
            else:
                lengths = lengths

            # Step 3: Assign object locations with random gaps and starting position
            total_spacing = 0.9 * L - sum(lengths)
            num_gaps = object_num + 1  # Gaps between objects and at the ends

            # Randomly divide the total spacing among the gaps
            random_gaps = np.random.uniform(0.5, 1, size=num_gaps)
            gaps_scaling_factor = total_spacing / random_gaps.sum()
            gap_sizes = random_gaps * gaps_scaling_factor

            # Randomize the starting y position within allowable range
            min_start_y = 0
            max_start_y = L - sum(lengths) - sum(gap_sizes)
            start_y = min_start_y + np.random.uniform() * (max_start_y - min_start_y)

            current_y = start_y + gap_sizes[0]  # Initial position including start gap
            regions = []
            for i, (w, l, rotation) in enumerate(object_dims):
                y = current_y
                x = table_edge_margin  # Distance from the left or right edge
                if object_relation == "near-right-edge":
                    x = W - w - table_edge_margin  # Adjust x for right edge
                regions.append((x, y, w, l, rotation))
                current_y += l + gap_sizes[i + 1]  # Move to next position
            
        for region in regions:
            x, y, w, l, rotation = region
            assert 0 <= x <= W - w, "Object exceeds table width"
            assert 0 <= y <= L - l, "Object exceeds table length"

        return "table-edge", regions

    def get_object_locations_table_side(object_num, table_dim):
        """
        Generate object locations at specific relative positions on the table:
        - left-side: objects are on the left half of the table.
        - right-side: objects are on the right half of the table.
        - front-side: objects are on the front half of the table.
        - back-side: objects are on the back half of the table.
        - central-column: objects' x-centroid follows a normal distribution centered at W/2.
        - central-row: objects' y-centroid follows a normal distribution centered at L/2.
        - centered-table: object's centroid is at the center of the table.
        """
        W, L = table_dim

        object_relation = rn.choice([
            "left-side",
            "right-side",
            "front-side",
            "back-side",
            "central-column",
            "central-row",
            "central-vertical-axis",
            "central-horizontal-axis",
            "centered-table"
        ])

        def partition(box, depth=3):

            min_distance = [W * 0.1, L * 0.1]
            max_distance = [W * 0.4, L * 0.4]

            if rand() < 0.15 or depth == 0 or (box[2] < 2 * min_distance[0] and box[3] < 2 * min_distance[1]):
                return [box]

            else:
                split_axis_prob = [0, 0]
                if box[2] > 2 * max_distance[0]:
                    split_axis_prob[0] = 2
                elif box[2] <= 2 * max_distance[0] and box[2] > 2 * min_distance[0]:
                    split_axis_prob[0] = 1
                else: 
                    split_axis_prob[0] = 0
                
                if box[3] > 2 * max_distance[1]:
                    split_axis_prob[1] = 2
                elif box[3] <= 2 * max_distance[1] and box[3] > 2 * min_distance[1]:
                    split_axis_prob[1] = 1
                else:
                    split_axis_prob[1] = 0
                
                split_axis_prob = split_axis_prob / np.sum(split_axis_prob)
                axis = np.random.choice([0, 1], p=split_axis_prob)

               
                split_point = rand() * (box[axis + 2] - 2 * min_distance[axis]) + min_distance[axis]

                rotations = np.random.choice([0, math.pi], size=2)
                if axis == 0:
                    regions =  partition((box[0], box[1], split_point, box[3], rotations[0]), depth - 1)
                    regions += partition((box[0] + split_point, box[1], box[2] - split_point, box[3], rotations[1]), depth - 1)
                else:
                    regions = partition((box[0], box[1], box[2], split_point, rotations[0]), depth - 1)
                    regions += partition((box[0], box[1] + split_point, box[2], box[3] - split_point, rotations[1]), depth - 1)

                return regions

        def pad_regions(regions):

            min_dims = [W * 0.09, L * 0.09]
            num_regions = len(regions)
            for i in range(num_regions):
                
                x, y, w, l, r = regions[i]
                max_spacing_w = w - min_dims[0]
                max_spacing_l = l - min_dims[1]
                x_padding = np.random.uniform(0.3, 1)*max_spacing_w/2
                y_padding = np.random.uniform(0.3, 1)*max_spacing_l/2
                region = (x + x_padding, y + y_padding, w - 2*x_padding, l - 2*y_padding, r)
                regions[i] = region

            return regions
        
        regions = []
            
        if object_relation == "left-side":
            regions = partition((0, 0, W/2, L, 0), 3)
            regions = pad_regions(regions)

        elif object_relation == "right-side":
            regions = partition((W/2, 0, W/2, L, 0), 3)
            regions = pad_regions(regions)

        elif object_relation == "front-side":
            regions = partition((0, 0, W, L/2, 0), 3)
            regions = pad_regions(regions)

        elif object_relation == "back-side":
            regions = partition((0, L/2, W, L/2, 0), 3)
            regions = pad_regions(regions)

        elif object_relation == "central-column" or object_relation == "central-vertical-axis":
            object_dims = get_object_dim(object_num, table_dim, facing="random", relation=object_relation)

            total_y_spacing = L - sum([l for w, l, r in object_dims])
            y_spacings = np.random.uniform(size=object_num+1)
            y_spacings = total_y_spacing * y_spacings / y_spacings.sum()

            start_y = y_spacings[0]
            for i in range(object_num):
                w, l, rotation = object_dims[i]
                if object_relation == "central-column":
                    min_x = W/3
                    max_x = 2*W/3 - w
                    x = np.random.uniform(min_x, max_x)
                else:
                    x = (W - w)/2
                y = start_y
                regions.append((x, y, w, l, rotation))
                start_y += l + y_spacings[i+1]

        elif object_relation == "central-row" or object_relation == "central-horizontal-axis":
            object_dims = get_object_dim(object_num, table_dim, facing="random", relation=object_relation)

            total_x_spacing = W - sum([w for w, l, r in object_dims])
            x_spacings = np.random.uniform(size=object_num+1)
            x_spacings = total_x_spacing * x_spacings / x_spacings.sum()

            start_x = x_spacings[0]
            for i in range(object_num):
                w, l, rotation = object_dims[i]
                if object_relation == "central-row":
                    min_y = L/3
                    max_y = 2*L/3 - l
                    y = np.random.uniform(min_y, max_y)
                else:
                    y = (L - l)/2
                x = start_x
                regions.append((x, y, w, l, rotation))
                start_x += w + x_spacings[i+1]

        elif object_relation == "centered-table":

            region_dim = (W, L)
            object_dims = get_object_dim(1, region_dim, facing="random")
            w, l, rotation = object_dims[0]
            x = (W - w)/2
            y = (L - l)/2
            regions.append((x, y, w, l, rotation))
        
        return "table-side", regions

    def get_object_locations_aligned(pair_num, table_dim):
        """
        Generate object locations for pairs of objects that are aligned according to specified relations:
        - "horizontally-aligned-bottom"
        - "horizontally-aligned-centroid"
        - "vertically-aligned-centroid"
        """
        W, L = table_dim
        object_relations = rn.choices(
            ["horizontally-aligned-bottom", "horizontally-aligned-centroid", "vertically-aligned-centroid"],
            k=pair_num
        )
        object_num = pair_num * 2  # Each pair consists of 2 objects
        # Get dimensions and orientations for all objects
        object_dims = get_object_dim(object_num, table_dim, facing="random")
        regions = []

        for i in range(pair_num):
            # Extract dimensions and orientations for the pair
            w1, l1, rotation1 = object_dims[2 * i]
            w2, l2, rotation2 = object_dims[2 * i + 1]
            relation = object_relations[i]

            # Calculate total dimensions and random spacing
            if relation.startswith("horizontally-aligned"):
                # Total width includes widths of both objects and spacing
                spacing = np.random.uniform(0.04 * W, max(0.05 * W, min(min(w1, w2) * 0.5, 0.1 * W)))
                
                total_width = w1 + w2 + spacing
                max_height = max(l1, l2)
                total_height = max_height
            elif relation == "vertically-aligned-centroid":
                # Total height includes heights of both objects and spacing
                spacing = np.random.uniform(0.04 * L, max(0.05 * L, min(min(l1, l2) * 0.5, 0.1 * L)))
                total_height = l1 + l2 + spacing
                max_width = max(w1, w2)
                total_width = max_width

            # Sample an empty space for this region within the table
            max_attempts = 1000
            placed = False

            for _ in range(max_attempts):
                if relation.startswith("horizontally-aligned"):
                    # Sample x and y within table bounds
                    x_min = 0
                    x_max = W - total_width
                    y_min = 0
                    y_max = L - total_height
                    x_region = np.random.uniform(x_min, x_max)
                    y_region = np.random.uniform(y_min, y_max)
                elif relation == "vertically-aligned-centroid":
                    x_min = 0
                    x_max = W - total_width
                    y_min = 0
                    y_max = L - total_height
                    x_region = np.random.uniform(x_min, x_max)
                    y_region = np.random.uniform(y_min, y_max)

                total_region = (x_region, y_region, total_width, total_height, rotation1)

                # Check for overlaps with existing regions
                overlap = False
                for region in regions:
                    if rectangles_overlap(region[:4], total_region[:4]):
                        overlap = True
                        break

                # Ensure objects are within table bounds
                if (0 <= x_region <= W - total_width and 0 <= y_region <= L - total_height and not overlap):

                    # Determine placements for each object within the region
                    if relation == "horizontally-aligned-bottom":
                        # Place first object
                        x1 = x_region
                        x2 = x1 + w1 + spacing
                        if rotation1 == 0:
                            y1 = y_region  # Align bottom edge at y_region
                            y2 = y_region  # Same y as first object
                        else:
                            y1 = y_region + total_height - l1  # Align top edge
                            y2 = y_region + total_height - l2
                            
                    elif relation == "horizontally-aligned-centroid":
                        # Align centroids along y-axis
                        x1 = x_region
                        y_center = y_region + total_height / 2
                        y1 = y_center - l1 / 2
                        x2 = x1 + w1 + spacing
                        y2 = y_center - l2 / 2
                    elif relation == "vertically-aligned-centroid":
                        # Align centroids along x-axis
                        y1 = y_region
                        x_center = x_region + total_width / 2
                        x1 = x_center - w1 / 2
                        y2 = y1 + l1 + spacing
                        x2 = x_center - w2 / 2

                    # Create regions for both objects
                    new_region1 = (x1, y1, w1, l1, rotation1)
                    new_region2 = (x2, y2, w2, l2, rotation1)

                    # Place objects
                    regions.extend([new_region1, new_region2])
                    placed = True
                    break   

            if not placed:
                print(f"Could not place pair {i+1} ({relation}) after {max_attempts} attempts.")

        return "aligned", regions

    def get_object_locations_side_of(pair_num, table_dim):
        """
        Generate object locations that satisfy the following side-of relations:
        - "right-of"
        - "left-of"
        - "front-of"
        - "back-of"
        """
        W, L = table_dim

        # Step 1: Generate the object relations
        object_relations = rn.choices(["right-of", "left-of", "front-of", "back-of"], k=pair_num)

        # Step 2: Get the object dimensions
        object_num = pair_num * 2  # Each pair consists of 2 objects
        object_dims = get_object_dim(object_num, table_dim, facing="random")

        regions = []

        for i in range(pair_num):
            # Extract dimensions and orientations for the pair
            wA, lA, rotationA = object_dims[2 * i]
            wB, lB, rotationB = object_dims[2 * i + 1]
            relation = object_relations[i]

            # Adjust object orientation if necessary to ensure they are facing the same direction
            # For simplicity, we'll assume both objects in a pair face the same direction
            rotation = rn.choice([0, math.pi])
            rotationA = rotationB = rotation

            # Step 3: Compute total dimensions and random spacing
            if relation in ["left-of", "right-of"]:
                # Occupy along x-axis, overlap along y-axis
                # Total width includes widths of both objects and spacing
                spacing = np.random.uniform(0.05 * W, 0.1 * W)
                total_width = wA + wB + spacing
                # For substantial overlap in y-axis, we define an overlap percentage
                min_overlap = 1.0  # At least 95% overlap
                max_overlap = 1.0  # Up to 100% overlap
                overlap_height = min(lA, lB) * np.random.uniform(min_overlap, max_overlap)
                total_height = max(lA, lB)
            elif relation in ["front-of", "back-of"]:
                # Occupy along y-axis, overlap along x-axis
                spacing = np.random.uniform(0.05 * L, 0.15 * L)
                total_height = lA + lB + spacing
                # For substantial overlap in x-axis, we define an overlap percentage
                min_overlap = 1.0
                max_overlap = 1.0
                overlap_width = min(wA, wB) * np.random.uniform(min_overlap, max_overlap)
                total_width = max(wA, wB)

            # Step 4: Sample a non-overlapping region within the table
            max_attempts = 1000
            placed = False

            for _ in range(max_attempts):
                x_region = np.random.uniform(0, W - total_width)
                y_region = np.random.uniform(0, L - total_height)

                # Step 5: Place objects within this region according to the side-of relation
                if relation == "left-of":
                    # Object A is on the left of Object B
                    # Place Object A
                    xA = x_region
                    yA = y_region + np.random.uniform(0, total_height - lA)
                    # Place Object B to the right of A with spacing
                    xB = xA + wA + spacing
                    yB = yA + np.random.uniform(- (lB - overlap_height), lA - overlap_height)
                elif relation == "right-of":
                    # Object A is on the right of Object B
                    # Place Object B
                    xB = x_region
                    yB = y_region + np.random.uniform(0, total_height - lB)
                    # Place Object A to the right of B with spacing
                    xA = xB + wB + spacing
                    yA = yB + np.random.uniform(- (lA - overlap_height), lB - overlap_height)
                elif relation == "front-of":
                    # Object A is in front of Object B
                    # Place Object A
                    yA = y_region
                    xA = x_region + np.random.uniform(0, total_width - wA)
                    # Place Object B behind A with spacing
                    yB = yA + lA + spacing
                    xB = xA + np.random.uniform(- (wB - overlap_width), wA - overlap_width)
                elif relation == "back-of":
                    # Object A is behind Object B
                    # Place Object B
                    yB = y_region
                    xB = x_region + np.random.uniform(0, total_width - wB)
                    # Place Object A behind B with spacing
                    yA = yB + lB + spacing
                    xA = xB + np.random.uniform(- (wA - overlap_width), wB - overlap_width)
                else:
                    continue  # Invalid relation

                # Adjust for substantial overlap
                if relation in ["left-of", "right-of"]:
                    # Ensure y-overlap
                    y_overlap_min = max(yA, yB)
                    y_overlap_max = min(yA + lA, yB + lB)
                    actual_overlap = y_overlap_max - y_overlap_min
                    required_overlap = overlap_height
                    if actual_overlap < required_overlap:
                        continue  # Overlap not sufficient, try again
                elif relation in ["front-of", "back-of"]:
                    # Ensure x-overlap
                    x_overlap_min = max(xA, xB)
                    x_overlap_max = min(xA + wA, xB + wB)
                    actual_overlap = x_overlap_max - x_overlap_min
                    required_overlap = overlap_width
                    if actual_overlap < required_overlap:
                        continue  # Overlap not sufficient, try again

                # Create regions for both objects
                new_regionA = (xA, yA, wA, lA, rotationA)
                new_regionB = (xB, yB, wB, lB, rotationB)

                # Check for overlaps with existing regions
                overlap = False
                for region in regions:
                    if (rectangles_overlap(region[:4], new_regionA[:4]) or
                        rectangles_overlap(region[:4], new_regionB[:4])):
                        overlap = True
                        break

                # Ensure objects are within table bounds
                if (0 <= xA <= W - wA and 0 <= yA <= L - lA and
                    0 <= xB <= W - wB and 0 <= yB <= L - lB and
                    not overlap):
                    # Place objects
                    regions.extend([new_regionA, new_regionB])
                    placed = True
                    break

            if not placed:
                print(f"Could not place pair {i+1} ({relation}) after {max_attempts} attempts.")

        return "side-of", regions

    def get_object_locations_on_top(pair_num, table_dim):
        W, L = table_dim
        object_num = pair_num * 2
        object_dims = get_object_dim(object_num, table_dim, facing="random")
        regions = []
        occupied_regions = []

        # Obj_A is the base

        for i in range(pair_num):
            # B is larger than A
            wA, lA, rotationA = object_dims[2 * i]
            wB, lB, rotationB = object_dims[2 * i + 1]
            if wB > wA:
                wB, wA = wA, wB
            if lB > lA:
                lB, lA = lA, lB
            placed = False
            max_attempts = 100

            for _ in range(max_attempts):
                xA = np.random.uniform(0, W - wA)
                yA = np.random.uniform(0, L - lA)

                # 50% probability to have centroids overlap
                if rn.random() < 0.5:
                    xB = xA + (wA - wB) / 2
                    yB = yA + (lA - lB) / 2
                else:
                    xB = xA + np.random.uniform(0, wA - wB)
                    yB = yA + np.random.uniform(0, lA - lB)

                new_regionA = (xA, yA, wA, lA, rotationA)
                new_regionB = (xB, yB, wB, lB, rotationA)

                overlap = False
                for base_region in occupied_regions:
                    if rectangles_overlap(base_region[:4], new_regionA[:4]):
                        overlap = True
                        break

                if not overlap:
                    regions.extend([new_regionA, new_regionB])
                    occupied_regions.append(new_regionA[:4])  # base object
                    placed = True
                    break

            if not placed:
                print("Could not place on-top objects without overlapping after max attempts")

        return "on-top-of", regions

    def get_object_locations_symmetry(pair_num, table_dim):
        """
        Generate object locations that satisfy symmetry relations:
        - "vertical-symmetry-on-table": mirror images along a vertical line x = b
        - "horizontal-symmetry-on-table": mirror images along a horizontal line y = b
        - "vertical-line-symmetry": symmetry around an axis object along x = b
        - "horizontal-line-symmetry": symmetry around an axis object along y = b
        """
        W, L = table_dim

        # Step 1: Sample relation
        object_relation = rn.choice([
            "vertical-symmetry-on-table",
            "horizontal-symmetry-on-table",
            "vertical-line-symmetry",
            "horizontal-line-symmetry"
        ])
        # Step 2: Get object dimensions
        if "line-symmetry" in object_relation:
            object_num = pair_num * 3  # Each group has 3 objects
        else:
            object_num = pair_num * 2  # Each group has 2 objects

        object_dims = get_object_dim(object_num, table_dim, facing="random", relation=object_relation)
        regions = []

        for i in range(pair_num):
            # Extract dimensions and orientations for the pair
            if "line-symmetry" in object_relation:
                axis_w, axis_l, r = object_dims[3 * i]
                w_pair, l_pair, r = object_dims[3 * i + 1]
                if "vertical" in object_relation:
                    spacing = np.random.uniform(0.05 * W, 0.15 * W)
                    total_width = axis_w + w_pair * 2 + spacing * 2
                    max_height = max(axis_l, l_pair)
                    total_height = max_height
                else:
                    spacing = np.random.uniform(0.05 * L, 0.15 * L)
                    total_height = axis_l + l_pair * 2 + spacing * 2
                    max_width = max(axis_w, w_pair)
                    total_width = max_width
            else:
                w_pair, l_pair, r = object_dims[2 * i]
                if "vertical" in object_relation:
                    spacing = np.random.uniform(0.05 * W, 0.15 * W)
                    total_width = w_pair * 2 + spacing
                    total_height = l_pair
                else:
                    spacing = np.random.uniform(0.05 * L, 0.15 * L)
                    total_height = l_pair * 2 + spacing
                    total_width = w_pair
            
            assert total_width <= W, "Total width exceeds table width"
            assert total_height <= L, "Total height exceeds table length"

            # Sample an empty space for this region within the table
            max_attempts = 1000
            placed = False

            for _ in range(max_attempts):

                if "line-symmetry" in object_relation:
                    x_min = 0
                    x_max = W - total_width
                    y_min = 0
                    y_max = L - total_height
                    x_region = np.random.uniform(x_min, x_max)
                    y_region = np.random.uniform(y_min, y_max)
                else:
                    if "vertical" in object_relation:
                        y_region = np.random.uniform(0, L - total_height)
                        x_region = (W - total_width) / 2
                    else:
                        x_region = np.random.uniform(0, W - total_width)
                        y_region = (L - total_height) / 2

                total_region = (x_region, y_region, total_width, total_height, r)

                # Check for overlaps with existing regions
                overlap = False
                for region in regions:
                    if rectangles_overlap(region[:4], total_region[:4]):
                        overlap = True
                        break
                if i == 0:
                    assert 0 <= x_region <= W - total_width and 0 <= y_region <= L - total_height and not overlap
    
                # Ensure objects are within table bounds
                if (0 <= x_region <= W - total_width and 0 <= y_region <= L - total_height and not overlap):

                    if object_relation == "vertical-line-symmetry":
                        # Place the axis object in the center of the region
                        x_axis = x_region + (total_width - axis_w) / 2
                        y_axis = y_region + (total_height - axis_l) / 2

                        # Place the pair of objects symmetrically around the axis object
                        x_pair_1 = x_axis - w_pair - spacing
                        y_pair_1 = y_region + (total_height - l_pair) / 2
                        x_pair_2 = x_axis + axis_w + spacing
                        y_pair_2 = y_pair_1

                        regions.extend([
                            (x_axis, y_axis, axis_w, axis_l, r),
                            (x_pair_1, y_pair_1, w_pair, l_pair, r),
                            (x_pair_2, y_pair_2, w_pair, l_pair, r)
                         ])

                    elif object_relation == "horizontal-line-symmetry":
                        # Place the axis object in the center of the region
                        x_axis = x_region + (total_width - axis_w) / 2
                        y_axis = y_region + (total_height - axis_l) / 2

                        # Place the pair of objects symmetrically around the axis object
                        y_pair_1 = y_axis - l_pair - spacing
                        x_pair_1 = x_region + (total_width - w_pair) / 2
                        y_pair_2 = y_axis + axis_l + spacing
                        x_pair_2 = x_pair_1

                        regions.extend([
                            (x_axis, y_axis, axis_w, axis_l, r),
                            (x_pair_1, y_pair_1, w_pair, l_pair, r),
                            (x_pair_2, y_pair_2, w_pair, l_pair, r)
                        ])

                    elif object_relation == "vertical-symmetry-on-table":
                        # place the pair of objects symmetrically along a vertical line
                        x_pair_1 = x_region
                        y_pair_1 = y_region
                        x_pair_2 = x_region + w_pair + spacing  
                        y_pair_2 = y_region

                        regions.extend([
                            (x_pair_1, y_pair_1, w_pair, l_pair, r),
                            (x_pair_2, y_pair_2, w_pair, l_pair, r)
                        ])
                    
                    elif object_relation == "horizontal-symmetry-on-table":
                        # place the pair of objects symmetrically along a horizontal line
                        x_pair_1 = x_region
                        y_pair_1 = y_region
                        x_pair_2 = x_region
                        y_pair_2 = y_region + l_pair + spacing


                        regions.extend([
                            (x_pair_1, y_pair_1, w_pair, l_pair, r),
                            (x_pair_2, y_pair_2, w_pair, l_pair, r)
                        ])
                    
                    placed = True
                    break   

            if not placed:
                print(f"Could not place pair {i+1} ({relation}) after {max_attempts} attempts.")

        return "symmetry", regions

    def get_object_locations_aligned_in_line(object_num, table_dim):
        """
        Generate object locations aligned in a line satisfying the following relations:
        - "aligned-in-horizontal-line-bottom"
        - "aligned-in-horizontal-line-centroid"
        - "aligned-in-vertical-line-centroid"
        """
        W, L = table_dim
        object_relation = rn.choice([
            "aligned-in-horizontal-line-bottom",
            "aligned-in-horizontal-line-centroid",
            "aligned-in-vertical-line-centroid"
        ])
        object_dims = get_object_dim(object_num, table_dim, facing="random", relation=object_relation)
        regions = []

        # Step 1: Determine the total dimension required, including equal spacing
        # Sample a reasonable spacing value

        if "horizontal" in object_relation:
            # Aligned along a horizontal line (objects placed side by side along x-axis)
            widths = [w for w, l, r in object_dims]
            heights = [l for w, l, r in object_dims]
            total_obj_width = sum(widths)
            num_spaces = object_num - 1
            # Sample spacing from a reasonable range
            spacing = np.random.uniform(0.03 * W, 0.07 * W)
            total_spacing = num_spaces * spacing
            total_width = total_obj_width + total_spacing
            max_height = max(heights)

            # Confirm that total_width fits within table width
            if total_width > 0.8 * W:
                # Scale down objects proportionally

                scale_factor = 0.8 * W / total_width
                object_dims = [(w * scale_factor, l, r) for (w, l, r) in object_dims]
                widths = [w * scale_factor for w in widths]
                heights = [l for w, l, r in object_dims]
                total_obj_width = sum(widths)
                spacing *= scale_factor
                total_spacing = num_spaces * spacing
                total_width = total_obj_width + total_spacing
                max_height = max(heights)


            # Now, sample a region where this total area fits within the table
            max_attempts = 1000
            placed = False

            
            x_min = 0
            x_max = W - total_width
            x_start = np.random.uniform(x_min, x_max)

            y_front = np.random.uniform(0, L - max_height)
            y_back = y_front + max_height

            # Determine y position based on relation
            if "bottom" in object_relation:
                # Align bottom edges
                x_current = x_start
                for w, l, r in object_dims:
                    x = x_current
                    if r == 0:
                    # Facing front, align bottom edge close to y=0
                        y = y_front
                    elif r == math.pi:
                        # Facing back, align top edge at y = L
                        y = y_back - l
                        
                    regions.append((x, y, w, l, r))
                    x_current += w + spacing
                    
            elif "centroid" in object_relation:
                # Align centroids along y-axis
                y_center = np.random.uniform(max_height / 2, L - max_height / 2)
                
                x_current = x_start
                for w, l, r in object_dims:
                    x = x_current
                    y = y_center - l / 2
                    
                    regions.append((x, y, w, l, r))
                    x_current += w + spacing         
            
        elif "vertical" in object_relation:
            # Aligned along a vertical line (objects placed atop one another along y-axis)
            lengths = [l for w, l, r in object_dims]
            widths = [w for w, l, r in object_dims]
            total_obj_length = sum(lengths)
            num_spaces = object_num - 1
            spacing = np.random.uniform(0.05 * L, 0.1 * L)
            total_spacing = num_spaces * spacing
            total_length = total_obj_length + total_spacing
            max_width = max(widths)

            # Confirm that total_length fits within table length
            if total_length > 0.8 * L:
                scale_factor = 0.8 * L / total_length
                object_dims = [(w, l * scale_factor, r) for (w, l, r) in object_dims]
                lengths = [l * scale_factor for l in lengths]
                widths = [w for w, l, r in object_dims]
                total_obj_length = sum(lengths)
                spacing *= scale_factor
                total_spacing = num_spaces * spacing
                total_length = total_obj_length + total_spacing
                max_width = max(widths)
            
            y_min = 0
            y_max = L - total_length
            y_start = np.random.uniform(y_min, y_max)

            # For centroid alignment along x-axis
            x_center = np.random.uniform(max_width / 2, W - max_width / 2)
            y_current = y_start
            for w, l, r in object_dims:
                x = x_center - w / 2
                y = y_current
                   
                regions.append((x, y, w, l, r))
                y_current += l + spacing

        return "aligned-in-line", regions

    def get_object_locations_regular_grid(object_num, table_dim):
        """
        Generate object locations arranged in a regular grid on the table,
        with consistent spacing between the objects.
        """
        W, L = table_dim

        assert object_num in [4, 6, 8, 9], "Number of objects must be 4, 6, 8, or 9 for a regular grid."

        # Step 1: Get object dimensions (same size for all objects)
        object_dims = get_object_dim(object_num, table_dim, facing="random", same_size=True)
        obj_w, obj_l, rotation = object_dims[0]  # Dimensions and rotation of the objects

        def find_smallest_divisor_above_sqrt(n):
            sqrt_n = np.sqrt(n)
            lower_bound = int(np.ceil(sqrt_n))  # Smallest integer greater than sqrt(n)
            # Iterate from lower_bound up to n to find the smallest divisor
            for b in range(lower_bound, n + 1):
                if n % b == 0:
                    return b
            # If no divisor is found (n is prime), n itself is the smallest divisor above sqrt(n)
            return n

        # Step 2: Determine the number of rows and columns
        n_cols = find_smallest_divisor_above_sqrt(object_num)
        n_rows = int(math.ceil(object_num / n_cols))

        # Step 3: Determine consistent spacing
        x_spacing = np.random.uniform(0.1, 0.3)
        y_spacing = np.random.uniform(0.1, 0.3)

        total_width = n_cols * obj_w + (n_cols - 1) * x_spacing
        total_height = n_rows * obj_l + (n_rows - 1) * y_spacing

        if total_width > 0.7 * W or total_height > 0.7 * L:
            # Need to scale down object dimensions to fit grid within table bounds
            scale_factor_x = 0.7 * W / total_width
            scale_factor_y = 0.7 * L / total_height
            obj_w *= scale_factor_x
            x_spacing *= scale_factor_x
            obj_l *= scale_factor_y
            y_spacing *= scale_factor_y
            object_dims = [(obj_w, obj_l, rotation)] * object_num

        # For consistent spacing, we'll use the minimum of max_spacing_x and max_spacing_y
        # Or sample a spacing less than or equal to the maximum possible spacing
        spacing_x = spacing_y = min(x_spacing, y_spacing)

        # Alternatively, you can fix the spacing to a specific value or sample from a range
        # For this implementation, we'll use the maximum possible spacing to center the grid

        # Step 4: Calculate the starting positions to center the grid
        total_width = n_cols * obj_w + (n_cols - 1) * spacing_x
        total_height = n_rows * obj_l + (n_rows - 1) * spacing_y

        start_x = (W - total_width) / 2
        start_y = (L - total_height) / 2

        # Step 5: Generate the grid positions
        regions = []
        idx = 0

        for i in range(n_rows):
            y = start_y + i * (obj_l + spacing_y)
            for j in range(n_cols):
                if idx >= object_num:
                    break
                x = start_x + j * (obj_w + spacing_x)
                regions.append((x, y, obj_w, obj_l, rotation))
                idx += 1

        return "regular-grid", regions

    def gen(relation, object_num=None):
        count = num_samples
        while count > 0:
            if relation == "all-composed":
                # Randomly choose a relation
                relation_options = ["table-edge", "table-side", "aligned", "side-of", "on-top-of", "symmetry", "aligned-in-line", "regular-grid"]
                probability = [0.125, 0.15, 0.125, 0.125, 0.05, 0.125, 0.175, 0.125]
                current_relation = rn.choices(relation_options, probability, k=1)[0]
            else:
                current_relation = relation

            if current_relation == "table-edge":
                object_num = rn.randint(3, 8)
                relation, regions = get_object_locations_table_edge(object_num, table_dim)

            elif current_relation == "table-side":
                object_num = rn.randint(3, 5)
                relation, regions = get_object_locations_table_side(object_num, table_dim)

            elif current_relation == "aligned":
                pair_num = rn.randint(2, 4)
                relation, regions = get_object_locations_aligned(pair_num, table_dim)

            elif current_relation == "side-of":
                pair_num = rn.randint(2, 4)
                relation, regions = get_object_locations_side_of(pair_num, table_dim)

            elif current_relation == "on-top-of":
                pair_num = rn.randint(2, 4)
                relation, regions = get_object_locations_on_top(pair_num, table_dim)

            elif current_relation == "symmetry":
                pair_num = rn.randint(2, 4)
                relation, regions = get_object_locations_symmetry(pair_num, table_dim)

            elif current_relation == "aligned-in-line":
                object_num = rn.randint(3, 8)
                relation, regions = get_object_locations_aligned_in_line(object_num, table_dim)

            elif current_relation == "regular-grid":
                object_num = rn.choice([4, 6, 8, 9])
                relation, regions = get_object_locations_regular_grid(object_num, table_dim)

            count -= 1
            print(f"{len(regions)} added! Relation: {relation}")

            for i in range(len(regions)):
                regions[i] = (regions[i], [])
    
            yield relation, regions

            if count == 0:
                break
        yield None
    return gen


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
