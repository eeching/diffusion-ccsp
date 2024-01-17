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
            fork_w, fork_l = 0.02*W, 0.22*L
            knife_w, knife_l = 0.02*W, 0.22*L
            spoon_w, spoon_l = 0.02*W, 0.2*L
            chopsticks_w, chopsticks_l = 0.02*W, 0.24*L
            glass_w, glass_l = 0.06*W, 0.09*L
            medium_plate_w, medium_plate_l = 0.12*W, 0.18*L
            small_plate_w, small_plate_l = 0.1*W, 0.15*L
            rice_bowl_w, rice_bowl_l = 0.09*W, 0.12*L
            ramen_bowl_w, ramen_bowl_l = 0.1*W, 0.15*L
            seasoning_w, seasoning_l = 0.04*W, 0.06*L
            baby_bowl_w, baby_bowl_l = 0.08*W, 0.12*L
            baby_plate_w, baby_plate_l = 0.1*W, 0.15*L
            baby_spoon_w, baby_spoon_l = 0.02*W, 0.15*L
            baby_cup_w, baby_cup_l = 0.04*W, 0.06*L

            front_base = 0.05*L
            back_base = 0.95*L


            if x == "1":
                # dinner table for 2
                serving_plate_1= (0.44*W, front_base, serving_plate_w, serving_plate_l)
                napkin_1 = (0.38*W, front_base, napkin_w, napkin_l)
                fork_1 = (0.395*W, front_base, fork_w, fork_l)
                knife_1 = (0.57*W, front_base, knife_w, knife_l)
                spoon_1 = (0.60*W, front_base, spoon_w, spoon_l)
                glass_1 = (0.65*W, 0.1*L, glass_w, glass_l)
                serving_plate_2 = (0.44*W, back_base - serving_plate_l, serving_plate_w, serving_plate_l)
                napkin_2 = (0.57*W, back_base - napkin_l, napkin_w, napkin_l)
                fork_2 = (0.585*W, back_base - fork_l, fork_w, fork_l)
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
                chopsticks_1 = (0.56*W, front_base, chopsticks_w, chopsticks_l)
                chopsticks_2 = (0.42*W, back_base - chopsticks_l, chopsticks_w, chopsticks_l)
                spoon_1 = (0.59*W, front_base, spoon_w, spoon_l)
                spoon_2 = (0.39*W, back_base - spoon_l, spoon_w, spoon_l)
                obj_list = [medium_plate_1, medium_plate_2, medium_plate_3, medium_plate_4, small_plate_1, small_plate_2, rice_bowl_1, rice_bowl_2, chopsticks_1, chopsticks_2, spoon_1, spoon_2]
                names = ["medium_plate_1", "medium_plate_2", "medium_plate_3", "medium_plate_4", "small_plate_1", 
                         "small_plate_2", "rice_bowl_1", "rice_bowl_2", "chopsticks_1", "chopsticks_2", "spoon_1", 
                         "spoon_2"]
            
            if x == "3":
                # dinner table for 2 same side
                serving_plate_1= (0.19*W, front_base, serving_plate_w, serving_plate_l)
                napkin_1 = (0.13*W, front_base, napkin_w, napkin_l)
                fork_1 = (0.145*W, front_base, fork_w, fork_l)
                knife_1 = (0.32*W, front_base, knife_w, knife_l)
                spoon_1 = (0.35*W, front_base, spoon_w, spoon_l)
                glass_1 = (0.37*W, 0.1*L, glass_w, glass_l)
                serving_plate_2= (0.69*W, front_base, serving_plate_w, serving_plate_l)
                napkin_2 = (0.63*W, front_base, napkin_w, napkin_l)
                fork_2 = (0.645*W, front_base, fork_w, fork_l)
                knife_2 = (0.82*W, front_base, knife_w, knife_l)
                spoon_2 = (0.85*W, front_base, spoon_w, spoon_l)
                glass_2 = (0.87*W, 0.1*L, glass_w, glass_l)
                obj_list = [serving_plate_1, napkin_1, fork_1, knife_1, spoon_1, glass_1, serving_plate_2, napkin_2, fork_2, knife_2, spoon_2, glass_2]
                names = ["serving_plate_1", "napkin_1", "fork_1", "knife_1", "spoon_1", "glass_1",
                         "serving_plate_2", "napkin_2", "fork_2", "knife_2", "spoon_2", "glass_2"]
            
            if x == "4":
                # ramen table for 1
                ramen_bowl = (0.45*W, front_base, ramen_bowl_w, ramen_bowl_l)
                chopsticks = (0.56*W, front_base, chopsticks_w, chopsticks_l)
                spoon = (0.58*W, front_base, spoon_w, spoon_l)
                medium_plate_1 = (0.35*W, 0.41*L, medium_plate_w, medium_plate_l)
                medium_plate_2 = (0.53*W, 0.41*L, medium_plate_w, medium_plate_l)
                seasoning_1 = (0.85*W, 0.64*L, seasoning_w, seasoning_l)
                seasoning_2 = (0.85*W, 0.56*L, seasoning_w, seasoning_l)
                seasoning_3 = (0.85*W, 0.48*L, seasoning_w, seasoning_l)
                seasoning_4 = (0.85*W, 0.4*L, seasoning_w, seasoning_l)
                obj_list = [ramen_bowl, chopsticks, spoon, medium_plate_1, medium_plate_2, seasoning_1, seasoning_2, seasoning_3, seasoning_4]
                names = ["ramen_bowl", "chopsticks", "spoon", "medium_plate_1", "medium_plate_2", "seasoning_1",
                          "seasoning_2", "seasoning_3", "seasoning_4"]

            if x == "5":
                # with baby
                baby_plate = (0.2*W, front_base, baby_plate_w, baby_plate_l)
                baby_bowl = (0.21*W, front_base + 0.015*L, baby_bowl_w, baby_bowl_l)
                baby_spoon = (0.31*W, front_base, baby_spoon_w, baby_spoon_l)
                baby_cup = (0.33*W, 0.1*L, baby_cup_w, baby_cup_l)
                serving_plate = (0.69*W, front_base, serving_plate_w, serving_plate_l)
                napkin = (0.63*W, front_base, napkin_w, napkin_l)
                fork = (0.645*W, front_base, fork_w, fork_l)
                knife = (0.82*W, front_base, knife_w, knife_l)
                spoon = (0.85*W, front_base, spoon_w, spoon_l)
                glass = (0.87*W, 0.1*L, glass_w, glass_l)
                seasoning_1 = (0.85*W, 0.56*L, seasoning_w, seasoning_l)
                seasoning_2 = (0.85*W, 0.48*L, seasoning_w, seasoning_l)
                seasoning_3 = (0.85*W, 0.4*L, seasoning_w, seasoning_l)
                obj_list = [baby_plate, baby_bowl, baby_spoon, baby_cup, serving_plate, napkin, fork, knife, spoon, glass, seasoning_1, seasoning_2, seasoning_3]
                names = ["baby_plate", "baby_bowl", "baby_spoon", "baby_cup", "serving_plate", "napkin", "fork", 
                         "knife", "spoon", "glass", "seasoning_1", "seasoning_2", "seasoning_3"]

            if x == "6":
                # left handed diner
                serving_plate_1= (0.44*W, front_base, serving_plate_w, serving_plate_l)
                napkin_1 = (0.38*W, front_base, napkin_w, napkin_l)
                fork_1 = (0.395*W, front_base, fork_w, fork_l)
                knife_1 = (0.57*W, front_base, knife_w, knife_l)
                spoon_1 = (0.60*W, front_base, spoon_w, spoon_l)
                glass_1 = (0.65*W, 0.1*L, glass_w, glass_l)
                serving_plate_2 = (0.44*W, back_base - serving_plate_l, serving_plate_w, serving_plate_l)
                napkin_2 = (0.57*W, back_base - napkin_l, napkin_w, napkin_l)
                fork_2 = (0.585*W, back_base - fork_l, fork_w, fork_l)
                knife_2 = (0.40*W, back_base - knife_l, knife_w, knife_l)
                spoon_2 = (0.36*W, back_base - spoon_l, spoon_w, spoon_l)
                glass_2 = (0.28*W, 0.81*L, glass_w, glass_l)
                obj_list = [serving_plate_1, napkin_1, fork_1, knife_1, spoon_1, glass_1, serving_plate_2, napkin_2, fork_2, knife_2, spoon_2, glass_2]
                names = ["serving_plate_1", "napkin_1", "fork_1", "knife_1", "spoon_1", "glass_1",
                         "serving_plate_2", "napkin_2", "fork_2", "knife_2", "spoon_2", "glass_2"]
            
            if x == "7":
                # ramen set with baby
                baby_bowl = (0.21*W, front_base + 0.015*L, baby_bowl_w, baby_bowl_l)
                baby_spoon = (0.31*W, front_base, baby_spoon_w, baby_spoon_l)
                baby_cup = (0.33*W, 0.1*L, baby_cup_w, baby_cup_l)

                ramen_bowl = (0.45*W, front_base, ramen_bowl_w, ramen_bowl_l)
                chopsticks = (0.56*W, front_base, chopsticks_w, chopsticks_l)
                spoon = (0.58*W, front_base, spoon_w, spoon_l)
                glass = (0.60*W, 0.4*L, glass_w, glass_l)

                medium_plate_1 = (0.35*W, 0.41*L, medium_plate_w, medium_plate_l)
                medium_plate_2 = (0.53*W, 0.41*L, medium_plate_w, medium_plate_l)
                
                seasoning_1 = (0.85*W, 0.64*L, seasoning_w, seasoning_l)
                seasoning_2 = (0.85*W, 0.56*L, seasoning_w, seasoning_l)
                seasoning_3 = (0.85*W, 0.48*L, seasoning_w, seasoning_l)
                seasoning_4 = (0.85*W, 0.4*L, seasoning_w, seasoning_l)

                obj_list = [baby_bowl, baby_spoon, baby_cup, 
                            ramen_bowl, chopsticks, spoon, 
                            medium_plate_1, medium_plate_2, 
                            glass, seasoning_1, seasoning_2, seasoning_3, seasoning_4]
                names = ["baby_bowl", "baby_spoon", "baby_cup", "ramen_bowl", "chopsticks", "spoon", 
                         "medium_plate_1", "medium_plate_2", "glass", "seasoning_1", 
                         "seasoning_2", "seasoning_3", "seasoning_4"]
            
            if x == "8":
                # 8 main dishes for sharing
                medium_plate_1 = (0.375*W, 0.315*L, medium_plate_w, medium_plate_l)
                medium_plate_2 = (0.505*W, 0.315*L, medium_plate_w, medium_plate_l)
                medium_plate_3 = (0.375*W, 0.505*L, medium_plate_w, medium_plate_l)
                medium_plate_4 = (0.505*W, 0.505*L, medium_plate_w, medium_plate_l)
                medium_plate_5 = (0.375*W, 0.315*L, medium_plate_w, medium_plate_l)
                medium_plate_6 = (0.505*W, 0.315*L, medium_plate_w, medium_plate_l)
                medium_plate_7 = (0.375*W, 0.505*L, medium_plate_w, medium_plate_l)
                medium_plate_8 = (0.505*W, 0.505*L, medium_plate_w, medium_plate_l)

                small_plate_1 = (0.45*W, front_base, small_plate_w, small_plate_l)
                small_plate_2 = (0.45*W, back_base - small_plate_l, small_plate_w, small_plate_l)
                rice_bowl_1 = (0.455*W, front_base + 0.015*L, rice_bowl_w, rice_bowl_l)
                rice_bowl_2 = (0.455*W, back_base - 0.015*L - rice_bowl_l, rice_bowl_w, rice_bowl_l)
                chopsticks_1 = (0.56*W, front_base, chopsticks_w, chopsticks_l)
                chopsticks_2 = (0.42*W, back_base - chopsticks_l, chopsticks_w, chopsticks_l)
                spoon_1 = (0.59*W, front_base, spoon_w, spoon_l)
                spoon_2 = (0.39*W, back_base - spoon_l, spoon_w, spoon_l)

                obj_list = [medium_plate_1, medium_plate_2, medium_plate_3, medium_plate_4, 
                            medium_plate_5, medium_plate_6, medium_plate_7, medium_plate_8, 
                            small_plate_1, small_plate_2, rice_bowl_1, 
                            rice_bowl_2, chopsticks_1, chopsticks_2, spoon_1, spoon_2]
                names = ["medium_plate_1", "medium_plate_2", "medium_plate_3", "medium_plate_4", 
                         "medium_plate_5", "medium_plate_6", "medium_plate_7", "medium_plate_8", 
                         "small_plate_1", "small_plate_2", "rice_bowl_1", "rice_bowl_2", "chopsticks_1", 
                         "chopsticks_2", "spoon_1", "spoon_2"]

            if x == "9":
                # ramen set for 2
                ramen_bowl_1 = (0.45*W, front_base, ramen_bowl_w, ramen_bowl_l)
                chopsticks_1 = (0.56*W, front_base, chopsticks_w, chopsticks_l)
                spoon_1 = (0.58*W, front_base, spoon_w, spoon_l)
                glass_1 = (0.60*W, 0.4*L, glass_w, glass_l)

                medium_plate_1 = (0.375*W, 0.315*L, medium_plate_w, medium_plate_l)
                medium_plate_2 = (0.505*W, 0.315*L, medium_plate_w, medium_plate_l)
                medium_plate_3 = (0.375*W, 0.505*L, medium_plate_w, medium_plate_l)
                medium_plate_4 = (0.505*W, 0.505*L, medium_plate_w, medium_plate_l)

                ramen_bowl_2 = (0.45*W, front_base, ramen_bowl_w, ramen_bowl_l)
                chopsticks_2 = (0.56*W, front_base, chopsticks_w, chopsticks_l)
                spoon_2 = (0.58*W, front_base, spoon_w, spoon_l)
                glass_2 = (0.60*W, 0.4*L, glass_w, glass_l)

                seasoning_1 = (0.85*W, 0.56*L, seasoning_w, seasoning_l)
                seasoning_2 = (0.85*W, 0.48*L, seasoning_w, seasoning_l)
                seasoning_3 = (0.85*W, 0.4*L, seasoning_w, seasoning_l)

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
                serving_plate_1= (0.44*W, front_base, serving_plate_w, serving_plate_l)
                napkin_1 = (0.38*W, front_base, napkin_w, napkin_l)
                fork_1 = (0.395*W, front_base, fork_w, fork_l)
                knife_1 = (0.57*W, front_base, knife_w, knife_l)
                spoon_1 = (0.60*W, front_base, spoon_w, spoon_l)
                glass_1 = (0.65*W, 0.1*L, glass_w, glass_l)
                serving_plate_2 = (0.44*W, back_base - serving_plate_l, serving_plate_w, serving_plate_l)
                napkin_2 = (0.57*W, back_base - napkin_l, napkin_w, napkin_l)
                fork_2 = (0.585*W, back_base - fork_l, fork_w, fork_l)
                knife_2 = (0.40*W, back_base - knife_l, knife_w, knife_l)
                spoon_2 = (0.36*W, back_base - spoon_l, spoon_w, spoon_l)
                glass_2 = (0.28*W, 0.81*L, glass_w, glass_l)

                serving_plate_3= (0.44*W, front_base, serving_plate_w, serving_plate_l)
                napkin_3 = (0.38*W, front_base, napkin_w, napkin_l)
                fork_3 = (0.395*W, front_base, fork_w, fork_l)
                knife_3 = (0.57*W, front_base, knife_w, knife_l)
                spoon_3 = (0.60*W, front_base, spoon_w, spoon_l)
                glass_3 = (0.65*W, 0.1*L, glass_w, glass_l)
                serving_plate_4 = (0.44*W, back_base - serving_plate_l, serving_plate_w, serving_plate_l)
                napkin_4 = (0.57*W, back_base - napkin_l, napkin_w, napkin_l)
                fork_4 = (0.585*W, back_base - fork_l, fork_w, fork_l)
                knife_4 = (0.40*W, back_base - knife_l, knife_w, knife_l)
                spoon_4 = (0.36*W, back_base - spoon_l, spoon_w, spoon_l)
                glass_4 = (0.28*W, 0.81*L, glass_w, glass_l)
                obj_list = [serving_plate_1, napkin_1, fork_1, knife_1, spoon_1, glass_1, 
                            serving_plate_2, napkin_2, fork_2, knife_2, spoon_2, glass_2,
                            serving_plate_3, napkin_3, fork_3, knife_3, spoon_3, glass_3,
                            serving_plate_4, napkin_4, fork_4, knife_4, spoon_4, glass_4]
                names =  ["serving_plate_1", "napkin_1", "fork_1", "knife_1", "spoon_1", "glass_1",
                         "serving_plate_2", "napkin_2", "fork_2", "knife_2", "spoon_2", "glass_2",
                            "serving_plate_3", "napkin_3", "fork_3", "knife_3", "spoon_3", "glass_3",
                            "serving_plate_4", "napkin_4", "fork_4", "knife_4", "spoon_4", "glass_4"]
                
            return obj_list, relation, names

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
            elif "dining_table" in relation:
                print("which idx")
                x = input()
                regions, relation_mode, names = dining_table(W, L, x, relation)
            elif "study_table" in relation:
                print("which idx")
                x = input()
                regions, relation_mode, names = study_table(W, L, x, relation)
            try:
                regions = filter_regions(regions, min_size)
            except:
                pdb.set_trace()
            # (("ccollide" in relation or "integrated" in relation) and len(regions) == 2) or
            if min_num_regions <= len(regions) <= max_num_regions or "study_table" in relation or "dining_table" in relation:
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
