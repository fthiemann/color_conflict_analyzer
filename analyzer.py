#import
import matplotlib.colors as matcolors
from colorspacious import cspace_convert
import numpy as np
#from colormath.color_diff import delta_e_cie2000
from collections import defaultdict
from qgis.core import QgsProject, QgsVectorLayer, QgsRendererCategory, QgsRendererRange, QgsGraduatedSymbolRenderer, QgsCategorizedSymbolRenderer
from math import sqrt, cos, sin, pi
from skimage.color import deltaE_ciede2000 as delta_e_cie2000
from qgis.PyQt.QtGui import QColor

########################
#colSpaces
########################

def create_cvd_spaces():
    # colorspace as dict
    # normal / prot / deuter / trit
    cvd_spaces_list = []
    # normal vision
    cvd_spaces_list.append({"name": "sRGB1", "cvd_type": "normal", "severity": 0})
    #protanomaly and protanopia
    sev = 25
    for i in range(4):
        space = {
        "name": "sRGB1+CVD",
        "cvd_type": "protanomaly",
        "severity": sev
        }
        sev += 25
        cvd_spaces_list.append(space)
        
    # deuteranomaly and deuteranopia
    sev = 25
    for i in range(4):
        space = {
        "name": "sRGB1+CVD",
        "cvd_type": "deuteranomaly",
        "severity": sev
        }
        sev += 25
        cvd_spaces_list.append(space)
        
    # tritanomaly and tritanopia#
    sev = 25
    for i in range(4):
        space = {
        "name": "sRGB1+CVD",
        "cvd_type": "tritanomaly",
        "severity": sev
        }
        sev += 25
        cvd_spaces_list.append(space)
    return cvd_spaces_list
        


########################
#functions
########################

def hexToCieLab(layer, cvdSpace):
    #return list with simulated colors
    col_list = []
    #check for diffrent renderer types
    if layer.renderer().type() == 'singleSymbol':
        #get Hexcode
        hexcode = layer.renderer().symbol().color().name()
        #convert to RGB 0-1 values
        rgb1 = matcolors.hex2color(hexcode)
        #convert to CVD color
        if cvdSpace.get('cvd_type', 'normal') == 'normal':
            simulated_rgb1 = rgb1
        else:
            simulated_rgb1 = cspace_convert(rgb1, "sRGB1", cvdSpace)
        #convert to CIE-LAB
        cieLab = cspace_convert(simulated_rgb1, "sRGB1", "CIELab")
        col_list.append({'layer_id': layer.id(), 'name': layer.name(), 'renderer': 'singleSymbol', 'label': 'single Symbol', 'cieLab': cieLab, 'CVD': cvdSpace, 'severity': cvdSpace.get('severity', 0)})
    elif layer.renderer().type() == 'categorizedSymbol':
        #for each category, get the symbol
        for category in layer.renderer().categories():
            label = category.label()
            #get Hexcode
            hexcode = category.symbol().color().name()
            #convert to RGB 0-1 values
            rgb1 = matcolors.hex2color(hexcode)
            #convert to CVD color
            if cvdSpace.get('cvd_type', 'normal') == 'normal':
                simulated_rgb1 = rgb1
            else:
                simulated_rgb1 = cspace_convert(rgb1, "sRGB1", cvdSpace)
            #convert to CIE-LAB
            cieLab = cspace_convert(simulated_rgb1, "sRGB1", "CIELab")
            col_list.append({'layer_id': layer.id(), 'name': layer.name(), 'renderer': 'categorizedSymbol', 'label': label, 'cieLab': cieLab, 'CVD': cvdSpace,  'severity': cvdSpace.get('severity', 0)})
    elif layer.renderer().type() == 'graduatedSymbol':
        #for each graduated category, get the symbol
        for graduated in layer.renderer().ranges():
            #get Label
            label = graduated.label()
            #get Hexcode
            hexcode = graduated.symbol().color().name()
            #convert to RGB 0-1 values
            rgb1 = matcolors.hex2color(hexcode)
            #convert to CVD color
            if cvdSpace.get('cvd_type', 'normal') == 'normal':
                simulated_rgb1 = rgb1
            else:
                simulated_rgb1 = cspace_convert(rgb1, "sRGB1", cvdSpace)
            #convert to CIE-LAB
            cieLab = cspace_convert(simulated_rgb1, "sRGB1", "CIELab")
            col_list.append({'layer_id': layer.id(), 'name': layer.name(), 'renderer': 'graduatedSymbol', 'label': label, 'cieLab': cieLab, 'CVD': cvdSpace,  'severity': cvdSpace.get('severity', 0)})
    # else:
        # print(f"Layer '{layer.name()}' has an unsupported type of renderer")
    return col_list

#helper to create a unique key for each color item
def create_color_key(item):
    return f"{item['name']}|{item['label']}|{item['renderer']}"

#helper to determine level of conflict
def level_of_conflict(delta_e):
    if delta_e < 5:
        return '!!!Critical conflict!!!'
    elif delta_e < 10:
        return "!Significant Konflikt!"
    elif delta_e < 15:
        return "Moderate conflict"
    else:
        return "Minor or no conflict"


########################
#creating base data
########################

def calculate_conflicts(selected_layers_ids, conflict_threshold=15.0):

    #retrieve all layers in project
    all_layers = QgsProject.instance().mapLayers()

    #selected layers
    layers = []
    for layer_id in selected_layers_ids:
        if layer_id in all_layers:
            layers.append(all_layers[layer_id])

    #simulate colors in different color vision deficiency spaces
    cvd_spaces_list = create_cvd_spaces()
    sim_results = {}
    for layer in layers:
        sim_results[layer.name()] = []
        for cvd_space in cvd_spaces_list:
            cieLab = hexToCieLab(layer, cvd_space)
            sim_results[layer.name()].append(cieLab)



    ########################
    #conflicts
    ########################

    #variables
    conflicts = []      #list of conflicts found

    #sim_results is a dict with layer names as keys and list of lists as values
    layer_names = list(sim_results.keys())
    num_layers = len(layer_names)
    num_spaces = len(cvd_spaces_list)
    col_headers = [
        f"{cvd.get('cvd_type', 'normal')} {cvd.get('severity', '')}"
        for cvd in cvd_spaces_list
        ]

    #layer <-> layer
    for i in range(num_layers):
        for j in range(i+1, num_layers):
            name1 = layer_names[i]
            name2 = layer_names[j]
            for idx, header in enumerate(col_headers):
                
                list1 = sim_results[name1][idx]
                list2 = sim_results[name2][idx]

                for item1 in list1:
                    for item2 in list2:
                        #calculate the distance in CIELab
                        lab1 = np.array(item1['cieLab'])
                        lab2 = np.array(item2['cieLab'])
                        delta_e_result = delta_e_cie2000(lab1[None,:], lab2[None,:])
                        delta_e = float(delta_e_result.flat[0])
                        
                        # if conflict, add to list
                        if delta_e < conflict_threshold:
                            conflicts.append({'color1': item1, 'color2': item2, 'delta_e': delta_e})

    #category <-> category within the same layer
    for layer_name in layer_names:
        for idx, header in enumerate(col_headers):
            # print(f"\nFarbraum: {header}")
            # print("-" * 50)

            entries = sim_results[layer_name][idx]

            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    item1 = entries[i]
                    item2 = entries[j]

                    # gleiche Kategorie nicht vergleichen
                    if item1['label'] == item2['label']:
                        continue

                    # Farbabstand berechnen
                    lab1 = np.array(item1['cieLab'])
                    lab2 = np.array(item2['cieLab'])
                    delta_e_result = delta_e_cie2000(lab1[None,:], lab2[None,:])
                    delta_e = float(delta_e_result.flat[0])

                    if delta_e < conflict_threshold:
                        conflicts.append({
                            'color1': item1,
                            'color2': item2,
                            'delta_e': delta_e
                        })

                    # print(f"{item1['label']} ↔ {item2['label']}: ΔE = {delta_e:.2f}")


    ########################
    #Aufarbeitung der Ergebnisse zur Darstellung
    ########################

    #Aufsummierung der Farbkonflikte Pro Farbe
    conflicts_summary = defaultdict(lambda: {'count': 0, 'sum_delta' : 0.0, 'min': float('inf'), 'max': float('-inf')})

    # Konflikte summieren
    for conflict in conflicts:
        for color in ['color1', 'color2']:
            item = conflict[color]
            key = create_color_key(item)
            conflicts_summary[key]['count'] += 1
            conflicts_summary[key]['sum_delta'] += conflict['delta_e']
            conflicts_summary[key]['min'] = min(conflicts_summary[key]['min'], conflict['delta_e'])
            conflicts_summary[key]['max'] = max(conflicts_summary[key]['max'], conflict['delta_e'])

    #######Option 3: Konfliktscore
    conflict_impact = defaultdict(float)  # key → impact score

    for conflict in conflicts:
        for color in ['color1', 'color2']:
            item = conflict[color]
            key = create_color_key(item)
            delta = conflict['delta_e']
            if delta > 0:
                conflict_impact[key] += 1 / delta  # je kleiner ΔE, desto höher der Score

    #########conflict groups
    matches = defaultdict(set) 
    for conflict in conflicts:
        key1 = create_color_key(conflict['color1'])
        key2 = create_color_key(conflict['color2'])
        #grouping by color key
        matches[key1].add(key2)
        matches[key2].add(key1)
    
    groups = []         #list of conflict groups
    visited = set()     #colors already visited

    for match in matches:
        if match not in visited:
            stack = [match] # stack for DFS
            x = set()  # current group
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    x.add(current)
                    for y in matches[current]:
                        if y not in visited:
                            stack.append(y)
            groups.append(x)

    ########################
    #Output
    ########################

    output = []
    output.append("Finished analysis:\n")
    
    if conflicts:
        output.append("Conflicts:")
        for conflict in conflicts:
            item1 = conflict['color1']
            item2 = conflict['color2']
            delta_e = conflict['delta_e']
            cvd_info = f"{item1['CVD'].get('cvd_type', 'normal')} ({item1['CVD'].get('severity', '')})"
            output.append(f"- {level_of_conflict(delta_e)} [{cvd_info}] {item1['name']} ({item1['label']}) ↔ {item2['name']} ({item2['label']}): ΔE = {delta_e:.2f}")
    else:
        output.append("No conflicts found.")

    # output conflict groups
    output.append("\nConflict groups:")
    if groups:
        for i, group in enumerate(groups, start=1):
            output.append(f"Group {i}: " + ", ".join(group))
    else:
        output.append("No Groups found.")

    #conflict summary
    output.append("\nConflict overview per color:")
    if conflicts_summary:
        for key, val in sorted(conflicts_summary.items(), key=lambda x: (-x[1]['count'], -x[1]['sum_delta'])):
            avg_delta = val['sum_delta'] / val['count']
            output.append(f"- {key} → Konflikte: {val['count']}, ⌀ΔE: {avg_delta:.2f}, min: {val['min']:.2f}, max: {val['max']:.2f}, Rating: {conflict_impact.get(key, 0.0):.2f}")
    else:
        output.append("No conclict overview available.")


   




    return "\n".join(output)


########################
#Recoloring
########################

############## helper functions ##############

#create list of colors to keep + cvd simulations
#input has to be the layers, that were not ticked

def create_keeping_colors_with_cvd_from_all_layers(selections, universe_layers):
    selected = defaultdict(set)
    for s in selections:
        selected[s['layer_id']].add(s['label'])

    keep_colors =[]
    cvd_spaces = create_cvd_spaces()

    for layer in universe_layers:
        renderer = layer.renderer().type()
        keep_entries = []

        if renderer == 'singleSymbol':
            if layer.id() not in selected:
                keep_entries.append(('single Symbol', layer.renderer().symbol().color().name()))
        elif renderer == 'categorizedSymbol':
            for cat in layer.renderer().categories():
                if cat.label() not in selected.get(layer.id(), set()):
                    keep_entries.append((cat.label(),cat.symbol().color().name()))
        elif renderer == 'graduatedSymbol':
            for rng in layer.renderer().ranges():
                if rng.label() not in selected.get(layer.id(), set()):
                    keep_entries.append((rng.label(),rng.symbol().color().name()))
        
        #convert these keep entries for each cvd
        for label, hexcode in keep_entries:
            rgb1 = matcolors.hex2color(hexcode)
            for cvd_space in cvd_spaces:
                rgb_sim = rgb1 if cvd_space.get('cvd_type', 'normal') == 'normal' else cspace_convert(rgb1, "sRGB1", cvd_space)
                cieLab = cspace_convert(rgb_sim, "sRGB1", "CIELab")
                keep_colors.append({'layer_id': layer.id(), 'name': layer.name(), 'renderer': renderer,'label': label, 'cieLab': cieLab, 'CVD': cvd_space, 'severity': cvd_space.get('severity',0)})
    return keep_colors

def lab_to_hex(lab):
    rgb1 = np.clip(cspace_convert(np.asarray(lab, dtype=float), "CIELab", "sRGB1"), 0, 1)
    r, g, b = (int(round(c * 255)) for c in rgb1)
    return f"#{r:02x}{g:02x}{b:02x}"



#create list of colors to be recolored
#input has to be layers that were ticked
def create_recoloring_colors(selections):
    #colors to be recolored
    recolor_colors = []
    hexcode = None
    for item in selections:
        item_layer = QgsProject.instance().mapLayer(item['layer_id'])
        #get hexcode from selected layer
        if item_layer.renderer().type() == 'singleSymbol':
            hexcode = item_layer.renderer().symbol().color().name()
        elif item_layer.renderer().type() == 'categorizedSymbol':
            for category in item_layer.renderer().categories():
                if category.label() == item['label']:
                    hexcode = category.symbol().color().name()
        elif item_layer.renderer().type() == 'graduatedSymbol':
            for graduated in item_layer.renderer().ranges():
                if graduated.label() == item['label']:
                    hexcode = graduated.symbol().color().name()
        rgb1 = matcolors.hex2color(hexcode)
        cieLab = cspace_convert(rgb1, "sRGB1", "CIELab")
        recolor_colors.append({'layer_id': item_layer.id(), 'name': item_layer.name(), 'renderer': item['renderer'], 'label': item['label'], 'cieLab': cieLab, 'CVD': "normal", 'severity': 0, 'hex': hexcode})
    return recolor_colors
                    
#reverse selection to get unselected layers
def create_unselected_layers(selections):
    all_layers = QgsProject.instance().mapLayers().values()
    selected_ids = [item['layer_id'] for item in selections]
    unselected_layers = []
    for layer in all_layers:
        if layer.id() not in selected_ids:
            unselected_layers.append(layer)
    return unselected_layers

############## building candidates ##############

def in_gamut_sRGB(rgb, eps = 1e-9):
    return np.all((rgb >= -eps) & (rgb <= 1 + eps))


#fibonacci sphere algorithm to generate directions on the sphere around the to-be-changed color
def golden_sphere_directions( k = 32):
    directions = []
    phi = (1 +sqrt(5)) / 2  # golden ratio
    golden_angle = 2 * pi * (1 - 1 / phi)
    for i in range(k):
        z = 1 - 2*(i+0.5)/k  # z goes from 1 to -1
        r = sqrt(1 - z*z)  # radius at z
        theta = golden_angle * i # golden angle increment
        x = r * cos(theta)
        y = r * sin(theta)
        v = np.array([x, y, z])
        v /= np.linalg.norm(v)  # normalize to unit vector
        directions.append(v)
    return directions

#get first step outside sphere
#original color
#direction: direction vector
#threshold: recolor threshold
#t_max: maximum distance to search
#step: step size
#pad: factor to increase the distance a bit to be sure to be outside the sphere
def get_step_outside_sphere(original_color, direction, threshold, t_max = 50.0, step=1.0, pad = 1.02):
    oc = np.asarray(original_color, dtype = float)
    v = np.asarray(direction, dtype = float)
    t = 0.0 #initial distance
    last_ok = None #last point inside the sphere
    #original_lab = LabColor(*original_color)
    while t <= t_max:
        candidate_lab = oc + v * t
        #convert to rgb
        candidate_rgb1 = cspace_convert(candidate_lab, "CIELab", "sRGB1")
        ### check if in gamut ###
        if not in_gamut_sRGB(candidate_rgb1):
            return (None, None)   #if it goes out of gamut, stop searching in this direction, diregarding re-entry (not likely)
        #distance to original color
        delta_result = delta_e_cie2000(oc[None,:], candidate_lab[None,:])
        delta_e_to_original = float(np.asarray(delta_result).item())
        if delta_e_to_original >= threshold:
            #if after first step outside sphere, return a small distance
            if last_ok is None:
                t_low = max(0.0, t - step)
                t_high = t
            else:
                t_low = last_ok[0]
                t_high = t
            ### Bisektion ### to find exact point, 10 iterations for accruacy
            for _ in range(10):
                t_mid = (t_low + t_high) / 2.0
                mid_cieLab = oc + v * t_mid
                if not in_gamut_sRGB(cspace_convert(mid_cieLab, "CIELab", "sRGB1")):
                    t_high = t_mid
                    continue
                mid_delta_result = delta_e_cie2000(oc[None,:], mid_cieLab[None,:])
                if float(np.asarray(mid_delta_result).item()) >= threshold:
                    t_high = t_mid
                else:
                    t_low = t_mid
            t_star = t_high * pad
            x_star = oc + v * t_star
            if in_gamut_sRGB(cspace_convert(x_star, "CIELab", "sRGB1")):
                return (x_star, t_star)
            return (None, None)
        else:
            last_ok = (t, delta_e_to_original)
            t += step
    return (None, None)

#### Outward push ####

#check candidate for distance to all views (normal + cvd)
#candidate_lab
#keep_colors_by_view: dict with keys per view
def lab_candidates_min_distance_all_views(candidate_lab, keep_colors_by_view):
    min_de = float('inf')   #initialize min distance
    candidate_lab_array = np.asarray(candidate_lab, dtype = float)
    total_comparisons = 0
    #normal
    arr = keep_colors_by_view.get('normal')
    if arr is not None and len(arr):
        delta_result = delta_e_cie2000(np.array(candidate_lab_array, ndmin = 2), arr)
        m = float(np.min(np.asarray(delta_result)))
        min_de = min(min_de, m)
        total_comparisons += len(arr)

    #cvds
    candidate_rgb = cspace_convert(candidate_lab_array, "CIELab", "sRGB1")
    for key, arr in keep_colors_by_view.items():
        if key == 'normal' or arr is None or not len(arr):
            continue
        cvd_type, sev = key.split(':',1)
        sev = int(sev)
        cvd_space = {"name": "sRGB1+CVD", "cvd_type": cvd_type, "severity": sev}
        if cvd_space.get('cvd_type', 'normal') == 'normal':
            candidate_cvd_rgb = candidate_rgb
        else:
            candidate_cvd_rgb = cspace_convert(candidate_rgb, "sRGB1", cvd_space)
        candidate_cvd_lab = cspace_convert(candidate_cvd_rgb, "sRGB1", "CIELab")
        delta_result = delta_e_cie2000(np.asarray(candidate_cvd_lab, dtype = float)[None,:], arr)
        m = float(np.min(np.asarray(delta_result)))
        min_de = min(min_de, m)
        total_comparisons += len(arr)
    

    ##
    #print(f"    Candidate Lab {candidate_lab_array} -> min distance: {min_de:.2f} from {total_comparisons} comparisons")
    ##

    return min_de

#pushing alongside directional vector until in gamut && outside threshold of keeping_colors
#original_color
#direction_lab: directional vektor
#candidate_lab: candidate to check
# threshold: recolorthreshold
#keep_colors_by_view: dict with keys per view
#step: size of steps in Lab
#max_push: max count of steps
def outward_push_until_free(original_color, direction_lab, candidate_lab, threshold, keep_colors_by_view, step = 0.75, max_push =1000):
    oc = np.asarray(original_color, dtype = float)
    v = np.asarray(direction_lab, dtype =float)
    candidate_base = np.asarray(candidate_lab, dtype = float)
    base_offset = candidate_base - oc
    candidate = candidate_base.copy()
    
    t_add = 0.0 #distance outside of sphere
    for _ in range (max_push):
        #gamut check
        if not in_gamut_sRGB(cspace_convert(candidate, "CIELab", "sRGB1")):
            return None
        min_de = lab_candidates_min_distance_all_views(candidate, keep_colors_by_view)  #smallest distance to all kept colors + cvds
        #smallest distance to kept colors + cvds > threshold --> valid candidate
        if min_de >= threshold:
            return candidate
        #otherwise go a step further outside
        t_add += step
        candidate = oc + t_add * v + base_offset
    return None

#### build candidates ####

#helpful dict for easier access later
def build_keep_arrays_by_view(keep_colors):
    output = {}
    for item in keep_colors:
        cvd_info = item.get('CVD', {})
        cvd_type = cvd_info.get('cvd_type', 'normal')
        severity = cvd_info.get('severity', 0)
        key = 'normal' if cvd_type == 'normal' else f"{cvd_type}:{int(severity)}"                                
        output.setdefault(key, []).append(np.array(item['cieLab'], dtype = float))
    for k in list (output.keys()):
        output[k] = np.vstack(output[k]) if len(output[k]) else np.empty((0,3), dtype=float)
    return output


#target_item: recolor_color
#keep_colors
#threshold: recolorthreshold
#kdirections: count of directions used in direction search
#pad: padding around sphere
#step_ring: size of step
def generate_ring_candidates_for_target(target_item, keep_colors, threshold, k_directions = 32, pad = 1.02, step_ring = 1.0, step_push = 0.75, max_push = 120):
    target_item_lab = np.array(target_item['cieLab'], dtype = float)
    keep_by_view = build_keep_arrays_by_view(keep_colors)

    ##
    #print(f"\n--- Generating candidates for {target_item['name']}|{target_item['label']} ---")
    #print(f"Target Lab: {target_item_lab}")
    #print(f"Keep arrays by view keys: {list(keep_by_view.keys())}")
    #for key, arr in keep_by_view.items():
    #    print(f"  {key}: {arr.shape[0]} colors")
    ##

    candidates = []
    successful_directions = 0
    distance_to_original_color = 5
    for v in golden_sphere_directions(k_directions):
        x0, t0 = get_step_outside_sphere(target_item_lab,  v, distance_to_original_color, step = step_ring, pad = pad)
        if x0 is None:
            continue
        x = outward_push_until_free(target_item_lab, v, x0, threshold, keep_by_view, step = step_push, max_push=max_push)
        if x is not None:
            candidates.append(x)
            successful_directions += 1

    ##
    #print(f"Successful directions: {successful_directions}/{k_directions}")
    #print(f"Total candidates found: {len(candidates)}")
    ##

    return candidates


def semantic_delta(original_lab, candidate_lab):
    delta_result = delta_e_cie2000(np.asarray(original_lab, dtype = float)[None, :], np.asarray(candidate_lab, dtype = float)[None, :])
    return float(np.asarray(delta_result).item())

def score_candidate(candidate_lab, original_lab, keep_by_view, threshold):

    min_de = lab_candidates_min_distance_all_views(candidate_lab, keep_by_view)
    reserve = float(min_de - threshold)
    sem = semantic_delta(original_lab, candidate_lab)
    dL = float(abs(candidate_lab[0] - original_lab[0]))

    #print(f"DEBUG score_candidate: threshold={threshold}, min_de={min_de:.2f}, reserve={reserve:.2f}")

    return {'lab': np.asarray(candidate_lab, dtype = float), 'min_de': float(min_de), 'reserve': reserve, 'sem': sem, 'dL': dL, 'feasible': bool(min_de >= threshold),}



def choose_best_candidate_for_target(target_item, candidates, keep_by_view, threshold):
    if not candidates:
        ##
        #print(f"No candidates for {target_item['name']}|{target_item['label']}")
        ##
        return None
    
    target_lab = np.asarray(target_item.get('cieLab'), dtype = float)
    scored = [score_candidate(cand, target_lab, keep_by_view, threshold) for cand in candidates]
    feasible = [s for s in scored if s['feasible']]

    ##
    #print(f"\nScoring {len(candidates)} candidates for {target_item['name']}|{target_item['label']}:")
    #print(f"Feasible candidates: {len(feasible)}/{len(scored)}")
    ##

    if feasible:
        feasible.sort(key=lambda s: (s['sem'], -s['reserve'], s['dL']))
        best = feasible[0]
        ##
        #print(f"Best candidate: min_de={best['min_de']:.2f}, reserve={best['reserve']:.2f}, sem={best['sem']:.2f}")
        ##
        return feasible[0]
    #Fallback, adds best close-to-feasible-case
    scored.sort(key=lambda s: (-s['min_de'], s['sem'], s['dL']))
    best = scored[0]
    ##
    #print(f"No feasible candidates! Best unfeasible: min_de={best['min_de']:.2f} (threshold: {threshold})")
    ##
    return scored[0]

#if multible colors to be recolored, add already recolored color to keep_colors/keep_by_view
def add_candidate_to_keep_by_view(keep_by_view, candidate_lab):
    candidate = np.asarray(candidate_lab, dtype = float)

    #normal
    if 'normal' not in keep_by_view or keep_by_view['normal'].size == 0:
        keep_by_view['normal'] = candidate[None,:]
    else:
        keep_by_view['normal'] = np.vstack([keep_by_view['normal'], candidate[None,:]])
    
    #cvds
    candidate_rgb = cspace_convert(candidate,"CIELab", "sRGB1")
    for key in list(keep_by_view.keys()):
        if key == 'normal':
            continue
        cvd_type, sev = key.split(':',1)
        spec = {"name": "sRGB1+CVD", "cvd_type": cvd_type, "severity": int(sev)}
        candidate_cvd_rgb = cspace_convert(candidate_rgb, "sRGB1", spec)
        candidate_cvd_lab = cspace_convert(candidate_cvd_rgb, "sRGB1", "CIELab")
        keep_by_view[key] = np.vstack([keep_by_view[key], candidate_cvd_lab[None,:]])


def difficulty_of_target(original_lab, keep_by_view):
    return lab_candidates_min_distance_all_views(np.asarray(original_lab, dtype = float), keep_by_view)

def recolor_targets(target_items, keep_colors, threshold, k_directions = 32, pad = 1.02, step_ring = 1.0, step_push = 0.75, max_push = 1000):
    results = []

    keep_by_view = build_keep_arrays_by_view(keep_colors)

    #sort to-be-recolored-colors for their closeness to other colors
    targets_sorted = sorted(target_items, key=lambda it: difficulty_of_target(np.asarray(it.get('cieLab', it.get('cielab')), dtype=float), keep_by_view))

    for target in targets_sorted:
        #generate candidates
        candidates = generate_ring_candidates_for_target(target, keep_colors,threshold, k_directions = k_directions, pad = pad, step_ring = step_ring, step_push = step_push, max_push = max_push)
        if not candidates:
            results.append({'target': target, 'best': None, 'status': 'no-candidate'})
            continue

        #scoring + best choice
        best = choose_best_candidate_for_target(target, candidates, keep_by_view, threshold)
        if best is None:
            results.append({'target': target, 'best': None, 'status': 'no-feasible'})
            continue
        
        #found candidate and appended
        results.append({'target': target, 'best': best, 'status': 'ok'})

        #add best candidate to keep_by_view
        add_candidate_to_keep_by_view(keep_by_view, best['lab'])

    

    return results

                            

########################
#Duplicate items in layer list
########################

#convert results to { layer_id: { label -> new_hex } }
def build_changes_by_layer(results):
    changes = {}
    for result in results:
        if result.get('status') != 'ok' or not result.get('best'):
            continue
        target = result['target']
        layer_id = target['layer_id']
        label = target['label']
        best = result.get('best') or {}
        new_hex = best.get('hex')
        if not new_hex:
            lab = best.get('lab')
            new_hex = lab_to_hex(lab)
        changes.setdefault(layer_id, {})[label] = new_hex
    return changes


def apply_colors_to_clone(clone_layer, label_to_hex):
    clone_r = clone_layer.renderer()
    clone_r_type = clone_r.type() if hasattr(clone_r, "type") else""
    if clone_r_type == "singleSymbol":
        hexcode = label_to_hex.get("single Symbol")
        if hexcode:
            sym = clone_r.symbol().clone()
            sym.setColor(QColor(hexcode))
            clone_r.setSymbol(sym)
            clone_layer.setRenderer(clone_r)

    elif clone_r_type == "categorizedSymbol":
        categories = clone_r.categories()
        new_categories = []
        changed =False
        for c in categories:
            hexcode = label_to_hex.get(c.label())
            sym = c.symbol().clone()
            if hexcode:
                sym.setColor(QColor(hexcode))
                changed = True
            new_categories.append(QgsRendererCategory(c.value(), sym, c.label()))
        if changed:
            try:
                class_attr = clone_r.classAttribute()
            except:
                class_attr = ""
            new_renderer = QgsCategorizedSymbolRenderer(class_attr, new_categories)
            clone_layer.setRenderer(new_renderer)

    elif clone_r_type == "graduatedSymbol":
        ranges = clone_r.ranges()
        new_ranges = []
        changed =False
        for rng in ranges:
            hexcode = label_to_hex.get(rng.label())
            sym = rng.symbol().clone()
            if sym is None:
                new_ranges.append(rng)
                continue
            if hexcode:
                sym.setColor(QColor(hexcode))
                changed = True
            new_ranges.append(QgsRendererRange(float(rng.lowerValue()), float(rng.upperValue()), sym, rng.label()))
        if changed:
            #building a new renderer
            class_attr = clone_r.classAttribute()
            new_renderer = QgsGraduatedSymbolRenderer(class_attr, new_ranges)
            try:
                new_renderer.setModer(clone_r.Mode())
            except Exception:
                pass
            clone_layer.setRenderer(new_renderer)

    
    clone_layer.triggerRepaint()

#duplicate each analyzed layer, apply the colors and put the clones into a subgroup
def duplicate_analyzed_layers_and_apply(analyzed_layer_ids, changes_by_layer, group_name = "CVD Recolored"):
    if not analyzed_layer_ids:
        return
    proj = QgsProject.instance()
    all_layers = proj.mapLayers()

    #create group
    root = proj.layerTreeRoot()
    group = root. findGroup(group_name)
    if group is None:
        group = root.insertGroup(0, group_name)
    
    for layer_id in analyzed_layer_ids:
        layer = all_layers.get(layer_id)
        #clone layer
        clone = layer.clone()
        clone.setName(f"{layer.name()} [CVD optimized]")
        #add clone to project
        proj.addMapLayer(clone, False)
        #apply recolor for layer if needed
        label_to_hex = changes_by_layer.get(layer_id, {})
        if label_to_hex:
            apply_colors_to_clone(clone, label_to_hex)
        #add clone to group
        group.insertLayer(0, clone)




############## main function to recolor layers ##############

def recolor_layers(selections, recolor_threshold, analyzed_layer_ids = None):
    if not selections:
        return " No items selected."
    
    #output lines
    lines = []
    lines.append("Starting recoloring...\n")

    all_layers = QgsProject.instance().mapLayers()
    if analyzed_layer_ids:
        universe_ids = set(analyzed_layer_ids)
        universe_layers = [all_layers[lid] for lid in universe_ids if lid in all_layers]
    else:
        universe_layers = list(all_layers.values())
      
    #list of colors to keep + cvd simulations
    #input has to be the layers, that were not ticked
    keep_colors = create_keeping_colors_with_cvd_from_all_layers(selections, universe_layers)
    #list of colors to be recolored
    #input has to be layers that were ticked
    recolor_colors = create_recoloring_colors(selections)
    

    #output info
    lines.append("Colors to be kept:")
    if keep_colors:
        for item in keep_colors:
            if item.get('CVD', {}).get('cvd_type', 'normal') == 'normal':
                lines.append(
                    f"- layer_name={item['name']} | renderer={item['renderer']} | label={item['label']} | ")
    else:
        lines.append("(leer)")

    lines.append("\nColors to be recolored:")
    if recolor_colors:
        for item in recolor_colors:
            lines.append(f"- layer_name={item['name']} | renderer={item['renderer']} | label={item['label']} | hex={item['hex']}")
    else:
        lines.append("(leer)")

    #recoloring
    results = recolor_targets(recolor_colors, keep_colors, recolor_threshold, k_directions = 32, pad=1.02, step_ring =1.0, step_push =0.75, max_push =120)

    lines.append("\n=== Results (per target) ===")
    if results:
        for res in results:
            tgt = res['target']
            status = res['status']
            if status != 'ok' or res['best'] is None:
                lines.append(
                    f"- {tgt['name']} | {tgt['renderer']} | {tgt['label']} → status={status}"
                )
                continue
            best = res['best']
            new_hex = lab_to_hex(best['lab'])
            lines.append(
                f"- {tgt['name']} | {tgt['renderer']} | {tgt['label']} → "
                f"new_lab={[round(float(x),2) for x in best['lab']]} | new_hex={new_hex} | "
                f"min_de={best['min_de']:.2f} | reserve={best['reserve']:.2f} | "
                f"sem={best['sem']:.2f} | dL={best['dL']:.2f} | feasible={best['feasible']}"
            )
    else:
        lines.append("(leer)")
    
    #duplicate and group
    try: 
        changes_by_layer = build_changes_by_layer(results)
        duplicate_analyzed_layers_and_apply(analyzed_layer_ids, changes_by_layer, group_name="CVD Recolored")
        lines.append("\n Duplicated analyzed layers into group 'CVD Recolored' and applied recolored styles.")
    except Exception as e:
        lines.append(f"\n Could not duplicate/apply recolors automatically: {e}")

    #debugging
    #print(f"\n=== DEBUG: Starting recoloring with threshold {recolor_threshold} ===")
    #print(f"Number of colors to keep: {len(keep_colors)}")
    #print(f"Number of colors to recolor: {len(recolor_colors)}")
    
    # for i, item in enumerate(keep_colors[:3]):  # Show first 3 kept colors
    #     cvd_info = item.get('CVD', {})
    #     print(f"Keep color {i}: {item['name']} | {item['label']} | CVD: {cvd_info.get('cvd_type', 'normal')} | Lab: {item['cieLab']}")
    
    # for i, item in enumerate(recolor_colors):
    #     print(f"Recolor {i}: {item['name']} | {item['label']} | Lab: {item['cieLab']} | Hex: {item['hex']}")




    return "\n".join(lines)




















