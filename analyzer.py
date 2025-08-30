#import
import matplotlib.colors as matcolors
from colorspacious import cspace_convert
import numpy as np
from colormath.color_objects import LabColor
#from colormath.color_diff import delta_e_cie2000
from collections import defaultdict
from qgis.core import QgsProject
from math import sqrt, cos, sin, pi
from skimage.color import deltaE_ciede2000 as delta_e_cie2000

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
                        lab1 = LabColor(lab_l=item1['cieLab'][0], lab_a=item1['cieLab'][1], lab_b=item1['cieLab'][2])
                        lab2 = LabColor(lab_l=item2['cieLab'][0], lab_a=item2['cieLab'][1], lab_b=item2['cieLab'][2])
                        delta_e = delta_e_cie2000(lab1, lab2)
                        
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
                    lab1 = LabColor(*item1['cieLab'])
                    lab2 = LabColor(*item2['cieLab'])
                    delta_e = delta_e_cie2000(lab1, lab2)

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
    output.append("Fnished analysis:\n")
    
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
    output.append("\n🔗 Conflict groups:")
    if groups:
        for i, group in enumerate(groups, start=1):
            output.append(f"Group {i}: " + ", ".join(group))
    else:
        output.append("No Groups found.")

    #conflict summary
    output.append("\n📊 Conflict overview per color:")
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
def create_keeping_colors_with_cvd(selections):
    #colors to keep
    keep_colors = []
    cvd_spaces = create_cvd_spaces()
    for item in selections:
        layer_id = item['layer_id']
        layer = QgsProject.instance().mapLayer(layer_id)
        if layer:
            for cvd_space in cvd_spaces:
                cieLab = hexToCieLab(layer, cvd_space)
                keep_colors.extend(cieLab)
    return keep_colors

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
        recolor_colors.append({'layer_id': item_layer.id(), 'name': item_layer.name(), 'renderer': item['renderer'], 'label': item['label'], 'cieLab': cieLab, 'cvd': "normal", 'severity': 0, 'hex': hexcode})
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
    t = 0.0 #initial distance
    last_ok = None #last point inside the sphere
    original_lab = LabColor(*original_color)
    while t <= t_max:
        candidate_lab = original_color + direction * t
        #convert to rgb
        candidate_rgb1 = cspace_convert(candidate_lab, "CIELab", "sRGB1")
        ### check if in gamut ###
        if not in_gamut_sRGB(candidate_rgb1):
            return (None, None)   #if it goes out of gamut, stop searching in this direction, diregarding re-entry (not likely)
        #distance to original color
        delta_e_to_original = delta_e_cie2000(original_lab, LabColor(*candidate_lab))
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
                mid_cieLab = original_color + direction * t_mid
                if not in_gamut_sRGB(cspace_convert(mid_cieLab, "CIELab", "sRGB1")):
                    t_high = t_mid
                    continue
                if delta_e_cie2000(LabColor(*original_color), LabColor(*mid_cieLab)) >= threshold:
                    t_high = t_mid
                else:
                    t_low = t_mid
            t_star = t_high * pad
            x_star = original_color + direction * t_star
            if in_gamut_sRGB(cspace_convert(x_star, "CIELab", "sRGB1")):
                return (x_star, t_star)
            return (None, None)
        else:
            last_ok = (t, delta_e_to_original)
            t += step
    return (None, None)

#### Outward push ####
#push each color outwards until it is outside the threshold for all keep colors and in gamut
#original_color: the color to be changed
#keep_colors_by_view:
def lab_candidates_outward_push(original_color, keep_colors_by_view):































############## main function to recolor layers ##############

def recolor_layers(selections, recolor_threshold):
    if not selections:
        return "⚠️ No items selected."
    
    #output lines
    lines = []
    lines.append("Starting recoloring...\n")

    all_layers = QgsProject.instance().mapLayers()
    #create list of layers that were not ticked
    unselected_layers = create_unselected_layers(selections)
                
    #list of colors to keep + cvd simulations
    #input has to be the layers, that were not ticked
    keep_colors = create_keeping_colors_with_cvd(unselected_layers)
    #list of colors to be recolored
    #input has to be layers that were ticked
    recolor_colors = create_recoloring_colors(selections)

    #output info
    lines.append("Colors to be kept:")
    if keep_colors:
        for item in keep_colors:
            if item['cvd'] == "normal":
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






    return "\n".join(lines)




















