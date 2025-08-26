#import
import matplotlib.colors as matcolors
from colorspacious import cspace_convert
import numpy as np
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
from collections import defaultdict
from qgis.core import QgsProject

########################
#colSpaces
########################

def create_cvd_spaces():
    # colorspace as dict
    # normal / prot / deuter / trit
    cvd_spaces_list = []
    # normal vision
    cvd_spaces_list.append({"name": "sRGB1"})
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
        simulated_rgb1 = cspace_convert(rgb1, cvdSpace, "sRGB1")
        #convert to CIE-LAB
        cieLab = cspace_convert(simulated_rgb1, "sRGB1", "CIELab")
        col_list.append({'name': layer.name(), 'renderer': 'singleSymbol', 'label': 'single Symbol', 'cieLab': cieLab, 'CVD': cvdSpace})
    elif layer.renderer().type() == 'categorizedSymbol':
        #for each category, get the symbol
        for category in layer.renderer().categories():
            label = category.label()
            #get Hexcode
            hexcode = category.symbol().color().name()
            #convert to RGB 0-1 values
            rgb1 = matcolors.hex2color(hexcode)
            #convert to CVD color
            simulated_rgb1 = cspace_convert(rgb1, cvdSpace, "sRGB1")
            #convert to CIE-LAB
            cieLab = cspace_convert(simulated_rgb1, "sRGB1", "CIELab")
            col_list.append({'name': layer.name(), 'renderer': 'categorizedSymbol', 'label': label, 'cieLab': cieLab, 'CVD': cvdSpace})
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
            simulated_rgb1 = cspace_convert(rgb1, cvdSpace, "sRGB1")
            #convert to CIE-LAB
            cieLab = cspace_convert(simulated_rgb1, "sRGB1", "CIELab")
            col_list.append({'name': layer.name(), 'renderer': 'graduatedSymbol', 'label': label, 'cieLab': cieLab, 'CVD': cvdSpace})
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

def recolor_layers(selections):
    if not selections:
        return "⚠️ No items selected."

    lines = ["🎨 Recolor-Stub – folgende Elemente würden angepasst:"]

    for sel in selections:
        lines.append(f" - LayerID={sel['layer_id']}, Renderer={sel['renderer']}, Label={sel['label']}")

    # später: hier Logik zum tatsächlichen Umfärben einbauen
    return "\n".join(lines)




















