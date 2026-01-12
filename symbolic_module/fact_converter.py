import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FactConverter:
    def __init__(self, config):
        self.config = config
        self.spatial_threshold = 50

    def convert_detections(self, detections, image_width, image_height, confidence_threshold=0.5):
        facts = []
        boxes = detections['boxes'].cpu().numpy()
        scores = detections['scores'].cpu().numpy()
        valid_indices = np.where(scores >= confidence_threshold)[0]
        object_ids = []
        object_centers = []
        object_sizes = []
        for i, idx in enumerate(valid_indices):
            box = boxes[idx]
            obj_id = f"obj{i+1}"
            object_ids.append(obj_id)
            facts.append(f"object({obj_id}).")
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            object_centers.append((center_x, center_y))
            facts.append(f"position({obj_id}, {center_x:.1f}, {center_y:.1f}).")
            width = box[2] - box[0]
            height = box[3] - box[1]
            area = width * height
            object_sizes.append(area)
            size = "large" if area > (image_width * image_height) / 16 else "small"
            facts.append(f"size({obj_id}, {size}).")
            if hasattr(detections, 'colors') and len(detections.colors) > idx:
                color_idx = detections.colors[idx].item()
                color = self.config.COLORS[color_idx]
                facts.append(f"color({obj_id}, {color}).")
            if hasattr(detections, 'shapes') and len(detections.shapes) > idx:
                shape_idx = detections.shapes[idx].item()
                shape = self.config.SHAPES[shape_idx]
                facts.append(f"shape({obj_id}, {shape}).")
            if hasattr(detections, 'materials') and len(detections.materials) > idx:
                material_idx = detections.materials[idx].item()
                material = self.config.MATERIALS[material_idx]
                facts.append(f"material({obj_id}, {material}).")
        for i, id1 in enumerate(object_ids):
            for j, id2 in enumerate(object_ids):
                if i != j:
                    x1, y1 = object_centers[i]
                    x2, y2 = object_centers[j]
                    if x1 + self.spatial_threshold < x2:
                        facts.append(f"left_of({id1}, {id2}).")
                    if x1 > x2 + self.spatial_threshold:
                        facts.append(f"right_of({id1}, {id2}).")
                    if y1 + self.spatial_threshold < y2:
                        facts.append(f"in_front_of({id1}, {id2}).")
                    if y1 > y2 + self.spatial_threshold:
                        facts.append(f"behind({id1}, {id2}).")
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < self.spatial_threshold * 2:
                        facts.append(f"close_to({id1}, {id2}).")
        for i, id1 in enumerate(object_ids):
            for j, id2 in enumerate(object_ids):
                if i != j:
                    size1 = object_sizes[i]
                    size2 = object_sizes[j]
                    if size1 > size2 * 1.2:
                        facts.append(f"larger_than({id1}, {id2}).")
                    if size1 * 1.2 < size2:
                        facts.append(f"smaller_than({id1}, {id2}).")
        if object_sizes:
            largest_idx = np.argmax(object_sizes)
            facts.append(f"largest({object_ids[largest_idx]}).")
            smallest_idx = np.argmin(object_sizes)
            facts.append(f"smallest({object_ids[smallest_idx]}).")
            x_coords = [center[0] for center in object_centers]
            y_coords = [center[1] for center in object_centers]
            leftmost_idx = np.argmin(x_coords)
            facts.append(f"leftmost({object_ids[leftmost_idx]}).")
            rightmost_idx = np.argmax(x_coords)
            facts.append(f"rightmost({object_ids[rightmost_idx]}).")
            frontmost_idx = np.argmin(y_coords)
            facts.append(f"frontmost({object_ids[frontmost_idx]}).")
            backmost_idx = np.argmax(y_coords)
            facts.append(f"backmost({object_ids[backmost_idx]}).")
        return facts

    def generate_knowledge_base(self, detections, image_width, image_height, confidence_threshold=0.5):
        facts = self.convert_detections(detections, image_width, image_height, confidence_threshold)
        # Group facts by predicate
        predicate_groups = {}
        for fact in facts:
            pred = fact.split('(')[0]
            predicate_groups.setdefault(pred, []).append(fact)
        # List of predicates that may be discontiguous
        discontiguous_preds = [
            "object/1", "position/3", "size/2", "color/2", "shape/2", "material/2",
            "left_of/2", "right_of/2", "in_front_of/2", "behind/2", "close_to/2",
            "larger_than/2", "smaller_than/2", "largest/1", "smallest/1",
            "leftmost/1", "rightmost/1", "frontmost/1", "backmost/1"
        ]
        # Add discontiguous declarations at the top
        prolog_lines = [f":- discontiguous {pred}." for pred in discontiguous_preds]
        # Add grouped facts
        for pred in sorted(predicate_groups.keys()):
            prolog_lines.extend(predicate_groups[pred])
        # Add rules (unchanged)
        rules = [
            "left_of(X, Z) :- left_of(X, Y), left_of(Y, Z).",
            "right_of(X, Z) :- right_of(X, Y), right_of(Y, Z).",
            "behind(X, Z) :- behind(X, Y), behind(Y, Z).",
            "in_front_of(X, Z) :- in_front_of(X, Y), in_front_of(Y, Z).",
            "right_of(X, Y) :- left_of(Y, X).",
            "left_of(X, Y) :- right_of(Y, X).",
            "behind(X, Y) :- in_front_of(Y, X).",
            "in_front_of(X, Y) :- behind(Y, X).",
            "between(Y, X, Z) :- left_of(X, Y), right_of(Z, Y).",
            "between(Y, X, Z) :- right_of(X, Y), left_of(Z, Y).",
            "between(Y, X, Z) :- in_front_of(X, Y), behind(Z, Y).",
            "between(Y, X, Z) :- behind(X, Y), in_front_of(Z, Y).",
            "surrounding(X, Y) :- object(X), object(Y), \\+ same(X, Y), close_to(X, Y), count_close_objects(Y, C), C >= 3.",
            "same(X, X).",
            "count_close_objects(Y, C) :- aggregate_all(count, close_to(_, Y), C).",
            "same_color(X, Y) :- color(X, C), color(Y, C), \\+ same(X, Y).",
            "same_shape(X, Y) :- shape(X, S), shape(Y, S), \\+ same(X, Y).",
            "same_material(X, Y) :- material(X, M), material(Y, M), \\+ same(X, Y).",
            "same_size(X, Y) :- size(X, S), size(Y, S), \\+ same(X, Y).",
            "row(X, Y, Z) :- left_of(X, Y), left_of(Y, Z), \\+ left_of(X, Z).",
            "column(X, Y, Z) :- in_front_of(X, Y), in_front_of(Y, Z), \\+ in_front_of(X, Z).",
            "visible_without(X, Y) :- object(X), object(Y), \\+ occludes(Y, X).",
            "occludes(X, Y) :- in_front_of(X, Y), close_to(X, Y)."
        ]
        prolog_lines.extend(rules)
        return "\n".join(prolog_lines)
