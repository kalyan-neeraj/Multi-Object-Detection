import os
import sys
import tempfile

from pyswip import Prolog

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# os.environ['LD_LIBRARY_PATH'] = '/usr/lib/swi-prolog/lib/x86_64-linux-gnu'
# os.environ['SWI_HOME_DIR'] = '/usr/lib/swi-prolog'

class PrologReasoner:
    def __init__(self):
        self.prolog = Prolog()
        self.facts_loaded = False

    def load_knowledge_base(self, knowledge_base_str):
        self._clear_knowledge_base()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pl', dir='.', delete=False) as temp_file:
            temp_file.write(knowledge_base_str)
            temp_file_path = temp_file.name
        try:
            self.prolog.consult(temp_file_path)
            self.facts_loaded = True
            print(f"Loaded knowledge base with {knowledge_base_str.count('.')} statements")
        except Exception as e:
            print(f"Error loading knowledge base: {str(e)}")
        finally:
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file: {e}")

    def _clear_knowledge_base(self):
        predicates = [
            "object", "position", "bottom_position", "dimensions", "distance",
            "color", "shape", "material", "size",
            "left_of", "right_of", "in_front_of", "behind", "close_to",
            "aligned_horizontally", "aligned_vertically",
            "larger_than", "smaller_than", "largest", "smallest",
            "leftmost", "rightmost", "frontmost", "backmost",
            "between", "surrounding", "same", "count_close_objects",
            "same_color", "same_shape", "same_material", "same_size",
            "row", "column", "visible_without", "occludes"
        ]
        for pred in predicates:
            for arity in range(1, 5):
                template = f"{pred}({'_,' * (arity-1) + '_'})"
                try:
                    list(self.prolog.query(f"retractall({template})"))
                except:
                    pass
        self.facts_loaded = False

    def query(self, query_str):
        if not self.facts_loaded:
            print("Warning: No knowledge base loaded. Results may be empty.")
        try:
            return list(self.prolog.query(query_str))
        except Exception as e:
            print(f"Error executing query: {query_str}")
            print(f"Error message: {str(e)}")
            return []

    def count_solutions(self, query_str):
        try:
            solutions = list(self.prolog.query(query_str))
            return len(solutions)
        except Exception as e:
            print(f"Error counting solutions for query: {query_str}")
            print(f"Error message: {str(e)}")
            return 0

    def answer_question(self, question_type, params):
        if not self.facts_loaded:
            return "No scene data loaded. Please analyze an image first."

        if question_type == 'count':
            base_query = "object(X)"
            constraints = []
            if 'color' in params:
                constraints.append(f"color(X, {params['color']})")
            if 'shape' in params:
                constraints.append(f"shape(X, {params['shape']})")
            if 'material' in params:
                constraints.append(f"material(X, {params['material']})")
            if 'size' in params:
                constraints.append(f"size(X, {params['size']})")
            if 'relation' in params and 'rel_object' in params:
                rel = params['relation']
                rel_obj_query = "object(Y)"
                rel_constraints = []
                if 'rel_color' in params:
                    rel_constraints.append(f"color(Y, {params['rel_color']})")
                if 'rel_shape' in params:
                    rel_constraints.append(f"shape(Y, {params['rel_shape']})")
                if 'rel_material' in params:
                    rel_constraints.append(f"material(Y, {params['rel_material']})")
                if 'rel_size' in params:
                    rel_constraints.append(f"size(Y, {params['rel_size']})")
                rel_query = rel_obj_query
                if rel_constraints:
                    rel_query += ", " + ", ".join(rel_constraints)
                constraints.append(f"{rel}(X, Y)")
                constraints.append(rel_query)
            query = base_query
            if constraints:
                query += ", " + ", ".join(constraints)
            print(f"Executing Prolog query: {query}")
            count = self.count_solutions(query)
            return count

        elif question_type == 'exist':
            base_query = "object(X)"
            constraints = []
            if 'color' in params:
                constraints.append(f"color(X, {params['color']})")
            if 'shape' in params:
                constraints.append(f"shape(X, {params['shape']})")
            if 'material' in params:
                constraints.append(f"material(X, {params['material']})")
            if 'size' in params:
                constraints.append(f"size(X, {params['size']})")
            if 'relation' in params and 'rel_object' in params:
                rel = params['relation']
                rel_obj_query = "object(Y)"
                rel_constraints = []
                if 'rel_color' in params:
                    rel_constraints.append(f"color(Y, {params['rel_color']})")
                if 'rel_shape' in params:
                    rel_constraints.append(f"shape(Y, {params['rel_shape']})")
                if 'rel_material' in params:
                    rel_constraints.append(f"material(Y, {params['rel_material']})")
                if 'rel_size' in params:
                    rel_constraints.append(f"size(Y, {params['rel_size']})")
                rel_query = rel_obj_query
                if rel_constraints:
                    rel_query += ", " + ", ".join(rel_constraints)
                constraints.append(f"{rel}(X, Y)")
                constraints.append(rel_query)
            query = base_query
            if constraints:
                query += ", " + ", ".join(constraints)
            print(f"Executing Prolog query: {query}")
            exists = self.count_solutions(query) > 0
            return exists

        elif question_type == 'query_attribute':
            base_query = f"object(X), {params['attribute']}(X, Value)"
            constraints = []
            if 'color' in params and params['attribute'] != 'color':
                constraints.append(f"color(X, {params['color']})")
            if 'shape' in params and params['attribute'] != 'shape':
                constraints.append(f"shape(X, {params['shape']})")
            if 'material' in params and params['attribute'] != 'material':
                constraints.append(f"material(X, {params['material']})")
            if 'size' in params and params['attribute'] != 'size':
                constraints.append(f"size(X, {params['size']})")
            if 'relation' in params and 'rel_object' in params:
                rel = params['relation']
                rel_obj_query = "object(Y)"
                rel_constraints = []
                if 'rel_color' in params:
                    rel_constraints.append(f"color(Y, {params['rel_color']})")
                if 'rel_shape' in params:
                    rel_constraints.append(f"shape(Y, {params['rel_shape']})")
                if 'rel_material' in params:
                    rel_constraints.append(f"material(Y, {params['rel_material']})")
                if 'rel_size' in params:
                    rel_constraints.append(f"size(Y, {params['rel_size']})")
                rel_query = rel_obj_query
                if rel_constraints:
                    rel_query += ", " + ", ".join(rel_constraints)
                constraints.append(f"{rel}(X, Y)")
                constraints.append(rel_query)
            query = base_query
            if constraints:
                query += ", " + ", ".join(constraints)
            print(f"Executing Prolog query: {query}")
            solutions = self.query(query)
            if solutions:
                return [solution['Value'] for solution in solutions]
            return None

        elif question_type == 'chain_reasoning':
            if 'steps' not in params or not params['steps']:
                return "No reasoning steps provided in the query."
            target_attr = params.get('target_attribute', 'color')
            steps = params['steps']
            current_bindings = self.query("object(X)")
            if not current_bindings:
                return "No objects found in the scene."
            for i, step in enumerate(steps):
                relation = step.get('relation')
                obj_constraints = [f"{attr}(X, {value})" for attr, value in step.get('object', {}).items()]
                if i == 0:
                    query_parts = ["object(X)"] + obj_constraints
                    query = ", ".join(query_parts)
                    results = self.query(query)
                    if not results:
                        return f"No objects found matching the description: {' and '.join(obj_constraints)}"
                    filtered_objects = [res['X'] for res in results]
                else:
                    new_objects = []
                    for prev_obj in filtered_objects:
                        rel_query = f"object(Y), {relation}(Y, {prev_obj})"
                        if obj_constraints:
                            rel_constraints = [c.replace('X', 'Y') for c in obj_constraints]
                            rel_query += ", " + ", ".join(rel_constraints)
                        rel_results = self.query(rel_query)
                        new_objects.extend([res['Y'] for res in rel_results])
                    filtered_objects = new_objects
                    if not filtered_objects:
                        return f"No objects found for step {i+1} in the reasoning chain."
            if filtered_objects:
                final_obj = filtered_objects[0]
                attr_query = f"{target_attr}({final_obj}, Value)"
                attr_results = self.query(attr_query)
                if attr_results:
                    return attr_results[0]['Value']
                return f"Could not determine the {target_attr} of the object."
            return "No objects match all the criteria in the reasoning chain."

        elif question_type == 'between':
            constraints1 = ["object(X)"]
            constraints2 = ["object(Z)"]
            if 'ref1_color' in params:
                constraints1.append(f"color(X, {params['ref1_color']})")
            if 'ref1_shape' in params:
                constraints1.append(f"shape(X, {params['ref1_shape']})")
            if 'ref1_material' in params:
                constraints1.append(f"material(X, {params['ref1_material']})")
            if 'ref1_size' in params:
                constraints1.append(f"size(X, {params['ref1_size']})")
            if 'ref2_color' in params:
                constraints2.append(f"color(Z, {params['ref2_color']})")
            if 'ref2_shape' in params:
                constraints2.append(f"shape(Z, {params['ref2_shape']})")
            if 'ref2_material' in params:
                constraints2.append(f"material(Z, {params['ref2_material']})")
            if 'ref2_size' in params:
                constraints2.append(f"size(Z, {params['ref2_size']})")
            ref1_query = ", ".join(constraints1)
            ref2_query = ", ".join(constraints2)
            between_query = f"{ref1_query}, {ref2_query}, between(Y, X, Z), object(Y)"
            count = self.count_solutions(between_query)
            return count

        return f"Unknown question type: {question_type}"
