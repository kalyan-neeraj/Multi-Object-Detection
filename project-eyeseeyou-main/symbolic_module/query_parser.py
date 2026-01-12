# symbolic_module/query_parser.py
import re
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class QueryParser:
    """
    Parses natural language questions into structured queries for the symbolic reasoner
    """
    def __init__(self, config):
        self.config = config
        
        # Define keywords for different question types
        self.count_keywords = ['how many', 'count']
        self.exist_keywords = ['is there', 'are there', 'exist', 'does']
        self.attribute_keywords = ['what is', "what's", 'what color', 'what shape', 'what material', 'what size']
        self.complex_keywords = ['what is the color of the object', 'which object', 'find the']
        
        # Define attribute keywords
        self.attribute_map = {
            'color': ['color', 'colored'],
            'shape': ['shape', 'cube', 'sphere', 'cylinder'],
            'material': ['material', 'made of', 'metallic', 'rubber', 'metal'],
            'size': ['size', 'large', 'small']
        }
        
        # Define spatial relationship keywords
        self.spatial_map = {
            'left_of': ['left of', 'to the left of'],
            'right_of': ['right of', 'to the right of'],
            'in_front_of': ['in front of', 'closer than', 'nearest to'],
            'behind': ['behind', 'further than', 'furthest from']
        }
        
        # Compile regex patterns for attributes
        self.color_pattern = re.compile(r'\b(' + '|'.join(self.config.COLORS) + r')\b')
        self.shape_pattern = re.compile(r'\b(' + '|'.join(self.config.SHAPES) + r')\b')
        self.material_pattern = re.compile(r'\b(' + '|'.join(self.config.MATERIALS) + r')\b')
        self.size_pattern = re.compile(r'\b(' + '|'.join(self.config.SIZES) + r')\b')
    
    def parse_question(self, question):
        """
        Parse a natural language question into a structured query
        
        Args:
            question: String containing the natural language question
            
        Returns:
            Dictionary with question type and parameters
        """
        # Convert to lowercase
        question = question.lower()
        
        # Determine question type
        question_type = self._determine_question_type(question)
        
        # Extract parameters based on question type
        if question_type == 'count':
            params = self._extract_count_params(question)
        elif question_type == 'exist':
            params = self._extract_exist_params(question)
        elif question_type == 'query_attribute':
            params = self._extract_attribute_params(question)
        elif question_type == 'chain_reasoning':
            params = self._extract_chain_params(question)
        elif question_type == 'between':
            params = self._extract_between_params(question)
        elif question_type == 'complex_spatial':
            params = self._extract_complex_spatial_params(question)
        else:
            params = {}
        
        return {
            'type': question_type,
            'params': params,
            'original_question': question
        }
    
    def _determine_question_type(self, question):
        """Determine the type of question"""
        # Check for complex reasoning patterns first
        for pattern in [
            r'what (is|are) the .+ of the .+ that (is|are) (to the )?(right|left|front|behind)',
            r'which object (is|are) .+ (of|from) the',
            r'between .+ and'
        ]:
            if re.search(pattern, question):
                # If question involves objects between other objects
                if 'between' in question:
                    return 'between'
                # Otherwise, it's a chain reasoning question
                return 'chain_reasoning'
        
        # Check for complex spatial patterns
        for pattern in ['row', 'surrounding', 'group', 'arrangement']:
            if pattern in question:
                return 'complex_spatial'
        
        # Check for count questions
        for keyword in self.count_keywords:
            if keyword in question:
                return 'count'
        
        # Check for existence questions
        for keyword in self.exist_keywords:
            if keyword in question:
                return 'exist'
        
        # Check for attribute questions
        for keyword in self.attribute_keywords:
            if keyword in question:
                return 'query_attribute'
        
        # Default
        return 'unknown'
    
    def _extract_count_params(self, question):
        """Extract parameters for count questions"""
        params = {}
        
        # Extract attributes directly using regex patterns
        color_matches = self.color_pattern.findall(question)
        if color_matches:
            params['color'] = color_matches[0]
        
        shape_matches = self.shape_pattern.findall(question)
        if shape_matches:
            params['shape'] = shape_matches[0]
        
        material_matches = self.material_pattern.findall(question)
        if material_matches:
            params['material'] = material_matches[0]
        
        size_matches = self.size_pattern.findall(question)
        if size_matches:
            params['size'] = size_matches[0]
        
        # Extract spatial relationships
        for rel_type, keywords in self.spatial_map.items():
            for keyword in keywords:
                if keyword in question:
                    # Split the question at the relation keyword
                    parts = question.split(keyword)
                    if len(parts) > 1:
                        params['relation'] = rel_type
                        params['rel_object'] = True
                        
                        # Extract attributes of the related object from the second part
                        rel_color = self.color_pattern.search(parts[1])
                        if rel_color:
                            params['rel_color'] = rel_color.group(0)
                        
                        rel_shape = self.shape_pattern.search(parts[1])
                        if rel_shape:
                            params['rel_shape'] = rel_shape.group(0)
                        
                        rel_material = self.material_pattern.search(parts[1])
                        if rel_material:
                            params['rel_material'] = rel_material.group(0)
                        
                        rel_size = self.size_pattern.search(parts[1])
                        if rel_size:
                            params['rel_size'] = rel_size.group(0)
        
        return params
    
    def _extract_exist_params(self, question):
        """Extract parameters for existence questions"""
        # Similar to count params but without spatial relationships
        params = {}
        
        # Extract attributes directly using regex patterns
        color_matches = self.color_pattern.findall(question)
        if color_matches:
            params['color'] = color_matches[0]
        
        shape_matches = self.shape_pattern.findall(question)
        if shape_matches:
            params['shape'] = shape_matches[0]
        
        material_matches = self.material_pattern.findall(question)
        if material_matches:
            params['material'] = material_matches[0]
        
        size_matches = self.size_pattern.findall(question)
        if size_matches:
            params['size'] = size_matches[0]
        
        # Check for spatial relationships
        for rel_type, keywords in self.spatial_map.items():
            for keyword in keywords:
                if keyword in question:
                    params['relation'] = rel_type
                    params['rel_object'] = True
                    
                    # Extract attributes of the related object
                    parts = question.split(keyword)
                    if len(parts) > 1:
                        # Check attributes in the second part
                        rel_color = self.color_pattern.search(parts[1])
                        if rel_color:
                            params['rel_color'] = rel_color.group(0)
                        
                        rel_shape = self.shape_pattern.search(parts[1])
                        if rel_shape:
                            params['rel_shape'] = rel_shape.group(0)
                        
                        rel_material = self.material_pattern.search(parts[1])
                        if rel_material:
                            params['rel_material'] = rel_material.group(0)
                        
                        rel_size = self.size_pattern.search(parts[1])
                        if rel_size:
                            params['rel_size'] = rel_size.group(0)
        
        return params
    
    def _extract_attribute_params(self, question):
        """Extract parameters for attribute query questions"""
        params = {}
        
        # Determine which attribute is being queried
        if 'what color' in question:
            params['attribute'] = 'color'
        elif 'what shape' in question:
            params['attribute'] = 'shape'
        elif 'what material' in question:
            params['attribute'] = 'material'
        elif 'what size' in question:
            params['attribute'] = 'size'
        else:
            # Default to a generic "what is" question - try to determine from context
            for attr in ['color', 'shape', 'material', 'size']:
                if attr in question:
                    params['attribute'] = attr
                    break
            
            if 'attribute' not in params:
                # Default to color if we can't determine
                params['attribute'] = 'color'
        
        # Extract constraints (attributes that are not being queried)
        if params['attribute'] != 'color':
            color_matches = self.color_pattern.findall(question)
            if color_matches:
                params['color'] = color_matches[0]
        
        if params['attribute'] != 'shape':
            shape_matches = self.shape_pattern.findall(question)
            if shape_matches:
                params['shape'] = shape_matches[0]
        
        if params['attribute'] != 'material':
            material_matches = self.material_pattern.findall(question)
            if material_matches:
                params['material'] = material_matches[0]
        
        if params['attribute'] != 'size':
            size_matches = self.size_pattern.findall(question)
            if size_matches:
                params['size'] = size_matches[0]
        
        # Check for spatial relationships
        for rel_type, keywords in self.spatial_map.items():
            for keyword in keywords:
                if keyword in question:
                    params['relation'] = rel_type
                    params['rel_object'] = True
                    
                    # Extract attributes of the related object
                    parts = question.split(keyword)
                    if len(parts) > 1:
                        rel_color = self.color_pattern.search(parts[1])
                        if rel_color:
                            params['rel_color'] = rel_color.group(0)
                        
                        rel_shape = self.shape_pattern.search(parts[1])
                        if rel_shape:
                            params['rel_shape'] = rel_shape.group(0)
                        
                        rel_material = self.material_pattern.search(parts[1])
                        if rel_material:
                            params['rel_material'] = rel_material.group(0)
                        
                        rel_size = self.size_pattern.search(parts[1])
                        if rel_size:
                            params['rel_size'] = rel_size.group(0)
        
        return params
    
    def _extract_chain_params(self, question):
        """Extract parameters for chain reasoning questions"""
        params = {
            'steps': []
        }
        
        # Example: "What is the color of the object that is to the right of the small cube?"
        
        # First, determine the queried attribute
        for attr in ['color', 'shape', 'material', 'size']:
            if f"what {attr}" in question or f"what is the {attr}" in question:
                params['target_attribute'] = attr
                break
        
        if 'target_attribute' not in params:
            params['target_attribute'] = 'color'  # Default
        
        # Find all spatial relations in the question
        relations = []
        for rel_type, keywords in self.spatial_map.items():
            for keyword in keywords:
                if keyword in question:
                    idx = question.find(keyword)
                    relations.append((idx, rel_type, keyword))
        
        # Sort relations by their position in the question (from end to beginning)
        relations.sort(reverse=True)
        
        # Extract object descriptions for each relation
        current_question = question
        for _, rel_type, keyword in relations:
            parts = current_question.split(keyword, 1)
            if len(parts) < 2:
                continue
            
            # The object after the relation keyword
            obj_desc = parts[1]
            
            # Extract attributes for this object
            obj_params = {}
            
            color_match = self.color_pattern.search(obj_desc)
            if color_match:
                obj_params['color'] = color_match.group(0)
            
            shape_match = self.shape_pattern.search(obj_desc)
            if shape_match:
                obj_params['shape'] = shape_match.group(0)
            
            material_match = self.material_pattern.search(obj_desc)
            if material_match:
                obj_params['material'] = material_match.group(0)
            
            size_match = self.size_pattern.search(obj_desc)
            if size_match:
                obj_params['size'] = size_match.group(0)
            
            # Add this step to the reasoning chain
            params['steps'].append({
                'relation': rel_type,
                'object': obj_params
            })
            
            # Update current_question to the part before this relation
            current_question = parts[0]
        
        return params
    
    def _extract_between_params(self, question):
        """Extract parameters for 'between' questions"""
        params = {}
        
        # Example: "How many objects are between the red cube and the blue sphere?"
        
        # Find the word "between" and split the question
        if "between" in question:
            parts = question.split("between", 1)
            if len(parts) < 2:
                return params
            
            # The part after "between" contains the two reference objects
            ref_part = parts[1]
            
            # Split by "and" to get the two reference objects
            if "and" in ref_part:
                ref_objs = ref_part.split("and", 1)
                
                # First reference object
                ref1 = ref_objs[0]
                color1 = self.color_pattern.search(ref1)
                if color1:
                    params['ref1_color'] = color1.group(0)
                
                shape1 = self.shape_pattern.search(ref1)
                if shape1:
                    params['ref1_shape'] = shape1.group(0)
                
                material1 = self.material_pattern.search(ref1)
                if material1:
                    params['ref1_material'] = material1.group(0)
                
                size1 = self.size_pattern.search(ref1)
                if size1:
                    params['ref1_size'] = size1.group(0)
                
                # Second reference object
                ref2 = ref_objs[1]
                color2 = self.color_pattern.search(ref2)
                if color2:
                    params['ref2_color'] = color2.group(0)
                
                shape2 = self.shape_pattern.search(ref2)
                if shape2:
                    params['ref2_shape'] = shape2.group(0)
                
                material2 = self.material_pattern.search(ref2)
                if material2:
                    params['ref2_material'] = material2.group(0)
                
                size2 = self.size_pattern.search(ref2)
                if size2:
                    params['ref2_size'] = size2.group(0)
        
        return params
    
    def _extract_complex_spatial_params(self, question):
        """Extract parameters for complex spatial reasoning questions"""
        params = {}
        
        # Look for specific patterns
        if "row" in question:
            params['pattern'] = 'row'
            
            # Check for additional constraints
            color_matches = self.color_pattern.findall(question)
            if color_matches:
                params['color'] = color_matches[0]
            
            shape_matches = self.shape_pattern.findall(question)
            if shape_matches:
                params['shape'] = shape_matches[0]
                
            # Check for count patterns
            count_match = re.search(r'(\d+|a|one|two|three|four|five) (objects|shapes)', question)
            if count_match:
                count_text = count_match.group(1)
                if count_text == 'a' or count_text == 'one':
                    params['count'] = 1
                elif count_text == 'two':
                    params['count'] = 2
                elif count_text == 'three':
                    params['count'] = 3
                elif count_text == 'four':
                    params['count'] = 4
                elif count_text == 'five':
                    params['count'] = 5
                else:
                    try:
                        params['count'] = int(count_text)
                    except:
                        params['count'] = 3  # Default for "row"
        
        elif "surrounding" in question or "surround" in question:
            params['pattern'] = 'surrounding'
            
            # Check what is being surrounded
            for attribute in ['color', 'shape', 'material', 'size']:
                pattern = re.compile(fr'(surrounded by|surrounding) (the|a) {attribute}')
                if pattern.search(question):
                    params['surrounded_attribute'] = attribute
        
        return params

    def convert_to_prolog_query(self, parsed_query):
        """
        Convert a parsed query to a Prolog query string
        
        Args:
            parsed_query: Dictionary with query type and parameters
            
        Returns:
            String containing a Prolog query
        """
        query_type = parsed_query['type']
        params = parsed_query.get('params', {})
        
        if query_type == 'count':
            base_query = "object(X)"
            constraints = []
            
            # Add attribute constraints
            for attr in ['color', 'shape', 'material', 'size']:
                if attr in params:
                    constraints.append(f"{attr}(X, {params[attr]})")
            
            # Add spatial relationship if specified
            if 'relation' in params and 'rel_object' in params:
                rel = params['relation']
                rel_obj_query = "object(Y)"
                
                # Add constraints for the related object
                rel_constraints = []
                for attr in ['color', 'shape', 'material', 'size']:
                    rel_attr = f'rel_{attr}'
                    if rel_attr in params:
                        rel_constraints.append(f"{attr}(Y, {params[rel_attr]})")
                
                # Build the relation constraint
                rel_query = rel_obj_query
                if rel_constraints:
                    rel_query += ", " + ", ".join(rel_constraints)
                
                constraints.append(f"{rel}(X, Y)")
                constraints.append(rel_query)
            
            # Build the final query
            query = base_query
            if constraints:
                query += ", " + ", ".join(constraints)
            
            return query
            
        elif query_type == 'exist':
            # Similar to count
            base_query = "object(X)"
            constraints = []
            
            # Add attribute constraints
            for attr in ['color', 'shape', 'material', 'size']:
                if attr in params:
                    constraints.append(f"{attr}(X, {params[attr]})")
            
            # Build the final query
            query = base_query
            if constraints:
                query += ", " + ", ".join(constraints)
            
            return query
            
        elif query_type == 'query_attribute':
            base_query = f"object(X), {params['attribute']}(X, Value)"
            constraints = []
            
            # Add constraints for other attributes
            for attr in ['color', 'shape', 'material', 'size']:
                if attr != params['attribute'] and attr in params:
                    constraints.append(f"{attr}(X, {params[attr]})")
            
            # Build the final query
            query = base_query
            if constraints:
                query += ", " + ", ".join(constraints)
            
            return query
            
        elif query_type == 'chain_reasoning':
            # For complex chain reasoning, we'll need to break it into steps
            # This is just a placeholder implementation
            if 'steps' not in params:
                return "object(X)"
            
            steps = []
            for i, step in enumerate(params['steps']):
                var1 = chr(88 + i)  # ASCII for 'X', 'Y', 'Z', etc.
                var2 = chr(88 + i + 1)
                
                constraints = [f"object({var1})"]
                
                # Add object constraints
                for attr, value in step.get('object', {}).items():
                    constraints.append(f"{attr}({var1}, {value})")
                
                # Add relation to the next object
                if i < len(params['steps']) - 1:
                    constraints.append(f"{step['relation']}({var2}, {var1})")
                    constraints.append(f"object({var2})")
                
                steps.append(", ".join(constraints))
            
            # Add the final target attribute query
            if 'target_attribute' in params:
                var = chr(88 + len(params['steps']))  # Last variable
                steps.append(f"{params['target_attribute']}({var}, Value)")
            
            return "), (".join(steps)
            
        elif query_type == 'between':
            # Create reference object queries
            ref1_constraints = ["object(X)"]
            ref2_constraints = ["object(Z)"]
            
            # Add constraints for reference objects
            for prefix, constraints in [('ref1_', ref1_constraints), ('ref2_', ref2_constraints)]:
                for attr in ['color', 'shape', 'material', 'size']:
                    key = prefix + attr
                    if key in params:
                        constraints.append(f"{attr}({'X' if prefix == 'ref1_' else 'Z'}, {params[key]})")
            
            # Build the query for objects between the references
            ref1_query = ", ".join(ref1_constraints)
            ref2_query = ", ".join(ref2_constraints)
            
            return f"{ref1_query}, {ref2_query}, between(Y, X, Z), object(Y)"
            
        elif query_type == 'complex_spatial':
            if 'pattern' in params:
                pattern = params['pattern']
                
                if pattern == 'row':
                    # Look for three objects in a row
                    constraints = ["object(X)", "object(Y)", "object(Z)", "row(X, Y, Z)"]
                    
                    # Add attribute constraints if specified
                    for attr in ['color', 'shape', 'material', 'size']:
                        if attr in params:
                            # Apply to all objects in the row
                            constraints.append(f"{attr}(X, {params[attr]})")
                            constraints.append(f"{attr}(Y, {params[attr]})")
                            constraints.append(f"{attr}(Z, {params[attr]})")
                    
                    return ", ".join(constraints)
                
                elif pattern == 'surrounding':
                    # Look for an object surrounded by others
                    return "object(X), surrounding(Y, X)"
            
        # Default case
        return "object(X)"