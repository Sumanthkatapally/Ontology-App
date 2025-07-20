#!/usr/bin/env python3
"""
survey_to_ontology.py

Reads a JSON survey dictionary and generates a Pydantic + Enum-based
ontology module (survey_ontology.py) with one Enum class per question,
one BaseModel per category, a Respondent model, and the SurveyOntology class
with field mappings and utility functions.
"""

import json
import re
from pathlib import Path
from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Any

# ─── Helpers ───────────────────────────────────────────────────────────────

def to_enum_member(label: str) -> str:
   """
   Convert an arbitrary label into a valid UPPER_SNAKE_CASE Enum member name.
   """
   s = re.sub(r'[^0-9a-zA-Z]+', '_', label.strip())
   s = re.sub(r'_{2,}', '_', s).strip('_')
   if not s:
       s = "UNKNOWN"
   if s[0].isdigit():
       s = '_' + s
   return s.upper()


def to_field_name(column: str) -> str:
   """
   Convert a column key into a snake_case Python identifier.
   """
   s = re.sub(r'[^0-9a-zA-Z]+', '_', column.strip())
   s = re.sub(r'_{2,}', '_', s).strip('_').lower()
   if not s:
       s = 'field'
   if s[0].isdigit():
       s = '_' + s
   return s


def to_class_name(name: str) -> str:
   """
   Convert an arbitrary string into a valid CamelCase class name.
   """
   parts = re.findall(r'[0-9A-Za-z]+', name)
   if not parts:
       return 'Unnamed'
   return ''.join(p.capitalize() for p in parts)


def generate_descriptive_name(column: str, meta: Dict[str, Any]) -> str:
   """
   Generate a descriptive field name using the class name key instead of full description.
   """
   # Use the generated class name as the descriptive field name
   class_name = meta.get('generated_class_name', '')
   if class_name:
       # Convert class name to snake_case
       return to_field_name(class_name)
   
   # Fallback to column name if no class name
   return to_field_name(column)

def generate_ontology_from_json(input_json_path: str, output_py_path: str):
    """
    Generate ontology from JSON file and save to specified output path.
    
    Args:
        input_json_path (str): Path to the input JSON file
        output_py_path (str): Path where the ontology Python file should be saved
    """
    
    # ─── Load survey dictionary ─────────────────────────────────────────────────

    INPUT_JSON = Path(input_json_path)
    if not INPUT_JSON.exists():
       raise FileNotFoundError(f"{INPUT_JSON!r} not found.")

    survey: Dict[str, Dict[str, Any]] = json.loads(INPUT_JSON.read_text())

    # ─── Group entries by category ───────────────────────────────────────────────

    by_category: Dict[str, Dict[str, Dict[str, Any]]] = {}
    field_name_mapping = {}
    field_descriptions = {}
    field_enum_values = {}  # Store enum values for each field
    category_mappings = {
       'Demographics': 'HAS_DEMOGRAPHIC',
       'PoliticalOpinions': 'HAS_POLITICAL_OPINION', 
       'LifestyleAndBehavioralOpinions': 'HAS_LIFESTYLE_BEHAVIOR',
       'ValuesAndSocialIssues': 'HAS_VALUES_SOCIAL_ISSUES',
       'PublicPolicyCivicEngagement': 'HAS_PUBLIC_POLICY',
       'TechnicalSurveyMetadata': 'HAS_METADATA'
    }

    for col, meta in survey.items():
       category_raw = meta.get('category', 'Uncategorized')
       by_category.setdefault(category_raw, {})[col] = meta
       
       # Generate field name mapping using class name instead of description
       descriptive_name = generate_descriptive_name(col, meta)
       field_name_mapping[col.upper()] = descriptive_name
       
       # Store enum values if they exist
       values = meta.get('labelled_values')
       if isinstance(values, dict):
           # Use the actual string values from the survey dictionary, not the enum member names
           enum_values = list(values.values())
           field_enum_values[descriptive_name] = enum_values
       
       # Fix field descriptions - handle potential None values and encoding issues
       label = meta.get('column_label', '')
       if label:
           # Clean up any problematic characters in descriptions
           clean_label = str(label).replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")
           field_descriptions[col.upper()] = clean_label
       else:
           field_descriptions[col.upper()] = col.lower().replace('_', ' ')

    # ─── Generate lines of the output module ────────────────────────────────────

    output_lines = [
       '# Auto-generated survey ontology module',
       'from enum import Enum',
       'from pydantic import BaseModel, Field',
       'from typing import Dict, List, Set, Tuple, Any',
       '',
       '# Respondent model',
       'class Respondent(BaseModel):',
       '    respondent_id: str = Field(..., description="Unique respondent identifier")',
       '',
    ]

    # 1) Enums
    for category_raw, entries in by_category.items():
       for col, meta in entries.items():
           enum_name = to_class_name(meta['generated_class_name'])
           label = meta.get('column_label', '')
           values = meta.get('labelled_values')
           if isinstance(values, dict):
               output_lines.append(f'class {enum_name}(str, Enum):')
               escaped_label = str(label).replace('\\', '\\\\').replace('"', '\\"') if label else ''
               output_lines.append(f'    """{escaped_label}"""')
               for code, text in values.items():
                   member = to_enum_member(text)
                   escaped_text = str(text).replace('\\', '\\\\').replace("'", "\\'")
                   output_lines.append(f'    {member} = {escaped_text!r}')
               output_lines.append('')

    # 2) BaseModels per category
    for category_raw, entries in by_category.items():
       class_name = to_class_name(category_raw)
       output_lines.append(f'class {class_name}(BaseModel):')
       for col, meta in entries.items():
           field_name = to_field_name(col)
           enum_name = to_class_name(meta['generated_class_name'])
           values = meta.get('labelled_values')
           desc = meta.get('column_label', '')
           escaped_desc = str(desc).replace('\\', '\\\\').replace('"', '\\"') if desc else ''
           type_hint = enum_name if isinstance(values, dict) else 'str'
           output_lines.append(
               f'    {field_name}: {type_hint} = Field(..., description="{escaped_desc}")'
           )
       output_lines.append('')

    # 3) Generate SurveyOntology class with all mappings and functions
    output_lines.extend([
       'class SurveyOntology:',
       '    """Enhanced survey domain ontology definition and mapping utilities"""',
       '',
    ])

    # Generate field sets by category
    for category_raw, entries in by_category.items():
       class_name = to_class_name(category_raw)
       field_set_name = f'{class_name.upper()}_FIELDS'
       
       fields = [f"'{col}'" for col in entries.keys()]
       output_lines.append(f'    {field_set_name} = {{')
       output_lines.extend([f'        {field},' for field in fields])
       output_lines.append('    }')
       output_lines.append('')

    # Generate relationship mappings
    output_lines.extend([
       '    # Relationship mappings',
       '    RELATIONSHIP_MAPPINGS = {',
    ])

    for category_raw, entries in by_category.items():
       class_name = to_class_name(category_raw)
       relationship_type = category_mappings.get(class_name, f'HAS_{class_name.upper()}')
       field_set_name = f'{class_name.upper()}_FIELDS'
       output_lines.append(f"        '{relationship_type}': {field_set_name},")

    output_lines.extend([
       '    }',
       '',
    ])

    # Generate entity types
    entity_types = ['Respondent'] + [to_class_name(cat) for cat in by_category.keys()]
    output_lines.extend([
       '    # Entity types for Neo4j labels',
       '    ENTITY_TYPES = {',
    ])
    for entity_type in entity_types:
       output_lines.append(f"        '{entity_type}',")
    output_lines.extend([
       '    }',
       '',
    ])

    # Generate field name mapping with proper escaping
    output_lines.extend([
       '    # Field name mappings (original -> descriptive)',
       '    FIELD_NAME_MAPPING = {',
    ])
    for original, descriptive in field_name_mapping.items():
       output_lines.append(f"        '{original}': '{descriptive}',")
    output_lines.extend([
       '    }',
       '',
    ])

    # Generate field descriptions with proper escaping
    output_lines.extend([
       '    # Field descriptions for better text generation',
       '    FIELD_DESCRIPTIONS = {',
    ])
    for field, description in field_descriptions.items():
       # Properly escape quotes in descriptions
       escaped_desc = str(description).replace('\\', '\\\\').replace('"', '\\"')
       output_lines.append(f'        "{field}": "{escaped_desc}",')
    output_lines.extend([
       '    }',
       '',
    ])

    # Generate node properties dictionary with descriptive field names
    output_lines.extend([
       '    # Node properties with descriptive field names (not codes)',
       '    NODE_PROPERTIES = {',
       '        "Respondent": ["respondent_id"],',
    ])

    # Add category-specific properties
    for category_raw, entries in by_category.items():
       class_name = to_class_name(category_raw)
       # Use descriptive field names instead of original column codes
       properties = [f'"{generate_descriptive_name(col, meta)}"' for col, meta in entries.items()]
       output_lines.append(f'        "{class_name}": [{", ".join(properties)}],')

    output_lines.extend([
       '    }',
       '',
    ])

    # Generate node properties values mapping from enum data
    output_lines.extend([
       '    # Node properties values mapping (field -> enum values)',
       '    NODE_PROPERTIES_VALUES = {',
    ])
    for field_name, enum_values in field_enum_values.items():
       enum_list = ', '.join([f'"{value}"' for value in enum_values])
       output_lines.append(f'        "{field_name}": [{enum_list}],')
    output_lines.extend([
       '    }',
       '',
    ])

    # Generate relationship structure
    output_lines.extend([
       '    # Relationship structure: source node -> relationship -> target node',
       '    RELATIONSHIPS_STRUCTURE = {',
       '        "Respondent": {',
    ])

    # Add relationships from Respondent to each category
    for category_raw in by_category.keys():
       class_name = to_class_name(category_raw)
       relationship_type = category_mappings.get(class_name, f'HAS_{class_name.upper()}')
       output_lines.append(f'            "{relationship_type}": "{class_name}",')

    output_lines.extend([
       '        }',
       '    }',
       '',
    ])

    # Generate utility methods
    output_lines.extend([
       '    @classmethod',
       '    def get_category_for_field(cls, field_name: str) -> str:',
       '        """Determine which category a CSV field belongs to"""',
       '        field_upper = field_name.upper()',
       '        ',
       '        for relationship, fields in cls.RELATIONSHIP_MAPPINGS.items():',
       '            if field_upper in fields:',
       '                return relationship',
       '        ',
       '        return "UNKNOWN"',
       '',
       '    @classmethod',
       '    def get_all_survey_fields(cls) -> Set[str]:',
       '        """Get all known survey fields"""',
       '        all_fields = set()',
       '        for fields in cls.RELATIONSHIP_MAPPINGS.values():',
       '            all_fields.update(fields)',
       '        return all_fields',
       '',
       '    @classmethod',
       '    def get_category_class_name(cls, relationship_type: str) -> str:',
       '        """Map relationship type to category class name"""',
       '        mapping = {',
    ])

    for category_raw in by_category.keys():
       class_name = to_class_name(category_raw)
       relationship_type = category_mappings.get(class_name, f'HAS_{class_name.upper()}')
       output_lines.append(f"            '{relationship_type}': '{class_name}',")

    output_lines.extend([
       '        }',
       '        return mapping.get(relationship_type, "Response")',
       '',
       '    @classmethod',
       '    def get_descriptive_field_name(cls, original_field: str) -> str:',
       '        """Convert original field name to descriptive name"""',
       '        return cls.FIELD_NAME_MAPPING.get(original_field.upper(), original_field.lower())',
       '',
       '    @classmethod',
       '    def get_field_description(cls, field_name: str) -> str:',
       '        """Get human-readable description for a field"""',
       '        return cls.FIELD_DESCRIPTIONS.get(field_name.upper(), field_name.lower().replace("_", " "))',
       '',
       '    @classmethod',
       '    def get_node_properties(cls, node_type: str) -> List[str]:',
       '        """',
       '        Get available properties for a node type.',
       '        ',
       '        Args:',
       '            node_type: The type of node to get properties for (e.g., "Demographics")',
       '            ',
       '        Returns:',
       '            List of property names for the specified node type',
       '        """',
       '        return cls.NODE_PROPERTIES.get(node_type, [])',
       '',
       '    @classmethod',
       '    def get_relationship_targets(cls, source_node: str) -> Dict[str, str]:',
       '        """',
       '        Get possible relationships and target node types for a source node type.',
       '        ',
       '        Args:',
       '            source_node: The source node type to get relationships for',
       '            ',
       '        Returns:',
       '            Dictionary mapping relationship names to target node types',
       '        """',
       '        return cls.RELATIONSHIPS_STRUCTURE.get(source_node, {})',
       '',
       '    @classmethod',
       '    def get_reverse_relationships(cls) -> Dict[str, List[Dict[str, str]]]:',
       '        """',
       '        Generate reverse relationship mappings for easy lookup.',
       '        ',
       '        Converts the forward relationship structure to a reverse lookup table',
       '        that allows finding which node types can connect to a given node type.',
       '        ',
       '        Returns:',
       '            Dictionary mapping target node types to lists of source types and relationships',
       '        """',
       '        reverse_rels = {}',
       '        for source, rels in cls.RELATIONSHIPS_STRUCTURE.items():',
       '            for rel, target in rels.items():',
       '                if target not in reverse_rels:',
       '                    reverse_rels[target] = []',
       '                reverse_rels[target].append({"source": source, "relationship": rel})',
       '        return reverse_rels',
       '',
       '    @classmethod',
       '    def validate_csv_fields(cls, csv_columns: List[str]) -> Dict[str, Any]:',
       '        """Validate CSV columns against ontology and categorize them"""',
       '        validation_result = {',
       '            "known_fields": {},',
       '            "unknown_fields": [],',
       '            "missing_fields": [],',
       '            "total_known": 0,',
       '            "total_unknown": 0',
       '        }',
       '        ',
       '        csv_upper = [col.upper() for col in csv_columns]',
       '        known_fields = cls.get_all_survey_fields()',
       '        ',
       '        for col in csv_columns:',
       '            category = cls.get_category_for_field(col)',
       '            if category != "UNKNOWN":',
       '                if category not in validation_result["known_fields"]:',
       '                    validation_result["known_fields"][category] = []',
       '                validation_result["known_fields"][category].append(col)',
       '                validation_result["total_known"] += 1',
       '            else:',
       '                validation_result["unknown_fields"].append(col)',
       '                validation_result["total_unknown"] += 1',
       '        ',
       '        # Check for missing expected fields',
       '        for field in known_fields:',
       '            if field not in csv_upper:',
       '                validation_result["missing_fields"].append(field)',
       '        ',
       '        return validation_result',
       '',
       '    @classmethod',
       '    def get_schema_summary(cls) -> Dict[str, Any]:',
       '        """Get a summary of the ontology schema"""',
       '        return {',
       '            "total_categories": len(cls.RELATIONSHIP_MAPPINGS),',
       '            "total_fields": len(cls.get_all_survey_fields()),',
       '            "categories": {',
       '                relationship: len(fields)',
       '                for relationship, fields in cls.RELATIONSHIP_MAPPINGS.items()',
       '            },',
       '            "entity_types": list(cls.ENTITY_TYPES)',
       '        }',
       '',
    ])

    # Add the get_ontology function at module level
    output_lines.extend([
       '',
       'def get_ontology() -> Dict[str, Any]:',
       '    """',
       '    Get information about the survey knowledge graph schema.',
       '    ',
       '    Returns:',
       '        Dictionary with node types, their properties, and relationship structure',
       '    """',
       '    # Get all node types from entity types and categories',
       '    node_types = list(SurveyOntology.ENTITY_TYPES)',
       '    ',
       '    # Get all relationship types',
       '    relationship_types = list(SurveyOntology.RELATIONSHIP_MAPPINGS.keys())',
       '    relationship_types.extend(["TOOK", "ANSWERED", "CONTAINS", "GENERATES", "BELONGS_TO"])',
       '    relationship_types = list(set(relationship_types))  # Remove duplicates',
       '    ',
       '    return {',
       '        "node_types": node_types,',
       '        "relationship_types": relationship_types,',
       '        "node_properties": SurveyOntology.NODE_PROPERTIES,',
       '        "node_properties_values": SurveyOntology.NODE_PROPERTIES_VALUES,',
       '        "relationship_structure": SurveyOntology.RELATIONSHIPS_STRUCTURE,',
       '    }',
       '',
    ])

    # Generate the SURVEY_ONTOLOGY dictionary structure as requested
    output_lines.extend([
       '',
       '# Generate SURVEY_ONTOLOGY dictionary',
       'def _generate_survey_ontology() -> Dict[str, Any]:',
       '    """Generate the SURVEY_ONTOLOGY dictionary from the SurveyOntology class data"""',
       '    ',
       '    # Generate NODE_TYPES list',
       '    node_types = list(SurveyOntology.ENTITY_TYPES)',
       '    ',
       '    # Generate RELATIONSHIP_TYPES mapping',
       '    relationship_types = {}',
       '    for relationship, fields in SurveyOntology.RELATIONSHIP_MAPPINGS.items():',
       '        # Convert field set to field set name',
       '        category_name = SurveyOntology.get_category_class_name(relationship)',
       '        field_set_name = f"{category_name.upper()}_FIELDS"',
       '        relationship_types[relationship] = field_set_name',
       '    ',
       '    return {',
       '        "NODE_TYPES": node_types,',
       '        "RELATIONSHIP_TYPES": relationship_types,',
       '        "NODE_PROPERTIES": SurveyOntology.NODE_PROPERTIES,',
       '        "NODE_PROPERTIES_VALUES": SurveyOntology.NODE_PROPERTIES_VALUES,',
       '        "RELATIONSHIPS_STRUCTURE": SurveyOntology.RELATIONSHIPS_STRUCTURE',
       '    }',
       '',
       '',
       '# Create the SURVEY_ONTOLOGY dictionary',
       'SURVEY_ONTOLOGY = _generate_survey_ontology()',
       '',
    ])

    # Write to file with UTF-8 encoding to handle special characters
    OUTPUT_PY = Path(output_py_path)
    try:
       OUTPUT_PY.write_text("\n".join(output_lines), encoding='utf-8')
       print(f'Generated {OUTPUT_PY.resolve()}')
       print(f'Generated {len(field_name_mapping)} field name mappings')
       print(f'Generated {len(by_category)} category mappings')
       print(f'Generated {len(field_enum_values)} field enum value mappings')
       print('SurveyOntology class includes all utility methods and field mappings')
       print('Added SURVEY_ONTOLOGY dictionary in the requested format')
       print('✅ get_ontology() function generated at module level')
       
       # Print a few example mappings to verify
       print('\nExample field name mappings:')
       for i, (original, descriptive) in enumerate(list(field_name_mapping.items())[:5]):
           print(f'  {original} -> {descriptive}')
       
       # Print enum values mappings for debugging
       if field_enum_values:
           print('\nExample enum value mappings:')
           for i, (field, values) in enumerate(list(field_enum_values.items())[:3]):
               print(f'  {field} -> {values}')
       else:
           print('\nWarning: No enum values found in survey data!')
       
       return str(OUTPUT_PY)
       
    except Exception as e:
       print(f'Error writing file: {e}')
       print('Falling back to basic encoding...')
       # Fallback with error handling
       safe_content = "\n".join(output_lines).encode('utf-8', errors='replace').decode('utf-8')
       OUTPUT_PY.write_text(safe_content)
       print(f'Generated {OUTPUT_PY.resolve()} (with encoding fallback)')
       return str(OUTPUT_PY)

def main():
    # ─── Load survey dictionary ─────────────────────────────────────────────────

    INPUT_JSON = Path("D:\MARIST\Research\Environment\myenv\Project\KnowledgeGraph\src\class_names_direct.json")
    OUTPUT_PY = Path('D:\MARIST\Research\Environment\myenv\Project\KnowledgeGraph\ontology\survey_ontology.py')
    
    generate_ontology_from_json(str(INPUT_JSON), str(OUTPUT_PY))

if __name__ == "__main__":
    main()