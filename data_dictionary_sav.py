#data_dictionary_sav.py
#Data dictionary extraction from SAV files for ontology generation
import pyreadstat
import json

def create_SAS_metadata_dict(file_path, output_file='survey_dictionary.json'):
    """
    Extract metadata from SAV file in format optimized for ontology generation.
    """
    print(f"📖 Reading SAV file: {file_path}")
    
    try:
        df, metadata = pyreadstat.read_sav(file_path)  # Fixed: read_sav instead of read_SAS
        print(f"✅ Successfully read {len(df)} rows, {len(df.columns)} columns")
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None
    
    metadata_dict = {}
    columns_with_labels = 0
    
    for index, column_name in enumerate(metadata.column_names):
        # Get labelled values - handle different formats
        labelled_values = metadata.variable_value_labels.get(column_name, "No labelled values")
        
        # Convert numeric keys to strings (required for JSON)
        if isinstance(labelled_values, dict):
            labelled_values = {str(k): str(v) for k, v in labelled_values.items()}
            columns_with_labels += 1
        
        column_info = {
            "column_name": column_name,
            "column_type": metadata.original_variable_types.get(column_name, "Unknown"),
            "column_label": metadata.column_labels[index] if (index < len(metadata.column_labels) and metadata.column_labels[index]) else "No label",
            "labelled_values": labelled_values
        }
        
        metadata_dict[column_name] = column_info
    
    # Print summary
    print(f"📊 Total columns: {len(metadata_dict)}")
    print(f"📊 Columns with labelled values: {columns_with_labels}")
    print(f"📊 Columns without labels: {len(metadata_dict) - columns_with_labels}")
    
    # Show sample columns
    sample_columns = list(metadata_dict.keys())[:5]
    print(f"📋 Sample columns: {sample_columns}")
    
    # Save to file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved metadata to: {output_file}")
        
        # Validate JSON
        with open(output_file, 'r', encoding='utf-8') as f:
            test_load = json.load(f)
        print(f"✅ JSON validation successful")
        
        return metadata_dict
        
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return metadata_dict

def preview_ontology_candidates(metadata_dict, max_preview=10):
    """Preview which columns will likely be used for ontology generation."""
    
    if not metadata_dict:
        return
    
    demographic_keywords = [
        "age", "gender", "race", "ethnic", "education", "marital", "income", 
        "birth", "residence", "state", "county", "zip", "household", "employment"
    ]
    
    print(f"\n🔍 ONTOLOGY PREVIEW (showing first {max_preview} relevant columns):")
    print("=" * 80)
    
    demographic_cols = []
    opinion_cols = []
    other_cols = []
    
    for col_name, col_data in metadata_dict.items():
        has_labels = isinstance(col_data.get("labelled_values"), dict)
        
        if not has_labels:
            continue
            
        label = (col_data.get("column_label") or "").lower()
        name = (col_data.get("column_name") or "").lower()
        
        is_demographic = any(keyword in label or keyword in name for keyword in demographic_keywords)
        
        if is_demographic:
            demographic_cols.append(col_name)
        else:
            opinion_cols.append(col_name)
    
    print(f"📍 DEMOGRAPHIC COLUMNS ({len(demographic_cols)}):")
    for col in demographic_cols[:max_preview]:
        label = metadata_dict[col]["column_label"]
        values = list(metadata_dict[col]["labelled_values"].keys())[:3]
        print(f"   • {col}: {label}")
        print(f"     Values: {values}{'...' if len(metadata_dict[col]['labelled_values']) > 3 else ''}")
    
    print(f"\n📍 OPINION/SURVEY COLUMNS ({len(opinion_cols)}):")
    for col in opinion_cols[:max_preview]:
        label = metadata_dict[col]["column_label"]
        values = list(metadata_dict[col]["labelled_values"].keys())[:3]
        print(f"   • {col}: {label}")
        print(f"     Values: {values}{'...' if len(metadata_dict[col]['labelled_values']) > 3 else ''}")
    
    print(f"\n🎯 READY FOR ONTOLOGY GENERATION:")
    print(f"   Total usable columns: {len(demographic_cols) + len(opinion_cols)}")
    print(f"   Expected Demographics model fields: {len(demographic_cols)}")
    print(f"   Expected Opinions model fields: {len(opinion_cols)}")

if __name__ == "__main__":
    # Your file path
    file_path = r'D:\MARIST\Research\Environment\myenv\Project\KnowledgeGraph\data\US_SURVEY\US20240903_FINAL.sav'
    output_file = r'D:\MARIST\Research\Environment\myenv\Project\KnowledgeGraph\data\US_SURVEY\survey_dictionary.json'
    
    print("🚀 Starting SAV metadata extraction...")
    
    # Extract metadata
    metadata = create_SAS_metadata_dict(file_path, output_file)
    
    if metadata:
        # Preview what the ontology will look like
        preview_ontology_candidates(metadata)
        
        print(f"\n🎉 SUCCESS! Ready to generate ontology:")
        print(f"   1. Deploy Modal service: modal deploy modal_ontology_service.py")
        print(f"   2. Generate ontology: python modal_client.py {output_file} my_ontology.py")
        
    else:
        print("❌ Failed to extract metadata")