#!/usr/bin/env python3
"""
Enhanced Survey Ontology Pipeline with Step Selection
Usage:
    python survey_ontology_pipeline.py filename.sav                           # Run all steps
    python survey_ontology_pipeline.py filename.sav --step dictionary        # Only dictionary extraction
    python survey_ontology_pipeline.py filename.sav --step csv_conversion    # Only CSV conversion
    python survey_ontology_pipeline.py filename.sav --step classification    # Only classification
    python survey_ontology_pipeline.py filename.sav --step class_creation    # Only class creation
    python survey_ontology_pipeline.py filename.sav --step ontology          # Only ontology creation
    python survey_ontology_pipeline.py filename.sav --step step1             # Only step 1
    python survey_ontology_pipeline.py filename.sav --step step3             # Only step 3
    python survey_ontology_pipeline.py filename.sav --steps 1,3,5            # Steps 1, 3, and 5 only
"""

import os
import sys
import json
import shutil
from pathlib import Path
import subprocess
import tempfile
import importlib.util
import argparse

def setup_argparse():
    """Setup command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Survey Ontology Pipeline with Step Selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1. dictionary      - Extract survey dictionary from .sav file
  2. csv_conversion  - Convert .sav to CSV using survey dictionary  
  3. classification  - Classify survey questions into categories
  4. class_creation  - Generate class names for each question
  5. ontology        - Create final ontology Python module

Examples:
  python survey_ontology_pipeline.py survey.sav                              # Run all steps
  python survey_ontology_pipeline.py survey.sav --step classification       # Only classification
  python survey_ontology_pipeline.py survey.sav --step step3                # Only step 3
  python survey_ontology_pipeline.py survey.sav --steps 1,3,5               # Steps 1, 3, and 5
        """
    )
    
    parser.add_argument("filename", help="Path to .sav file")
    
    # Step selection options
    step_group = parser.add_mutually_exclusive_group()
    step_group.add_argument(
        "--step", 
        choices=["dictionary", "csv_conversion", "classification", "class_creation", "ontology", 
                "step1", "step2", "step3", "step4", "step5"],
        help="Run only a specific step"
    )
    step_group.add_argument(
        "--steps",
        help="Run multiple steps (comma-separated: 1,2,3 or dictionary,classification,ontology)"
    )
    
    # Additional options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--force", "-f", action="store_true", help="Force overwrite existing files")
    
    return parser

def normalize_step_names(steps):
    """Convert step names to standardized format"""
    step_mapping = {
        "dictionary": "step1",
        "step1": "step1",
        "1": "step1",
        
        "csv_conversion": "step2",
        "csv": "step2", 
        "step2": "step2",
        "2": "step2",
        
        "classification": "step3",
        "categories": "step3",
        "step3": "step3",
        "3": "step3",
        
        "class_creation": "step4",
        "classes": "step4",
        "step4": "step4", 
        "4": "step4",
        
        "ontology": "step5",
        "step5": "step5",
        "5": "step5"
    }
    
    normalized = []
    for step in steps:
        if step.lower() in step_mapping:
            normalized.append(step_mapping[step.lower()])
        else:
            print(f"⚠️ Unknown step: {step}")
    
    return list(set(normalized))  # Remove duplicates

def check_file_exists(filepath, step_name):
    """Check if required input file exists for a step"""
    if not os.path.exists(filepath):
        print(f"❌ Error: Required file for {step_name} not found: {filepath}")
        print(f"💡 Tip: Run previous steps first or check the file path")
        return False
    return True

def detect_file_type_and_setup(input_file_path):
    """Detect if input is SAV or intermediate JSON and setup accordingly"""
    input_path = Path(input_file_path)
    
    if input_path.suffix.lower() == '.sav':
        # Standard SAV file input
        return 'sav', input_path.parent
    elif input_path.suffix.lower() == '.json':
        # JSON file input - determine which step this represents
        if 'categorized' in input_path.name:
            # This is a categorized survey from step 3
            return 'categorized_json', input_path.parent.parent  # Go up to the main survey folder
        elif 'class_names' in input_path.name or 'with_classes' in input_path.name:
            # This is class names from step 4  
            return 'class_names_json', input_path.parent.parent
        elif 'survey_dictionary' in input_path.name:
            # This is survey dictionary from step 1
            return 'dictionary_json', input_path.parent.parent
        else:
            # Generic JSON, assume it's survey dictionary
            return 'dictionary_json', input_path.parent
    else:
        raise ValueError(f"Unsupported file type: {input_path.suffix}")

def setup_output_directories_smart(input_file_path: str, file_type: str):
    """Smart directory setup based on input file type"""
    input_path = Path(input_file_path)
    
    if file_type == 'sav':
        # Standard SAV workflow - create directories next to SAV file
        base_path = input_path.parent
    else:
        # JSON input - use existing directory structure
        if 'CategoryClassification' in str(input_path):
            # We're in CategoryClassification folder, go up to main folder
            base_path = input_path.parent.parent
        elif 'ClassCreation' in str(input_path):
            # We're in ClassCreation folder, go up to main folder  
            base_path = input_path.parent.parent
        elif 'SurveyDictionary' in str(input_path):
            # We're in SurveyDictionary folder, go up to main folder
            base_path = input_path.parent.parent
        else:
            # Default to parent directory
            base_path = input_path.parent
    
    print(f"📁 Using base directory: {base_path}")
    
    directories = [
        "SurveyDictionary",
        "CategoryClassification", 
        "ClassCreation",
        "Ontology"
    ]
    
    created_dirs = {}
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(exist_ok=True)
        created_dirs[directory] = dir_path
        print(f"📁 Verified directory: {dir_path}")
    
    return created_dirs

def execute_step4_class_creation_smart(input_file_path: str, file_type: str, output_dirs: dict):
    """Step 4: Generate class names - smart input detection"""
    print("\n" + "="*60)
    print("🚀 STEP 4: Class Name Generation")
    print("="*60)
    
    # Determine the correct categorized file path
    if file_type == 'categorized_json':
        # Input is already the categorized file
        categorized_file = input_file_path
        print(f"📂 Using provided categorized file: {categorized_file}")
    else:
        # Look for categorized file in standard location
        categorized_file = output_dirs["CategoryClassification"] / "categorized_survey.json"
        if not check_file_exists(categorized_file, "Step 4 (Class Creation)"):
            print(f"💡 Run Steps 1-3 first: --steps 1,2,3")
            return None
    
    # Load categorized survey data
    try:
        with open(categorized_file, 'r') as f:
            survey_data = json.load(f)
        survey_json = json.dumps(survey_data)
        print(f"✅ Loaded {len(survey_data)} questions from: {categorized_file}")
    except Exception as e:
        print(f"❌ Error loading categorized survey: {e}")
        return None
    
    # Import the class creation module
    try:
        spec = importlib.util.spec_from_file_location("class_creation_azure", "class_creation_azure.py")
        creation_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(creation_module)
    except Exception as e:
        print(f"❌ Error importing class_creation_azure.py: {e}")
        return None
    
    try:
        print("🤖 Starting Azure OpenAI class name generation service...")
        
        # Use the Azure function for class name generation
        print("🔧 Running direct tool class name generation...")
        results_json = creation_module.generate_class_names_direct(survey_json)
        results = json.loads(results_json)
        
        # Save class names results
        class_names_file = output_dirs["ClassCreation"] / "class_names_direct.json"
        with open(class_names_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create complete survey with class names
        complete_survey = survey_data.copy()
        for column_name, result in results.items():
            if column_name in complete_survey:
                complete_survey[column_name]["class_name"] = result.get("class_name_key", "UnknownCategory")
                complete_survey[column_name]["generated_class_name"] = result.get("generated_class_name", "UnknownCategory")
                complete_survey[column_name]["generation_metadata"] = {
                    "approach": result.get("approach", "direct_tool"),
                    "model_used": result.get("model_used", "DeepSeek-R1-0528")
                }
                if "error" in result:
                    complete_survey[column_name]["generation_metadata"]["error"] = result["error"]
        
        # Save complete survey with class names
        complete_file = output_dirs["ClassCreation"] / "complete_survey_with_class_names.json"
        with open(complete_file, 'w') as f:
            json.dump(complete_survey, f, indent=2)
        
        print(f"✅ Class names saved to: {class_names_file}")
        print(f"✅ Complete survey with class names saved to: {complete_file}")
        
        # Print summary
        successful = len([r for r in results.values() if "error" not in r])
        total = len(results)
        print(f"\n📊 SUMMARY: Successfully generated {successful}/{total} class names")
        
        return str(class_names_file)
        
    except Exception as e:
        print(f"❌ Error in class name generation: {e}")
        return None

def execute_step1_dictionary_extraction(sav_file_path: str, output_dirs: dict):
    """Step 1: Extract survey dictionary from .sav file"""
    print("\n" + "="*60)
    print("🚀 STEP 1: Survey Dictionary Extraction")
    print("="*60)
    
    # Import the data_dictionary_sav module
    try:
        spec = importlib.util.spec_from_file_location("data_dictionary_sav", "data_dictionary_sav.py")
        dict_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dict_module)
    except Exception as e:
        print(f"❌ Error importing data_dictionary_sav.py: {e}")
        return None
    
    # Set output path
    output_file = output_dirs["SurveyDictionary"] / "survey_dictionary.json"
    
    # Execute the dictionary extraction
    try:
        print(f"📖 Processing SAV file: {sav_file_path}")
        metadata = dict_module.create_SAS_metadata_dict(sav_file_path, str(output_file))
        
        if metadata:
            print(f"✅ Survey dictionary saved to: {output_file}")
            # Also run the preview
            dict_module.preview_ontology_candidates(metadata)
            return str(output_file)
        else:
            print("❌ Failed to extract survey dictionary")
            return None
            
    except Exception as e:
        print(f"❌ Error in dictionary extraction: {e}")
        return None

def execute_step2_csv_conversion(sav_file_path: str, output_dirs: dict):
    """Step 2: Convert .sav to CSV using the survey dictionary"""
    print("\n" + "="*60)
    print("🚀 STEP 2: CSV Conversion")
    print("="*60)
    
    # Check for required input from step 1
    survey_dict_path = output_dirs["SurveyDictionary"] / "survey_dictionary.json"
    if not check_file_exists(survey_dict_path, "Step 2 (CSV Conversion)"):
        print(f"💡 Run Step 1 first: --step dictionary")
        return None
    
    # Import the por_to_csv module
    try:
        spec = importlib.util.spec_from_file_location("por_to_csv", "por_to_csv.py")
        csv_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(csv_module)
    except Exception as e:
        print(f"❌ Error importing por_to_csv.py: {e}")
        return None
    
    # Set output path
    output_csv = output_dirs["SurveyDictionary"] / "survey_data.csv"
    
    try:
        print(f"📊 Converting SAV to CSV: {sav_file_path}")
        
        # Load the .sav file
        df, meta = csv_module.load_sav_file(sav_file_path)
        
        # Load the metadata
        metadata = csv_module.load_metadata(str(survey_dict_path))
        
        # Apply the labelled values mappings
        df = csv_module.apply_labelled_values(df, metadata)
        
        # Save to CSV
        csv_module.save_to_csv(df, str(output_csv))
        
        print(f"✅ CSV file saved to: {output_csv}")
        return str(output_csv)
        
    except Exception as e:
        print(f"❌ Error in CSV conversion: {e}")
        return None

def execute_step3_classification(output_dirs: dict):
    """Step 3: Classify survey questions into categories using Azure OpenAI"""
    print("\n" + "="*60)
    print("🚀 STEP 3: Question Classification")
    print("="*60)
    
    # Check for required input from step 1
    survey_dict_path = output_dirs["SurveyDictionary"] / "survey_dictionary.json"
    if not check_file_exists(survey_dict_path, "Step 3 (Classification)"):
        print(f"💡 Run Step 1 first: --step dictionary")
        return None
    
    # Load survey dictionary
    try:
        with open(survey_dict_path, 'r') as f:
            survey_data = json.load(f)
        survey_json = json.dumps(survey_data)
    except Exception as e:
        print(f"❌ Error loading survey dictionary: {e}")
        return None
    
    # Import the classification module
    try:
        spec = importlib.util.spec_from_file_location("react_classification_azure", "react_classification_azure.py")
        class_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(class_module)
    except Exception as e:
        print(f"❌ Error importing react_classification_azure.py: {e}")
        return None
    
    try:
        print("🤖 Starting Azure OpenAI classification service...")
        
        # Extract base filename for output
        base_name = Path(survey_dict_path).stem
        
        # Use the Azure function for classification
        print("🔧 Running direct tool classification...")
        results_json = class_module.classify_with_direct_tool(survey_json, base_name)
        results = json.loads(results_json)
        
        # Save categorized results
        categorized_file = output_dirs["CategoryClassification"] / "categorized_survey.json"
        with open(categorized_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save uncategorized questions separately
        uncategorized_questions = {k: v for k, v in results.items() if v.get('category') == 'Other/Uncategorized'}
        if uncategorized_questions:
            uncategorized_file = output_dirs["CategoryClassification"] / "uncategorized_survey.json"
            with open(uncategorized_file, 'w') as f:
                json.dump(uncategorized_questions, f, indent=2)
            print(f"📄 Saved {len(uncategorized_questions)} uncategorized questions to: {uncategorized_file}")
        
        print(f"✅ Categorized survey saved to: {categorized_file}")
        
        # Print category summary
        category_counts = {}
        for result in results.values():
            category = result.get('category', 'Unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        print("\n📊 CATEGORY DISTRIBUTION:")
        for category, count in sorted(category_counts.items()):
            print(f"   {category}: {count} questions")
        
        return str(categorized_file)
        
    except Exception as e:
        print(f"❌ Error in classification: {e}")
        return None

def execute_step4_class_creation(output_dirs: dict):
    """Step 4: Generate class names using Azure OpenAI"""
    print("\n" + "="*60)
    print("🚀 STEP 4: Class Name Generation")
    print("="*60)
    
    # Check for required input from step 3
    categorized_file = output_dirs["CategoryClassification"] / "categorized_survey.json"
    if not check_file_exists(categorized_file, "Step 4 (Class Creation)"):
        print(f"💡 Run Steps 1-3 first: --steps 1,2,3")
        return None
    
    # Load categorized survey data
    try:
        with open(categorized_file, 'r') as f:
            survey_data = json.load(f)
        survey_json = json.dumps(survey_data)
    except Exception as e:
        print(f"❌ Error loading categorized survey: {e}")
        return None
    
    # Import the class creation module
    try:
        spec = importlib.util.spec_from_file_location("class_creation_azure", "class_creation_azure.py")
        creation_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(creation_module)
    except Exception as e:
        print(f"❌ Error importing class_creation_azure.py: {e}")
        return None
    
    try:
        print("🤖 Starting Azure OpenAI class name generation service...")
        
        # Use the Azure function for class name generation
        print("🔧 Running direct tool class name generation...")
        results_json = creation_module.generate_class_names_direct(survey_json)
        results = json.loads(results_json)
        
        # Save class names results
        class_names_file = output_dirs["ClassCreation"] / "class_names_direct.json"
        with open(class_names_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create complete survey with class names
        complete_survey = survey_data.copy()
        for column_name, result in results.items():
            if column_name in complete_survey:
                complete_survey[column_name]["class_name"] = result.get("class_name_key", "UnknownCategory")
                complete_survey[column_name]["generated_class_name"] = result.get("generated_class_name", "UnknownCategory")
                complete_survey[column_name]["generation_metadata"] = {
                    "approach": result.get("approach", "direct_tool"),
                    "model_used": result.get("model_used", "DeepSeek-R1-0528")
                }
                if "error" in result:
                    complete_survey[column_name]["generation_metadata"]["error"] = result["error"]
        
        # Save complete survey with class names
        complete_file = output_dirs["ClassCreation"] / "complete_survey_with_class_names.json"
        with open(complete_file, 'w') as f:
            json.dump(complete_survey, f, indent=2)
        
        print(f"✅ Class names saved to: {class_names_file}")
        print(f"✅ Complete survey with class names saved to: {complete_file}")
        
        # Print summary
        successful = len([r for r in results.values() if "error" not in r])
        total = len(results)
        print(f"\n📊 SUMMARY: Successfully generated {successful}/{total} class names")
        
        return str(class_names_file)
        
    except Exception as e:
        print(f"❌ Error in class name generation: {e}")
        return None

def execute_step5_ontology_creation(output_dirs: dict):
    """Step 5: Generate final ontology Python module"""
    print("\n" + "="*60)
    print("🚀 STEP 5: Ontology Generation")
    print("="*60)
    
    # Check for required input from step 4
    class_names_file = output_dirs["ClassCreation"] / "class_names_direct.json"
    if not check_file_exists(class_names_file, "Step 5 (Ontology Creation)"):
        print(f"💡 Run Steps 1-4 first: --steps 1,2,3,4")
        return None
    
    # Import the ontology creation module
    try:
        spec = importlib.util.spec_from_file_location("survey_to_ontology", "survey_to_ontology.py")
        ontology_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ontology_module)
    except Exception as e:
        print(f"❌ Error importing survey_to_ontology.py: {e}")
        return None
    
    try:
        print(f"🏗️ Generating ontology from: {class_names_file}")
        
        # Set output path for the ontology
        output_py = output_dirs["Ontology"] / "survey_ontology.py"
        
        # Use the function from the ontology module
        result_path = ontology_module.generate_ontology_from_json(str(class_names_file), str(output_py))
        
        print(f"✅ Ontology generated successfully: {output_py}")
        
        return str(output_py)
        
    except Exception as e:
        print(f"❌ Error in ontology generation: {e}")
        return None

def main():
    """Main pipeline execution with step selection"""
    parser = setup_argparse()
    args = parser.parse_args()
    
    input_file_path = args.filename
    
    # Validate input file
    if not Path(input_file_path).exists():
        print(f"❌ Error: File not found: {input_file_path}")
        sys.exit(1)
    
    # Detect file type and setup accordingly
    try:
        file_type, base_dir = detect_file_type_and_setup(input_file_path)
        print(f"🔍 Detected file type: {file_type}")
    except Exception as e:
        print(f"❌ Error detecting file type: {e}")
        sys.exit(1)
    
    # Get absolute path for better display
    input_file_abs = Path(input_file_path).resolve()
    
    print("🌟 SURVEY ONTOLOGY GENERATION PIPELINE")
    print("=" * 60)
    print(f"📁 Input file: {input_file_abs}")
    print(f"📁 Base directory: {base_dir}")
    print(f"📁 Working directory: {Path.cwd()}")
    
    # Determine which steps to run
    steps_to_run = []
    
    if args.step:
        steps_to_run = [args.step]
    elif args.steps:
        steps_to_run = [s.strip() for s in args.steps.split(",")]
    else:
        # Default: run all steps
        steps_to_run = ["step1", "step2", "step3", "step4", "step5"]
    
    # Normalize step names
    normalized_steps = normalize_step_names(steps_to_run)
    
    if not normalized_steps:
        print("❌ No valid steps specified")
        sys.exit(1)
    
    print(f"🎯 Steps to run: {', '.join(normalized_steps)}")
    print()
    
    # Setup output directories using smart detection
    output_dirs = setup_output_directories_smart(input_file_path, file_type)
    
    # Execute steps in order with smart input handling
    results = {}
    
    try:
        for step in ["step1", "step2", "step3", "step4", "step5"]:  # Always check in order
            if step in normalized_steps:
                
                if step == "step1":
                    if file_type == 'sav':
                        result = execute_step1_dictionary_extraction(input_file_path, output_dirs)
                    else:
                        print("⏭️ Skipping Step 1 - JSON input provided")
                        result = "skipped"
                        
                elif step == "step2":
                    if file_type == 'sav':
                        result = execute_step2_csv_conversion(input_file_path, output_dirs)
                    else:
                        print("⏭️ Skipping Step 2 - JSON input provided")
                        result = "skipped"
                        
                elif step == "step3":
                    if file_type in ['sav', 'dictionary_json']:
                        result = execute_step3_classification(output_dirs)
                    else:
                        print("⏭️ Skipping Step 3 - categorized input provided")
                        result = "skipped"
                        
                elif step == "step4":
                    result = execute_step4_class_creation_smart(input_file_path, file_type, output_dirs)
                    
                elif step == "step5":
                    result = execute_step5_ontology_creation(output_dirs)
                
                results[step] = result
                
                if result is None:
                    print(f"\n❌ Pipeline stopped due to {step} failure")
                    sys.exit(1)
                
                print()  # Add spacing between steps
        
        # Success summary
        print("\n" + "🎉" * 60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("🎉" * 60)
        print(f"📁 All outputs created in: {base_dir}")
        
        # Summary
        step_names = {
            "step1": "Survey Dictionary", 
            "step2": "CSV Data",
            "step3": "Categorized Survey", 
            "step4": "Class Names", 
            "step5": "Final Ontology"
        }
        
        print("📋 Summary:")
        for step in ["step1", "step2", "step3", "step4", "step5"]:
            if step in results and results[step]:
                if results[step] == "skipped":
                    print(f"  ⏭️ {step_names[step]}: Skipped (using existing)")
                else:
                    print(f"  ✅ {step_names[step]}: {results[step]}")
            elif step in normalized_steps:
                print(f"  ❌ {step_names[step]}: Failed")
            else:
                print(f"  ⏭️ {step_names[step]}: Not requested")
        
        print("\n✅ Your survey ontology pipeline execution completed!")
        
        # Show the final directory structure if any steps completed
        if len([s for s in normalized_steps if results.get(s) and results[s] != "skipped"]) > 0:
            print(f"\n📂 Directory Structure:")
            print(f"{base_dir}/")
            for dir_name in ["SurveyDictionary", "CategoryClassification", "ClassCreation", "Ontology"]:
                dir_path = base_dir / dir_name
                if dir_path.exists():
                    print(f"├── {dir_name}/")
                    for file in sorted(dir_path.iterdir()):
                        if file.is_file():
                            print(f"│   ├── {file.name}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    step_functions = {
        "step1": run_step1_classification,
        "step2": run_step2_class_creation, 
        "step3": run_step3_kg_creation
    }
    
    results = {}
    
    for step in ["step1", "step2", "step3"]:  # Always check in order
        if step in normalized_steps:
            result = step_functions[step](args.filename, args)
            results[step] = result
            
            if result is None:
                print(f"\n❌ Pipeline stopped due to {step} failure")
                sys.exit(1)
            
            print()  # Add spacing between steps
    
    # Summary
    print("🎉 Pipeline execution completed!")
    print("📋 Summary:")
    for step, output in results.items():
        step_names = {"step1": "Classification", "step2": "Class Creation", "step3": "Knowledge Graph"}
        if output:
            print(f"  ✅ {step_names[step]}: {output}")
        else:
            print(f"  ⏭️ {step_names[step]}: Skipped")

if __name__ == "__main__":
    main()