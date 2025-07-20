#class_creation_azure.py
import json
import re
from typing import Dict, List, Any, Optional
import os
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─── AZURE CREDENTIALS FROM ENVIRONMENT ───────────────────────────────────────────────────────
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "DeepSeek-R1-0528")

# Validate credentials
if not AZURE_ENDPOINT or not AZURE_API_KEY:
    raise ValueError("Missing Azure credentials! Please set AZURE_ENDPOINT and AZURE_API_KEY in your .env file")

print(f"🔐 Loaded Azure credentials from .env file")
print(f"📡 Endpoint: {AZURE_ENDPOINT}")
print(f"🤖 Model: {MODEL_NAME}")

client = ChatCompletionsClient(
    endpoint=AZURE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_API_KEY),
    api_version="2024-05-01-preview"
)

# ============================================================
# UTILITY FUNCTIONS FOR PREVENTING HALLUCINATIONS
# ============================================================
def extract_class_name(output: str) -> str:
    """Extract only the class name from the output with improved regex"""
    # Clean the output first
    output = output.strip().strip('"').strip("'")
    
    # First try: Look for class name after explicit markers
    match = re.search(r'^(?:Class Name:|Output:|Answer:)?\s*([A-Z][a-zA-Z0-9]+)$', output.strip(), re.MULTILINE)
    if match:
        return match.group(1).strip()
    
    # Second try: Look for the first PascalCase word at the beginning of a line
    lines = output.strip().split('\n')
    for line in lines:
        line = line.strip().strip('"').strip("'")
        if re.match(r'^[A-Z][a-zA-Z0-9]+$', line):
            return line
    
    # Third try: Look for first valid PascalCase word in the entire output
    words = re.findall(r'[A-Z][a-zA-Z0-9]*', output)
    for word in words:
        # Skip common prefixes and ensure minimum length
        if (len(word) >= 3 and 
            word.lower() not in ['the', 'class', 'name', 'is', 'for', 'question', 'survey'] and
            not word.endswith('Approval') or len([w for w in words if w.endswith('Approval')]) == 1):
            return word
    
    return "UnknownCategory"  # fallback

def validate_class_name(class_name: str) -> str:
    """Ensure class name follows PascalCase convention"""
    # Remove any non-alphanumeric characters
    clean_name = re.sub(r'[^a-zA-Z0-9]', '', class_name)
    
    # Ensure it starts with uppercase
    if clean_name and not clean_name[0].isupper():
        clean_name = clean_name[0].upper() + clean_name[1:]
    
    # Ensure it's not empty and has minimum length
    if not clean_name or len(clean_name) < 3:
        return "UnknownCategory"
    
    return clean_name

def create_focused_prompt(column_name: str, column_label: str, values_list: List[str], category: str = None) -> str:
    """Create a simplified prompt for DeepSeek-R1"""
    
    values_text = ', '.join(values_list[:5]) if values_list else "None"
    category_text = category if category else "Unknown"
    
    prompt = f"""Generate a PascalCase class name for this survey question.

Question: {column_label}
Category: {category_text}
Answer Options: {values_text}

Rules:
- Use PascalCase (e.g., PresidentialApproval)
- Be specific to what the question measures
- Maximum 4-5 words combined
- No spaces or special characters
- Take into account the answer options and category, if provided.

Give your answer in this exact format (with its index):
Final Answer: <your_class_name_here>

Examples:
Q.Biden approval question 
- Final Answer: JoeBidenApproval
Q.Voter registration question → VoterRegistration
- Final Answer: VoterRegistration
Q.Urban/suburban question → ResidenceType
- Final Answer: ResidenceType
Q.Income question → IncomeLevel
- Final Answer: IncomeLevel

Class creation analysis for this question:"""
    
    return prompt

# ============================================================
# AZURE OPENAI LLM CLASS
# ============================================================
class AzureLLM:
    """Azure OpenAI LLM with controlled generation for class name creation"""
    
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Generate text with Azure OpenAI"""
        try:
            response = self.client.complete(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates descriptive class names for survey questions. Always respond with a single PascalCase class name as specified."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                temperature=0.2,
                max_tokens=8192,  # INCREASED from 500 to 1000
                top_p=0.9
            )
            
            generated_text = response.choices[0].message.content.strip()
            
            return generated_text or "UnknownCategory"
            
        except Exception as e:
            print(f"❌ Azure OpenAI API error: {str(e)}")
            return f"Error: {str(e)}"

# ============================================================
# CLASS NAME GENERATION TOOL WITH UNIQUENESS
# ============================================================
class GenerateClassNameTool:
    """Tool for generating appropriate class names from survey questions"""
    
    def __init__(self, llm: AzureLLM):
        self.llm = llm
        self.name = "generate_class_name"
        self.description = "Generate a PascalCase class name for a survey question based on its label and possible values"
        self.used_class_names = set()  # Track used class names

    def run(self, question_data: str) -> str:
        try:
            # Parse the question data
            data = json.loads(question_data) if isinstance(question_data, str) else question_data
            
            column_name = data.get('column_name', '')
            column_label = data.get('column_label', '')
            category = data.get('category', '')
            raw_values = data.get('labelled_values', {})
            
            # Robustly handle non-dict labelled_values
            if not isinstance(raw_values, dict):
                labelled_values = {}
                values_list = []
            else:
                labelled_values = raw_values
                values_list = list(labelled_values.values())
            
            print(f"▶️ Generating class name for: {column_name}")
            print(f"   Label: {column_label}")
            print(f"   Category: {category}")
            if values_list:
                print(f"   Values: {values_list[:3]}{'...' if len(values_list) > 3 else ''}")
            else:
                print("   Values: (no labeled values)")
            
            # Handle empty or missing labels - generate from column name only
            if not column_label or column_label.strip() == '' or column_label.lower() == 'no label':
                clean_name = re.sub(r'[^a-zA-Z0-9]', '', column_name)
                if clean_name:
                    class_name = ''.join(word.capitalize() for word in re.split(r'[_\s]+', clean_name))
                    unique_class_name = self._ensure_unique_class_name(validate_class_name(class_name))
                    print(f"▶️ Generated from column name: {unique_class_name}")
                    return unique_class_name
                else:
                    unique_class_name = self._ensure_unique_class_name("UnknownCategory")
                    print(f"▶️ Using fallback for empty column name: {unique_class_name}")
                    return unique_class_name
            
            # Try up to 3 times to get a unique class name
            for attempt in range(3):
                # Create focused prompt with uniqueness information
                prompt = self._create_unique_prompt(column_name, column_label, values_list, category, attempt)
                
                # Single LLM call
                response = self.llm._call(prompt)
                print(f"▶️ LLM response length: {len(response)} characters")
                print(f"▶️ LLM response: {repr(response)}")
                
                # ONLY extract from "Final Answer:" format
                response = response.strip()
                
                # Extract from "Final Answer:" format
                final_answer_match = re.search(r'Final Answer:\s*(.+?)(?=\n|$)', response, re.IGNORECASE)
                if final_answer_match:
                    extracted_class = final_answer_match.group(1).strip()
                    print(f"▶️ Extracted from Final Answer: {extracted_class}")
                    
                    # Validate and clean the extracted class name
                    validated_name = validate_class_name(extracted_class)
                    
                    # Check if it's unique
                    if validated_name != "UnknownCategory" and validated_name not in self.used_class_names:
                        self.used_class_names.add(validated_name)
                        print(f"▶️ Final unique class name: {validated_name}")
                        return validated_name
                    elif validated_name in self.used_class_names:
                        print(f"⚠️ Class name '{validated_name}' already exists, retrying...")
                        continue
                
                print(f"▶️ Attempt {attempt + 1} failed, retrying...")
            
            # If all attempts failed, create a unique name with suffix
            base_name = "UnknownCategory"
            unique_name = self._ensure_unique_class_name(base_name)
            print(f"▶️ Final fallback unique class name: {unique_name}")
            return unique_name
            
        except Exception as e:
            print(f"▶️ Tool error: {str(e)}")
            unique_name = self._ensure_unique_class_name("UnknownCategory")
            return unique_name

    def _create_unique_prompt(self, column_name: str, column_label: str, values_list: List[str], category: str = None, attempt: int = 0) -> str:
        """Create prompt with uniqueness constraints"""
        
        values_text = ', '.join(values_list[:5]) if values_list else "None"
        category_text = category if category else "Unknown"
        
        # Add uniqueness information to prompt
        used_names_text = ""
        if self.used_class_names:
            used_names_list = list(self.used_class_names)[:10]  # Show first 10
            used_names_text = f"\nALREDY USED CLASS NAMES (do NOT use these): {', '.join(used_names_list)}"
            if len(self.used_class_names) > 10:
                used_names_text += f" ... and {len(self.used_class_names) - 10} more"
        
        # Add retry information
        retry_text = ""
        if attempt > 0:
            retry_text = f"\nATTEMPT {attempt + 1}: Previous attempts failed. Try a different approach or add suffixes like 'Survey', 'Question', 'Response', etc."
        
        prompt = f"""Generate a PascalCase class name for this survey question.

Question: {column_label}
Category: {category_text}
Answer Options: {values_text}{used_names_text}{retry_text}

Rules:
- Use PascalCase (e.g., PresidentialApproval)
- Be specific to what the question measures
- Maximum 4-5 words combined
- No spaces or special characters
- Take into account the answer options and category, if provided
- MUST be unique - do not use any of the already used class names listed above

Give your answer in this exact format (with its index):
Final Answer: <your_class_name_here>

Examples:
Q.Biden approval question 
- Final Answer: JoeBidenApproval
Q.Voter registration question → VoterRegistration
- Final Answer: VoterRegistration
Q.Urban/suburban question → ResidenceType
- Final Answer: ResidenceType
Q.Income question → IncomeLevel
- Final Answer: IncomeLevel

Class creation analysis for this question:"""
        
        return prompt

    def _ensure_unique_class_name(self, base_name: str) -> str:
        """Ensure class name is unique by adding suffixes if needed"""
        if base_name not in self.used_class_names:
            self.used_class_names.add(base_name)
            return base_name
        
        # Try adding suffixes to make it unique
        suffixes = ["Question", "Response", "Survey", "Data", "Field", "Item", "Value", "Info"]
        
        for suffix in suffixes:
            candidate = base_name + suffix
            if candidate not in self.used_class_names:
                self.used_class_names.add(candidate)
                return candidate
        
        # If all suffixes are used, add numbers
        counter = 1
        while True:
            candidate = f"{base_name}{counter}"
            if candidate not in self.used_class_names:
                self.used_class_names.add(candidate)
                return candidate
            counter += 1

# ============================================================
# DIRECT TOOL APPROACH FOR CLASS NAME GENERATION WITH UNIQUENESS
# ============================================================

def generate_class_names_direct(survey_json: str) -> str:
    """Direct tool approach for generating unique class names using Azure OpenAI"""
    
    survey_data = json.loads(survey_json)
    
    print(f"🤖 Using Azure OpenAI {MODEL_NAME} for class name generation...")
    
    # Create LLM and tool (tool will track unique names)
    llm = AzureLLM(client, MODEL_NAME)
    tool = GenerateClassNameTool(llm=llm)
    
    print(f"🔧 Direct tool created, processing {len(survey_data)} questions...")
    print(f"📊 Ensuring all {len(survey_data)} class names are unique...")
    
    # Process questions
    results = {}
    processed_count = 0
    
    for i, (name, q) in enumerate(survey_data.items(), 1):
        try:
            print(f"\n📝 [{i}/{len(survey_data)}] Generating class name for: {name}")
            print(f"   📊 Already generated: {len(tool.used_class_names)} unique class names")
            
            # Direct tool call
            class_name = tool.run(q)
            
            results[name] = {
                "original_column": name,
                "generated_class_name": class_name,
                "class_name_key": class_name,  # Store for easy access
                "column_label": q.get('column_label', ''),
                "category": q.get('category', ''),
                "labelled_values": q.get('labelled_values', {}),
                "approach": "direct_tool",
                "model_used": MODEL_NAME,
                "uniqueness_ensured": True
            }
            
            processed_count += 1
            print(f"   ✅ Generated unique class name: {class_name}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            # Even for errors, ensure unique name
            error_class_name = tool._ensure_unique_class_name("UnknownCategory")
            results[name] = {
                "original_column": name,
                "generated_class_name": error_class_name,
                "class_name_key": error_class_name,  # Store for easy access
                "error": str(e),
                "column_label": q.get('column_label', ''),
                "category": q.get('category', ''),
                "labelled_values": q.get('labelled_values', {}),
                "approach": "direct_tool",
                "uniqueness_ensured": True
            }
    
    # Verify uniqueness
    all_class_names = [r["generated_class_name"] for r in results.values()]
    unique_class_names = set(all_class_names)
    
    print(f"\n🎉 Completed all {len(survey_data)} class name generations!")
    print(f"📊 UNIQUENESS VERIFICATION:")
    print(f"   Total questions: {len(survey_data)}")
    print(f"   Total class names: {len(all_class_names)}")
    print(f"   Unique class names: {len(unique_class_names)}")
    print(f"   ✅ All class names are unique: {len(all_class_names) == len(unique_class_names)}")
    
    if len(all_class_names) != len(unique_class_names):
        # Find duplicates
        duplicates = []
        seen = set()
        for name in all_class_names:
            if name in seen:
                duplicates.append(name)
            seen.add(name)
        print(f"   ⚠️ Duplicates found: {duplicates}")
    
    # Save complete results to local file
    output_dir = os.path.dirname(os.path.abspath(__file__)) if __file__ else "."
    output_file = os.path.join(output_dir, "survey_with_class_names_direct.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(results, indent=2, ensure_ascii=False))
    
    print(f"💾 Complete results saved to: {output_file}")
    
    return json.dumps(results, indent=2)
# ============================================================
# AGENTIC APPROACH FOR CLASS NAME GENERATION
# ============================================================

def generate_class_names_agent(survey_json: str) -> str:
    """Agentic approach for generating class names with reasoning using Azure OpenAI"""
    
    survey_data = json.loads(survey_json)
    
    print(f"🤖 Using Azure OpenAI {MODEL_NAME} for agentic class name generation...")
    
    # Create LLM and tools
    llm = AzureLLM(client, MODEL_NAME)
    tool = GenerateClassNameTool(llm=llm)
    
    print(f"🔧 Agent created, processing {len(survey_data)} questions...")
    
    # Process questions
    results = {}
    for i, (name, q) in enumerate(survey_data.items(), 1):
        try:
            print(f"\n📝 [{i}/{len(survey_data)}] Generating class name for: {name}")
            
            # Prepare input for agent
            question_input = json.dumps(q)
            
            # Agent reasoning prompt
            agent_prompt = f"""You have one tool to generate class names for survey questions:

generate_class_name: Generate a PascalCase class name for a survey question based on its label and possible values

Use this format:

Thought: I need to analyze this survey question to generate an appropriate class name.
Action: generate_class_name
Action Input: {question_input}
Observation: <tool output>
Thought: The tool generated a class name. I should verify it follows the conventions.
Final Answer: <the generated class name>

STOP IMMEDIATELY AFTER THE FIRST "Final Answer:" LINE.

Begin!

Question Data: {question_input}"""
            
            # Run agent
            agent_response = llm._call(agent_prompt)
            
            print(f"   🧠 Agent reasoning: {agent_response[:100]}...")
            
            # Extract class name from agent response or run tool directly
            class_name = "UnknownCategory"
            
            # Look for Action Input and run the tool
            if "Action Input:" in agent_response:
                # Run the classification tool
                class_name = tool.run(question_input)
            
            # Also try to extract from Final Answer
            final_answer_match = re.search(r'Final Answer:\s*(.+?)(?=\n|$)', agent_response, re.IGNORECASE)
            if final_answer_match:
                potential_class = extract_class_name(final_answer_match.group(1))
                validated_potential = validate_class_name(potential_class)
                if validated_potential != "UnknownCategory":
                    class_name = validated_potential
            
            results[name] = {
                "original_column": name,
                "generated_class_name": class_name,
                "class_name_key": class_name,  # Store for easy access
                "column_label": q.get('column_label', ''),
                "category": q.get('category', ''),
                "labelled_values": q.get('labelled_values', {}),
                "agent_reasoning": {
                    "raw_output": agent_response
                },
                "approach": "agentic",
                "model_used": MODEL_NAME
            }
            
            print(f"   ✅ Generated class name: {class_name}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            results[name] = {
                "original_column": name,
                "generated_class_name": "UnknownCategory",
                "class_name_key": "UnknownCategory",  # Store for easy access
                "error": str(e),
                "column_label": q.get('column_label', ''),
                "category": q.get('category', ''),
                "labelled_values": q.get('labelled_values', {}),
                "approach": "agentic"
            }
    
    print(f"\n🎉 Completed all {len(survey_data)} agentic class name generations!")
    
    # Save complete results to local file
    output_dir = os.path.dirname(os.path.abspath(__file__)) if __file__ else "."
    output_file = os.path.join(output_dir, "survey_with_class_names_agentic.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(results, indent=2, ensure_ascii=False))
    
    print(f"💾 Complete results saved to: {output_file}")
    
    return json.dumps(results, indent=2)

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Test both approaches for class name generation using Azure OpenAI"""
    
    # Test data based on your examples
    test_survey = {
        "BIDJP105": {
            "column_name": "BIDJP105",
            "column_type": "F8.2",
            "column_label": "Do you approve or disapprove of the job Joe Biden is doing as president?",
            "labelled_values": {
                "1.0": "Approve",
                "2.0": "Disapprove",
                "8.0": "Vol: Unsure",
                "9.0": "Refused"
            },
            "category": "Political Opinions"
        },
        "TOSS1A": {
            "column_name": "TOSS1A",
            "column_type": "F8.2",
            "column_label": "If November's general election for president were held today, whom would you support if the candidates are:",
            "labelled_values": {
                "1.0": "Joe Biden, the Democrat",
                "2.0": "Donald Trump, the Republican",
                "8.0": "Vol: Undecided",
                "9.0": "Vol: REFUSED"
            },
            "category": "Political Opinions"
        },
        "STATE": {
            "column_name": "STATE",
            "column_type": "F8.2",
            "column_label": "Which state do you live in?",
            "labelled_values": {
                "1.0": "ALABAMA",
                "2.0": "ARIZONA",
                "3.0": "ARKANSAS"
            },
            "category": "Demographics"
        },
        "USRRR1": {
            "column_name": "USRRR1",
            "column_type": "F8.2", 
            "column_label": "Do you consider the area where YOU live to be urban, suburban, or rural?",
            "labelled_values": {
                "1.0": "Urban",
                "2.0": "Suburban", 
                "3.0": "Rural"
            },
            "category": "Demographics"
        }
    }
    
    # Load your actual survey data if file exists
    try:
        with open("survey_dictionary_categorized.json", "r") as f:
            survey_json = f.read()  # already a JSON string
    except FileNotFoundError:
        print("Using test data (survey_dictionary_categorized.json not found)")
        survey_json = json.dumps(test_survey)
    
    print("🚀 APPROACH 1: Direct Tool Usage with Azure OpenAI")
    print("=" * 50)
    print("🔧 Direct class name generation without agent overhead")
    print()
    
    # Test direct tool approach (single call)
    print("💾 Running class name generation and saving results...")
    results_json = generate_class_names_direct(survey_json)
    results = json.loads(results_json)
    
    print("\n📊 DIRECT TOOL RESULTS:")
    for column_name, result in results.items():
        if "error" in result:
            print(f"❌ {column_name}: {result['error']}")
        else:
            print(f"✅ {column_name} → {result['generated_class_name']}")
    
    # Save results to additional local files (using already generated results)
    print("\n💾 Creating additional output files...")
    
    # Save both the results and create a complete survey file
    with open("class_names_direct.json", "w", encoding="utf-8") as f:
        f.write(results_json)
    
    # Create complete survey data with class names integrated
    complete_survey = json.loads(survey_json)
    
    # Add class names back to original survey structure
    for column_name, result in results.items():
        if column_name in complete_survey:
            complete_survey[column_name]["class_name"] = result.get("class_name_key", "UnknownCategory")
            complete_survey[column_name]["generation_metadata"] = {
                "approach": result.get("approach", "direct_tool"),
                "model_used": result.get("model_used", MODEL_NAME),
                "generated_class_name": result.get("generated_class_name", "UnknownCategory")
            }
            if "error" in result:
                complete_survey[column_name]["generation_metadata"]["error"] = result["error"]
    
    # Save complete survey with class names
    with open("complete_survey_with_class_names.json", "w", encoding="utf-8") as f:
        json.dump(complete_survey, f, indent=2, ensure_ascii=False)
    
    print("📁 Results saved to:")
    print("   - class_names_direct.json (detailed results)")
    print("   - complete_survey_with_class_names.json (original survey + class names)")
    print("   - survey_with_class_names_direct.json (created by function)")
    
    # Print summary
    print(f"\n📊 SUMMARY:")
    successful = len([r for r in results.values() if "error" not in r])
    total = len(results)
    print(f"✅ Successfully generated: {successful}/{total} class names")
    if successful < total:
        failed = [name for name, r in results.items() if "error" in r]
        print(f"❌ Failed: {failed}")
    
    return results

if __name__ == "__main__":
    main()