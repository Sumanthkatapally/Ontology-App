#react_classification_azure.py
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
def extract_final_answer(output: str) -> str:
    """Extract only the first 'Final Answer:' and ignore any text after"""
    match = re.search(r'Final Answer:\s*(.+?)(?=\n|$)', output, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return output.strip()  # fallback

def save_uncategorized_questions(uncategorized_data: Dict, survey_filename: str):
    """Save uncategorized questions to a separate file"""
    if uncategorized_data:
        # Extract base filename without extension
        base_name = os.path.splitext(survey_filename)[0] if survey_filename else "survey"
        uncategorized_filename = f"{base_name}_uncategorized_columns.json"
        
        try:
            with open(uncategorized_filename, 'w') as f:
                json.dump(uncategorized_data, f, indent=2)
            print(f"📄 Saved {len(uncategorized_data)} uncategorized questions to: {uncategorized_filename}")
        except Exception as e:
            print(f"❌ Error saving uncategorized file: {str(e)}")

def save_categorized_results(results_data: Dict, survey_filename: str):
    """Save the categorized survey dictionary to a file"""
    # Extract base filename without extension
    base_name = os.path.splitext(survey_filename)[0] if survey_filename else "survey"
    categorized_filename = f"{base_name}_categorized.json"
    
    try:
        with open(categorized_filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"📄 Saved {len(results_data)} categorized questions to: {categorized_filename}")
        return categorized_filename
    except Exception as e:
        print(f"❌ Error saving categorized file: {str(e)}")
        return None

# ============================================================
# AZURE OPENAI LLM CLASS
# ============================================================
class AzureLLM:
    """Azure OpenAI LLM with controlled generation and deterministic parameters"""
    
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Generate text with Azure OpenAI"""
        try:
            response = self.client.complete(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that classifies survey questions into categories."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                temperature=0.1,
                max_tokens=1000,  # Increased from 300 to 500
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ Azure OpenAI API error: {str(e)}")
            return f"Error: {str(e)}"

# ============================================================
# CLASSIFICATION TOOL WITH LABELLED VALUES
# ============================================================
class ClassifyQuestionTool:
    """Direct classification tool with improved robustness and labelled values support"""
    
    def __init__(self, llm: AzureLLM):
        self.llm = llm
        self.name = "classify_question"
        self.description = "Classify a survey question into exactly one of these seven categories"

    def run(self, question_data: str) -> str:
        try:
            # Parse question data (expecting JSON string with question_text and labelled_values)
            data = json.loads(question_data)
            question_text = data.get('question_text', '')
            labelled_values = data.get('labelled_values', {})
            
            # Input sanitization
            question_text = re.sub(r"[^\w\s?.!,;:-]", "", str(question_text))[:500]
            print(f"▶️ Tool received question: {repr(question_text)}")
            print(f"▶️ Tool received labelled_values: {repr(labelled_values)}")
            
            # Format labelled values for the prompt
            if labelled_values and isinstance(labelled_values, dict) and labelled_values:
                values_text = "\nAnswer Options:\n"
                for key, value in labelled_values.items():
                    values_text += f"  {key}: {value}\n"
                print(f"▶️ Formatted answer options: {len(labelled_values)} options")
            else:
                values_text = "\nAnswer Options: No label values"
                print(f"▶️ No labelled values found")
            
            # Enhanced prompt with numbered categories and specific format
            prompt = f"""You are a survey classification expert. Classify this survey question into EXACTLY ONE category.

Question: {question_text}{values_text}

Categories:
1. Demographics - Personal characteristics (age, gender, location, income, education)
2. Political Opinions - Political views, approval ratings, voting preferences  
3. Values and Social Issues - Beliefs, social attitudes, agreement with statements
4. Lifestyle and Behavioral Opinions - Personal habits, behaviors, activities
5. Public Policy & Civic Engagement - Government policies, civic participation
6. Technical / Survey Metadata - Survey mechanics, data collection, identifiers
7. Other/Uncategorized - Questions that don't fit clearly into any above category

Give your answer in this exact format (with its index):
Final Answer: <category_number>. <one of the seven category names exactly>

Examples:
- Final Answer: 1. Demographics
- Final Answer: 2. Political Opinions
- Final Answer: 6. Technical / Survey Metadata

Category analysis for this question:"""

            response = self.llm._call(prompt)
            print(f"▶️ LLM raw response: {repr(response)}")
            
            # Improved extraction logic - look for "Final Answer:" first
            response = response.strip()
            
            # First try: Extract from "Final Answer:" format
            final_answer_match = re.search(r'Final Answer:\s*(.+?)(?=\n|$)', response, re.IGNORECASE)
            if final_answer_match:
                extracted_category = final_answer_match.group(1).strip()
                print(f"▶️ Extracted from Final Answer: {extracted_category}")
                
                # Validate against known categories
                valid_categories = [
                    "Demographics",
                    "Political Opinions", 
                    "Values and Social Issues",
                    "Lifestyle and Behavioral Opinions",
                    "Public Policy & Civic Engagement",
                    "Technical / Survey Metadata",
                    "Other/Uncategorized"
                ]
                
                for category in valid_categories:
                    if category.lower() == extracted_category.lower():
                        print(f"▶️ Tool returning category (Final Answer match): {category}")
                        return category
                    if extracted_category.startswith(tuple(str(i) for i in range(1, 8))):
                        # Match numbered format
                        for cat in valid_categories:
                            if cat.lower() in extracted_category.lower():
                                print(f"▶️ Tool returning category (Numbered match): {cat}")
                                return cat
            
            
            print(f"▶️ Tool returning category (fallback): Other/Uncategorized")
            return "Other/Uncategorized"
            
        except Exception as e:
            print(f"▶️ Tool error: {str(e)}")
            return "Other/Uncategorized"

# ============================================================
# DIRECT TOOL APPROACH WITH LABELLED VALUES
# ============================================================

def classify_with_direct_tool(survey_json: str, survey_filename: str = "survey") -> str:
    """APPROACH 1: Direct tool usage with labelled values support using Azure OpenAI"""
    
    survey_data = json.loads(survey_json)
    
    print(f"🤖 Using Azure OpenAI {MODEL_NAME} for direct tool usage...")
    
    # Create LLM and tool
    llm = AzureLLM(client, MODEL_NAME)
    tool = ClassifyQuestionTool(llm=llm)
    
    print(f"🔧 Direct tool created, processing {len(survey_data)} questions...")
    
    # Process questions with direct tool calls
    results = {}
    uncategorized_questions = {}
    
    for i, (name, q) in enumerate(survey_data.items(), 1):
        try:
            print(f"\n📝 [{i}/{len(survey_data)}] Classifying: {name}")
            question_text = q.get('column_label', '')
            labelled_values = q.get('labelled_values', {})
            
            # Handle case where labelled_values might be None or not a dict
            if not isinstance(labelled_values, dict):
                labelled_values = {}
            
            print(f"   Question: {question_text}")
            if labelled_values:
                values_list = list(labelled_values.values()) if isinstance(labelled_values, dict) else []
                print(f"   Labels: {values_list}")
            else:
                print(f"   Labels: No label values")
            
            # Prepare data for tool - ensure proper JSON structure
            tool_input = json.dumps({
                'question_text': str(question_text),
                'labelled_values': labelled_values if isinstance(labelled_values, dict) else {}
            })
            
            print(f"   🔧 Tool input prepared: question_text='{question_text}', labelled_values={len(labelled_values)} items")
            
            # Direct tool call
            category = tool.run(tool_input)
            
            # Add category to existing question data
            q_with_category = q.copy()
            q_with_category['category'] = category
            
            results[name] = q_with_category
            
            # Track uncategorized questions
            if category == "Other/Uncategorized":
                uncategorized_questions[name] = q_with_category
            
            print(f"   ✅ Classified as: {category}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            q_with_error = q.copy()
            q_with_error['category'] = "Other/Uncategorized"
            q_with_error['error'] = str(e)
            results[name] = q_with_error
            uncategorized_questions[name] = q_with_error
    
    # Save uncategorized questions to separate file
    save_uncategorized_questions(uncategorized_questions, survey_filename)
    
    # Save the main categorized results to file
    save_categorized_results(results, survey_filename)
    
    print(f"\n🎉 Completed all {len(survey_data)} direct tool classifications!")
    print(f"📊 Summary: {len(uncategorized_questions)} questions classified as Other/Uncategorized")
    
    return json.dumps(results)

def classify_with_simple_agent(survey_json: str, survey_filename: str = "survey") -> str:
    """APPROACH 2: Improved ultra-simple agent with labelled values support using Azure OpenAI"""
    
    survey_data = json.loads(survey_json)
    
    print(f"🤖 Using Azure OpenAI {MODEL_NAME} for simple agent...")
    
    # Create LLM and tool
    llm = AzureLLM(client, MODEL_NAME)
    tool = ClassifyQuestionTool(llm=llm)
    
    print(f"🔧 Simple agent created, processing {len(survey_data)} questions...")
    
    # Process questions
    results = {}
    uncategorized_questions = {}
    
    for i, (name, q) in enumerate(survey_data.items(), 1):
        try:
            print(f"\n📝 [{i}/{len(survey_data)}] Classifying: {name}")
            question_text = q.get('column_label', '')
            labelled_values = q.get('labelled_values', {})
            
            # Handle case where labelled_values might be None or not a dict
            if not isinstance(labelled_values, dict):
                labelled_values = {}
            
            # Prepare data for agent - ensure proper JSON structure
            question_data = json.dumps({
                'question_text': str(question_text),
                'labelled_values': labelled_values if isinstance(labelled_values, dict) else {}
            })
            
            print(f"   Question: {question_text}")
            if labelled_values:
                values_list = list(labelled_values.values()) if isinstance(labelled_values, dict) else []
                print(f"   Labels: {values_list}")
            
            print(f"   🔧 Agent input prepared: question_text='{question_text}', labelled_values={len(labelled_values)} items")
            
            # Simple agent reasoning prompt
            agent_prompt = f"""You have exactly one tool:

classify_question: Classify a survey question into exactly one of these seven categories:
  1. Demographics – Personal characteristics (age, gender, location, income, education)
  2. Political Opinions – Political views, approval ratings, voting preferences
  3. Values and Social Issues – Beliefs, social attitudes, agreement with statements
  4. Lifestyle and Behavioral Opinions – Personal habits, behaviors, activities
  5. Public Policy & Civic Engagement – Government policies, civic participation
  6. Technical / Survey Metadata – Survey mechanics, data collection
  7. Other/Uncategorized – Questions that don't fit clearly into any above category

You can use the following format for reasoning:

Thought: I need to analyze this question to determine its category.
Action: classify_question
Action Input: {question_data}
Observation: <tool output>
Thought: Based on the tool output, I can determine the final category.
Final Answer: <one of the seven category names exactly>

STOP IMMEDIATELY AFTER THE FIRST "Final Answer:" LINE. DO NOT GENERATE ANYTHING ELSE.

Begin!

Question Data: {question_data}"""
            
            # Get agent response
            agent_response = llm._call(agent_prompt)
            
            print(f"   🧠 Agent reasoning: {agent_response[:100]}...")
            
            # Extract tool action and run it
            category = "Other/Uncategorized"
            
            # Look for Action Input and run the tool
            if "Action Input:" in agent_response:
                # Run the classification tool
                category = tool.run(question_data)
            
            # Extract final answer to prevent hallucinations
            final_answer = extract_final_answer(agent_response)
            
            if final_answer:
                final_answer_clean = final_answer.strip()
                
                # Check if it's one of our valid categories
                categories = [
                    "Demographics",
                    "Political Opinions", 
                    "Values and Social Issues",
                    "Lifestyle and Behavioral Opinions",
                    "Public Policy & Civic Engagement",
                    "Technical / Survey Metadata",
                    "Other/Uncategorized"
                ]
                
                for cat in categories:
                    if cat.lower() in final_answer_clean.lower():
                        category = cat
                        break
            
            print(f"   ✅ Classified as: {category}")
            
            # Add category to existing question data
            q_with_category = q.copy()
            q_with_category['category'] = category
            
            results[name] = q_with_category
            
            # Track uncategorized questions
            if category == "Other/Uncategorized":
                uncategorized_questions[name] = q_with_category
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            q_with_error = q.copy()
            q_with_error['category'] = "Other/Uncategorized"
            q_with_error['error'] = str(e)
            results[name] = q_with_error
            uncategorized_questions[name] = q_with_error
    
    # Save uncategorized questions to separate file
    save_uncategorized_questions(uncategorized_questions, survey_filename)
    
    # Save the main categorized results to file
    save_categorized_results(results, survey_filename)
    
    print(f"\n🎉 Completed all {len(survey_data)} improved agent classifications!")
    print(f"📊 Summary: {len(uncategorized_questions)} questions classified as Other/Uncategorized")
    
    return json.dumps(results)

def main():
    """Test both approaches with improvements and labelled values using Azure OpenAI"""
    
    # Load your survey data
    survey_filename = "survey_dictionary.json"  # Update this with your actual filename
    file_path = r"D:\MARIST\Research\Environment\myenv\Project\KnowledgeGraph\data\US_SURVEY\survey_dictionary.json"
    
    try:
        with open(file_path, "r") as f:
            survey_json = f.read()  # already a JSON string
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        # Fallback to test data
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
                }
            },
            "STATE": {
                "column_name": "STATE",
                "column_type": "F8.2",
                "column_label": "Which state do you live in?",
                "labelled_values": {
                    "1.0": "Alabama",
                    "2.0": "Arizona",
                    "3.0": "California"
                }
            },
            "WEIRD_Q": {
                "column_name": "WEIRD_Q",
                "column_type": "F8.2",
                "column_label": "What is the airspeed velocity of an unladen swallow?",
                "labelled_values": {
                    "1.0": "African or European?",
                    "2.0": "I don't know that",
                    "3.0": "42"
                }
            }
        }
        survey_json = json.dumps(test_survey)
        survey_filename = "test_survey.json"
    
    print("🚀 APPROACH 1: Direct Tool Usage with Azure OpenAI")
    print("=" * 60)
    print("🔧 Enhanced with labelled_values support")
    print("📁 Saves uncategorized questions to separate file")
    print("📊 Added 'Other/Uncategorized' category")
    print()
    
    # Test direct tool approach
    results_json = classify_with_direct_tool(survey_json, survey_filename)
    results = json.loads(results_json)
    
    # Save results locally
    base_name = os.path.splitext(survey_filename)[0]
    local_categorized_file = f"{base_name}_categorized.json"
    local_uncategorized_file = f"{base_name}_uncategorized_columns.json"
    
    # Save main categorized results locally
    with open(local_categorized_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Saved main results locally to: {local_categorized_file}")
    
    # Save uncategorized questions locally
    uncategorized_questions = {k: v for k, v in results.items() if v.get('category') == 'Other/Uncategorized'}
    if uncategorized_questions:
        with open(local_uncategorized_file, 'w') as f:
            json.dump(uncategorized_questions, f, indent=2)
        print(f"💾 Saved uncategorized questions locally to: {local_uncategorized_file}")
    
    print("\n📊 DIRECT TOOL RESULTS WITH CATEGORIES:")
    category_counts = {}
    for column_name, result in results.items():
        category = result.get('category', 'Unknown')
        category_counts[category] = category_counts.get(category, 0) + 1
        
        if "error" in result:
            print(f"❌ {column_name}: {category} (Error: {result['error']})")
        else:
            print(f"✅ {column_name}: {category}")
    
    print(f"\n📈 CATEGORY DISTRIBUTION:")
    for category, count in sorted(category_counts.items()):
        print(f"   {category}: {count} questions")
    
    print(f"\n🎉 Classification complete!")
    print(f"💾 Files saved in current directory:")
    print(f"   📄 Main results: {local_categorized_file}")
    if uncategorized_questions:
        print(f"   📄 Uncategorized: {local_uncategorized_file}")
    print(f"\n📍 Full path: {os.path.abspath(local_categorized_file)}")

if __name__ == "__main__":
    main()