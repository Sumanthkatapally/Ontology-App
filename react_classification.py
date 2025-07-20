import modal
import json
import re
from typing import Dict, List, Any, Optional
import os

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

# Modal app setup
app = modal.App("direct-tool-survey-classifier")

# Container image with optimized dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "transformers>=4.35.0",
    "torch>=2.0.0",
    "accelerate>=0.24.0",
    "langchain-community>=0.2.0",
    "pydantic>=2.0.0",
    "bitsandbytes>=0.39.0"
])

# Model configuration
MODEL_NAME = "ibm-granite/granite-4.0-tiny-preview"
GPU_TYPE = "A10G"
MEMORY_GB = 24

# ============================================================
# IMPROVED GRANITE LLM WITH DETERMINISTIC PARAMETERS
# ============================================================
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.callbacks import StdOutCallbackHandler
from langchain.llms.base import LLM
from pydantic import Field, PrivateAttr

class GraniteLLM(LLM):
    """Granite LLM with controlled generation and deterministic parameters"""
    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self._model = model
        self._tokenizer = tokenizer

    @property
    def _llm_type(self) -> str:
        return "granite-tiny"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Generate text with deterministic parameters for consistency"""
        import torch
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=False
            ).to(self._model.device)
            
            with torch.no_grad():
                outputs = self._model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=300,
                    temperature=0.01,  # Move these parameters directly here
                    top_p=0.9,
                    do_sample=False,
                    repetition_penalty=1.1,
                    pad_token_id=self._tokenizer.eos_token_id,
                    use_cache=False,
                    num_return_sequences=1,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            
            generated_text = self._tokenizer.decode(
                outputs[0][inputs.input_ids.shape[-1]:], 
                skip_special_tokens=True
            ).strip()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return generated_text or "No response"
            
        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return f"Error: {str(e)}"
# ============================================================
# IMPROVED CLASSIFICATION TOOL WITH LABELLED VALUES
# ============================================================
class ClassifyQuestionTool(BaseTool):
    """Direct classification tool with improved robustness and labelled values support"""
    name: str = Field(default="classify_question")
    description: str = Field(default="Classify a survey question into exactly one of these seven categories: 1. Demographics, 2. Political Opinions, 3. Values and Social Issues, 4. Lifestyle and Behavioral Opinions, 5. Public Policy & Civic Engagement, 6. Technical / Survey Metadata, 7. Other/Uncategorized")
    _llm: Any = PrivateAttr()

    def __init__(self, llm: LLM):
        super().__init__()
        self._llm = llm

    def _run(self, question_data: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
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
            
            # Enhanced prompt with labelled values
            prompt = f"""Classify this survey question into exactly one category:

Question: {question_text}{values_text}

Categories:
1. Demographics - Personal characteristics (age, gender, location, income, education)
2. Political Opinions - Political views, approval ratings, voting preferences  
3. Values and Social Issues - Beliefs, social attitudes, agreement with statements
4. Lifestyle and Behavioral Opinions - Personal habits, behaviors, activities
5. Public Policy & Civic Engagement - Government policies, civic participation
6. Technical / Survey Metadata - Survey mechanics, data collection
7. Other/Uncategorized - Questions that don't fit clearly into any above category

Answer with the number and category name only. For example: "2. Political Opinions"

Answer:"""

            response = self._llm._call(prompt)
            print(f"▶️ LLM raw response: {repr(response)}")
            
            # Parse response using same logic as your working approach
            response = response.strip()
            
            # Expected categories (updated with new category)
            categories = [
                "Demographics",
                "Political Opinions", 
                "Values and Social Issues",
                "Lifestyle and Behavioral Opinions",
                "Public Policy & Civic Engagement",
                "Technical / Survey Metadata",
                "Other/Uncategorized"
            ]
            
            # Look for category name directly in response
            category = "Other/Uncategorized"  # default changed to new category
            response_lower = response.lower()
            
            for cat in categories:
                if cat.lower() in response_lower:
                    category = cat
                    break
            
            # Look for numbered responses (1-7) - updated for new category
            if category == "Other/Uncategorized":
                if "1" in response or "demographics" in response_lower:
                    category = "Demographics"
                elif "2" in response or "political" in response_lower:
                    category = "Political Opinions"
                elif "3" in response or "values" in response_lower or "social" in response_lower:
                    category = "Values and Social Issues"
                elif "4" in response or "lifestyle" in response_lower or "behavioral" in response_lower:
                    category = "Lifestyle and Behavioral Opinions"
                elif "5" in response or "policy" in response_lower or "civic" in response_lower:
                    category = "Public Policy & Civic Engagement"
                elif "6" in response or "technical" in response_lower or "metadata" in response_lower:
                    category = "Technical / Survey Metadata"
                elif "7" in response or "other" in response_lower or "uncategorized" in response_lower:
                    category = "Other/Uncategorized"
            
            print(f"▶️ Tool returning category: {category}")
            return category
            
        except Exception as e:
            print(f"▶️ Tool error: {str(e)}")
            return "Other/Uncategorized"

# ============================================================
# UPDATED DIRECT TOOL APPROACH WITH LABELLED VALUES
# ============================================================

@app.function(
    image=image,
    gpu=GPU_TYPE,
    memory=MEMORY_GB * 1024,
    timeout=3600
)
def classify_with_direct_tool(survey_json: str, survey_filename: str = "survey") -> str:
    """APPROACH 1: Direct tool usage with labelled values support"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import json
    
    survey_data = json.loads(survey_json)
    
    print(f"🤖 Loading {MODEL_NAME} for direct tool usage...")
    
    # Load model (same as working approach)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        print("✅ Model loaded without quantization")
    except Exception as e:
        print(f"Trying with quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        print("✅ Model loaded with quantization")
    
    # Create LLM and tool with improved parameters
    llm = GraniteLLM(model=model, tokenizer=tokenizer)
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
            category = tool._run(tool_input)
            
            # Add category to existing question data
            q_with_category = q.copy()
            q_with_category['category'] = category
            
            results[name] = q_with_category
            
            # Track uncategorized questions
            if category == "Other/Uncategorized":
                uncategorized_questions[name] = q_with_category
            
            print(f"   ✅ Classified as: {category}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
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
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"\n🎉 Completed all {len(survey_data)} direct tool classifications!")
    print(f"📊 Summary: {len(uncategorized_questions)} questions classified as Other/Uncategorized")
    
    return json.dumps(results)

@app.function(
    image=image,
    gpu=GPU_TYPE,
    memory=MEMORY_GB * 1024,
    timeout=3600
)
def classify_with_simple_agent(survey_json: str, survey_filename: str = "survey") -> str:
    """APPROACH 2: Improved ultra-simple agent with labelled values support"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import json
    
    survey_data = json.loads(survey_json)
    
    print(f"🤖 Loading {MODEL_NAME} for simple agent...")
    
    # Load model (same as working approach)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        print("✅ Model loaded without quantization")
    except Exception as e:
        print(f"Trying with quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        print("✅ Model loaded with quantization")
    
    # Create LLM and tool with improved parameters
    llm = GraniteLLM(model=model, tokenizer=tokenizer)
    tools = [ClassifyQuestionTool(llm=llm)]
    
    # Updated ReAct template with new category
    react_template = """You have exactly one tool:

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
Action Input: <the question data as JSON>
Observation: <tool output>
Thought: Based on the tool output, I can determine the final category.
Final Answer: <one of the seven category names exactly>

STOP IMMEDIATELY AFTER THE FIRST "Final Answer:" LINE. DO NOT GENERATE ANYTHING ELSE.

Begin!

Tools: {tools}
Tool Names: {tool_names}

Question Data: {input}
{agent_scratchpad}"""
    
    react_prompt = PromptTemplate(
        input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
        template=react_template
    )
    
    # Create improved agent with multi-step reasoning capability
    agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_execution_time=30,
        callbacks=[StdOutCallbackHandler()],
    )
    
    print(f"🔧 Improved agent created, processing {len(survey_data)} questions...")
    
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
            
            # Feed question data to agent
            result = agent_executor.invoke({"input": question_data})
            
            # Extract reasoning steps and show the agent's thought process
            raw_output = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])
            
            print(f"   🧠 Agent Reasoning Steps:")
            for j, (action, observation) in enumerate(intermediate_steps, 1):
                print(f"      Step {j}:")
                print(f"        Action: {action.tool}")
                print(f"        Observation: {observation}")
            
            # Extract only the final answer to prevent hallucinations
            final_answer = extract_final_answer(raw_output)
            
            # Extract category from final answer
            category = "Other/Uncategorized"  # default updated
            
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
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
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
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"\n🎉 Completed all {len(survey_data)} improved agent classifications!")
    print(f"📊 Summary: {len(uncategorized_questions)} questions classified as Other/Uncategorized")
    
    return json.dumps(results)

@app.local_entrypoint()
def main():
    """Test both approaches with improvements and labelled values"""
    
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
    
    print("🚀 APPROACH 1: Direct Tool Usage with Labelled Values")
    print("=" * 60)
    print("🔧 Enhanced with labelled_values support")
    print("📁 Saves uncategorized questions to separate file")
    print("📊 Added 'Other/Uncategorized' category")
    print()
    
    # Test direct tool approach
    results_json = classify_with_direct_tool.remote(survey_json, survey_filename)
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