import streamlit as st
import os
import json
import requests
import time
import datetime
import duckdb
import pandas as pd
from streamlit_ace import st_ace

# Set page configuration
st.set_page_config(
    page_title="AI Code Generator",
    page_icon="💻",
    layout="wide",
)

# Database configuration
DB_PATH = "codecraftai.duckdb"

# Initialize DuckDB connection and tables
@st.cache_resource
def init_db():
    # Connect to DuckDB database (will be created if it doesn't exist)
    conn = duckdb.connect(DB_PATH)
    
    # Create tables if they don't exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS code_history (
            id INTEGER PRIMARY KEY,
            timestamp TIMESTAMP,
            prompt TEXT,
            generated_code TEXT,
            edited_code TEXT,
            temperature FLOAT,
            max_length INTEGER
        )
    """)
    
    # Create a table for user tags/bookmarks
    conn.execute("""
        CREATE TABLE IF NOT EXISTS code_tags (
            code_id INTEGER,
            tag TEXT,
            PRIMARY KEY (code_id, tag),
            FOREIGN KEY (code_id) REFERENCES code_history(id)
        )
    """)
    
    return conn

# Get DuckDB connection
conn = init_db()

# Function to save code to database
def save_code_to_db(prompt, generated_code, edited_code, temperature, max_length, model_name=None):
    timestamp = datetime.datetime.now()
    
    # If model_name is not provided, use the current selected model from session state
    if model_name is None and 'selected_model' in st.session_state:
        model_name = st.session_state.selected_model
    
    # Check if the code_history table has the model_name column
    table_info = conn.execute("PRAGMA table_info(code_history)").fetchall()
    has_model_column = any(col[1] == 'model_name' for col in table_info)
    
    # Add the model_name column if it doesn't exist
    if not has_model_column:
        conn.execute("ALTER TABLE code_history ADD COLUMN model_name TEXT")
    
    # Insert the record with model_name
    conn.execute("""
        INSERT INTO code_history (timestamp, prompt, generated_code, edited_code, temperature, max_length, model_name)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, prompt, generated_code, edited_code, temperature, max_length, model_name))
    
    # Get the ID of the inserted record
    result = conn.execute("SELECT last_insert_rowid()").fetchone()
    return result[0] if result else None

# Function to update edited code in database
def update_edited_code_in_db(code_id, edited_code):
    if code_id is None:
        return False
        
    try:
        conn.execute("""
            UPDATE code_history
            SET edited_code = ?
            WHERE id = ?
        """, (edited_code, code_id))
        return True
    except Exception as e:
        st.error(f"Error updating code in database: {str(e)}")
        return False

# Function to get code history
def get_code_history(limit=10):
    # Check if the code_history table has the model_name column
    table_info = conn.execute("PRAGMA table_info(code_history)").fetchall()
    has_model_column = any(col[1] == 'model_name' for col in table_info)
    
    if has_model_column:
        df = conn.execute(f"""
            SELECT id, timestamp, prompt, temperature, max_length, model_name
            FROM code_history
            ORDER BY timestamp DESC
            LIMIT {limit}
        """).fetchdf()
    else:
        df = conn.execute(f"""
            SELECT id, timestamp, prompt, temperature, max_length
            FROM code_history
            ORDER BY timestamp DESC
            LIMIT {limit}
        """).fetchdf()
    return df

# Function to get code by ID
def get_code_by_id(code_id):
    # Check if the code_history table has the model_name column
    table_info = conn.execute("PRAGMA table_info(code_history)").fetchall()
    has_model_column = any(col[1] == 'model_name' for col in table_info)
    
    if has_model_column:
        result = conn.execute("""
            SELECT prompt, generated_code, edited_code, temperature, max_length, model_name
            FROM code_history
            WHERE id = ?
        """, (code_id,)).fetchone()
        
        if result:
            return {
                'prompt': result[0],
                'generated_code': result[1],
                'edited_code': result[2],
                'temperature': result[3],
                'max_length': result[4],
                'model_name': result[5]
            }
    else:
        result = conn.execute("""
            SELECT prompt, generated_code, edited_code, temperature, max_length
            FROM code_history
            WHERE id = ?
        """, (code_id,)).fetchone()
        
        if result:
            return {
                'prompt': result[0],
                'generated_code': result[1],
                'edited_code': result[2],
                'temperature': result[3],
                'max_length': result[4],
                'model_name': DEFAULT_MODEL  # Default for backward compatibility
            }
    
    return None

# Initialize session state variables if they don't exist
if 'generated_code' not in st.session_state:
    st.session_state.generated_code = ""
if 'user_prompt' not in st.session_state:
    st.session_state.user_prompt = ""
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'edited_code' not in st.session_state:
    st.session_state.edited_code = ""
if 'huggingface_token' not in st.session_state:
    st.session_state.huggingface_token = os.getenv("HUGGINGFACE_TOKEN", "")
if 'current_code_id' not in st.session_state:
    st.session_state.current_code_id = None
if 'selected_history_id' not in st.session_state:
    st.session_state.selected_history_id = None

# Define available CodeGen models with latest versions (CodeGen1, 2, and 2.5)
CODEGEN_MODELS = {
    # CodeGen 1.0 models (original)
    "CodeGen1-350M-mono": "Salesforce/codegen-350M-mono",
    "CodeGen1-2B-mono": "Salesforce/codegen-2B-mono",
    "CodeGen1-6B-mono": "Salesforce/codegen-6B-mono",
    "CodeGen1-16B-mono": "Salesforce/codegen-16B-mono",
    
    # CodeGen 2.0 models
    "CodeGen2-1B": "Salesforce/codegen2-1B",
    "CodeGen2-3.7B": "Salesforce/codegen2-3_7B",
    "CodeGen2-7B": "Salesforce/codegen2-7B", 
    "CodeGen2-16B": "Salesforce/codegen2-16B",
    
    # CodeGen 2.5 models
    "CodeGen2.5-7B": "Salesforce/codegen25-7b-mono",
    "CodeGen2.5-7B-instruct": "Salesforce/codegen25-7b-instruct", 
}

# Default model - CodeGen 2.5 is the latest and best performing
DEFAULT_MODEL = "CodeGen2.5-7B"

# Function to check if HuggingFace API token is valid
def check_huggingface_token(token):
    try:
        headers = {"Authorization": f"Bearer {token}"}
        # Check against the default CodeGen model
        model_id = CODEGEN_MODELS[DEFAULT_MODEL]
        response = requests.get(
            f"https://huggingface.co/api/models/{model_id}",
            headers=headers
        )
        return response.status_code == 200
    except Exception:
        return False

# Function to generate code via Hugging Face API using CodeGen
def generate_code(prompt, api_token, model_name=DEFAULT_MODEL, max_length=1024, temperature=0.2):
    try:
        # Get the full model ID
        model_id = CODEGEN_MODELS.get(model_name, CODEGEN_MODELS[DEFAULT_MODEL])
        
        # Format prompt based on model version
        if "CodeGen1" in model_name:
            # Original CodeGen 1.0 format
            prompt_text = f"# Generate Python code for: {prompt}\n\n```python\n"
        elif "instruct" in model_name.lower():
            # CodeGen 2.5 instruct format
            prompt_text = f"<human>: Write Python code to {prompt}\n<assistant>:\n```python\n"
        elif "2.5" in model_name:
            # CodeGen 2.5 standard format  
            prompt_text = f"def solution(problem):\n    \"\"\"\n    {prompt}\n    \"\"\"\n    "
        else:
            # CodeGen 2.0 format
            prompt_text = f"# Task: {prompt}\n\n# Python solution:\n"
        
        # API request to Hugging Face Inference API
        API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
        headers = {"Authorization": f"Bearer {api_token}"}
        payload = {
            "inputs": prompt_text,
            "parameters": {
                "max_length": int(max_length),
                "temperature": float(temperature),
                "top_p": 0.95,
                "return_full_text": False,
                "num_return_sequences": 1
            }
        }
        
        # Make the request
        response = requests.post(API_URL, headers=headers, json=payload)
        
        # Check if the response is successful
        if response.status_code == 200:
            output = response.json()
            
            if isinstance(output, list) and len(output) > 0:
                generated_text = output[0].get("generated_text", "")
            else:
                generated_text = output.get("generated_text", "")
            
            # Clean up the output to get only the code
            if generated_text.startswith(prompt_text):
                code_part = generated_text[len(prompt_text):]
            else:
                code_part = generated_text
                
            # Further clean up: remove any markdown code block markers
            if "```python" in code_part:
                code_part = code_part.split("```python")[1]
                if "```" in code_part:
                    code_part = code_part.split("```")[0]
            elif "```" in code_part:
                code_part = code_part.split("```")[0]
                
            return code_part.strip()
        elif response.status_code == 503:
            return "# The model is currently loading. Please try again in a moment."
        else:
            return f"# Error: Received status code {response.status_code} from the API.\n# Response: {response.text}"
    except Exception as e:
        st.error(f"Error generating code: {str(e)}")
        return f"# Error generating code: {str(e)}"

# Title and description
st.title("CodeCraftAI: Salesforce CodeGen")
st.markdown("""
This application uses the Salesforce CodeGen models to generate Python code based on your prompt.
CodeGen offers three generations of models:
- **CodeGen 1.0**: Original models released in March 2022
- **CodeGen 2.0**: Improved models with better infill capability (May 2023)
- **CodeGen 2.5**: Latest models with cutting-edge performance (July 2023)

Choose your preferred model and customize generation parameters. You can edit the generated code in the built-in editor and save your work.
""")

# Create a sidebar for API settings and model options
with st.sidebar:
    # Create tabs for Settings and History
    settings_tab, history_tab = st.tabs(["⚙️ Settings", "📜 History"])
    
    with settings_tab:
        st.header("Hugging Face API Settings")
        
        # Input field for HuggingFace API token
        api_token = st.text_input(
            "HuggingFace API Token", 
            value=st.session_state.huggingface_token,
            type="password",
            help="Enter your HuggingFace API token to use the model. Get one at huggingface.co/settings/tokens"
        )
        
        st.session_state.huggingface_token = api_token
        
        # Model settings
        st.header("Model Options")
        
        # Add model selection
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = DEFAULT_MODEL
            
        # Create model selection with improved formatting
        selected_model = st.selectbox(
            "CodeGen Model",
            options=list(CODEGEN_MODELS.keys()),
            format_func=lambda x: x.replace("CodeGen", "CodeGen ").replace("-mono", "").replace("-instruct", " (Instruct)"),
            index=list(CODEGEN_MODELS.keys()).index(st.session_state.selected_model),
            help="Select the CodeGen model to use. Larger models may produce better results but take longer to generate. CodeGen 2.5 provides best performance."
        )
        
        st.session_state.selected_model = selected_model
        
        temperature = st.slider(
            "Temperature", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.2, 
            step=0.1,
            help="Higher values make the output more random, lower values make it more deterministic."
        )
        
        max_length = st.slider(
            "Max Length", 
            min_value=128, 
            max_value=2048, 
            value=1024, 
            step=128,
            help="Maximum length of the generated code."
        )
        
        # Check if token is valid and update model status
        if api_token:
            if check_huggingface_token(api_token):
                st.session_state.model_loaded = True
                st.success("API Token is valid. Model is ready to use!")
            else:
                st.session_state.model_loaded = False
                st.error("Invalid API Token. Please enter a valid HuggingFace API token.")
    
    with history_tab:
        st.header("Code Generation History")
        
        # Get code generation history from the database
        history_df = get_code_history(limit=20)
        
        if history_df.empty:
            st.info("No code generation history yet. Generate some code first!")
        else:
            # Format the timestamps for better display
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            
            # Display the history as a dataframe with selection
            if 'model_name' in history_df.columns:
                st.dataframe(
                    history_df[['timestamp', 'prompt', 'model_name']],
                    column_config={
                        "timestamp": st.column_config.TextColumn("Time"),
                        "prompt": st.column_config.TextColumn("Prompt"),
                        "model_name": st.column_config.TextColumn("Model")
                    },
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.dataframe(
                    history_df[['timestamp', 'prompt']],
                    column_config={
                        "timestamp": st.column_config.TextColumn("Time"),
                        "prompt": st.column_config.TextColumn("Prompt")
                    },
                    use_container_width=True,
                    hide_index=True
                )
            
            # Allow user to select a history entry
            if 'selected_history_id' not in st.session_state:
                st.session_state.selected_history_id = None
                
            selected_id = st.selectbox(
                "Select an entry to view:",
                options=history_df['id'].tolist(),
                format_func=lambda x: f"ID: {x} - {history_df[history_df['id'] == x]['prompt'].iloc[0][:50]}..."
            )
            
            if st.button("Load Selected Code"):
                st.session_state.selected_history_id = selected_id
                code_data = get_code_by_id(selected_id)
                
                if code_data:
                    st.session_state.user_prompt = code_data['prompt']
                    st.session_state.generated_code = code_data['generated_code']
                    st.session_state.edited_code = code_data['edited_code']
                    st.success(f"Loaded code from history (ID: {selected_id})")
                    # Trigger a rerun to update the main UI
                    st.rerun()

# Create two columns for prompt and generated code
col1, col2 = st.columns(2)

with col1:
    st.subheader("Enter your prompt")
    prompt = st.text_area(
        "Describe the Python code you want to generate",
        height=150,
        key="prompt_input",
        help="Example: 'Create a function to calculate the Fibonacci sequence'"
    )
    
    # Generate button
    if st.button("Generate Code", key="generate_button", disabled=not st.session_state.model_loaded):
        if not prompt:
            st.warning("Please enter a prompt first.")
        elif not st.session_state.huggingface_token:
            st.error("Please enter a valid HuggingFace API token first.")
        else:
            with st.spinner("Generating code..."):
                st.session_state.user_prompt = prompt
                generated_code = generate_code(
                    prompt=prompt, 
                    api_token=st.session_state.huggingface_token,
                    model_name=st.session_state.selected_model,
                    max_length=max_length,
                    temperature=temperature
                )
                st.session_state.generated_code = generated_code
                st.session_state.edited_code = generated_code
                
                # Save to database
                code_id = save_code_to_db(
                    prompt=prompt,
                    generated_code=generated_code,
                    edited_code=generated_code,  # Initially, edited code is the same as generated
                    temperature=temperature,
                    max_length=max_length,
                    model_name=st.session_state.selected_model
                )
                st.session_state.current_code_id = code_id
                
                st.success("Code generated successfully and saved to history!")

with col2:
    st.subheader("Generated Code")
    if st.session_state.generated_code:
        # Create a button to copy the code to clipboard
        st.button(
            "Copy to Clipboard",
            key="copy_button",
            help="Copy the generated code to clipboard"
        )
        
        # Display the generated code in a code block with syntax highlighting
        st.code(st.session_state.generated_code, language="python")

# Code Editor Section
if st.session_state.generated_code:
    st.header("Code Editor")
    st.markdown("You can edit the generated code below:")
    
    # Create an Ace editor for Python code
    edited_code = st_ace(
        value=st.session_state.edited_code,
        language="python",
        theme="monokai",
        key="ace_editor",
        height=400,
        font_size=14,
        auto_update=True,
        wrap=True,
    )
    
    st.session_state.edited_code = edited_code
    
    # Save edited code button
    save_col, run_col = st.columns(2)
    
    with save_col:
        if st.button("Save Edited Code", key="save_edited_code"):
            # Check if we have a current code ID (either from generation or loaded from history)
            code_id = st.session_state.current_code_id or st.session_state.selected_history_id
            
            if code_id and update_edited_code_in_db(code_id, edited_code):
                st.success(f"Edited code saved to database (ID: {code_id})!")
            else:
                st.error("Failed to save edited code. No active code session.")
    
    with run_col:
        # Run Code button
        if st.button("Run Code (Preview)"):
            st.subheader("Code Output Preview")
            st.info("This is a preview of what running this code might look like. For security reasons, we don't execute arbitrary code.")
            st.code("# Code execution preview would appear here")
            st.warning("In a production environment, you would implement a secure sandboxed execution environment.")

# Footer
st.markdown("---")
st.markdown("Powered by Salesforce CodeGen models via HuggingFace")
