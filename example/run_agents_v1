import autogen
import os  # We need this to read the environment variable

# 1. Get your Gemini API key from an environment variable
#    (We'll set this in the terminal in the next step)
gemini_api_key = os.getenv("GOOGLE_API_KEY")

if not gemini_api_key:
    # This check prevents the script from failing deep inside AutoGen
    raise ValueError("GOOGLE_API_KEY environment variable not set. \n"
                     "Run 'export GOOGLE_API_KEY=\"your_key_here\"' in your terminal first.")

# 2. Define the LLM configuration for Gemini
llm_config = {
    # This is the key: AutoGen's "google" type
    # will find and use your 'google-genai' library.
    "api_type": "google",
    
    # The API key you're loading
    "api_key": gemini_api_key,
    
    # The model you want to use
    "model": "gemini-2.5-flash",
}

# 3. Create the "Assistant" agent (The Coder)
assistant = autogen.AssistantAgent(
    name="Assistant_Coder",
    llm_config=llm_config,  # Pass the Gemini config
    system_message="You are a helpful AI assistant. You write Python code to solve tasks. You must provide your answer in a markdown code block ```python."
)

# 4. Create the "User Proxy" agent (The Executor)
user_proxy = autogen.UserProxyAgent(
    name="Code_Executor",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding", # A safe directory to run code
        "use_docker": False,
    },
    llm_config=llm_config,  # Pass the Gemini config here too
    system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet."""
)

# 5. Start the conversation!
print("🤖 Starting the agent chat...")
user_proxy.initiate_chat(
    assistant,
    message="""What is today's date?
Compare the stock price of Tesla (TSLA) and Microsoft (MSFT) for all of 2024.
Plot the result as a line chart and save it to a file named 'stock_prices.png'."""
)
print("✅ Agent chat finished.")
