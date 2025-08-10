# Test script for DirectPromptAgent class

from workflow_agents.base_agents import DirectPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the Capital of France?"

direct_prompt_agent = DirectPromptAgent(openai_api_key)
direct_agent_response = direct_prompt_agent.respond(prompt)

# Print the response from the agent
print(direct_agent_response)

# Print the source of the knowledge of the answer
print("**Notes**")
print("The knowledge source of the LLM Pre-Training. It has not been fine tuned or augmented with new knowledge.")
