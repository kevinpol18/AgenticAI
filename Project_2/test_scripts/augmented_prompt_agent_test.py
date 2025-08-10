from workflow_agents.base_agents import AugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

augmented_prompt_agent = AugmentedPromptAgent(openai_api_key, persona)
augmented_agent_response = augmented_prompt_agent.respond(prompt)


# Print the agent's response
print(augmented_agent_response)

print("**Notes**")
print("The agent still used its pre training knowledge base from the LLM but now used the persona in the system prompt to influence its response")
print("The response now starts with 'Dear students' and ends with the sign-off")