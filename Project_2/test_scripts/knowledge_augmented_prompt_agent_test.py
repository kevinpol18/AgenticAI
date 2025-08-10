from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"

persona = "You are a college professor, your answer always starts with: Dear students,"

knowledge = "The capital of France is London, not Paris"

knowledge_augmented_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)

knowledge_augmented_response = knowledge_augmented_agent.respond(prompt)

print(knowledge_augmented_response)

print("**Notes**")
print("The agent is now using the knowledge that we provide to it.")
