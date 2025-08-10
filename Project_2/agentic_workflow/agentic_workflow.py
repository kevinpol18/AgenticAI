# agentic_workflow.py

from workflow_agents.base_agents import ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load the OpenAI key into a variable called openai_api_key
openai_api_key = os.getenv("OPENAI_API_KEY")

# load the product spec
with open('/workspace/cd14525-agentic-workflows-classroom/project/starter/phase_2/Product-Spec-Email-Router.txt', 'r') as file:
    product_spec = file.read()

# Instantiate all the agents

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)

# Instantiate an action_planning_agent using the 'knowledge_action_planning'
action_planning_agent = ActionPlanningAgent(openai_api_key, knowledge_action_planning)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."

knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    "Product Spec: \n"
    f"{product_spec}"
)

# Instantiate a product_manager_knowledge_agent using 'persona_product_manager' and the completed 'knowledge_product_manager'
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_product_manager, knowledge_product_manager)

# Product Manager - Evaluation Agent

# Define the persona for the evaluation agent
persona_product_manager_evaluator = (
    "You are an evaluation agent that checks the answers of other worker agents"
)

# Define the evaluation criteria for user stories
evaluation_criteria = (
"The answer should be stories that follow the following structure: As a [type of user], I want [an action or feature] so that [benefit/value]."
)

# Instantiate the evaluation agent
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_product_manager_evaluator,
    evaluation_criteria,
    product_manager_knowledge_agent,
    10
)

# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."

# Instantiate a program_manager_knowledge_agent using 'persona_program_manager' and 'knowledge_program_manager'
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_program_manager, knowledge_program_manager)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."

evaluation_criteria_program_manager = (
    "The answer should be product features that follow the following structure:\n"
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user"
)

program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_program_manager_eval,
    evaluation_criteria_program_manager,
    program_manager_knowledge_agent,
    10
)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."

# Instantiate a development_engineer_knowledge_agent using 'persona_dev_engineer' and 'knowledge_dev_engineer'
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_dev_engineer, knowledge_dev_engineer )

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."

evaluation_criteria_dev_engineer = (
    "The answer should be tasks following this exact structure:\n"
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first"
)

development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_dev_engineer_eval,
    evaluation_criteria_dev_engineer,
    development_engineer_knowledge_agent,
    10
)


# Routing Agent

# Instantiate the routing agent with the OpenAI API key and empty initial config
routing_agent = RoutingAgent(openai_api_key, {})

# Job function persona support functions
def product_manager_support_function(query):
    response = product_manager_knowledge_agent.respond(query)    
    evaluation = product_manager_evaluation_agent.evaluate(response)
    return evaluation['final_response']

def program_manager_support_function(query):
    response = program_manager_knowledge_agent.respond(query)
    evaluation = program_manager_evaluation_agent.evaluate(response)
    return evaluation['final_response']

def development_engineer_support_function(query):
    response = development_engineer_knowledge_agent.respond(query)
    evaluation = development_engineer_evaluation_agent.evaluate(response)
    return evaluation['final_response']

# Define the list of agents with their routing metadata and response functions
agents = [
    {
        "name": "product manager agent",
        "description": "Generates user stories for the product based on the spec",
        "func": product_manager_support_function
    },
    {
        "name": "program manager agent",
        "description": "Defines features by organizing related user stories",
        "func": program_manager_support_function
    },
    {
        "name": "development engineer agent",
        "description": "Defines development tasks needed to implement user stories",
        "func": development_engineer_support_function
    }
]

# Assign the agents list to the routing_agent's 'agents' attribute
routing_agent.agents = agents

# Open a file in write mode once
with open("workflow_output.txt", "w", encoding="utf-8") as f:

    def print_to_file(*args, **kwargs):
        # Convert all args to string and join with space
        text = " ".join(str(arg) for arg in args)
        # Write to file with newline
        f.write(text + "\n")
        f.flush()  # flush after every write to ensure immediate write to file

    print_to_file("\n*** Workflow execution started ***\n")

    # Workflow Prompt
    workflow_prompt = "What would the development tasks for this product be?"
    print_to_file(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

    print("\nDefining workflow steps from the workflow prompt")

    workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)

    completed_steps = []

    for i, step in enumerate(workflow_steps, start=1):
        step = step.strip()
        if not step:
            continue

        print(f"\nExecuting Step {i}: {step}")
        result = routing_agent.route(step)
        completed_steps.append(result)
        print(f"Result for Step {i}:\n{result}")

    if completed_steps:
        print_to_file("\n*** Workflow execution completed ***")
        # Log all results
        print_to_file("\nFinal Output of All Steps:")
        for idx, output in enumerate(completed_steps, start=1):
            print_to_file(f"Step {idx} Output:\n{output}")
    else:
        print_to_file("No workflow steps were executed.")