# AgenticAI

Welcome to **AgenticAI**, a collection of projects developed as part of the Udacity Agentic AI Nanodegree program. These projects showcase practical applications of multi-agent systems and AI-driven workflows across various domains.

## Projects Overview

### Project 1: Agentsville Trip Planner  
In this project, you step into the role of an AI Engineer to build the **AgentsVille Trip Planner**, a sophisticated multi-stage AI assistant designed to create personalized travel itineraries.

#### The Scenario: Your Adventure in AgentsVille Awaits!  
Imagine a traveler eager to explore the fictional city of AgentsVille with specific preferences — whether a long weekend focused on art galleries and tech meetups or a week immersed in cultural experiences and street food, all within a budget. They turn to your AI-powered Trip Planner for help.

#### Your Challenge  
Build an AI system that can:  
- **Understand and Interpret:** Take into account user preferences and constraints.  
- **Plan Comprehensively:** Generate a detailed, day-by-day itinerary that forms a coherent plan tailored to the traveler.  
- **Evaluate and Enhance:** Use a set of tools intelligently to evaluate, fetch new information, and refine the itinerary.

This involves designing a system that reasons and interacts with “external” information sources rather than just generating text outputs.

#### Project Description: Building Your AI Travel Assistant  
Your "AgentsVille Trip Planner" is implemented as a Jupyter Notebook application orchestrating interactions with a Large Language Model (LLM) to perform two main functions:

- **The Expert Planner (Initial Itinerary Generation):**  
  - Based on user-defined preferences (destination, duration, interests, budget), prompt the AI to act as an expert travel planner.  
  - The AI generates a detailed, structured day-by-day itinerary in JSON format conforming to a Pydantic model.  
  - Success depends on crafting prompts that guide the LLM through a structured planning process.

- **The Resourceful Assistant (Itinerary Enhancement with a Tool-Using ReAct Agent):**  
  - Handles follow-up questions or modification requests on the itinerary.  
  - Decides if external “tools” (e.g., simulated activities API) can assist.  
  - Generates structured requests (THINK and ACT steps), receives OBSERVATIONS from tool simulations, and iteratively refines answers or itinerary accordingly.

---

### Project 2: AI-Powered Agentic Workflow for Project Management

#### Introduction  
Welcome to the project **AI-Powered Agentic Workflow for Project Management**! Imagine yourself as a highly sought-after AI Workflow Architect who specializes in implementing intelligent agentic systems that don't just automate tasks, but dynamically manage them. Your newest client, **InnovateNext Solutions**, a rapidly scaling startup brimming with brilliant ideas but hampered by inconsistent project execution, has a critical challenge they believe only you can solve.

They are seeking a revolutionary way to manage their entire product development lifecycle. Your goal is to engineer a sophisticated, reusable agentic workflow to assist their technical project managers (TPMs) in consistently and scalably transforming product ideas into well-defined user stories, product features, and detailed engineering tasks. You will pilot this system on their upcoming "Email Router" project.

#### The Challenge: Building a Scalable Engine for Innovation  
TPMs at InnovateNext Solutions face a bottleneck: converting multiple product ideas into actionable development plans leads to miscommunications, varied output quality, and project delays. They need an AI-driven project management framework that can be applied company-wide.

Your role as an AI Workflow Architect is twofold:  
1. Construct a robust library of diverse, reusable AI agents—the core adaptable toolkit.  
2. Deploy a selection of these agents to build a general-purpose agentic workflow for technical project management, demonstrated through the "Email Router" pilot.

#### Your Product: AI-Powered Agentic Workflow for Project Management (Pilot: Email Router)  
You will deliver a two-part solution:

- **Phase 1: The Agentic Toolkit**  
  - A Python package (`workflow_agents`) containing seven meticulously crafted, individually tested agent classes (`base_agents.py`):  
    - DirectPromptAgent  
    - AugmentedPromptAgent  
    - KnowledgeAugmentedPromptAgent  
    - RAGKnowledgePromptAgent (provided)  
    - EvaluationAgent  
    - RoutingAgent  
    - ActionPlanningAgent  
  - Standalone test scripts with evidence of successful runs.

- **Phase 2: The Project Management Workflow Implementation**  
  - A primary Python script (`agentic_workflow.py`) orchestrating selected agents (`ActionPlanningAgent`, `KnowledgeAugmentedPromptAgent`, `EvaluationAgent`, `RoutingAgent`) to execute multi-step technical project management tasks.  
  - Workflow features:  
    - Accepts high-level prompts simulating TPM requests and the product spec for the Email Router.  
    - Breaks down goals into sub-tasks via the Action Planning Agent.  
    - Routes sub-tasks intelligently to specialized agent teams.  
    - Simulates Product Manager, Program Manager, and Development Engineer teams with KnowledgeAugmentedPromptAgents paired with EvaluationAgents to ensure quality.  
    - Produces a comprehensive, structured project plan demonstrating workflow capabilities.

#### Project Submission  
You will submit:  
- The reusable agent library and test scripts (Phase 1).  
- The orchestrating workflow script and output evidence (Phase 2).

---

### Project 3: UdaPlay - An AI Research Agent for the Video Game Industry

#### Project Scenario  
You’ve been hired as an AI Engineer at a gaming analytics company developing an assistant called **UdaPlay**. Executives, analysts, and gamers want to ask natural language questions like:

- “Who developed FIFA 21?”  
- “When was God of War Ragnarok released?”  
- “What platform was Pokémon Red launched on?”  
- “What is Rockstar Games working on right now?”

#### Agent Capabilities  
UdaPlay is designed to:

- Attempt to answer questions from internal knowledge based on a pre-loaded list of companies and games.  
- If information is missing or confidence is low, perform a web search using the Tavily API.  
- Parse and store retrieved information in long-term memory for future use.  
- Generate clean, structured, and well-cited answers or reports.

#### Project Specifications  
- Answer user questions about video games including titles, details, release dates, platforms, descriptions, genres, and publisher information.  
- Utilize a two-tier information retrieval system:  
  - Primary: Retrieval Augmented Generation (RAG) over a local dataset of games.  
  - Secondary: Web search fallback with the Tavily API when internal data is insufficient.  
- Implement a robust evaluation system to:  
  - Assess the quality of retrieved information.  
  - Decide when to fallback to web search.  
  - Provide confidence levels in answers.  
- Generate clear, natural, and readable responses that cite sources and combine multiple information sources when needed.

---

### Project 4: The Beaver's Choice Paper Company Sales Team

**The Beaver's Choice Paper Company Needs Your Help!**

#### Introduction  
Imagine yourself as a trusted and seasoned consultant specializing in building smart, efficient, and powerful agent-based workflows. Businesses rely on you to solve their complex operational challenges with cutting-edge solutions. Your latest client, the Beaver's Choice Paper Company, urgently requires your expertise to revolutionize their inventory management and quoting system. Your role is to develop a multi-agent system that streamlines their operations, enabling quick quote generation, accurate inventory tracking, and ultimately driving increased sales.

#### The Challenge  
The Beaver's Choice Paper Company struggles with managing their paper supplies, responding promptly to customer inquiries, and generating competitive quotes. They are overwhelmed and losing potential sales due to inefficiencies. Your challenge is to design and implement a multi-agent solution, restricted to at most five agents, capable of handling inquiries, checking inventory status, providing accurate quotations, and completing transactions seamlessly. Your solution must ensure responsiveness, accuracy, and reliability in managing requests and maintaining optimal stock levels.

#### Core Components of the Multi-agent System  
- Handle strictly text-based inputs and outputs.  
- Manage inventory questions and reorder supplies when necessary by using database information effectively to make purchase decisions.  
- Provide accurate and intelligent quotes for customers by considering historical quote data and pricing strategies.  
- Finalize sales transactions efficiently based on inventory availability and delivery timelines.  

You may choose among three Python frameworks for your agent implementation: `smolagents`, `pydantic-ai`, or `npcpy`. Your submission includes:  
- A workflow diagram outlining the system design (image file).  
- Source code (one Python file).  
- A detailed document explaining your system and how it meets all requirements.

#### Project Summary  
This 6-hour project involves:  
1. Diagramming & Planning: Create a detailed flow diagram of your multi-agent system.  
2. Implementation: Develop the system using Python and a chosen agent framework.  
3. Testing & Debugging: Test agents with sample inputs for robust performance.  
4. Documentation: Write a comprehensive report on your design and implementation.

Ready to revolutionize inventory and quoting management for the Beaver's Choice Paper Company? Let's get started!

---

Feel free to explore each project to see how agentic AI concepts are applied to solve real-world challenges creatively and efficiently.
