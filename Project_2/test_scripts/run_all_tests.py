import subprocess
import sys
import os

# We will run this notebook from the phase_1 directory
phase1_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

test_modules = [
    "test_scripts.evaluation_agent_test",
    "test_scripts.action_planning_agent_test",
    "test_scripts.knowledge_augmented_prompt_agent_test",
    "test_scripts.augmented_prompt_agent_test",
    "test_scripts.rag_knowledge_prompt_agent_test",
    "test_scripts.direct_prompt_agent_test",
    "test_scripts.routing_agent_test"
]

output_file = os.path.join(phase1_dir, "all_agents_test_output.txt")

with open(output_file, "w", encoding="utf-8") as f_out:

    for module in test_modules:
        f_out.write(f"\n{'='*40}\nRunning {module}\n{'='*40}\n")
        print(f"Running {module} ...")
        result = subprocess.run(
            [sys.executable, "-m", module],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=phase1_dir,
        )
        f_out.write(result.stdout)
        f_out.write("\n\n")
print(f"All tests completed. Output saved to {output_file}")