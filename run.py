#!/usr/bin/env python3
# run.py

import argparse
import subprocess
import time
import sys
from Host.host import HostAgent

def install_dependencies():
    """Installs dependencies from requirements.txt."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        exit(1)

def start_agent_service(agent_name, script_path, port):
    """Starts an agent service as a background process."""
    try:
        process = subprocess.Popen(
            [sys.executable, script_path, '--port', str(port)],
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        print(f"{agent_name} service started on port {port} (PID: {process.pid})")
        time.sleep(2)  # Give the server a moment to start
        return process
    except Exception as e:
        print(f"Failed to start {agent_name}: {e}")
        return None

def main():
    install_dependencies()
    
    parser = argparse.ArgumentParser(description="Run the full Clinical Treatment Plan Review System")
    parser.add_argument(
        "treatment_plan",
        type=str,
        help="The raw text of the treatment plan to be reviewed."
    )
    args = parser.parse_args()

    # Agent service configurations
    plan_review_agent_port = 5001
    intake_agent_port = 5002
    
    plan_review_script = "Agents/Plan_review_Agent/client.py"
    intake_agent_script = "Agents/intake_agent/client.py"

    # Start services
    print("--- Starting Agent Services ---")
    plan_review_process = start_agent_service("PlanReviewAgent", plan_review_script, plan_review_agent_port)
    intake_process = start_agent_service("IntakeAgent", intake_agent_script, intake_agent_port)

    if not plan_review_process or not intake_process:
        print("Could not start all agent services. Exiting.")
        if plan_review_process: plan_review_process.terminate()
        if intake_process: intake_process.terminate()
        return

    try:
        # Initialize and run the HostAgent
        print("\n--- Initializing Host Agent ---")
        host = HostAgent(
            intake_agent_url=f"http://localhost:{intake_agent_port}",
            plan_review_agent_url=f"http://localhost:{plan_review_agent_port}"
        )
        
        print("\n--- Processing Treatment Plan ---")
        final_review = host.process_treatment_plan(args.treatment_plan)
        
        print("\n--- Final Review ---")
        print(final_review)

    finally:
        # Shutdown services
        print("\n--- Shutting Down Agent Services ---")
        if plan_review_process:
            plan_review_process.terminate()
            print("PlanReviewAgent service stopped.")
        if intake_process:
            intake_process.terminate()
            print("IntakeAgent service stopped.")

if __name__ == "__main__":
    main()
