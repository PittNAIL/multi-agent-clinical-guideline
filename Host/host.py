# Host/host.py

import requests
import json
import uuid
import time

class HostAgent:
    def __init__(self, intake_agent_url, plan_review_agent_url):
        self.intake_agent_url = intake_agent_url
        self.plan_review_agent_url = plan_review_agent_url
        self.session_id = str(uuid.uuid4())
        
        # Discover agent endpoints from their cards
        self.intake_endpoint = self._discover_endpoint(intake_agent_url)
        self.plan_review_endpoint = self._discover_endpoint(plan_review_agent_url)

    def _discover_endpoint(self, agent_base_url, retries=5, delay=1):
        """Tries to discover an agent's endpoint, with retries."""
        for i in range(retries):
            try:
                response = requests.get(f"{agent_base_url}/agent")
                response.raise_for_status()
                card = response.json()
                print(f"Successfully discovered {card.get('name')} at {agent_base_url}")
                return f"{agent_base_url}/task"
            except requests.exceptions.RequestException as e:
                print(f"Could not discover agent at {agent_base_url} (attempt {i+1}/{retries}): {e}")
                if i < retries - 1:
                    time.sleep(delay)
        return None

    def process_treatment_plan(self, raw_text: str):
        """
        Orchestrates the full treatment plan review process.
        """
        if not self.intake_endpoint or not self.plan_review_endpoint:
            return {"error": "Failed to discover one or more agent services."}
            
        print("\n1. Contacting Intake Agent to structure the plan...")
        structured_response = self._call_intake_agent(raw_text)
        if not structured_response:
            return {"error": "Failed to get structured plan from Intake Agent."}

        print("\n2. Intake Agent returned structured plan:")
        intake_output = structured_response.get('intake_output', {})
        print(json.dumps(intake_output, indent=2))

        diagnosis = intake_output.get("plan", {}).get("diagnosis")
        steps = intake_output.get("plan", {}).get("steps")

        if not diagnosis or not steps:
            return {"error": "Structured plan is missing diagnosis or steps."}

        print("\n3. Contacting Plan Review Agent for compliance check...")
        review_response = self._call_plan_review_agent(diagnosis, steps)
        if not review_response:
            return {"error": "Failed to get review from Plan Review Agent."}
            
        print("\n4. Plan Review Agent returned:")
        review_result = review_response.get('review_result', {})
        print(json.dumps(review_result, indent=2))
        return review_result

    def _call_intake_agent(self, raw_text):
        try:
            payload = {
                "id": str(uuid.uuid4()),
                "sessionId": self.session_id,
                "message": {"raw_text": raw_text}
            }
            response = requests.post(self.intake_endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error calling Intake Agent: {e}")
            return None

    def _call_plan_review_agent(self, diagnosis, steps):
        try:
            payload = {
                "id": str(uuid.uuid4()),
                "sessionId": self.session_id,
                "message": {"diagnosis": diagnosis, "steps": steps}
            }
            response = requests.post(self.plan_review_endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error calling Plan Review Agent: {e}")
            return None 