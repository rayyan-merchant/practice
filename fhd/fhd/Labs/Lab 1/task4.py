import random

class Environment:
    def __init__(self, vulner):
        self.vulner = vulner

    def display(self):
        print("System State:")
        for i, state in enumerate(self.vulner):
            print(f"Component {chr(65 + i)}: {state}")
        print()

    def get_vulnerability(self, i):
        return self.vulner[i]


class SecurityAgent:
    def __init__(self):
        self.patching = []

    def Scan(self, env):
        for i, state in enumerate(env.vulner):
            if state == "Safe":
                print(f"Component {chr(65 + i)} is Safe. Logging success.")
            else:
                print(f"Component {chr(65 + i)} has vulnerabilities. Logging warning.")
                self.patching.append(i)

    def PatchingVulnerabilities(self, env):
        for i in self.patching:
            if env.vulner[i] == "Low Risk Vulnerable":
                env.vulner[i] = "Safe"
                print(f"Patched Low Risk Vulnerability in Component {chr(65 + i)}.")
            elif env.vulner[i] == "High Risk Vulnerable":
                print(f"Component {chr(65 + i)} requires premium service to patch High Risk Vulnerabilities.")


def random_vulnerabilities():
    states = ["Safe", "Low Risk Vulnerable", "High Risk Vulnerable"]
    return [random.choice(states) for _ in range(9)]


def Run_Agent(agent, env):
    print("Initial System State:")
    env.display()
    print("Starting System Scan...")
    agent.Scan(env)
    print("\nPatching Vulnerabilities...")
    agent.PatchingVulnerabilities(env)
    print("\nFinal System State:")
    env.display()


env = Environment(random_vulnerabilities())
agent = SecurityAgent()
Run_Agent(agent, env)
