"""
EXPERIMENT 2: Five Types of Intelligent Agents
Simple demonstration using a small synthetic gym-member database.
"""

members = {
    "M01": {"name": "Asha", "goal": "weight loss", "workout": "Cardio", "attendance": 3},
    "M02": {"name": "Rohan", "goal": "muscle gain", "workout": "Strength training", "attendance": 1},
    "M03": {"name": "Neha", "goal": "fitness", "workout": "Mixed workout", "attendance": 5},
}

# 1. SIMPLE REFLEX: action depends only on the current command
def simple_reflex(command, member_id):
    if member_id not in members:
        return "Member not found"
    if command == "profile":
        return members[member_id]
    if command == "workout":
        return members[member_id]["workout"]
    return "Unknown command"


# 2. MODEL-BASED: remembers the selected member
class ModelBasedAgent:
    def __init__(self):
        self.current_member = None

    def run(self, command, member_id=None):
        if command == "select":
            self.current_member = member_id
            return "Selected " + str(member_id)
        if self.current_member not in members:
            return "Select a valid member first"
        return simple_reflex(command, self.current_member)


# 3. GOAL-BASED: makes a complete report to achieve a goal
def goal_based_report(member_id):
    m = members[member_id]
    return f"{m['name']}: goal={m['goal']}, workout={m['workout']}"


# 4. UTILITY-BASED: chooses the member who needs attention most
def utility_based_agent():
    # Lower attendance means greater need for attention
    return min(members, key=lambda x: members[x]["attendance"])


# 5. LEARNING AGENT: learns a simple goal-to-workout mapping from examples
def learning_agent():
    examples = [("weight loss", "Cardio"), ("muscle gain", "Strength training"),
                ("fitness", "Mixed workout")]
    learned_rules = dict(examples)
    new_goal = "weight loss"
    return learned_rules.get(new_goal, "General workout")


print("1. Simple reflex:", simple_reflex("workout", "M01"))
agent = ModelBasedAgent()
print("2. Model-based:", agent.run("select", "M02"), "|", agent.run("workout"))
print("3. Goal-based:", goal_based_report("M03"))
print("4. Utility-based: member needing attention =", utility_based_agent())
print("5. Learning-based: suggested workout =", learning_agent())
