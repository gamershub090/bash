"""
EXPERIMENT 1: Logic-based Expert System (Python version)
Official syllabus asks for Prolog. This Python version demonstrates the same
facts + rules + query idea, but it is NOT a Prolog program.
"""

# FACTS: sample customer information
customers = {
    "rahul": {"income": 12, "credit_score": 780, "years_employed": 5, "existing_debt": 2},
    "amit":  {"income": 4,  "credit_score": 610, "years_employed": 1, "existing_debt": 5},
}

# RULES: conditions used to make a decision
def loan_decision(name):
    if name not in customers:
        return "Customer not found"

    c = customers[name]

    # A simple rule-based decision (illustrative thresholds)
    if (c["income"] >= 6 and c["credit_score"] >= 700
            and c["years_employed"] >= 2 and c["existing_debt"] <= 4):
        return "Loan Approved"
    return "Loan Rejected"


# QUERY: ask the system about a customer
name = input("Enter customer name (rahul/amit): ").strip().lower()
print("Customer:", name)
print("Decision:", loan_decision(name))

# Try another query by changing the name above.
