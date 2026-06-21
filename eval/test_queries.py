"""
eval/test_queries.py
---------------------

PURPOSE:
    A hand-crafted dataset of test queries with "ground truth" answers.
    RAGAS (and our benchmark) use this to measure how well the agent performs.

WHAT IS A TEST DATASET IN RAG EVALUATION?
    In normal ML, you have training data + test data.
    In RAG evaluation, your "test data" is a list of:

        question        → what the user asked
        expected_tools  → which tools SHOULD be retrieved (ground truth)
        expected_answer → what the final answer SHOULD contain (keywords)
        context_desc    → description of the tools that are relevant

    You run the agent on each question, then compare:
        - Did it retrieve the right tools?   → Retrieval accuracy
        - Does the answer contain expected keywords? → Answer quality
        - Did RAGAS score it as faithful?    → RAGAS metrics

WHY HAND-CRAFT THIS INSTEAD OF AUTO-GENERATING?
    Auto-generated test sets often have errors.
    Hand-crafted = you KNOW what the right answer is.
    For a portfolio project: 20 good queries is enough to show rigour.
"""

# Each test case is a dict with these keys:
#   query:          What the user asks (input to the agent)
#   expected_tools: List of tool names that SHOULD be in the top-3 retrieved
#   expected_keywords: Words that should appear in the final answer
#   category:       Which enterprise system is being tested
#   difficulty:     "easy" (obvious keywords), "medium" (needs semantics), "hard" (multi-step)

TEST_QUERIES = [

    # ─── JIRA ────────────────────────────────────────────────────────────────

    {
        "query": "Create a high priority bug ticket titled 'Login crash on Safari iOS 17' in the MOBILE project",
        "expected_tools": ["create_jira_ticket"],
        "expected_keywords": ["MOBILE", "ticket", "created", "bug"],
        "category": "Jira",
        "difficulty": "easy",
    },
    {
        "query": "What is the current status of ticket MOBILE-145?",
        "expected_tools": ["get_jira_ticket"],
        "expected_keywords": ["status", "MOBILE-145"],
        "category": "Jira",
        "difficulty": "easy",
    },
    {
        "query": "Find all open bugs that are currently blocked in our backlog",
        "expected_tools": ["search_jira_tickets"],
        "expected_keywords": ["blocked", "issues", "found"],
        "category": "Jira",
        "difficulty": "medium",
    },
    {
        "query": "Log a new feature request for dark mode support in the UI project",
        "expected_tools": ["create_jira_ticket"],
        "expected_keywords": ["ticket", "UI", "feature"],
        "category": "Jira",
        "difficulty": "easy",
    },
    {
        "query": "Show me all unresolved issues assigned to nobody on the backend team",
        "expected_tools": ["search_jira_tickets"],
        "expected_keywords": ["unassigned", "issues"],
        "category": "Jira",
        "difficulty": "medium",
    },

    # ─── DATABASE / ANALYTICS ─────────────────────────────────────────────────

    {
        "query": "What were our MRR and ARR numbers last quarter?",
        "expected_tools": ["get_revenue_metrics"],
        "expected_keywords": ["MRR", "ARR"],
        "category": "Database",
        "difficulty": "easy",
    },
    {
        "query": "Show me monthly revenue for the past 3 months",
        "expected_tools": ["run_sql_query", "get_revenue_metrics"],
        "expected_keywords": ["revenue", "month"],
        "category": "Database",
        "difficulty": "medium",
    },
    {
        "query": "What is our current customer churn rate and how does it compare to last month?",
        "expected_tools": ["get_revenue_metrics"],
        "expected_keywords": ["churn", "rate"],
        "category": "Database",
        "difficulty": "medium",
    },
    {
        "query": "Run a query to count how many customers signed up this month from the analytics database",
        "expected_tools": ["run_sql_query"],
        "expected_keywords": ["query", "customers", "count"],
        "category": "Database",
        "difficulty": "easy",
    },

    # ─── DEVOPS / GITHUB ──────────────────────────────────────────────────────

    {
        "query": "Is the production deployment currently healthy?",
        "expected_tools": ["get_deployment_status"],
        "expected_keywords": ["production", "healthy", "status"],
        "category": "DevOps",
        "difficulty": "easy",
    },
    {
        "query": "Did the latest CI pipeline pass on the main branch of the backend repo?",
        "expected_tools": ["get_ci_pipeline_status"],
        "expected_keywords": ["pipeline", "passed", "main"],
        "category": "DevOps",
        "difficulty": "easy",
    },
    {
        "query": "Check the error logs for the auth service from the last 30 minutes",
        "expected_tools": ["get_error_logs"],
        "expected_keywords": ["error", "auth", "logs"],
        "category": "DevOps",
        "difficulty": "easy",
    },
    {
        "query": "Are there any critical errors in production right now? Check the API gateway service",
        "expected_tools": ["get_error_logs", "get_deployment_status"],
        "expected_keywords": ["error", "production", "API gateway"],
        "category": "DevOps",
        "difficulty": "hard",
    },

    # ─── SLACK ────────────────────────────────────────────────────────────────

    {
        "query": "Send a message to the #engineering channel saying the deployment is complete",
        "expected_tools": ["send_slack_message"],
        "expected_keywords": ["message", "sent", "engineering"],
        "category": "Slack",
        "difficulty": "easy",
    },
    {
        "query": "Notify the team on Slack that we have a production incident with the login service",
        "expected_tools": ["send_slack_message"],
        "expected_keywords": ["message", "incident", "sent"],
        "category": "Slack",
        "difficulty": "medium",
    },

    # ─── SALESFORCE ───────────────────────────────────────────────────────────

    {
        "query": "What deals are currently in the negotiation stage of our sales pipeline?",
        "expected_tools": ["get_sales_pipeline"],
        "expected_keywords": ["negotiation", "pipeline", "deals"],
        "category": "Salesforce",
        "difficulty": "medium",
    },
    {
        "query": "Show me the total value of our current sales pipeline",
        "expected_tools": ["get_sales_pipeline"],
        "expected_keywords": ["pipeline", "value", "total"],
        "category": "Salesforce",
        "difficulty": "easy",
    },

    # ─── MULTI-STEP (hard) ────────────────────────────────────────────────────

    {
        "query": "Check if the auth service has errors, and if it does, create a Jira ticket for it",
        "expected_tools": ["get_error_logs", "create_jira_ticket"],
        "expected_keywords": ["error", "ticket", "auth"],
        "category": "Multi-step",
        "difficulty": "hard",
    },
    {
        "query": "Is our deployment healthy and what are our revenue metrics this month?",
        "expected_tools": ["get_deployment_status", "get_revenue_metrics"],
        "expected_keywords": ["deployment", "revenue", "healthy"],
        "category": "Multi-step",
        "difficulty": "hard",
    },

    # ─── EDGE CASES ───────────────────────────────────────────────────────────

    {
        "query": "I need to report a problem with our software tracking system",
        # Tests semantic retrieval — "problem with software" → Jira
        # BM25 alone would miss this (no exact keywords match)
        "expected_tools": ["create_jira_ticket", "search_jira_tickets"],
        "expected_keywords": ["ticket", "issue", "created"],
        "category": "Jira",
        "difficulty": "hard",   # purely semantic, no keywords
    },
    {
        "query": "Hi, my name is Akshat and I am the lead developer for the MOBILE project.",
        "expected_tools": [],
        "expected_keywords": ["Akshat", "MOBILE"],
        "category": "Memory",
        "difficulty": "easy",
    },
    {
        "query": "Create a high priority bug ticket for a login crash on Safari iOS 17.",
        "expected_tools": ["create_jira_ticket"],
        "expected_keywords": ["ticket", "Safari", "iOS"],
        "category": "Jira",
        "difficulty": "easy",
    },
]


def get_test_queries() -> list[dict]:
    """Returns the full test dataset."""
    return [q for q in TEST_QUERIES if isinstance(q, dict)]


def get_queries_by_difficulty(difficulty: str) -> list[dict]:
    """Filter test queries by difficulty: 'easy', 'medium', or 'hard'."""
    return [q for q in TEST_QUERIES if q["difficulty"] == difficulty]


def get_queries_by_category(category: str) -> list[dict]:
    """Filter test queries by category: 'Jira', 'Database', 'DevOps', etc."""
    return [q for q in TEST_QUERIES if q["category"] == category]


if __name__ == "__main__":
    from collections import Counter
    queries = get_test_queries()
    print(f"Total test queries: {len(queries)}")
    print(f"\nBy difficulty:")
    for d, c in Counter(q["difficulty"] for q in queries).items():
        print(f"  {d:8} → {c}")
    print(f"\nBy category:")
    for cat, c in Counter(q["category"] for q in queries).items():
        print(f"  {cat:15} → {c}")
