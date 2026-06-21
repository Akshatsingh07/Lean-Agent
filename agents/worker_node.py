import json
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from graph.state import Agentstate

@tool
def create_jira_ticket(
    project_key: str,
    summary: str,
    issue_type: str = "Bug",
    description: str = "",
    priority: str = "Medium"
) -> dict:
    """
    Create a new Jira issue or ticket in a specified project.
    Use when a user wants to log a bug, create a task, report an incident,
    or add a feature request to Jira.
    """
    num = abs(hash(summary)) % 900 + 100
    return {
        "ticket_id": f"{project_key}-{num}",
        "url": f"https://company.atlassian.net/browse/{project_key}-{num}",
        "status": "Open",
        "issue_type": issue_type,
        "priority": priority,
        "summary": summary,
    }

@tool
def get_jira_ticket(ticket_id: str) -> dict:
    """Fetch details of an existing Jira ticket by its ID."""
    return {
        "ticket_id": ticket_id,
        "summary": "Login page crashes on Safari iOS 17",
        "status": "In Progress",
    }

@tool
def search_jira_tickets(jql_query: str, max_results: int = 10) -> dict:
    """Search for Jira tickets using JQL or plain text."""
    return {
        "total": 3,
        "issues": [
            {"id": "PROJ-145", "summary": "Login crash on Safari", "status": "In Progress"},
            {"id": "PROJ-132", "summary": "Slow dashboard load", "status": "Open"},
        ],
        "jql_used": jql_query,
    }

@tool
def send_slack_message(channel_or_user: str, message: str) -> dict:
    """Send a message to a Slack channel or direct message to a user."""
    return {
        "success": True,
        "channel": channel_or_user,
        "message_preview": message[:100],
        "timestamp": "2025-06-14T10:30:00Z",
    }

@tool
def run_sql_query(sql_query: str, database: str = "analytics") -> dict:
    """Execute a read-only SQL SELECT query against the analytics warehouse."""
    return {
        "rows": [{"month": "2025-06", "revenue": 485000}],
        "row_count": 1,
    }

@tool
def get_revenue_metrics(time_period: str = "last_quarter") -> dict:
    """Retrieve key revenue metrics: MRR, ARR, churn rate, growth rate."""
    return {"period": time_period, "mrr": 485000, "arr": 5820000}

@tool
def get_ci_pipeline_status(repo: str, branch_or_commit: str = "main") -> dict:
    """Check the status of CI/CD pipeline runs for a repository or commit."""
    return {"repo": repo, "status": "passed"}

@tool
def get_deployment_status(environment: str, repo: str = "") -> dict:
    """Retrieve the status of the latest deployment to any environment."""
    return {"environment": environment, "status": "healthy", "version": "v2.14.3"}

@tool
def get_error_logs(service_name: str, severity: str = "ERROR", time_range_minutes: int = 60) -> dict:
    """Retrieve recent error logs for a service from the logging system."""
    return {
        "service": service_name,
        "total_errors": 2,
        "logs": [{"time": "2025-06-14T10:22Z", "message": "NullPointerException"}],
    }

@tool
def get_sales_pipeline(stage_filter: str = "") -> dict:
    """Retrieve the current sales pipeline with open opportunities."""
    return {"total_value": 2450000}

ALL_MOCK_TOOLS: dict = {
    "create_jira_ticket":    create_jira_ticket,
    "get_jira_ticket":       get_jira_ticket,
    "search_jira_tickets":   search_jira_tickets,
    "send_slack_message":    send_slack_message,
    "run_sql_query":         run_sql_query,
    "get_revenue_metrics":   get_revenue_metrics,
    "get_ci_pipeline_status": get_ci_pipeline_status,
    "get_deployment_status": get_deployment_status,
    "get_error_logs":        get_error_logs,
    "get_sales_pipeline":    get_sales_pipeline,
}

def get_llm():
    return ChatOllama(
        model="llama3.2",
        temperature=0,
    )


def execute_tool_calls(tool_calls: list, bound_tools: dict) -> list[ToolMessage]:
    messages = []
 
    for tc in tool_calls:
        name = tc["name"]
        args = tc["args"]
        call_id = tc["id"] 
 
        print(f"\nCalling: {name}")
        print(f"Args: {json.dumps(args, indent=8)}")
 
        if name not in bound_tools:
            result = {"error": f"Tool '{name}' has no mock implementation yet."}
        else:
            try:
                result = bound_tools[name].invoke(args)
                print(f"Result: {str(result)[:150]}...")
            except Exception as e:
                result = {"error": str(e)}
                print(f"Error: {e}")
 
        messages.append(ToolMessage(
            content=json.dumps(result), 
            tool_call_id=call_id,
            name=name,
        ))
 
    return messages

def worker_node(state: Agentstate) -> dict:
    print("[NODE 2 - WORKER]")
    retrieved_tools = state.get("retrieved_tools", [])
    query = state["user_query"]
    memory_context = state.get("memory_context", "")
    
    if not retrieved_tools:
        return {
            "messages": [AIMessage(content="No relevant tools found for your request.")],
            "tool_calls_made": [],
            "tool_results": [],
        }

    bound_tool_functions = {}
    for tool_info in retrieved_tools:
        name = tool_info["name"]
        if name in ALL_MOCK_TOOLS:
            bound_tool_functions[name] = ALL_MOCK_TOOLS[name]
        else:
            print(f"{name} has no mock implementation")
            
    if not bound_tool_functions:
        return {
            "messages": [AIMessage(content="Retrieved tools have no implementations yet.")],
            "tool_calls_made": [],
            "tool_results": [],
        }
    llm = get_llm()
    tools_lists = list(bound_tool_functions.values())
    llm_with_tools = llm.bind_tools(tools_lists)
    
    bound_names = [t.name for t in tools_lists]
    implemented_tool_count = len(ALL_MOCK_TOOLS)
    print(f"\n Tools bound to LLM: {bound_names}")
    print(f" Tool implementations available: {implemented_tool_count}")
    print(f" Hidden from LLM:   {implemented_tool_count - len(tools_lists)} implemented tools")
    
    memory_section = f"\n\n{memory_context}" if memory_context else ""
    
    system_prompt = (
        f"You are an enterprise AI assistant with access to {len(tools_lists)} "
        f"relevant company tools. Use them to complete the user's request precisely. "
        f"If tool data is needed, call the best available tool before answering. "
        f"Summarize concrete tool results in a concise final answer."
        f"{memory_section}"
    )
    
    messages = [HumanMessage(content=query)]
    all_tool_calls_made = []
    all_tool_results = []
    
    max_iterations = 5
    for iteration in range(1, max_iterations + 1):
        print(f"LLM iteration {iteration}/{max_iterations}")
        ai_response = llm_with_tools.invoke(
            [{"role": "system", "content": system_prompt}] + messages
        )
        messages.append(ai_response)
        
        if not ai_response.tool_calls:
            print("LLM Done- no more tool calls")
            print(f"Response: {str(ai_response.content)[:200]}")
            break
        print(f"LLM requested {len(ai_response.tool_calls)} tool call(s)")
        all_tool_calls_made.extend([tc["name"] for tc in ai_response.tool_calls])
 
        tool_messages = execute_tool_calls(ai_response.tool_calls, bound_tool_functions)
 
        for tm in tool_messages:
            all_tool_results.append({"tool": tm.name, "result": tm.content})
        messages.extend(tool_messages)
    else:
        messages.append(AIMessage(content="Reached max iterations — task may be incomplete."))
 
    return {
        "messages": messages,
        "tool_calls_made": all_tool_calls_made,
        "tool_results": all_tool_results,
    }
