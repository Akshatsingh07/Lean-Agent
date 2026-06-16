from pydantic import BaseModel, Field
from typing import Any

class ToolSchema(BaseModel):
    name: str=Field(description="Unique Tool Identifier, snake_case")
    description: str=Field(description="Rich natural language description- this gets embedded")
    category: str=Field(description="Department/system this tool belongs to")
    parameters: dict[str,Any]=Field(description="Expected input parameters and their types.")
    returns: str=Field(description="What data this tool gives back")
    
ENTERPRISE_TOOLS: list[ToolSchema]=[
    ToolSchema(
        name="create_jira_ticket",
        description="Create a new Jira issue or ticket in a specified project. Use when a user wants to log a bug, create a task, report an incident, or add a feature request to Jira. Requires project key, summary title, and issue type.",
        category="Jira",
        parameters={"project_key": "str", "summary": "str", "issue_type": "str", "description": "str", "priority": "str"},
        returns="Ticket ID and URL of the created Jira issue"
    ),
    ToolSchema(
        name="get_jira_ticket",
        description="Fetch details of an existing Jira ticket by its ID. Use when a user asks about the status, assignee, comments, or description of a specific Jira issue. Returns full ticket metadata.",
        category="Jira",
        parameters={"ticket_id": "str"},
        returns="Full Jira ticket object with status, assignee, description, comments"
    ),
    ToolSchema(
        name="update_jira_ticket_status",
        description="Transition a Jira ticket to a new status such as In Progress, Done, Blocked, or In Review. Use when a user wants to move a ticket through the workflow or mark it as complete.",
        category="Jira",
        parameters={"ticket_id": "str", "new_status": "str"},
        returns="Confirmation of status transition"
    ),
    ToolSchema(
        name="assign_jira_ticket",
        description="Assign a Jira ticket to a specific team member or user. Use when a user wants to delegate a task or change who is responsible for a Jira issue.",
        category="Jira",
        parameters={"ticket_id": "str", "assignee_email": "str"},
        returns="Confirmation of assignment"
    ),
    ToolSchema(
        name="search_jira_tickets",
        description="Search for Jira tickets using JQL (Jira Query Language) or plain text. Use when a user wants to find all open bugs, unassigned tasks, tickets in a sprint, or issues by label.",
        category="Jira",
        parameters={"jql_query": "str", "max_results": "int"},
        returns="List of matching Jira tickets with key, summary, status"
    ),
    ToolSchema(
        name="add_jira_comment",
        description="Add a comment to an existing Jira ticket. Use when a user wants to leave a note, update stakeholders, or provide context on a Jira issue.",
        category="Jira",
        parameters={"ticket_id": "str", "comment_body": "str"},
        returns="Confirmation with comment ID"
    ),
    ToolSchema(
        name="get_jira_sprint",
        description="Retrieve the current active sprint for a Jira board including all tickets in that sprint. Use when a user asks what's in the current sprint or wants a sprint overview.",
        category="Jira",
        parameters={"board_id": "str"},
        returns="Sprint details with all tickets, story points, and assignees"
    ),
    ToolSchema(
        name="get_jira_project_velocity",
        description="Calculate the velocity of a Jira project by analyzing completed story points over recent sprints. Use for sprint planning or when the user asks about team performance metrics.",
        category="Jira",
        parameters={"project_key": "str", "num_sprints": "int"},
        returns="Average velocity, completed story points per sprint, trend"
    ),
    ToolSchema(
        name="send_slack_message",
        description="Send a message to a Slack channel or direct message to a user. Use when a user wants to notify a team, post an update, or send a DM on Slack.",
        category="Slack",
        parameters={"channel_or_user": "str", "message": "str", "thread_ts": "str | None"},
        returns="Message timestamp and delivery confirmation"
    ),
    ToolSchema(
        name="get_slack_channel_history",
        description="Retrieve recent message history from a Slack channel. Use when a user wants to see what was discussed, find a decision made in Slack, or review conversation history.",
        category="Slack",
        parameters={"channel_name": "str", "limit": "int"},
        returns="List of messages with sender, content, and timestamp"
    ),
    ToolSchema(
        name="create_slack_channel",
        description="Create a new Slack channel with a given name and invite initial members. Use when a user wants to set up a new project channel or team space in Slack.",
        category="Slack",
        parameters={"channel_name": "str", "members": "list[str]", "is_private": "bool"},
        returns="Channel ID and invite link"
    ),
    ToolSchema(
        name="search_slack_messages",
        description="Search across all Slack messages for a keyword, phrase, or topic. Use when a user wants to find a past conversation, decision, or mention of a specific topic in Slack.",
        category="Slack",
        parameters={"query": "str", "channel_filter": "str | None"},
        returns="List of matching messages with context"
    ),
    ToolSchema(
        name="get_slack_user_status",
        description="Check the current status, availability, and timezone of a Slack user. Use when a user wants to know if someone is online, on vacation, or in a meeting.",
        category="Slack",
        parameters={"user_email": "str"},
        returns="User status, availability, local time, and status emoji"
    ),
    ToolSchema(
        name="schedule_slack_message",
        description="Schedule a Slack message to be sent at a future date and time. Use when a user wants to send a reminder or announcement at a specific time.",
        category="Slack",
        parameters={"channel": "str", "message": "str", "send_at": "datetime"},
        returns="Scheduled message ID and confirmation"
    ),
    ToolSchema(
        name="add_slack_reaction",
        description="Add an emoji reaction to a specific Slack message. Use when automating acknowledgements or voting reactions on a Slack post.",
        category="Slack",
        parameters={"channel": "str", "message_ts": "str", "emoji": "str"},
        returns="Confirmation of reaction added"
    ),
    ToolSchema(
        name="run_sql_query",
        description="Execute a read-only SQL SELECT query against the company's analytics data warehouse. Use when a user wants to retrieve data, generate reports, or answer data questions. Never use for write operations.",
        category="Database",
        parameters={"sql_query": "str", "database": "str", "limit": "int"},
        returns="Query results as a list of rows with column names"
    ),
    ToolSchema(
        name="get_table_schema",
        description="Retrieve the schema and column definitions for a database table. Use when a user asks what columns exist in a table, or when you need to understand data structure before writing SQL.",
        category="Database",
        parameters={"table_name": "str", "database": "str"},
        returns="Column names, data types, nullable flags, and sample values"
    ),
    ToolSchema(
        name="list_database_tables",
        description="List all available tables in a given database or data warehouse. Use when a user doesn't know what tables are available or wants to explore the data catalogue.",
        category="Database",
        parameters={"database": "str", "filter_prefix": "str | None"},
        returns="List of table names with row counts and last updated timestamps"
    ),
    ToolSchema(
        name="get_revenue_metrics",
        description="Retrieve key revenue metrics including MRR, ARR, churn rate, and growth rate from the financial database. Use when a user asks about company revenue, financial performance, or growth metrics.",
        category="Database",
        parameters={"time_period": "str", "breakdown_by": "str | None"},
        returns="Revenue metrics with time series data"
    ),
    ToolSchema(
        name="get_user_analytics",
        description="Fetch user behavior analytics including DAU, MAU, retention rate, and feature adoption from the product analytics database. Use when a user asks about product usage, user activity, or retention metrics.",
        category="Database",
        parameters={"metric": "str", "start_date": "str", "end_date": "str"},
        returns="Analytics time series with breakdowns"
    ),
    ToolSchema(
        name="run_funnel_analysis",
        description="Analyze conversion funnel performance for a given user flow such as signup, onboarding, or purchase. Returns step-by-step conversion rates and drop-off points.",
        category="Database",
        parameters={"funnel_name": "str", "start_date": "str", "end_date": "str", "segment": "str | None"},
        returns="Funnel steps with conversion rates and absolute user counts"
    ),
    ToolSchema(
        name="export_data_to_csv",
        description="Export query results or a dataset to a CSV file and return a download link. Use when a user wants to download data for Excel or external analysis.",
        category="Database",
        parameters={"sql_query": "str", "filename": "str"},
        returns="Download URL of the generated CSV file"
    ),
    ToolSchema(
        name="create_data_snapshot",
        description="Create a point-in-time snapshot of a database table for auditing or comparison purposes. Use when a user needs to compare current data against a historical baseline.",
        category="Database",
        parameters={"table_name": "str", "snapshot_label": "str"},
        returns="Snapshot ID and record count"
    ),
    ToolSchema(
        name="get_salesforce_lead",
        description="Retrieve details of a lead or prospect in Salesforce CRM by name, email, or ID. Use when a user wants to check the status, owner, or details of a sales lead.",
        category="Salesforce",
        parameters={"identifier": "str", "id_type": "str"},
        returns="Lead object with contact info, status, source, and owner"
    ),
    ToolSchema(
        name="create_salesforce_opportunity",
        description="Create a new sales opportunity in Salesforce linked to an account. Use when a user wants to log a new deal, potential sale, or upsell opportunity in the CRM.",
        category="Salesforce",
        parameters={"account_name": "str", "opportunity_name": "str", "amount": "float", "close_date": "str", "stage": "str"},
        returns="Opportunity ID and Salesforce URL"
    ),
    ToolSchema(
        name="update_salesforce_stage",
        description="Update the stage of an existing Salesforce opportunity such as moving from Prospecting to Proposal or to Closed Won. Use when a deal progresses in the sales pipeline.",
        category="Salesforce",
        parameters={"opportunity_id": "str", "new_stage": "str"},
        returns="Confirmation of stage update"
    ),
    ToolSchema(
        name="get_sales_pipeline",
        description="Retrieve the current sales pipeline overview including all open opportunities, their stages, values, and expected close dates. Use when a user asks about deals in progress or pipeline health.",
        category="Salesforce",
        parameters={"owner_filter": "str | None", "stage_filter": "str | None"},
        returns="List of opportunities with stage, value, and close date"
    ),
    ToolSchema(
        name="log_salesforce_activity",
        description="Log a call, meeting, or email activity against a Salesforce contact or opportunity. Use when a user wants to record that they spoke with a customer or completed a sales activity.",
        category="Salesforce",
        parameters={"record_id": "str", "activity_type": "str", "notes": "str", "date": "str"},
        returns="Activity ID and confirmation"
    ),
    ToolSchema(
        name="get_account_health",
        description="Retrieve account health score, renewal risk, product usage, and customer success metrics for a Salesforce account. Use when a user asks about customer health or churn risk.",
        category="Salesforce",
        parameters={"account_name": "str"},
        returns="Health score, risk level, usage metrics, and renewal date"
    ),
    ToolSchema(
        name="search_salesforce_contacts",
        description="Search Salesforce for contacts by name, company, title, or email. Use when a user wants to find a customer's contact information or look up who to reach at an account.",
        category="Salesforce",
        parameters={"search_query": "str", "account_filter": "str | None"},
        returns="List of matching contacts with email, phone, and title"
    ),
    ToolSchema(
        name="get_quota_attainment",
        description="Calculate a sales rep's quota attainment percentage for the current or specified quarter. Use when a user asks how a rep is performing against their sales target.",
        category="Salesforce",
        parameters={"rep_email": "str", "quarter": "str"},
        returns="Quota target, achieved revenue, attainment percentage"
    ),
    ToolSchema(
        name="create_salesforce_case",
        description="Create a support or service case in Salesforce linked to a customer account. Use when a customer reports an issue and a formal case needs to be opened in the CRM.",
        category="Salesforce",
        parameters={"account_name": "str", "subject": "str", "description": "str", "priority": "str"},
        returns="Case ID and URL"
    ),
    ToolSchema(
        name="create_calendar_event",
        description="Create a new Google Calendar event and send invites to attendees. Use when a user wants to schedule a meeting, standup, review, or any calendar appointment.",
        category="Calendar",
        parameters={"title": "str", "start_time": "datetime", "end_time": "datetime", "attendees": "list[str]", "description": "str"},
        returns="Event ID, calendar link, and confirmation of invites sent"
    ),
    ToolSchema(
        name="get_calendar_availability",
        description="Check the free/busy availability of one or more users on a given date. Use when a user wants to schedule a meeting and needs to know when everyone is free.",
        category="Calendar",
        parameters={"user_emails": "list[str]", "date": "str", "duration_minutes": "int"},
        returns="Available time slots when all attendees are free"
    ),
    ToolSchema(
        name="get_upcoming_meetings",
        description="Retrieve a list of upcoming calendar events for a user. Use when a user asks what meetings they have today, tomorrow, or this week.",
        category="Calendar",
        parameters={"user_email": "str", "days_ahead": "int"},
        returns="List of upcoming events with time, attendees, and location"
    ),
    ToolSchema(
        name="cancel_calendar_event",
        description="Cancel an existing calendar event and notify all attendees. Use when a user wants to cancel a meeting or appointment.",
        category="Calendar",
        parameters={"event_id": "str", "cancellation_message": "str | None"},
        returns="Confirmation that event was cancelled and invites were notified"
    ),
    ToolSchema(
        name="send_email",
        description="Send an email via Gmail to one or more recipients. Use when a user wants to send an email, reply to a thread, or forward an email to someone.",
        category="Calendar",
        parameters={"to": "list[str]", "subject": "str", "body": "str", "cc": "list[str] | None"},
        returns="Message ID and delivery confirmation"
    ),
    ToolSchema(
        name="search_emails",
        description="Search a user's Gmail inbox for emails matching a query. Use when a user wants to find an email from a specific person, about a topic, or from a date range.",
        category="Calendar",
        parameters={"query": "str", "max_results": "int"},
        returns="List of matching emails with sender, subject, date, and snippet"
    ),
    ToolSchema(
        name="get_email_thread",
        description="Retrieve the full thread of an email conversation including all replies. Use when a user wants to review the complete context of an email discussion.",
        category="Calendar",
        parameters={"thread_id": "str"},
        returns="Complete email thread with all messages and attachments"
    ),
    ToolSchema(
        name="get_github_pr",
        description="Retrieve details of a GitHub Pull Request including diff, review status, comments, and CI checks. Use when a user asks about the status of a PR or wants to review code changes.",
        category="GitHub",
        parameters={"repo": "str", "pr_number": "int"},
        returns="PR details with diff, reviews, CI status, and merge readiness"
    ),
    ToolSchema(
        name="create_github_pr",
        description="Create a new GitHub Pull Request from a branch. Use when a user wants to open a PR, request code review, or merge a feature branch.",
        category="GitHub",
        parameters={"repo": "str", "title": "str", "head_branch": "str", "base_branch": "str", "body": "str"},
        returns="PR URL and number"
    ),
    ToolSchema(
        name="get_ci_pipeline_status",
        description="Check the status of CI/CD pipeline runs for a repository or specific commit. Use when a user asks if the build passed, tests are green, or deployment succeeded.",
        category="GitHub",
        parameters={"repo": "str", "branch_or_commit": "str"},
        returns="Pipeline status, step results, failed test names if any"
    ),
    ToolSchema(
        name="get_deployment_status",
        description="Retrieve the status of the latest deployment to production, staging, or any environment. Use when a user asks if a feature is deployed, what version is live, or if a deploy failed.",
        category="GitHub",
        parameters={"environment": "str", "repo": "str | None"},
        returns="Deployment status, version, timestamp, and who triggered it"
    ),
    ToolSchema(
        name="trigger_deployment",
        description="Trigger a deployment pipeline to push code to a specific environment. Use when a user wants to deploy to staging or production. Requires confirmation before executing.",
        category="GitHub",
        parameters={"repo": "str", "environment": "str", "branch": "str"},
        returns="Deployment job ID and monitoring URL"
    ),
    ToolSchema(
        name="get_error_logs",
        description="Retrieve recent application error logs from the logging system for a service or time period. Use when a user wants to investigate an error, check for exceptions, or debug a production issue.",
        category="GitHub",
        parameters={"service_name": "str", "severity": "str", "time_range_minutes": "int"},
        returns="List of error log entries with timestamps, messages, and stack traces"
    ),
    ToolSchema(
        name="get_system_metrics",
        description="Retrieve infrastructure metrics including CPU usage, memory, latency, and error rates for a service. Use when a user wants to check system health or investigate performance issues.",
        category="GitHub",
        parameters={"service_name": "str", "metric_type": "str", "time_range": "str"},
        returns="Time series metric data with current value, average, and anomalies"
    ),
    ToolSchema(
        name="rollback_deployment",
        description="Rollback a service to the previous stable deployment version. Use when there is an incident, a broken deploy, or a user explicitly requests a rollback.",
        category="GitHub",
        parameters={"service_name": "str", "target_version": "str | None"},
        returns="Confirmation of rollback with new version hash"
    ),
    ToolSchema(
        name="get_on_call_schedule",
        description="Retrieve the current on-call rotation schedule for engineering teams. Use when a user asks who is on call, who to page in an incident, or when their on-call shift is.",
        category="GitHub",
        parameters={"team_name": "str", "date": "str | None"},
        returns="Current on-call engineer, backup, and upcoming rotation schedule"
    ),
    ToolSchema(
        name="create_incident",
        description="Create a new incident in the incident management system with severity level and assign responders. Use when a user reports a production outage, data loss, or critical service degradation.",
        category="GitHub",
        parameters={"title": "str", "severity": "str", "description": "str", "affected_services": "list[str]"},
        returns="Incident ID, severity, assigned responders, and war room link"
    ),
    ToolSchema(
        name="get_feature_flags",
        description="Retrieve the current state of feature flags for a service or user segment. Use when a user wants to know if a feature is enabled, who can see it, or what percentage of users it's rolled out to.",
        category="GitHub",
        parameters={"service_name": "str", "flag_name": "str | None"},
        returns="Feature flag states with rollout percentages and enabled segments"
    ),
]

def get_all_tools() -> list[ToolSchema]:
    return ENTERPRISE_TOOLS

def get_tool_by_name(name: str) -> ToolSchema | None:
    for tool in ENTERPRISE_TOOLS:
        if tool.name==name:
            return tool
    return None

if __name__=="__main__":
    tools=get_all_tools()
    print(f"Total tools registered: {len(tools)}")
    
    from collections import Counter
    category_counts=Counter(t.category for t in tools)
    print("\nTools per category:")
    
    for category,count in sorted(category_counts.items()):
        print(f"{category:15}-> {count} tools")
        
    print(f"\nSample tool schema:")
    print(tools[0].model_dump_json(indent=2))