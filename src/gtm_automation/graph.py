"""Define a data enrichment agent.

Works with a chat model with tool calling support.
"""

import json
from typing import Any, Dict, List, Literal, Optional, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Send
from pydantic import BaseModel, Field

from gtm_automation import prompts
from gtm_automation.configuration import Configuration
from gtm_automation.state import (
    Industries,
    IndustryGoToMarketResearch,
    IndustryPicks,
    ResearchOutput,
    StartupDescription,
)
from gtm_automation.tools import scrape_website, search
from gtm_automation.utils import init_model


async def branch_out(
    state: StartupDescription, *, config: Optional[RunnableConfig] = None
) -> IndustryPicks:
    """Branch out to a list of industries that the startup should target."""
    p = prompts.BRANCH_OUT_PROMPT.format(startup=state.startup)
    model = init_model(config)
    response = model.with_structured_output(Industries)
    actual_response = await response.ainvoke(p)
    return cast(IndustryPicks, actual_response)

async def continue_to_research(state: IndustryPicks, *, config: Optional[RunnableConfig] = None) -> list[Send]:
    """Pass the specific industry to a research agent."""
    to_send: list[IndustryGoToMarketResearch] = [
        IndustryGoToMarketResearch(startup=state.startup, industry=industry)
        for industry in state.industries if industry
    ]
    return [Send("call_agent_model", send) for send in to_send]

async def call_agent_model(
    state: IndustryGoToMarketResearch, *, config: Optional[RunnableConfig] = None
) -> ResearchOutput:
    """Call the primary Language Model (LLM) to decide on the next research action.

    This asynchronous function performs the following steps:
    1. Initializes configuration and sets up the 'Info' tool, which is the user-defined extraction schema.
    2. Prepares the prompt and message history for the LLM.
    3. Initializes and configures the LLM with available tools.
    4. Invokes the LLM and processes its response.
    5. Handles the LLM's decision to either continue research or submit final info.
    """
    # Load configuration from the provided RunnableConfig
    configuration = Configuration.from_runnable_config(config)

    # Define the 'Info' tool, which is the user-defined extraction schema
    info_tool = {
        "name": "Info",
        "description": "Call this when you have gathered all the relevant info",
        "parameters": state.model_json_schema(),
    }

    # Format the prompt defined in prompts.py with the extraction schema and topic
    p = configuration.prompt.format(
        info=json.dumps(state.model_json_schema(), indent=2), topic=state.industry, startup=state.startup
    )

    # Create the messages list with the formatted prompt and the previous messages
    messages = [HumanMessage(content=p)] + state.messages

    # Initialize the raw model with the provided configuration and bind the tools
    raw_model = init_model(config)
    model = raw_model.bind_tools([scrape_website, search, info_tool], tool_choice="any")
    response = cast(AIMessage, await model.ainvoke(messages))
    # Initialize info to None
    info = None

    # Check if the response has tool calls
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "Info":
                info = tool_call["args"]
                break
    if info is not None:
        # The agent is submitting their answer;
        # ensure it isn't erroneously attempting to simultaneously perform research
        response.tool_calls = [
            next(tc for tc in response.tool_calls if tc["name"] == "Info")
        ]
    response_messages: List[BaseMessage] = [response]
    if not response.tool_calls:  # If LLM didn't respect the tool_choice
        response_messages.append(
            HumanMessage(content="Please respond by calling one of the provided tools.")
        )

    return ResearchOutput(
        startup=state.startup,
        industry_research=[info] if info else [],
        loop_step=1,
    )

class InfoIsSatisfactory(BaseModel):
    """Validate whether the current extracted info is satisfactory and complete."""

    reason: List[str] = Field(
        description="First, provide reasoning for why this is either good or bad as a final result. Must include at least 3 reasons."
    )
    is_satisfactory: bool = Field(
        description="After providing your reasoning, provide a value indicating whether the result is satisfactory. If not, you will continue researching."
    )
    improvement_instructions: Optional[str] = Field(
        description="If the result is not satisfactory, provide clear and specific instructions on what needs to be improved or added to make the information satisfactory."
        " This should include details on missing information, areas that need more depth, or specific aspects to focus on in further research.",
        default=None,
    )


async def reflect(
    state: ResearchOutput, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Validate the quality of the data enrichment agent's output.

    This asynchronous function performs the following steps:
    1. Prepares the initial prompt using the main prompt template.
    2. Constructs a message history for the model.
    3. Prepares a checker prompt to evaluate the presumed info.
    4. Initializes and configures a language model with structured output.
    5. Invokes the model to assess the quality of the gathered information.
    6. Processes the model's response and determines if the info is satisfactory.
    """
    p = prompts.MAIN_PROMPT.format(
        info=json.dumps(state.model_json_schema(), indent=2), topic=state.startup
    )
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"{reflect.__name__} expects the last message in the state to be an AI message with tool calls."
            f" Got: {type(last_message)}"
        )
    messages = [HumanMessage(content=p)] + state.messages[:-1]
    presumed_info = state.industry_research[-1].model_dump_json()
    checker_prompt = """I am thinking of calling the info tool with the info below. \
Is this good? Give your reasoning as well. \
You can encourage the Assistant to look at specific URLs if that seems relevant, or do more searches.
If you don't think it is good, you should be very specific about what could be improved.

{presumed_info}"""
    p1 = checker_prompt.format(presumed_info=json.dumps(presumed_info or {}, indent=2))
    messages.append(HumanMessage(content=p1))
    raw_model = init_model(config)
    bound_model = raw_model.with_structured_output(InfoIsSatisfactory)
    response = cast(InfoIsSatisfactory, await bound_model.ainvoke(messages))
    if response.is_satisfactory and presumed_info:
        return {
            "info": presumed_info,
            "messages": [
                ToolMessage(
                    tool_call_id=last_message.tool_calls[0]["id"],
                    content="\n".join(response.reason),
                    name="Info",
                    additional_kwargs={"artifact": response.model_dump()},
                    status="success",
                )
            ],
        }
    else:
        return {
            "messages": [
                ToolMessage(
                    tool_call_id=last_message.tool_calls[0]["id"],
                    content=f"Unsatisfactory response:\n{response.improvement_instructions}",
                    name="Info",
                    additional_kwargs={"artifact": response.model_dump()},
                    status="error",
                )
            ]
        }


def route_after_agent(
    state: IndustryGoToMarketResearch,
) -> Literal["reflect", "tools", "call_agent_model", "__end__"]:
    """Schedule the next node after the agent's action.

    This function determines the next step in the research process based on the
    last message in the state. It handles three main scenarios:

    1. Error recovery: If the last message is unexpectedly not an AIMessage.
    2. Info submission: If the agent has called the "Info" tool to submit findings.
    3. Continued research: If the agent has called any other tool.
    """
    last_message = state.messages[-1]

    # "If for some reason the last message is not an AIMessage (due to a bug or unexpected behavior elsewhere in the code),
    # it ensures the system doesn't crash but instead tries to recover by calling the agent model again.
    if not isinstance(last_message, AIMessage):
        return "call_agent_model"
    # If the "Into" tool was called, then the model provided its extraction output. Reflect on the result
    if last_message.tool_calls and last_message.tool_calls[0]["name"] == "Info":
        return "reflect"
    # The last message is a tool call that is not "Info" (extraction output)
    else:
        return "tools"


def route_after_checker(
    state: IndustryGoToMarketResearch, config: RunnableConfig
) -> Literal["__end__", "call_agent_model"]:
    """Schedule the next node after the checker's evaluation.

    This function determines whether to continue the research process or end it
    based on the checker's evaluation and the current state of the research.
    """
    configurable = Configuration.from_runnable_config(config)
    try:
        last_message = state.messages[-1]
    except IndexError:
        raise ValueError(f"{route_after_checker.__name__} expected at least one message in the state.")

    if state.loop_step < configurable.max_loops:
        if not state.specific_steps:
            return "call_agent_model"
        if not isinstance(last_message, ToolMessage):
            raise ValueError(
                f"{route_after_checker.__name__} expected a tool messages. Received: {type(last_message)}."
            )
        if last_message.status == "error":
            # Research deemed unsatisfactory
            return "call_agent_model"
        # It's great!
        return "__end__"
    else:
        return "__end__"


# Create the graph
workflow = StateGraph(
    ResearchOutput, input=StartupDescription, output=ResearchOutput, config_schema=Configuration
)
workflow.add_node(branch_out, input=IndustryPicks)
workflow.add_node(call_agent_model, input=IndustryGoToMarketResearch)
workflow.add_node(reflect, input=ResearchOutput)
workflow.add_node("tools", ToolNode([search, scrape_website]), input=IndustryGoToMarketResearch)

workflow.add_edge("__start__", "branch_out")
workflow.add_conditional_edges("branch_out", continue_to_research, ["call_agent_model"])
workflow.add_conditional_edges("call_agent_model", route_after_agent)
workflow.add_edge("tools", "call_agent_model")
workflow.add_conditional_edges("reflect", route_after_checker)

graph = workflow.compile()
graph.name = "ResearchTopic"
