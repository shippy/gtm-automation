# LangGraph Data Enrichment Template

[![CI](https://github.com/langchain-ai/data-enrichment/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/langchain-ai/data-enrichment/actions/workflows/unit-tests.yml)
[![Integration Tests](https://github.com/langchain-ai/data-enrichment/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/langchain-ai/data-enrichment/actions/workflows/integration-tests.yml)

This is a starter project to help you get started with developing a data enrichment agent using [LangGraph](https://github.com/langchain-ai/langgraph) in [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio).

![](/static/studio.png)

It contains an example graph exported from `src/enrichment_agent/graph.py` that implements a research assistant capable of automatically gathering information on various topics from the web.

## What it does

The enrichment agent:

1. Takes a research topic as input
2. Searches the web for relevant information
3. Reads and extracts key details from websites
4. Organizes the findings into a structured format
5. Validates the gathered information for completeness and accuracy

By default, it's set up to gather information based on the user-provided schema passed through the `template_schema` key in the state.


## Getting Started

This template was designed for [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio). To use, clone this repo locally and open the folder in LangGraph Studio.

Note that the `Deploy` button is currently not supported, but will be soon!

You will need the latest versions of `langgraph` and `langchain`. See these instructions for help upgrading an [existing project](https://github.com/langchain-ai/langgraph#installation).

You can also [click here](https://www.loom.com/share/81cafa32d57f4933bd5d9b08c70f460c?sid=4ebcb366-f27a-4c49-854d-169106b4f6fe) to see a (rough) video tour of Studio.

To set up:

1. Set up your API keys for the LLM (Claude) and search tool (Tavily) in the `.env` file.
2. Install the required dependencies (`poetry install`)
3. Customize whatever you'd like in the code.
4. Run the script with your research topic as input.

## Repo Structure

```txt
├── LICENSE
├── README.md
├── langgraph.json
├── pyproject.toml
├── src
│   └── enrichment_agent
│       ├── __init__.py
│       ├── configuration.py
│       ├── graph.py
│       ├── state.py
│       ├── tools.py
│       └── utils.py
├── static
│   └── studio_ui.png
└── tests
    ├── integration_tests
    │   ├── __init__.py
    │   └── test_graph.py
    └── unit_tests
        └── __init__.py
```

## Development

While iterating on your graph, you can edit past state and rerun your app from past states to debug specific nodes. Local changes will be automatically applied via hot reload. Try adding an interrupt before the agent calls tools, updating the default system message in `src/enrichment_agent/utils.py` to take on a persona, or adding additional nodes and edges!

Follow up requests will be appended to the same thread. You can create an entirely new thread, clearing previous history, using the `+` button in the top right.

You can find the latest (under construction) docs on [LangGraph](https://github.com/langchain-ai/langgraph) here, including examples and other references. Using those guides can help you pick the right patterns to adapt here for your use case.

LangGraph Studio also integrates with [LangSmith](https://smith.langchain.com/) for more in-depth tracing and collaboration with teammates.

## How to extend it

1. **Customize research targets**: Modify the `InfoSchema` to gather different types of information.
2. **Enhance data sources and validation**: Add new tools, APIs, or implement more rigorous fact-checking.
3. **Improve output and interactivity**: Develop custom formatting and user feedback mechanisms.