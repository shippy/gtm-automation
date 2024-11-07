"""Default prompts used in this project."""

MAIN_PROMPT = """You are doing web research on behalf of a startup that's entering the market. The startup is as follows:

<startup>
{startup}
</startup>

You are trying to figure out this information:

<info>
{info}
</info>

You have access to the following tools:

- `Search`: call a search tool and get back some results
- `ScrapeWebsite`: scrape a website and get relevant notes about the given request. This will update the notes above.
- `Info`: call this when you are done and have gathered all the relevant info

Here is the information you have about the topic you are researching:

Topic: {topic}"""

BRANCH_OUT_PROMPT = """
You are about to start researching a go-to-market strategy for a startup.
Your task is to generate a list of industries that the startup should target 
based on the following description of the startup:

<startup>
{startup}
</startup>
"""
