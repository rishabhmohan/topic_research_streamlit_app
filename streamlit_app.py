import streamlit as st
from crewai import LLM, Agent, Crew, Task
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Topic Research with Mistral AI",
    page_icon=":bar_chart:",
    layout="wide",
)

st.title("Topic Research with Mistral AI")


with st.sidebar:
    st.header("Settings")

    topic = st.text_area(
        "Enter your topic",
        placeholder="Enter a topic related to any industry",
        height=100,
    )
    st.markdown("### LLM Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.5)

    # Add some spacing
    st.markdown("---")


    # Make the generate button more prominent in the sidebar
    generate_button = st.button(
        "Generate Report", type="primary", use_container_width=True
    )

    # Add some helpful text
    with st.expander("What is this app about?"):
        st.markdown(
            """
        1. Enter your topic and click the button to generate a report.
        2. The report will be generated using a combination of search and LLM tools.
        3. The report will include the latest trends, challenges, and opportunities in the desired industry.
        """
        )


def generate_content(topic):
    """Generates content using the LLM and search tools."""
    llm = LLM(
        model="mistral/mistral-small-latest",
        temperature=0.5,
        max_tokens=1500,
    )

    # Tool 2: Define the search tool
    search_tool = SerperDevTool(n=10)

    # Agent 1
    senior_resaerch_analyst = Agent(
        role="Senior Research Analyst",
        goal=f"Research and analyze the latest trends on {topic}",
        backstory="You are a senior research analyst with expertise in finance. You are tasked with researching and"
        " analyzing the latest trends in the finance industry. You will use the search tool to gather information"
        " and summarize your findings. You will also provide insights and recommendations based on your analysis.",
        allow_delegation=False,
        verbose=True,
        tools=[search_tool],
        llm=llm,
    )

    # Agent 2
    content_writer = Agent(
        role="Content Writer",
        goal="Write a comprehensive article based on the research findings while maintaining accuracy.",
        backstory="You are a content writer with expertise in finance. You will write a comprehensive article based on"
        " research findings provided by the senior research analyst. You will ensure that the article is well-"
        "structured, informative, and engaging. You will also ensure that the article is accurate and free of"
        " errors. You will use the search tool to gather additional information if needed. You will also provide"
        "insights and recommendations based on your analysis. You will also ensure that the article is optimized"
        "",
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    # task 1: Research tasks
    research_task = Task(
        description=(
            """
                1. Conduct research on the {topic} including:
                    -  latest trends, challenges, and opportunities.
                    -  key players and competitors.
                    -  emerging technologies and innovations.
                    -  market dynamics and economic factors.
                2. EvaLuate the credibility and reliability of the sources.
                3. Summarize the findings in a clear and concise manner.
                4. Provide insights and recommendations based on the analysis.
                5. Include all relevant sources and references.
                """
        ),
        expected_output=""" A detailed research report on the latest trends in the finance industry, including:
                -  latest trends, challenges, and opportunities.
                -  key players and competitors.
                -  emerging technologies and innovations.
                -  market dynamics and economic factors.
                -  insights and recommendations based on the analysis.
                -  all relevant sources and references.
                -  a summary of the findings in a clear and concise manner.
                """,
        agent=senior_resaerch_analyst,
    )

    # task 2: Writing tasks
    writing_task = Task(
        description=(
            """
                1. Write a comprehensive article based on the research findings provided by the senior research analyst.
                2. Ensure that the article is well-structured, informative, and engaging.
                3. Ensure that the article is accurate and free of errors.
                4. Use the search tool to gather additional information if needed.
                5. Provide insights and recommendations based on your analysis.
                """
        ),
        expected_output=""" A comprehensive article based on the research findings provided by senior research analyst,
                including:
                -  well-structured, informative, and engaging content.
                -  accurate and free of errors.
                -  insights and recommendations based on the analysis.
                """,
        agent=content_writer,
    )
    crew = Crew(
        name="Finance Industry Research Crew",
        agents=[senior_resaerch_analyst, content_writer],
        tasks=[research_task, writing_task],
        verbose=True,
    )

    return crew.kickoff(inputs={"topic": topic})


if generate_button:
    with st.spinner("Generating report..."):
        try:
            result = generate_content(topic)
            st.markdown("### Generated Report")
            st.markdown(result)

            # Add download button
            st.download_button(
                label="Download Report",
                data=result.raw,
                file_name=f"{topic}.md",
                mime="text/markdown",
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Footer
st.markdown(
    """
    ---
    Made with ❤️ by Rishabh Mohan
    """
)
st.markdown(
    """
    ---
    This app is powered by [CrewAI](https://crewai.com).
    """
)
