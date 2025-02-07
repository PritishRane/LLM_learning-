import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
SERPER_API_KEY = os.getenv('SERPER_API_KEY')

# Initialize LLM
llm = ChatGroq(model='llama3-70b-8192')

# Define Agents
planner = Agent(
    llm=llm,
    role="Clinical Research Planner",
    goal="Plan well-structured and scientifically accurate research content on {topic}",
    backstory="You're working on planning a research document about the topic: {topic}.\n"
              "You collect information that helps researchers, clinicians, and healthcare professionals\n"
              "understand key aspects of the topic, including methodologies, study designs, and clinical relevance.\n"
              "Your work forms the foundation for the Clinical Research Writer to develop a detailed and well-supported document.",
    allow_delegation=False,
    verbose=True
)

writer = Agent(
    llm=llm,
    role="Clinical Research Writer",
    goal="Write a precise, scientifically sound, and well-supported research document about {topic}",
    backstory="You're working on drafting a research document about the topic: {topic}.\n"
              "You base your writing on the structured outline provided by the Clinical Research Planner,\n"
              "ensuring accuracy, adherence to scientific standards, and proper citation of sources.\n"
              "Your work should present objective, evidence-based insights, avoiding speculation unless explicitly stated.\n"
              "Your writing should be clear and accessible to both clinicians and researchers.",
    allow_delegation=False,
    verbose=True
)

editor = Agent(
    llm=llm,
    role="Clinical Research Editor",
    goal="Edit a given research document to align with scientific writing standards and ensure clarity, accuracy, and consistency.",
    backstory="You are an editor reviewing a clinical research document prepared by the Clinical Research Writer.\n"
              "Your goal is to refine the document by ensuring scientific rigor, proper structure, logical flow,\n"
              "and adherence to research publication standards.\n"
              "You check for consistency, clarity, and proper citation of sources while ensuring the content remains unbiased\n"
              "and grounded in evidence-based medicine.",
    allow_delegation=False,
    verbose=True
)

# Define Tasks
plan = Task(
    description=(
        "1. Identify the latest research trends, key studies, "
        "and significant findings related to {topic}.\n"
        "2. Define the target audience, including researchers, clinicians, "
        "and healthcare professionals, considering their knowledge level and interests.\n"
        "3. Develop a structured research outline, including "
        "an introduction, methodology, key findings, discussion, and conclusion.\n"
        "4. Gather relevant references, guidelines, and supporting data "
        "from reputable sources (e.g., PubMed, clinical trials, systematic reviews)."
    ),
    expected_output="A detailed research plan document with an outline, audience analysis, "
                    "key references, and structured methodology.",
    agent=planner,
)

write = Task(
    description=(
        "1. Use the research plan to draft a well-structured and "
        "scientifically accurate document on {topic}.\n"
        "2. Ensure adherence to research writing standards (e.g., IMRAD format if applicable).\n"
        "3. Provide clear section headings with concise and informative content.\n"
        "4. Support statements with evidence from high-quality sources.\n"
        "5. Ensure clarity, logical flow, and proper citation of references (e.g., AMA, APA, Vancouver styles).\n"
    ),
    expected_output="A well-written research document in a structured format, "
                    "ready for peer review or publication.",
    agent=writer,
)

edit = Task(
    description=("Proofread the research document for "
                 "scientific accuracy, logical consistency, "
                 "and adherence to research writing standards."),
    expected_output="A well-polished research document, "
                    "free of grammatical errors and aligned with "
                    "scientific writing conventions.",
    agent=editor
)

# Streamlit UI
st.title("Clinical Research Document Generator")

# User input
topic = st.text_input("Enter the research topic:")

if st.button("Generate Research Document"):
    crew = Crew(
        agents=[planner, writer, editor],
        tasks=[plan, write, edit],
        verbose=2
    )
    
    with st.spinner("Generating research document..."):
        result = crew.kickoff(inputs={"topic": topic})
    
    st.subheader("Generated Research Document")
    st.text_area("Output", result, height=500)

st.markdown("---")
st.markdown("Created with ❤️ using CrewAI, LangChain, and Streamlit")