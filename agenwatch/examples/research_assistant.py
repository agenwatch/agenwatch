"""
AgenWatch Example: AI Research Assistant
Demonstrates real-world usage with budget control
"""

from agenwatch._kernel.context_manager import ContextManager, ContextConfig
from agenwatch._kernel.execution_manager import ToolExecutionManager
from agenwatch._kernel.tools.registry import ToolRegistry
from agenwatch._kernel.safety.budget_manager import BudgetManager, CostCalculator

import agenwatch
print("agenwatch FILE:", AgenWatch.__file__)
print("agenwatch CONTENTS:", dir(AgenWatch))


# =============================================================================
# DEFINE TOOLS
# =============================================================================
from agenwatch import Agent, tool
from datetime import datetime
@tool("Search arXiv for academic papers")
def search_papers(query: str, max_results: int = 5) -> str:
    """
    Search arXiv for academic papers
    Returns paper titles, authors, and abstracts
    """
    # Real implementation would call arXiv API
    # This is simplified for demo
    
    papers = [
        {
            "title": f"Paper about {query} (1)",
            "authors": "Smith et al.",
            "abstract": f"A comprehensive study of {query}...",
            "url": "https://arxiv.org/abs/2024.12345"
        },
        {
            "title": f"Advances in {query} (2)",
            "authors": "Johnson & Lee",
            "abstract": f"Novel approaches to {query}...",
            "url": "https://arxiv.org/abs/2024.67890"
        }
    ]
    
    result = f"Found {len(papers)} papers on '{query}':\n\n"
    for i, paper in enumerate(papers[:max_results], 1):
        result += f"{i}. {paper['title']}\n"
        result += f"   Authors: {paper['authors']}\n"
        result += f"   Abstract: {paper['abstract']}\n"
        result += f"   URL: {paper['url']}\n\n"
    
    return result


@tool("Summarize text content")
def summarize(text: str, max_words: int = 100) -> str:
    """
    Summarize long text into key points
    """
    # In production, this might call another LLM or use extractive summarization
    words = text.split()
    if len(words) <= max_words:
        return text
    
    summary = " ".join(words[:max_words]) + "..."
    return f"Summary ({max_words} words): {summary}"


@tool("Save content to file")
def save_to_file(filename: str, content: str) -> str:
    """
    Save text content to a file
    """
    try:
        with open(filename, 'w') as f:
            f.write(content)
        return f"✅ Saved to {filename}"
    except Exception as e:
        return f"❌ Error saving file: {str(e)}"


@tool("Search Wikipedia")
def search_wikipedia(topic: str) -> str:
    """
    Search Wikipedia for a topic
    Returns summary and key information
    """
    # Simplified - real implementation would use Wikipedia API
    return f"""
Wikipedia Summary for '{topic}':

{topic} is a fundamental concept in computer science and artificial intelligence.
It involves the study of algorithms and systems that can perform tasks typically
requiring human intelligence.

Key aspects:
- Machine Learning
- Neural Networks
- Natural Language Processing
- Computer Vision

Last updated: {datetime.now().strftime('%Y-%m-%d')}
"""


@tool("Compare two topics")
def compare_topics(topic1: str, topic2: str) -> str:
    """
    Compare and contrast two topics
    """
    return f"""
Comparison: {topic1} vs {topic2}

Similarities:
- Both are important areas of study
- Both have practical applications
- Both continue to evolve

Differences:
- {topic1} focuses on X
- {topic2} focuses on Y

Conclusion: While related, they serve different purposes.
"""


# =============================================================================
# CREATE RESEARCH ASSISTANT
# =============================================================================

def create_research_assistant():
    """
    Create a research assistant agent with budget controls
    """
    return Agent(
        tools=[
            search_papers,
            summarize,
            save_to_file,
            search_wikipedia,
            compare_topics
        ],
        model="claude-sonnet-4-20250514",
        max_iterations=15,
        enable_budget=True,
        max_budget=2.00,  # $2 max cost
        verbose=True
    )


# =============================================================================
# EXAMPLE TASKS
# =============================================================================

def example_1_paper_search():
    """Example: Search and summarize papers"""
    print("=" * 70)
    print("EXAMPLE 1: Paper Search & Summary")
    print("=" * 70)
    print()
    
    agent = create_research_assistant()
    
    task = """
    Search for recent papers about 'transformer models' in machine learning.
    Summarize the top 3 papers and save the summary to 'transformer_papers.txt'.
    """
    
    result = agent.run(task)
    print()
    print("RESULT:")
    print(result)
    print()
    print("STATS:")
    import json
    print(json.dumps(agent.get_stats(), indent=2))


def example_2_comparative_research():
    """Example: Comparative analysis"""
    print("=" * 70)
    print("EXAMPLE 2: Comparative Research")
    print("=" * 70)
    print()
    
    agent = create_research_assistant()
    
    task = """
    Compare 'supervised learning' and 'unsupervised learning'.
    Then search Wikipedia for more details on each.
    Provide a comprehensive comparison.
    """
    
    result = agent.run(task)
    print()
    print("RESULT:")
    print(result)


def example_3_budget_limit():
    """Example: Hit budget limit"""
    print("=" * 70)
    print("EXAMPLE 3: Budget Enforcement")
    print("=" * 70)
    print()
    
    # Create agent with very low budget
    agent = Agent(
        tools=[search_papers, search_wikipedia, compare_topics],
        enable_budget=True,
        max_budget=0.01,  # Only $0.01
        verbose=True
    )
    
    task = """
    Search for papers on 'quantum computing', 'machine learning', 
    'neural networks', 'deep learning', and 'reinforcement learning'.
    Compare all of them.
    """
    
    try:
        result = agent.run(task)
        print()
        print("RESULT:")
        print(result)
    except Exception as e:
        print()
        print(f"⚠️  Agent stopped due to: {e}")
        print()
        print("BUDGET STATUS:")
        import json
        print(json.dumps(agent.get_stats(), indent=2))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print()
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║           AgenWatch - Research Assistant Demo                 ║")
    print("║  Demonstrating: Tools, Budget Control, Real Tasks             ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    # Run examples
    example_1_paper_search()
    print("\n" + "="*70 + "\n")
    
    # Uncomment to run more examples:
    # example_2_comparative_research()
    # example_3_budget_limit()
    
    print()
    print("✅ Demo complete!")
    print()
    print("💡 Key Features Demonstrated:")
    print("   - Simple @tool decorator")
    print("   - One-line agent creation")
    print("   - Automatic budget tracking")
    print("   - Clean, readable code")
    print()
    print("📚 Next Steps:")
    print("   - Try your own tools")
    print("   - Adjust budget limits")
    print("   - Enable execution recording")
    print("   - Use agent.replay() to debug")




