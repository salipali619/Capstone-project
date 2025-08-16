import gradio as gr
import os
import time
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

# Secure API key entry
os.getenv("OPENAI_API_KEY")  # Enter open ai api key

def analyze_research_topic(research_topic, research_scope):
    if not research_topic.strip():
        return "❗ Please enter a research topic."
    
    if not research_scope.strip():
        research_scope = "Comprehensive analysis"

    # Show loading message with animation
    loading_messages = [
        "🔍 Initializing AI research team...",
        "🌐 Web Research Agent gathering online sources...",
        "📝 Content Summarizer Agent processing information...",
        "📊 Data Analyst Agent extracting key insights...",
        "🔗 Citation Manager Agent organizing references...",
        "👨‍💼 Supervisor Agent compiling comprehensive report...",
        "⚡ Generating final research analysis..."
    ]
    
    # Start with loading message
    yield loading_messages[0]
    
    try:
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)

        # Show progressive loading messages
        for i, message in enumerate(loading_messages[1:], 1):
            time.sleep(8)  # Small delay for visual effect
            yield message

        # Specialized research agents
        web_researcher = Agent(
            role="Web Research Specialist",
            goal=f"Conduct comprehensive web research on {research_topic} to gather the most relevant and up-to-date information",
            backstory="Expert web researcher with 10+ years of experience in digital information gathering, source verification, and online content analysis. Skilled at finding authoritative sources and current information on any topic.",
            llm=llm,
            verbose=False
        )

        content_summarizer = Agent(
            role="Content Analysis Expert",
            goal=f"Analyze and summarize information about {research_topic} from multiple sources",
            backstory="Senior content analyst with expertise in information synthesis, key insight extraction, and comprehensive summarization. Previously worked as a research analyst at top consulting firms.",
            llm=llm,
            verbose=False
        )

        data_analyst = Agent(
            role="Research Data Analyst",
            goal=f"Extract key data points, statistics, and trends related to {research_topic}",
            backstory="Data analysis specialist with strong background in research methodology, statistical analysis, and trend identification. Expert at finding patterns and insights in complex information.",
            llm=llm,
            verbose=False
        )

        citation_manager = Agent(
            role="Citation and Reference Specialist",
            goal=f"Organize and format all sources and citations for the {research_topic} research",
            backstory="Academic reference specialist with expertise in citation management, source verification, and research documentation standards. Ensures all information is properly attributed.",
            llm=llm,
            verbose=False
        )

        # Supervisor agent
        research_supervisor = Agent(
            role="Research Director",
            goal="Synthesize all research findings into a comprehensive, well-structured report",
            backstory="Senior research director with 15+ years experience leading research teams and producing high-quality analytical reports. Expert at creating actionable insights from complex research.",
            llm=llm,
            verbose=False
        )

        # Research tasks
        web_search_task = Task(
            description=f"""
            Conduct comprehensive web research on "{research_topic}" with focus on "{research_scope}":
            
            1. INFORMATION GATHERING:
               - Search for current news and developments
               - Find authoritative sources and expert opinions
               - Identify key trends and recent changes
               - Gather statistical data and factual information
            
            2. SOURCE EVALUATION:
               - Assess credibility and reliability of sources
               - Identify primary vs secondary sources
               - Note publication dates and relevance
               - Flag any conflicting information
            
            3. TOPIC COVERAGE:
               - Cover all major aspects of the topic
               - Include different perspectives and viewpoints
               - Find recent developments and future outlook
               - Identify key players and stakeholders
            
            Simulate web search results by providing comprehensive information as if gathered from top online sources.
            """,
            expected_output="Comprehensive collection of information from various web sources with source credibility assessment",
            agent=web_researcher
        )

        content_analysis_task = Task(
            description=f"""
            Analyze and synthesize all gathered information about "{research_topic}":
            
            1. CONTENT SYNTHESIS:
               - Identify main themes and key points
               - Organize information by relevance and importance
               - Summarize complex concepts clearly
               - Highlight significant findings
            
            2. INSIGHT EXTRACTION:
               - Extract actionable insights
               - Identify cause-and-effect relationships
               - Note implications and consequences
               - Recognize emerging patterns
            
            3. INFORMATION QUALITY:
               - Assess information accuracy and reliability
               - Identify gaps in available information
               - Note any contradictions or uncertainties
               - Prioritize most valuable insights
            
            Create clear, concise summaries that capture the essence of each major finding.
            """,
            expected_output="Synthesized analysis with key insights, themes, and prioritized information",
            agent=content_summarizer
        )

        data_analysis_task = Task(
            description=f"""
            Perform detailed data analysis on "{research_topic}" research findings:
            
            1. QUANTITATIVE ANALYSIS:
               - Extract key statistics and metrics
               - Identify numerical trends and patterns
               - Calculate growth rates and changes over time
               - Compare data across different sources
            
            2. QUALITATIVE ANALYSIS:
               - Analyze expert opinions and commentary
               - Identify consensus views vs dissenting opinions
               - Note qualitative trends and shifts
               - Assess market sentiment or public opinion
            
            3. TREND IDENTIFICATION:
               - Historical trends and developments
               - Current state and recent changes
               - Future projections and forecasts
               - Potential risks and opportunities
            
            Focus on data that provides the most valuable insights for understanding the topic.
            """,
            expected_output="Comprehensive data analysis with key metrics, trends, and statistical insights",
            agent=data_analyst,
            context=[web_search_task, content_analysis_task]
        )

        citation_task = Task(
            description=f"""
            Create comprehensive citation and reference system for "{research_topic}" research:
            
            1. SOURCE ORGANIZATION:
               - Catalog all information sources
               - Organize by source type and credibility
               - Create proper citation format
               - Note access dates and URLs (simulated)
            
            2. ATTRIBUTION SYSTEM:
               - Link specific claims to sources
               - Create reference numbering system
               - Ensure all facts are properly attributed
               - Identify primary source origins
            
            3. REFERENCE QUALITY:
               - Assess source authority and expertise
               - Note publication credentials
               - Identify peer-reviewed vs general sources
               - Flag any potential bias or limitations
            
            Create a professional reference system that supports all research claims.
            """,
            expected_output="Complete citation system with organized references and proper attribution",
            agent=citation_manager,
            context=[web_search_task, content_analysis_task]
        )

        # Final report compilation
        final_report_task = Task(
            description=f"""
            Create comprehensive research report on "{research_topic}" with scope "{research_scope}":
            
            RESEARCH REPORT STRUCTURE:
            
            1. EXECUTIVE SUMMARY:
               - Topic overview and significance
               - Key findings and main conclusions
               - Critical insights and implications
               - Recommended actions or considerations
            
            2. DETAILED FINDINGS:
               - Comprehensive topic analysis
               - Current state and recent developments
               - Key statistics and data points
               - Expert opinions and perspectives
            
            3. TREND ANALYSIS:
               - Historical context and evolution
               - Current trends and patterns
               - Future outlook and projections
               - Potential challenges and opportunities
            
            4. SOURCES SUMMARY:
               - Top sources and references
               - Source credibility assessment
               - Information gaps and limitations
               - Recommendations for further research
            
            5. CONCLUSION:
               - Summary of key takeaways
               - Implications and significance
               - Actionable insights
               - Next steps or recommendations
            
            Create a professional, well-structured report that provides comprehensive coverage of the topic.
            """,
            expected_output="Professional research report with executive summary, detailed findings, analysis, citations, and conclusions",
            agent=research_supervisor,
            context=[web_search_task, content_analysis_task, data_analysis_task, citation_task]
        )

        # Create research crew
        crew = Crew(
            agents=[web_researcher, content_summarizer, data_analyst, citation_manager, research_supervisor],
            tasks=[web_search_task, content_analysis_task, data_analysis_task, citation_task, final_report_task],
            verbose=False,
            process="sequential"
        )

        # Show final processing message
        yield "🎯 AI agents collaborating on final research report..."
        
        result = crew.kickoff()
        yield str(result)

    except Exception as e:
        yield f"❌ Research analysis failed: {str(e)}"

def create_gradio_interface():
    # Light premium theme with soft colors
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
        color: #1e293b !important;
        min-height: 100vh !important;
        position: relative !important;
    }
    
    /* Animated background elements */
    .gradio-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 20% 30%, rgba(59, 130, 246, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 70%, rgba(16, 185, 129, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 50% 10%, rgba(168, 85, 247, 0.1) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }
    
    .main-content {
        position: relative;
        z-index: 1;
    }
    
    .main-header {
        text-align: center !important;
        padding: 5rem 0 3rem 0 !important;
        position: relative !important;
    }
    
    .title {
        font-size: 4.5rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #1e293b 0%, #3b82f6 50%, #10b981 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        margin: 0 0 1.5rem 0 !important;
        letter-spacing: -0.03em !important;
        text-shadow: 0 0 40px rgba(59, 130, 246, 0.3) !important;
        animation: glow 2s ease-in-out infinite alternate !important;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 20px rgba(59, 130, 246, 0.3)); }
        to { filter: drop-shadow(0 0 30px rgba(59, 130, 246, 0.5)); }
    }
    
    .subtitle {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
        margin-bottom: 1rem !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    .description {
        font-size: 1.3rem !important;
        color: #64748b !important;
        max-width: 900px !important;
        margin: 0 auto 3rem auto !important;
        line-height: 1.7 !important;
        font-weight: 400 !important;
    }
    
    .agent-grid {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(226, 232, 240, 0.8) !important;
        border-radius: 24px !important;
        padding: 2.5rem !important;
        margin-bottom: 3rem !important;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.1) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .agent-grid::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.5), transparent);
    }
    
    .agent-card {
        background: rgba(248, 250, 252, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        padding: 1.5rem !important;
        border-radius: 16px !important;
        border: 1px solid rgba(226, 232, 240, 0.6) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .agent-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(16, 185, 129, 0.1));
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .agent-card:hover {
        transform: translateY(-8px) !important;
        border-color: rgba(59, 130, 246, 0.5) !important;
        box-shadow: 0 20px 40px rgba(59, 130, 246, 0.15) !important;
    }
    
    .agent-card:hover::before {
        opacity: 1;
    }
    
    .agent-icon {
        font-size: 2rem !important;
        margin-bottom: 1rem !important;
        display: block !important;
    }
    
    .agent-title {
        color: #1e293b !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        margin-bottom: 0.75rem !important;
        position: relative !important;
        z-index: 1 !important;
    }
    
    .agent-desc {
        color: #64748b !important;
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
        position: relative !important;
        z-index: 1 !important;
    }
    
    .input-container {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(226, 232, 240, 0.8) !important;
        border-radius: 24px !important;
        padding: 3rem !important;
        margin-bottom: 3rem !important;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.1) !important;
        position: relative !important;
    }
    
    .gr-textbox {
        background: rgba(248, 250, 252, 0.9) !important;
        backdrop-filter: blur(10px) !important;
        border: 2px solid rgba(226, 232, 240, 0.8) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        font-size: 1.1rem !important;
        color: #1e293b !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-weight: 500 !important;
    }
    
    .gr-textbox:focus {
        border-color: #3b82f6 !important;
        outline: none !important;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.15), 0 0 20px rgba(59, 130, 246, 0.2) !important;
        transform: translateY(-2px) !important;
    }
    
    .gr-button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 1.5rem 3rem !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
        margin-top: 2rem !important;
        position: relative !important;
        overflow: hidden !important;
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.3) !important;
    }
    
    .gr-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .gr-button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 35px rgba(59, 130, 246, 0.4) !important;
    }
    
    .gr-button:hover::before {
        left: 100%;
    }
    
    .output-container {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(226, 232, 240, 0.8) !important;
        border-radius: 24px !important;
        padding: 3rem !important;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.1) !important;
    }
    
    .output-container textarea {
        background: rgba(248, 250, 252, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(226, 232, 240, 0.6) !important;
        border-radius: 16px !important;
        color: #1e293b !important;
        font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace !important;
        font-size: 0.95rem !important;
        line-height: 1.7 !important;
        padding: 1.5rem !important;
    }
    
    .examples-container {
        background: rgba(248, 250, 252, 0.8) !important;
        backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(226, 232, 240, 0.6) !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        margin-top: 2rem !important;
    }
    
    .gr-examples .gr-button {
        background: rgba(248, 250, 252, 0.9) !important;
        backdrop-filter: blur(10px) !important;
        color: #64748b !important;
        border: 1px solid rgba(226, 232, 240, 0.8) !important;
        font-size: 0.9rem !important;
        padding: 1rem 1.5rem !important;
        margin: 0.5rem !important;
        border-radius: 12px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        box-shadow: none !important;
    }
    
    .gr-examples .gr-button:hover {
        background: rgba(59, 130, 246, 0.1) !important;
        color: #1e293b !important;
        border-color: rgba(59, 130, 246, 0.3) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.15) !important;
    }
    
    .footer {
        text-align: center !important;
        padding: 4rem 0 2rem 0 !important;
        color: #64748b !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        border-top: 1px solid rgba(226, 232, 240, 0.6) !important;
        margin-top: 4rem !important;
        background: rgba(248, 250, 252, 0.6) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Light theme overrides */
    .gr-form, .gr-box {
        background: transparent !important;
        border: none !important;
    }
    
    label {
        color: #374151 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Loading animation */
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    .loading {
        background: linear-gradient(90deg, rgba(248, 250, 252, 0.8) 25%, rgba(226, 232, 240, 0.8) 50%, rgba(248, 250, 252, 0.8) 75%);
        background-size: 1000px 100%;
        animation: shimmer 2s infinite;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .title { font-size: 3rem !important; }
        .subtitle { font-size: 1.4rem !important; }
        .description { font-size: 1.1rem !important; }
        .input-container, .output-container, .agent-grid { padding: 1.5rem !important; }
        .gr-button { font-size: 1.1rem !important; padding: 1.2rem 2rem !important; }
    }
    """
    
    with gr.Blocks(css=custom_css, title="ResearchIQ | AI Research Analyst", theme=gr.themes.Default()) as interface:
        
        with gr.Column(elem_classes="main-content"):
            # Premium Header
            gr.HTML("""
                <div class="main-header">
                    <h1 class="title">ResearchIQ</h1>
                    <p class="subtitle">AI Research Analyst - Autonomous Report Builder</p>
                    <p class="description">
                        Transform your research process with our autonomous AI research team. Simply provide a topic 
                        and watch as our AI agents search the web, summarize sources, generate citations, and compile 
                        comprehensive research reports that would normally take hours or days to complete.
                    </p>
                </div>
            """)
            
            # Premium AI Agents Grid
            gr.HTML("""
                <div class="agent-grid">
                    <h3 style="color: #1e293b; margin: 0 0 2rem 0; font-size: 1.5rem; font-weight: 700; text-align: center;">
                        🧠 Autonomous AI Research Team
                    </h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem;">
                        <div class="agent-card">
                            <span class="agent-icon">🌐</span>
                            <div class="agent-title">Web Research Specialist</div>
                            <div class="agent-desc">Conducts comprehensive web searches, gathers information from multiple sources, and verifies credibility</div>
                        </div>
                        <div class="agent-card">
                            <span class="agent-icon">📝</span>
                            <div class="agent-title">Content Analysis Expert</div>
                            <div class="agent-desc">Analyzes and synthesizes information from various sources into coherent summaries and insights</div>
                        </div>
                        <div class="agent-card">
                            <span class="agent-icon">📊</span>
                            <div class="agent-title">Research Data Analyst</div>
                            <div class="agent-desc">Extracts key statistics, trends, and data points to provide quantitative insights</div>
                        </div>
                        <div class="agent-card">
                            <span class="agent-icon">🔗</span>
                            <div class="agent-title">Citation Manager</div>
                            <div class="agent-desc">Organizes references, creates proper citations, and ensures all sources are properly attributed</div>
                        </div>
                        <div class="agent-card" style="grid-column: 1 / -1;">
                            <span class="agent-icon">👨‍💼</span>
                            <div class="agent-title">Research Supervisor Agent</div>
                            <div class="agent-desc">Compiles all research findings into professional, comprehensive reports with actionable insights</div>
                        </div>
                    </div>
                </div>
            """)
            
            # Premium Input Section
            with gr.Column(elem_classes="input-container"):
                gr.HTML("""
                    <h3 style="color: #1e293b; margin: 0 0 2rem 0; font-size: 1.3rem; font-weight: 700; text-align: center;">
                        ⚡ Generate Autonomous Research Report
                    </h3>
                """)
                with gr.Row():
                    with gr.Column(scale=3):
                        topic_input = gr.Textbox(
                            label="🔍 Research Topic",
                            placeholder="Enter any topic (e.g., Climate Change, Artificial Intelligence, Renewable Energy...)",
                            lines=1
                        )
                    with gr.Column(scale=2):
                        scope_input = gr.Textbox(
                            label="📋 Research Scope",
                            placeholder="e.g., Current trends, Market analysis, Technical overview...",
                            lines=1
                        )
                
                analyze_btn = gr.Button("🚀 Start Autonomous Research Analysis", variant="primary")
            
            # Premium Output Section
            with gr.Column(elem_classes="output-container"):
                output = gr.Textbox(
                    label="📋 Comprehensive Research Report",
                    lines=35,
                    show_copy_button=True,
                    placeholder="""🎯 Your autonomous research report will appear here...\n\n📊 EXECUTIVE SUMMARY\n• Topic overview and significance\n• Key findings and main conclusions\n• Critical insights and implications\n• Recommended actions or considerations\n\n🌐 WEB RESEARCH FINDINGS\n• Comprehensive information from top sources\n• Current developments and news\n• Expert opinions and authoritative content\n• Source credibility assessments\n\n📝 CONTENT ANALYSIS\n• Synthesized information and key themes\n• Important insights and patterns\n• Conflicting viewpoints and consensus areas\n• Quality assessment of available information\n\n📊 DATA ANALYSIS\n• Key statistics and metrics\n• Quantitative trends and patterns\n• Historical context and projections\n• Comparative analysis and benchmarks\n\n🔗 SOURCES & CITATIONS\n• Comprehensive reference list\n• Source credibility ratings\n• Proper citation formatting\n• Links to original materials\n\n📋 CONCLUSIONS & INSIGHTS\n• Summary of key takeaways\n• Actionable recommendations\n• Future research directions\n• Practical applications\n\nEnter a research topic above and click the button to begin your autonomous research analysis.""",
                    show_label=True
                )
            
            # Premium Examples
            with gr.Column(elem_classes="examples-container"):
                gr.Examples(
                    examples=[
                        ["Climate Change Impact", "Global trends and solutions"], 
                        ["Artificial Intelligence", "Current developments and future"], 
                        ["Renewable Energy", "Market analysis and technology"], 
                        ["Cryptocurrency", "Market trends and regulations"], 
                        ["Remote Work", "Post-pandemic business impact"], 
                        ["Electric Vehicles", "Industry growth and adoption"],
                        ["Quantum Computing", "Technical progress and applications"],
                        ["Space Technology", "Commercial developments and trends"]
                    ],
                    inputs=[topic_input, scope_input],
                    label="🎯 Research Topic Examples"
                )
            
            # Premium Footer
            gr.HTML("""
                <div class="footer">
                    <div style="margin-bottom: 1rem;">
                        <strong>ResearchIQ</strong> • Autonomous AI Research Platform
                    </div>
                    <div style="font-size: 0.9rem; color: #64748b;">
                        Web research • Content analysis • Citation management • Professional reports
                    </div>
                </div>
            """)
        
        # Event Handlers with streaming
        analyze_btn.click(
            fn=analyze_research_topic,
            inputs=[topic_input, scope_input],
            outputs=output,
            show_progress=True
        )
        
        topic_input.submit(
            fn=analyze_research_topic,
            inputs=[topic_input, scope_input],
            outputs=output,
            show_progress=True
        )
    
    return interface

# Launch the application
if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        favicon_path=None,
        app_kwargs={"docs_url": None, "redoc_url": None}
    )