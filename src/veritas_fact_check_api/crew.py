from crewai import Agent, Crew, Process, Task
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain.tools import Tool
import os
import yaml
from datetime import datetime

class InstagramFactCheckCrew:
	def __init__(self):
		self.load_config()
		
	def load_config(self):
		# Get the directory where the current file (crew.py) is located
		current_dir = os.path.dirname(os.path.abspath(__file__))
		
		# Load agents config
		agents_path = os.path.join(current_dir, 'config', 'agents.yaml')
		with open(agents_path, 'r') as f:
			self.agents_config = yaml.safe_load(f)
			
		# Load tasks config
		tasks_path = os.path.join(current_dir, 'config', 'tasks.yaml')
		with open(tasks_path, 'r') as f:
			self.tasks_config = yaml.safe_load(f)

	def create_agents(self):
		# Get API key from environment
		serpapi_key = os.getenv('SERPAPI_API_KEY')
		if not serpapi_key:
			raise ValueError("SERPAPI_API_KEY not found in environment variables")
		
		# Create tools using langchain tools
		search = SerpAPIWrapper(serpapi_api_key=serpapi_key)
		wiki = WikipediaAPIWrapper()
		
		# Convert to CrewAI compatible tools
		search_tools = [
			Tool(
				name="Search",
				func=search.run,
				description="Search the internet for current information and facts."
			),
			Tool(
				name="Wikipedia",
				func=wiki.run,
				description="Search Wikipedia for information."
			)
		]
		
		self.format_checker = Agent(
			role=self.agents_config['format_checker']['role'],
			goal=self.agents_config['format_checker']['goal'],
			backstory=self.agents_config['format_checker']['backstory'],
			verbose=True,
			llm_model="gpt-4"
		)
		
		self.fact_checker = Agent(
			role=self.agents_config['fact_checker']['role'],
			goal=self.agents_config['fact_checker']['goal'],
			backstory=self.agents_config['fact_checker']['backstory'],
			tools=search_tools,
			verbose=True,
			llm_model="gpt-4"
		)
		
		self.analysis_writer = Agent(
			role=self.agents_config['analysis_writer']['role'],
			goal=self.agents_config['analysis_writer']['goal'],
			backstory=self.agents_config['analysis_writer']['backstory'],
			verbose=True,
			llm_model="gpt-4"
		)

	def create_tasks(self, post_data: dict):
		verify_config = self.tasks_config['verify_content']
		analysis_config = self.tasks_config['create_analysis_report']
		validate_config = self.tasks_config['validate_format']
		
		verify_description = verify_config['description'].format(
			username=post_data['username'],
			description=post_data['description'],
			post_url=post_data['post_url']
		)
		
		self.verify_task = Task(
			description=verify_description,
			expected_output=verify_config['expected_output'],
			agent=self.fact_checker
		)
		
		self.analysis_task = Task(
			description=analysis_config['description'],
			expected_output=analysis_config['expected_output'],
			agent=self.analysis_writer
		)
		
		self.validate_task = Task(
			description=validate_config['description'],
			expected_output=validate_config['expected_output'],
			agent=self.format_checker
		)

	def run(self, post_data: dict):
		self.create_agents()
		self.create_tasks(post_data)
		
		try:
			crew = Crew(
				agents=[self.fact_checker, self.analysis_writer, self.format_checker],
				tasks=[self.verify_task, self.analysis_task, self.validate_task],
				process=Process.sequential,
				verbose=True
			)
			
			result = crew.kickoff()
			
			# Format the result as JSON with content as string
			formatted_result = {
				"message": [
					{
						"content": str(result),  # Convert CrewOutput to string
						"timestamp": datetime.now().isoformat()
					}
				]
			}
			
			return formatted_result
			
		except Exception as e:
			# Return error in the same JSON format
			return {
				"message": [
					{
						"error": str(e),
						"status": "failed",
						"timestamp": datetime.now().isoformat()
					}
				]
			}
	

