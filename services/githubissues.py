import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables (e.g., GitHub token)
load_dotenv()

# GitHub API settings
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')  # Set your token in .env file
HEADERS = {'Authorization': f'token {GITHUB_TOKEN}'} if GITHUB_TOKEN else {}

def get_github_issues(owner, repo, state='open'):
    """
    Fetch issues from a GitHub repository.
    
    :param owner: Repository owner (e.g., 'WoolawayWx')
    :param repo: Repository name (e.g., 'showmefire-api')
    :param state: Issue state ('open', 'closed', or 'all')
    :return: List of issues
    """
    url = f'https://api.github.com/repos/{owner}/{repo}/issues'
    params = {'state': state, 'per_page': 100}  # Max 100 per page
    issues = []
    
    while url:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        issues.extend(response.json())
        
        # Check for next page
        if 'next' in response.links:
            url = response.links['next']['url']
            params = {}  # Params are in the URL now
        else:
            url = None
    
    return issues

def save_issues_to_json(issues, filename='data/issues.json'):
    """
    Save issues to a JSON file.
    
    :param issues: List of issues
    :param filename: Output file name
    """
    with open(filename, 'w') as f:
        json.dump(issues, f, indent=4)
    print(f"Saved {len(issues)} issues to {filename}")

if __name__ == '__main__':
    # Example usage: Replace with your repo details
    owner = 'WoolawayWx'  # or 'Cade417'
    repo = 'showmefire-api'  # or 'showmefire'
    state = 'all'  # 'open', 'closed', or 'all'
    
    try:
        issues = get_github_issues(owner, repo, state)
        save_issues_to_json(issues)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching issues: {e}")