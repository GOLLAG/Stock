import requests
from bs4 import BeautifulSoup

# Define the URL
url = 'https://www.wincalendar.com/India/date/4-June-2024'

# Send a request to fetch the content
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Print the title of the page
    print("Title of the page:", soup.title.string)
    
    # Example: Extracting calendar information
    # Find the main content of the page
    main_content = soup.find('div', class_='mm-wrapper__blocker mm-slideout')
    if main_content:
        # Print the text of the main content
        print("Main content:")
        print(main_content.get_text(strip=True))
    else:
        print("Main content not found.")
else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")
