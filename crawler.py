import requests
from bs4 import BeautifulSoup

url = "https://www.asha.org/practice-portal/professional-issues/documentation-in-health-care/common-medical-abbreviations/"

response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')

    table = soup.find('table', class_='table table-striped')
    abbreviations = []

    for row in table.find_all('tr'):
        cells = row.find_all('td')
        if len(cells) == 2:  
            abbreviation = cells[0].get_text(strip=True)
            meaning = cells[1].get_text(strip=True)
            abbreviations.append((abbreviation, meaning))

    # for abbr, meaning in abbreviations:
    #     print(f"{abbr}: {meaning}")
else:
    print("Failed to retrieve the webpage")

print(abbreviations[0][0:2])
# with open('abbreviation.txt', 'w', encoding='utf-8') as f:
#     for abbr in abbreviations:
#         f.write(str(abbr) + "\n")

