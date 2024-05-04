import re

def load_abbreviations(file_path):
    abbr_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                parts = line.strip().split(', ')
                abbr = parts[0].strip("('")  
                full_form = parts[1].strip("')")  
                abbr_dict[abbr.lower()] = full_form
                abbr_dict[full_form.lower()] = abbr
    return abbr_dict

def replace_terms(text, abbr_dict):
    pattern = re.compile(r'(\b)(\w+)(\b)', re.IGNORECASE)
    
    def replace_match(match):
        word = match.group(2)
        if word.lower() in abbr_dict:
            replacement = abbr_dict[word.lower()]
            if word[0].isupper():
                return match.group(1) + replacement.capitalize() + match.group(3)
            return match.group(1) + replacement + match.group(3)
        return match.group(0) 
    
    return pattern.sub(replace_match, text)

abbr_dict = load_abbreviations('abbreviation.txt')
input_text = "Patient shows signs of A.Fib. and requires frequent monitoring of A.B.G levels before meals."
output_text = replace_terms(input_text, abbr_dict)

print(output_text)
