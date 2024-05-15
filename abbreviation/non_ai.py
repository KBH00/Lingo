def load_abbreviations(file_path):
    """Load abbreviations from a given file into a dictionary."""
    abbrev_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith('(') and line.endswith(')'):
                line = line[1:-1]  
                parts = line.split("', '")
                if len(parts) == 2:
                    key = parts[0].strip("('")
                    value = parts[1].strip("')")
                    # Handle series like etc.'
                    if 'etc.' in key:
                        base_key = key.split(', etc.')[0]
                        base_value = value.split(', etc.')[0]

                        key_series = base_key.split(', ')
                        value_series = base_value.split(', ')
                        for k, v in zip(key_series, value_series):
                            abbrev_dict[k.strip()] = v.strip()
                    else:
                        abbrev_dict[key] = value
    return abbrev_dict

def replace_abbreviations(sentence, abbrev_dict):
    """Replace abbreviations in a sentence with their full forms."""
    import re
    words = re.split('(\W+)', sentence)
    replaced_words = []
    for word in words:
        stripped_word = word.strip('.,;:!?"')
        if stripped_word in abbrev_dict:
            # Ensure to match the full word only
            full_word = abbrev_dict[stripped_word]
            replaced_words.append(full_word)
        else:
            replaced_words.append(word)
    return ''.join(replaced_words)

# abbreviations = load_abbreviations('abbreviation.txt')
# input_sentence = "The patient's C1 and C2 vertebrae are aligned, and BP, CT, CHF is stable."

# output_sentence = replace_abbreviations(input_sentence, abbreviations)
# print("Original:", input_sentence)
# print("Processed:", output_sentence)
