import yaml

# Load the environment.yml file with binary mode
with open('environment.yml', 'rb') as file:
    content = file.read()

# Decode the content
decoded_content = content.decode('utf-8', errors='replace')

# Load the YAML content
env = yaml.safe_load(decoded_content)

# Extract the dependencies
dependencies = env['dependencies']

# Open the requirements.txt file
with open('requirements.txt', 'w') as file:
    for dep in dependencies:
        if isinstance(dep, str):
            file.write(f"{dep}\n")
        elif isinstance(dep, dict) and 'pip' in dep:
            for pip_dep in dep['pip']:
                file.write(f"{pip_dep}\n")
