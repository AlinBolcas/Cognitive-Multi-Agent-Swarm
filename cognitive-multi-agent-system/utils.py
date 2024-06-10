import os, re, json, markdown, html, glob, time, math
from pathlib import Path
import subprocess

def quick_look(path):
    # On macOS, use 'qlmanage' to show the GIF using Quick Look
    subprocess.run(['qlmanage', '-p', path])

def markdown_to_html(markdown_text):
    # Convert Markdown to HTML
    html_text = markdown.markdown(markdown_text)
    return html_text

def escape_html(text):
    """
    Escapes HTML special characters in text for safe display in HTML content.

    Args:
        text (str): The text to escape.

    Returns:
        str: The escaped text with HTML special characters converted to their corresponding HTML entities.
    """
    return html.escape(text)

def extract_json(output, retry_function, attempt=1, max_attempts=10):
    """
    Tries to parse the entire output as JSON. If it fails, looks for a JSON block in the output and attempts to parse it.
    If parsing fails or no JSON block is found, it retries by calling the provided retry function.
    
    Args:
        output (str): The output string, potentially containing JSON or a JSON block.
        retry_function (callable): Function to retry generating the output.
        attempt (int): Current attempt number.
        max_attempts (int): Maximum number of attempts allowed.
    
    Returns:
        dict or None: Parsed JSON object if successful, or None if unsuccessful after max attempts.
    """
    try:
        # First, attempt to parse the entire output as JSON
        parsed_json = json.loads(output)
        return parsed_json
    except json.JSONDecodeError:
        # If it fails, look for a JSON block within the output
        try:
            json_block_match = re.search(r'```json\n([\s\S]*?)\n```', output)
            if json_block_match:
                json_block = json_block_match.group(1)  # Extract the JSON block
                parsed_json = json.loads(json_block)  # Attempt to parse the JSON block
                return parsed_json
            else:
                raise ValueError("No JSON block found in the output.")
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < max_attempts:
                print("Retrying...")
                new_output = retry_function()  # Call the retry function to generate new output
                return extract_json(new_output, retry_function, attempt + 1, max_attempts)
            else:
                print("Maximum attempts reached. Unable to extract or parse JSON.")
                print(output)  # Print the output for debugging purposes
                return None

def save_markdown(contents, file_path="output/undefined.md"):
    # Implement the logic to compile contents into a Markdown document
    # md_path = f"output/{file_name}.md"
    md_path = file_path
    # Check if the file exists and delete it
    if os.path.exists(md_path):
        os.remove(md_path)
    
    with open(md_path, "w") as md_file:
        for content in contents:
            md_file.write(content)

def save_json(data, file_name="undefined"):
    # Implement the logic to compile contents into a Markdown document
    json_path = f"output/{file_name}.json"
    # Check if the file exists and delete it
    if os.path.exists(json_path):
        os.remove(json_path)
    
    with open(json_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

def read_file(file_base_name, extensions=['.md', '.txt', '.json']):
    """
    Attempts to read a file, trying different extensions until the file is found.
    
    Args:
        file_base_name (str): The base name of the file without extension.
        extensions (list): A list of file extensions to try.
    
    Returns:
        str or None: The content of the file if found, None otherwise.
    """
    for ext in extensions:
        file_path = f"output/{file_base_name}{ext}"
        if os.path.exists(file_path):
            print(f"Reading {file_path} from file.")
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                return content
    print(f"File {file_base_name} with extensions {extensions} does not exist.")
    return None

def json_to_markdown(json_obj):
    """
    Converts a JSON object to a markdown string with selective use of bullet points, bold text,
    and numbers for better readability, while maintaining thematic breaks and depth-based headers
    for structure, ensuring header levels do not surpass ###.
    """
    def process_item(key, value, depth=1, is_list=False):
        """
        Processes each item, applying markdown based on its type, context, and whether it's part of a list,
        adjusting the depth to manage header levels.
        """
        md = ""
        prefix = ""
        # Adjust the maximum depth for headers to not surpass level 3
        adjusted_depth = min(depth, 4)  # Ensures we don't go beyond ### headers
        
        if adjusted_depth == 0:
            adjusted_depth = 1
        if adjusted_depth > 0:  # Adjust prefix based on depth to create numbered lists at deeper levels
            prefix = f"{'#' * (adjusted_depth)} "

        if key:
            md += f"{prefix}{key}\n\n"

        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                # Increase depth but do not let it surpass the maximum for headers
                md += process_item(sub_key, sub_value, depth + 1 if depth < 3 else depth, is_list)
        elif isinstance(value, list):
            md += "\n".join([process_item(None, item, depth + 1 if depth < 3 else depth, is_list=True) for item in value]) # + "\n"
        else:
            # Format simple values, applying bold for keys if within a list for clarity
            if key and is_list:
                md += f"{value}\n\n" #\n" # f"**{key}:** {value}\n\n"
            else:
                md += f"- {value}\n\n" #\n"

        # Include thematic breaks after top-level sections for clear separation
        if depth == 2:
            md += "---\n\n" # \n"

        return md

    markdown = process_item(None, json_obj)
    return markdown.strip()  # Ensure clean output without leading/trailing whitespace

def save_image(image, base_name="img_", extension=".png"):
    if image:
        output_directory = output_dir()  # Get the output directory
        existing_files = glob.glob(os.path.join(output_directory, base_name + "*"))
        next_number = len(existing_files) + 1
        filename = f"{base_name}{next_number:03}{extension}"
        path = os.path.join(output_directory, filename)

        image.save(path)
        print(f"Image saved as {path}\n")
        # os.system(f"open {path}")  # Opens the image; adjust command based on your OS
    else:
        print("Failed to save image.")

def save_speech(speech_data, filename):
    try:
        output_directory = output_dir()  # Get the output directory
        file_path = os.path.join(output_directory, filename)
        
        # Assuming speech_data is raw audio data that needs to be written to a file
        with open(file_path, "wb") as file:
            file.write(speech_data)
        
        print(f"Speech saved as {file_path}")
    except Exception as e:
        print(f"Failed to save speech: {e}")

def output_path():
    # Calculate the project root (which is three directories up from this file)
    output_dir = Path(__file__).resolve().parents[3] / "output"
    print(f"Project root: {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def chaos_trigger():
    # Rename the chaotic trigger function for consistency
    t = time.time() % 30  
    return abs(math.sin(t))

from colorama import Fore, Style, init as colorama_init

# Initialize colorama
colorama_init(autoreset=True)

def printColoured(text, color_name):
    """
    Color the given text with the specified color name.
    Available colors: red, green, blue, magenta, yellow, white, grey, default.
    """
    color_dict = {
        "red": f"{Style.BRIGHT}{Fore.RED}",
        "green": f"{Style.BRIGHT}{Fore.GREEN}",
        "cyan": f"{Style.BRIGHT}{Fore.CYAN}",
        "blue": f"{Style.BRIGHT}{Fore.BLUE}",
        "magenta": f"{Style.BRIGHT}{Fore.MAGENTA}",
        "yellow": f"{Style.BRIGHT}{Fore.YELLOW}",
        "white": f"{Fore.LIGHTWHITE_EX}",
        "grey": f"{Fore.LIGHTBLACK_EX}",
        "default": Style.RESET_ALL
    }
    
    color_code = color_dict.get(color_name.lower(), Style.RESET_ALL)
    return f"{color_code}{text}{Style.RESET_ALL}"

# Example usage
if __name__ == "__main__":

    read_file()
    
    
    