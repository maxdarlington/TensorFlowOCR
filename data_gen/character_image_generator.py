from PIL import Image, ImageDraw, ImageFont
import os
import string

# Settings
size = 28
img_size = (size, size)
font_size = 32

# Define the base output and fonts directories
root_dir = os.path.dirname(os.path.dirname(__file__))  # Get TensorFlowOCR directory
output_base_dir = os.path.join(root_dir, "content", "data", "generated_images")
fonts_dir = os.path.join(os.path.dirname(__file__), "fonts")

# Create base output directory structure
os.makedirs(output_base_dir, exist_ok=True)

# Character sets
symbols = ['!', '@', '#', '$', '%', '&', '*', '+', '-', '?', '<', '>']
symwords = ['exclmark', 'at', 'hash', 'dollar', 'percent', 'ampersand', 
            'asterisk', 'plus', 'minus', 'quesmark', 'lessthan', 'greaterthan']

upperLetters = [chr(i) for i in range(65,91)]
lowerLetters = [chr(i) for i in range(97,123)]
numbers = [chr(i) for i in range(48,58)]
characters = upperLetters + lowerLetters + numbers

def create_character_image(char, folder_name, font, font_name):
    # Create folder for this character in base output directory
    char_dir = os.path.join(output_base_dir, folder_name)
    os.makedirs(char_dir, exist_ok=True)
    
    # Create image
    img = Image.new("L", img_size, color="white")
    draw = ImageDraw.Draw(img)

    # Get size of text to center it
    bbox = draw.textbbox((0, 0), char, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((0-bbox[0]+((size-text_width)//2)),
                (0-bbox[1]+((size-text_height)//2)))
    
    # Draw the character
    draw.text(position, char, fill="black", font=font)
    
    # Save image using case-specific folder names
    if char.isalpha():
        case_prefix = 'lower' if char.islower() else 'upper'
        safe_name = f"{case_prefix}{char.upper()}"
    elif char in symbols:
        symbol_index = symbols.index(char)
        safe_name = symwords[symbol_index]
    else:
        safe_name = f"{char}"
        
    filename = f"{font_name}_{safe_name}.png"
    img.save(os.path.join(char_dir, filename))

def process_fonts():
    # Get all font files from fonts directory
    font_files = [f for f in os.listdir(fonts_dir) if f.endswith(('.ttf', '.otf'))]
    
    for font_file in font_files:
        font_path = os.path.join(fonts_dir, font_file)
        font_name = os.path.splitext(font_file)[0]
        print(f"\nProcessing font: {font_name}")
        
        try:
            font = ImageFont.truetype(font_path, font_size)
            
            # Process all characters
            for char in characters:
                if char.isalpha():
                    case_prefix = 'lower' if char.islower() else 'upper'
                    folder_name = f"{case_prefix}{char.upper()}"
                else:
                    folder_name = f"{char}"
                print(f"Creating image of: {char} in folder: {folder_name}")
                create_character_image(char, folder_name, font, font_name)

            # Process symbols
            for sym, word in zip(symbols, symwords):
                print(f"Creating image of: {sym} in folder: {word}")
                create_character_image(sym, word, font, font_name)
                
        except Exception as e:
            print(f"Error processing font {font_file}: {str(e)}")

if __name__ == "__main__":
    process_fonts()
