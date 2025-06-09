from PIL import Image, ImageDraw, ImageFont
import os
import random  # Add this import at the top

class CharacterImageGenerator:
    def __init__(self):
        # Settings
        self.size = 28
        self.img_size = (self.size, self.size)
        self.font_size = 32

        # Define the base output and fonts directories
        root_dir = os.path.dirname(os.path.dirname(__file__))
        self.output_base_dir = os.path.join(root_dir, "content", "data", "generated_images")
        self.fonts_dir = os.path.join(root_dir, "content", "fonts")

        # Create base output directory structure
        os.makedirs(self.output_base_dir, exist_ok=True)

        # Character sets
        self.symbols = ['!', '@', '#', '$', '%', '&', '*', '+', '-', '?', '<', '>']
        self.symwords = ['exclmark', 'at', 'hash', 'dollar', 'percent', 'ampersand', 
                    'asterisk', 'plus', 'minus', 'quesmark', 'lessthan', 'greaterthan']

        self.upperLetters = [chr(i) for i in range(65,91)]
        self.lowerLetters = [chr(i) for i in range(97,123)]
        self.numbers = [chr(i) for i in range(48,58)]
        self.characters = self.upperLetters + self.lowerLetters + self.numbers

    def create_character_image(self, char, folder_name, font, font_name):
        # Create folder for this character in base output directory
        char_dir = os.path.join(self.output_base_dir, folder_name)
        os.makedirs(char_dir, exist_ok=True)
        
        # Create image with padding to prevent cutoff during rotation
        padding = 4  # Add padding to prevent character cutoff
        padded_size = (self.size + padding, self.size + padding)
        img = Image.new("L", padded_size, color="white")
        draw = ImageDraw.Draw(img)

        # Get size of text to center it
        bbox = draw.textbbox((0, 0), char, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        position = ((padding//2-bbox[0]+((self.size-text_width)//2)),
                   (padding//2-bbox[1]+((self.size-text_height)//2)))
        
        # Draw the character
        draw.text(position, char, fill="black", font=font)
        
        # Apply random rotation
        rotation = random.uniform(-20, 20)
        img = img.rotate(rotation, resample=Image.Resampling.BILINEAR, expand=False)
        
        # Crop to final size
        left = padding//2
        top = padding//2
        right = left + self.size
        bottom = top + self.size
        img = img.crop((left, top, right, bottom))
        
        # Save image using case-specific folder names
        if char.isalpha():
            case_prefix = 'lower' if char.islower() else 'upper'
            safe_name = f"{case_prefix}{char.upper()}"
        elif char in self.symbols:
            symbol_index = self.symbols.index(char)
            safe_name = self.symwords[symbol_index]
        else:
            safe_name = f"{char}"
            
        filename = f"{font_name}_{safe_name}.png"
        img.save(os.path.join(char_dir, filename))

    def process_fonts(self):
        # Get all font files from fonts directory
        font_files = [f for f in os.listdir(self.fonts_dir) if f.endswith(('.ttf', '.otf'))]
        
        for font_file in font_files:
            font_path = os.path.join(self.fonts_dir, font_file)
            font_name = os.path.splitext(font_file)[0]
            print(f"\nProcessing font: {font_name}")
            
            try:
                font = ImageFont.truetype(font_path, self.font_size)
                
                # Process all characters
                for char in self.characters:
                    if char.isalpha():
                        case_prefix = 'lower' if char.islower() else 'upper'
                        folder_name = f"{case_prefix}{char.upper()}"
                    else:
                        folder_name = f"{char}"
                    print(f"Creating image of: {char} in folder: {folder_name}")
                    self.create_character_image(char, folder_name, font, font_name)

                # Process symbols
                for sym, word in zip(self.symbols, self.symwords):
                    print(f"Creating image of: {sym} in folder: {word}")
                    self.create_character_image(sym, word, font, font_name)
                    
            except Exception as e:
                print(f"Error processing font {font_file}: {str(e)}")