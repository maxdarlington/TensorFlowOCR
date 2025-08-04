# TensorFlow OCR - Character Recognition System

A comprehensive Optical Character Recognition (OCR) system built with TensorFlow, designed to recognize handwritten and printed characters with support for multiple font types, rotated images, and various character styles.

## Features

- **Multi-font Support**: Handles serif, sans-serif, and monospace fonts
- **Rotation Tolerance**: Processes images rotated up to ±20 degrees
- **Character Variety**: Supports letters (A-Z, a-z), numbers (0-9), and symbols (!@#$%&\*+-?<>)
- **User-friendly Interface**: Interactive menu system with progress indicators
- **System-friendly Processing**: Conservative multiprocessing for stable performance
- **Dataset Generation**: Built-in character image generator with font variations

## Requirements

- **Python**: 3.11
- **Operating System**: Windows, macOS, or Linux
- **Memory**: 8GB+ RAM recommended for large datasets

## Installation

1. **Clone the repository**:

   ```cmd
   git clone https://github.com/maxdarlington/TensorFlowOCR.git
   cd TensorFlowOCR
   ```

3. **Install dependencies**:

   ```cmd
   pip install -r requirements.txt
   ```

## Project Structure

```
TensorFlowOCR/
├── src/                    # Source code
│   ├── main.py            # Main application entry point
│   ├── model.py           # Neural network model definition
│   ├── dataUtil.py        # Dataset processing utilities
│   └── imgUtil.py         # Image generation utilities
├── content/
│   ├── data/              # Dataset storage
│   ├── fonts/             # Font files for character generation
│   ├── results/           # Test results and outputs
│   └── saved_models/      # Trained model files
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Font Setup

### Adding Custom Fonts

1. **Place font files** in the `content/fonts/` directory
2. **Supported formats**: `.ttf`, `.otf`, `.woff2`

### Included Fonts

The program comes with a curated selection of fonts optimized for OCR training:

- **AmaticSC** (Regular, Bold) - Handwritten style
- **Caveat** (Bold) - Natural handwriting simulation
- **Comic Sans MS** (Regular, Bold) - Casual, readable style
- **Courier New** (Regular, Bold) - Monospace font
- **Georgia** (Regular, Bold) - Serif font
- **Helvetica** (Regular, Bold) - Clean sans-serif
- **Times New Roman** (Regular, Bold) - Classic serif
- **Verdana** (Regular, Bold) - Web-optimized sans-serif

_Note: Only essential Regular and Bold variants are included to reduce redundancy and improve processing efficiency._

## Dataset Organization

### For Training/Testing

1. **Create dataset folders** in `content/data/`
2. **Organize by character class**:

   ```
   content/data/your_dataset/
   ├── upperA/          # Uppercase A images
   ├── upperB/          # Uppercase B images
   ├── lowera/          # Lowercase a images
   ├── lowerb/          # Lowercase b images
   ├── 0/               # Number 0 images
   ├── 1/               # Number 1 images
   ├── exclmark/        # Exclamation mark images
   ├── quesmark/        # Question mark images
   └── ...              # Other character classes
   ```

3. **Image requirements**:
   - **Format**: PNG
   - **Size**: 28x28 pixels (will be automatically resized)
   - **Color**: Grayscale or RGB (will be converted to grayscale)

### Supported Character Classes

- **Letters**: `upperA` through `upperZ`, `lowera` through `lowerz`
- **Numbers**: `0` through `9`
- **Symbols**: `exclmark`, `at`, `hash`, `dollar`, `percent`, `ampersand`, `asterisk`, `plus`, `minus`, `quesmark`, `lessthan`, `greaterthan`

## Usage

### Starting the Program

**Bash (Linux/macOS):**

```bash
python src/main.py
```

**Windows Command Prompt:**

```cmd
python src\main.py
```

### Main Menu Options

1. **Train a new model** - Create and train a neural network
2. **Test an existing model** - Evaluate trained models
3. **Generate custom dataset** - Create character images from fonts
4. **Exit** - Close the program

### Training Mode

1. **Select "Train a new model"**
2. **Choose dataset source**:
   - **Process dataset**: Load and process raw image folders
   - **Load processed dataset**: Use pre-processed `.npz` files
3. **Configure training**:
   - **Optimal epochs** (recommended): Automatically calculated based on dataset size
   - **Custom epochs**: Manually specify training cycles
4. **Monitor progress**: View real-time training metrics
5. **Save model**: Provide a name for your trained model

### Testing Mode

1. **Select "Test an existing model"**
2. **Choose dataset source** (same options as training)
3. **Select model**: Choose from saved `.keras` files
4. **Choose test type**:
   - **Visualized test cases**: Interactive testing with image display
   - **Automated test cases**: Batch processing with progress tracking
5. **Configure options**:
   - **Randomization**: Shuffle test order
   - **CSV export**: Save results to file
6. **View results**: Accuracy metrics and detailed analysis

### Dataset Generation

1. **Select "Generate custom dataset"**
2. **Automatic processing**: Creates character images from all fonts in `content/fonts/`
3. **Features**:
   - **Multiple fonts**: All available fonts are used
   - **Character coverage**: Letters, numbers, and symbols
   - **System-friendly**: Conservative multiprocessing

## Performance Tips

### For Large Datasets

- **Use processed `.npz` files** for faster loading
- **Consider dataset size** when choosing epoch count

### For Best Results

- **Diverse fonts**: Include serif, sans-serif, and monospace fonts
- **Quality images**: Use clear, well-lit character images
- **Balanced classes**: Ensure equal representation of all characters

## Output Files

### Generated Files

- **Models**: Saved as `.keras` files in `content/saved_models/`
- **Processed data**: Saved as `.npz` files in `content/data/`
- **Test results**: CSV files with detailed analysis
- **Character images**: Generated in `content/data/generated_images/`

### File Naming Convention

- **Models**: `model_name.keras`
- **Datasets**: `dataset_name.npz`
- **Character images**: `FontName_CharacterClass.png`

## Troubleshooting

### Common Issues

1. **"No font files found"**

   - Ensure fonts are in `content/fonts/` directory
   - Check file extensions (.ttf, .otf, .woff2)

2. **"No datasets found"**

   - Create folders in `content/data/`
   - Follow the character class naming convention

3. **Memory errors during processing**

   - Reduce batch size in multiprocessing
   - Close other applications
   - Use smaller datasets for testing

**Happy OCR Training! (^\_^)**
