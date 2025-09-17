# Background Remover

A Python script that uses AI to remove backgrounds from images in a folder.

## Author

Pablo R (murapadev)

## Features

- Loads a pre-trained AI model from Hugging Face
- Processes all images in an input folder
- Saves background-removed images to an output folder
- Supports common image formats (JPG, PNG, BMP, TIFF)

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/murapadev/background-remover.git
   cd background-remover
   ```

2. Create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your images in the `input` folder.

2. Run the script:

   ```bash
   python background_remover.py
   ```

   Or specify custom folders:

   ```bash
   python background_remover.py --input /path/to/input --output /path/to/output
   ```

3. Processed images will be saved in the `output` folder with "\_no_bg" suffix.

## Model

Uses the RMBG-1.4 model from BRIA AI on Hugging Face, which is one of the best for background removal.

## License

MIT License - see LICENSE file for details.

## Contributing

Feel free to open issues or submit pull requests.
