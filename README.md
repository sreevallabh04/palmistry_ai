# Palmistry AI 👋🔮✨

![Project Status: In Development](https://img.shields.io/badge/Project%20Status-In%20Development-yellow.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

## Description 📖

So, this project is kinda cool. It's an AI thingy that tries to read palm lines and stuff, you know, like in palmistry? My dad's really into that, so I thought it'd be a funny little project to mess around with. The idea is to use computers to do the old-school palm reading, making it easier for anyone to get a reading. 🤖✋

Basically, the code in [`palmistry_ai.py`](palmistry_ai.py) does a bunch of steps:

*   **Image Loading and Preprocessing:** 📸 First, it loads up a picture of a palm and cleans it up a bit. Makes it grayscale, blurs it, boosts the contrast, that kinda thing. (Check out [`load_and_preprocess_image()`](palmistry_ai.py:140))
*   **Palm Region Detection:** 🖐️ Then, it tries to find just the palm part in the picture. Gotta ignore the fingers and thumb, right? (See [`detect_palm_region()`](palmistry_ai.py:185))
*   **Palm Line Extraction:** 📏 This is where it gets tricky. It pulls out the lines from the palm using some fancy edge detection and other image magic. (Look at [`extract_palm_lines()`](palmistry_ai.py:248))
*   **Palm Line Classification:** 🔮 Once it has the lines, it tries to figure out which is which – the Life Line, Head Line, Heart Line, and Fate Line. It uses some rules based on where they are and how they look. (See [`classify_palm_lines()`](palmistry_ai.py:313))
*   **Line Property Analysis:** ✨ After identifying the lines, it looks at their details – how long they are, if they're deep, if they have breaks. It uses some old palmistry info I put in there. (See [`_analyze_line_properties()`](palmistry_ai.py:526) and [`__init__()`](palmistry_ai.py:88))
*   **Palm Mount Detection (Advanced):** There's also a part that tries to find the bumps on your palm, like the Mount of Venus or Jupiter. That's in the `AdvancedPalmistryAI` class. (See [`AdvancedPalmistryAI`](palmistry_ai.py:989) and [`detect_palm_mounts()`](palmistry_ai.py:1007))
*   **Report Generation:** 📜 Finally, it puts together a report based on what it found, giving you a personalized reading. (Check out [`generate_palmistry_report()`](palmistry_ai.py:636) and [`generate_advanced_report()`](palmistry_ai.py:1061))
*   **Result Visualization:** 🎨 Oh, and it also makes a cool picture showing the lines it found on your palm. (See [`create_annotated_image()`](palmistry_ai.py:580))

## Features ✨

*   Reads the main palm lines.
*   Can look at palm bumps too (the advanced bit).
*   Gives you a personalized reading based on the AI.
*   You can use it from the command line.
*   Might even have a simple web thingy with Streamlit later.
*   Detailed reports explaining what the AI thinks.
*   Shows you the lines it detected on your palm image.

## Technologies Used 💻

*   Python (obvs)
*   OpenCV (`cv2`) - For all the image stuff.
*   NumPy (`np`) - For numbers and arrays.
*   Matplotlib (`plt`) - To show pictures.
*   Scikit-image (`skimage`) - More image processing tools.
*   SciPy (`scipy`) - Some science-y math stuff.
*   PIL (Pillow) (`PIL`) - Helps with images too.
*   Streamlit - For the potential web interface.

## Installation 🛠️

**(Still figuring out the best way to install this, but here's the basic idea.)**

To get it running on your computer:

1.  Clone the repo (if you know how to do that):
    ```bash
    git clone https://github.com/sreevallabh04/palmistry_ai.git
    ```
2.  Go into the project folder:
    ```bash
    cd palmistry_ai
    ```
3.  Install the libraries it needs:
    ```bash
    pip install -r requirements.txt
    ```
    *(Need to make a `requirements.txt` file with all the libraries listed: `opencv-python`, `numpy`, `matplotlib`, `scikit-image`, `scipy`, `Pillow`, `streamlit`)*

## Usage ▶️

**(How to actually use this thing. Will add more details later.)**

**Command Line:**

```bash
python palmistry_ai.py --image path/to/your/palm/image.jpg
```
*(Replace with the right command once I finalize it)*

**Streamlit Web App (if I get around to it):**

```bash
streamlit run palmistry_ai.py
```
*(This is just a guess, might change)*

## Deployment 🚀

To deploy the production-ready web app:

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Run the Streamlit app:
    ```bash
    streamlit run palmistry_ai.py
    ```
3. Open the provided local URL in your browser to use the app.

Make sure you have a clear palm image ready to upload for best results!

## Contributing 🤝

Want to help out? That'd be awesome! This is an open source project, so anyone can jump in.

If you have ideas or find bugs, just open an issue. Or, if you're feeling brave, fork the repo and send a pull request!

Don't forget to star the project if you like it! ⭐

1.  Fork the Project (click the button up there!)
2.  Make a new branch for your changes (`git checkout -b feature/YourCoolIdea`)
3.  Save your changes (`git commit -m 'Add your cool idea'`)
4.  Push your branch (`git push origin feature/YourCoolIdea`)
5.  Open a Pull Request (compare your branch to the main one here)

## License 📄

It's under the MIT License. Pretty standard stuff. (Gotta make a `LICENSE` file with the actual text.)

## Contact 📧

Sreevallabh Kakarala - srivallabhkakarala@gmail.com

Project Link: [https://github.com/sreevallabh04/palmistry_ai](https://github.com/sreevallabh04/palmistry_ai)