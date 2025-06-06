# 🔮 AI Palmistry Reader ✨

Unlock the secrets written in the lines of your palm with the power of AI and ancient palmistry wisdom!

---

## 🌟 What is this?

**AI Palmistry Reader** is a modern web app that analyzes a photo of your palm, detects your major palm lines, and generates a mystical palmistry report based on traditional rules—all with a beautiful, engaging interface. The application uses computer vision techniques to identify palm lines and AI to generate personalized readings.

---

## ✨ Features

- 📸 **Palm Line Detection**: Upload a clear photo of your palm and let the AI find your Life Line, Head Line, Heart Line, and Fate Line.
- 🤖 **AI-Powered Palmistry Report**: Get a detailed, personalized reading based on the detected lines and real palmistry rules.
- 🎨 **Beautiful, Modern UI**: Enjoy a mystical, aesthetic web experience with gradients, emojis, and a user-friendly layout.
- 🖼️ **Annotated Results**: See your palm image with detected lines and download both the annotated image and your report.
- 📱 **Mobile Friendly**: Access via QR code on your phone for easy palm photo capture.
- 🔒 **Privacy-Focused**: Your palm images are processed locally and not stored permanently.

---

## 🛠️ Technologies Used

- Python 3.9+
- OpenCV for computer vision
- NumPy & SciPy for numerical processing
- Matplotlib for visualization
- scikit-image for image processing
- Pillow (PIL) for image manipulation
- Streamlit for the web interface
- Groq API for AI-powered readings

---

## 🚀 Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/sreevallabh04/palmistry_ai.git
   cd palmistry_ai
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   # On Windows
   python -m venv palmistry_env
   palmistry_env\Scripts\activate

   # On macOS/Linux
   python -m venv palmistry_env
   source palmistry_env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API keys:**
   
   Option 1: Create a `.env` file in the project root with your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```
   
   Option 2: Add your API keys to the `config.json` file:
   ```json
   {
     "api_keys": ["your_api_key_here"]
   }
   ```

---

## ▶️ Usage

1. **Start the web app:**
   ```bash
   streamlit run palmistry_ai.py
   ```

2. **Open your browser** to the local URL shown (usually [http://localhost:8501](http://localhost:8501)).

3. **For mobile access**, scan the QR code shown in the app with your phone.

4. **Upload a clear photo of your palm** (fingers spread, good lighting, plain background recommended).

5. **Click 'Analyze My Palm'** and wait for your mystical reading!

6. **View and download** your annotated palm image and personalized palmistry report.

### Tips for Best Results

- Use natural lighting when taking your palm photo
- Spread your fingers slightly apart
- Hold your palm flat, not curved
- Use a plain background for better contrast
- Make sure your palm fills most of the frame

---

## 🔒 Security Considerations

- API keys should never be committed to version control
- Use environment variables for production deployments
- The application does not permanently store palm images
- Consider adding user authentication for production use

---

## 🖼️ Screenshots

> _Add your screenshots here!_
>
> ![Palmistry App Screenshot](screenshots/demo.png)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Credits

- Developed by Sreevallabh Kakarala
- Inspired by the wisdom of traditional palmists and the magic of modern AI
- Special thanks to the open-source computer vision community

---

## 📬 Contact

- Email: srivallabhkakarala@gmail.com
- GitHub: [https://github.com/sreevallabh04/palmistry_ai](https://github.com/sreevallabh04/palmistry_ai)

---

> _"The lines of your palm are the map of your soul."_
