from deepface import DeepFace

# images in the same folder as analyze_image.py
images = ["sad.png", "angr.png", "happ.png", "natr.png"]

for img in images:
    result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
    print(f"Image: {img} → Predicted emotion: {result[0]['dominant_emotion']}")
