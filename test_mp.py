import mediapipe as mp
print(f"MediaPipe version: {mp.__version__}")

try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    print("✅ Tasks API import ho gaya!")

    # Try initializing the landmarker
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    landmarker = vision.HandLandmarker.create_from_options(options)
    print("✅ MediaPipe HandLandmarker successfully initialize ho gaya!")
    
except Exception as e:
    print(f"❌ Error aaya: {e}")