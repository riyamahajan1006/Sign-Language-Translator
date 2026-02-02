import os
import cv2
import numpy as np
from tensorflow import keras
from flask import Flask,request,jsonify,render_template,Response
from keras.models import Sequential, load_model
from keras.layers import Conv2D,Dense,Flatten,MaxPooling2D,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings("ignore")

data_dir="data"
img_size=224
model_path="modelnet_model.h5"
upload_folder="uploads"
allowed_extensions=["mp4","avi","mov","mkv","jpg","jpeg","png"]

labels=sorted(os.listdir(data_dir))

app=Flask(__name__)
app.config["upload_folder"]=upload_folder
os.makedirs(upload_folder,exist_ok=True)

def build_and_train_model():
    global labels
    datagen=ImageDataGenerator(rescale=1./255,validation_split=0.2)

    train_gen=datagen.flow_from_directory(
        data_dir,target_size=(img_size,img_size),
        batch_size=32,class_mode="categorical",subset="training"
    )

    val_gen=datagen.flow_from_directory(
        data_dir,target_size=(img_size,img_size),
        batch_size=32,class_mode="categorical",subset="validation"
    )

    labels=list(train_gen.class_indices.keys())
    print("Detected classes:",labels)

    model= Sequential([
        Conv2D(32,(3,3),activation="relu",input_shape=(img_size,img_size,3)),
        MaxPooling2D(2,2),
        Conv2D(64,(3,3),activation="relu"),
        MaxPooling2D(2,2),
        Conv2D(128,(3,3),activation="relu"),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(120,activation="relu"),
        Dropout(0.5),
        Dense(len(labels),activation="softmax")
    ])

    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
    model.fit(train_gen,epochs=20,validation_data=val_gen)
    model.save(model_path)
    print(f"Model trained and saved as {model_path}")
    return model

if os.path.exists(model_path):
    model=load_model(model_path)
    print("Loaded pretrained model")
else:
    print("model not found")
    model=build_and_train_model()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in allowed_extensions

def detect_and_crop_hand(frame):
    # Convert to HSV for skin detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Skin color range (works decently under normal lighting)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create skin mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Find contours
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Largest contour assumed to be hand
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 2000:
        return None

    x, y, w, h = cv2.boundingRect(cnt)
    return frame[y:y+h, x:x+w]


def preprocess_frame(frame):
    img=cv2.resize(frame,(img_size,img_size))
    img=img.astype("float32")/255.0
    img=np.expand_dims(img,axis=0)
    return img

def predict_frame(frame):
    cropped=detect_and_crop_hand(frame)
    if cropped is None:
        return "No Hand Detected",0.0
    processed=preprocess_frame(cropped)
    preds=model.predict(processed,verbose=0)
    class_index=np.argmax(preds)
    confidence=float(np.max(preds))
    return labels[class_index],confidence

def extract_frames_and_predict(video_path,step=5):
    cap=cv2.VideoCapture(video_path)
    sequence=[]
    frame_count=0
    while True:
        ret,frame=cap.read()
        if not ret:
            break
        if frame_count%step==0:
            label,_= predict_frame(frame)
            if label !="No Hand Detected":
                sequence.append(label)
        frame_count +=1
    cap.release()
    collapsed=[]
    for char in sequence:
        if not collapsed or char!=collapsed[-1]:
            collapsed.append(char)
    return ".".join(collapsed)

@app.get("/")
def index():
    return render_template("index.html")

@app.route("/predict_image",methods=["POST"])
def predict_image():
    if "file" not in request.files:
        return jsonify({"error":"No File uploaded"}),400
    file=request.files["file"]
    npimg=np.frombuffer(file.read(),np.uint8)
    img=cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    label,conf=predict_frame(img)
    return jsonify({"prediction":label,"confidence":conf})

@app.route("/predict_video",methods=["POST"])
def predict_video():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config["upload_folder"], filename)
    file.save(video_path)
    sequence = extract_frames_and_predict(video_path, step=5)
    return jsonify({"prediction": sequence})

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        label, conf = predict_frame(frame)
        display_text = f"{label} ({conf:.2f})" if label != "No Hand Detected" else label
        cv2.putText(frame, display_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.route("/predict_live")
def predict_live():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)