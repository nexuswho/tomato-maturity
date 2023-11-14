from ultralytics import YOLO
from flask import Flask, flash, request, render_template, redirect
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw
import cv2
import os
import io
import base64

from waitress import serve


model = YOLO(model="best.pt")

ALLOWED_EXTENSIONS = set(["jpg", "jpeg", "png"])
PORT_NUMBER = 5000

app = Flask(__name__)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def predict():
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)

    file = request.files["file"]
    print(file)

    if file.filename == "":
        flash("No image selected for uploading")
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upl_img = Image.open(file)
        extension = upl_img.format.lower()

        # use yolov8 model on image
        result = model.predict(source=upl_img, save=False)[0]

        with open("read.txt", "w") as file:
            file.writelines(str(result))

        print("-------------------")
        res_img = Image.fromarray(result.plot())
        image_byte_stream = io.BytesIO()
        res_img.save("D:/DUK/tomato-maturity/webapp/saves/img.png", format="PNG")
        res_img.save(
            image_byte_stream, format="PNG"
        )  # You can use a different format if desired, such as 'JPEG'
        image_byte_stream.seek(0)
        image_base64 = base64.b64encode(image_byte_stream.read()).decode("utf-8")

        return render_template("index.html", detection_results=image_base64)


if __name__ == "__main__":
    #app.run(port=PORT_NUMBER)

    serve(app, host='0.0.0.0', port=PORT_NUMBER)
