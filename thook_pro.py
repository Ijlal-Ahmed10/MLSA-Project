import cv2
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from time import time
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
import supervision as sv
import secrets


# Email configuration
from_email = secrets.FROM_EMAIL
password = secrets.PASSWORD
to_email = secrets.TO_EMAIL

# Create a simple box annotator to use in our custom sink
annotator = sv.BoxAnnotator()

def send_email(to_email, from_email, object_detected):
    """Sends an email notification indicating the number of objects detected."""
    message = MIMEMultipart()
    message["From"] = from_email
    message["To"] = to_email
    message["Subject"] = "Security Alert"
    message_body = f"ALERT - {object_detected} objects have been detected!!"
    message.attach(MIMEText(message_body, "plain"))
    
    server = smtplib.SMTP("smtp.gmail.com: 587")
    server.starttls()
    server.login(from_email, password)
    server.sendmail(from_email, to_email, message.as_string())
    server.quit()

class ThiefDetection:
    def __init__(self, capture_index):
        """Initializes the ThiefDetection instance with a camera index."""
        self.capture_index = capture_index
        self.email_sent = False
        
        # Load the YOLO model for object detection
        self.model = YOLO("yolo11n.pt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize inference pipeline for weapon detection
        self.pipeline = InferencePipeline.init(
            model_id="thief-detection-dataset/1",#weapon-detection-aoxpz/5", "weapons v2/4"
            video_reference="demo.mp4",
            on_prediction=self.custom_sink,
            api_key=secrets.API_KEY,
        )
        
    def custom_sink(self, predictions: dict, video_frame: VideoFrame):
        """Handles predictions for weapon detection."""
        labels = [p["class"] for p in predictions["predictions"]]
        detections = sv.Detections.from_inference(predictions)
        image = annotator.annotate(
            scene=video_frame.image.copy(), detections=detections)
        cv2.imshow("Weapons Predictions", image)
        cv2.waitKey(1)

    def predict(self, im0):
        """Run prediction using the YOLO model for the input image."""
        results = self.model(im0)
        return results

    def plot_bboxes(self, results, im0):
        """Plots bounding boxes on an image given detection results; returns annotated image and class IDs."""
        class_ids = []
        annotator = Annotator(im0, 3, results[0].names)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names

        for box, cls in zip(boxes, clss):
            class_ids.append(cls)
            color = colors(int(cls), True)
            im0 = cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            label = f"{names[int(cls)]}"
            im0 = cv2.putText(im0, label, (int(box[0]), int(box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return im0, class_ids

    def __call__(self):
        """Run the thief detection on video frames from a camera stream."""
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Start the inference pipeline for weapon detection in a separate thread
        self.pipeline.start()

        while True:
            ret, im0 = cap.read()
            assert ret

            results = self.predict(im0)
            im0, class_ids = self.plot_bboxes(results, im0)

            # Check for detected objects and send email if not sent before
            if len(class_ids) > 0 and not self.email_sent:
                send_email(to_email, from_email, len(class_ids))
                self.email_sent = True
            elif len(class_ids) == 0:
                self.email_sent = False

            cv2.imshow("Object Detection", im0)

            if cv2.waitKey(5) & 0xFF == 27:  # Break loop on 'ESC' key
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.pipeline.join()

# Initialize and run the detection
detector = ThiefDetection(capture_index=0)
detector()