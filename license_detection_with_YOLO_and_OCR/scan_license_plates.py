import yolov5
import cv2 as cv
import keras_ocr


def get_plate_images_from_image(img_path: str, boxes: list):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Get only the license plates
    plates = []
    for box in boxes:
        x1, y1, x2, y2 = box
        plate = img[int(y1):int(y2), int(x1):int(x2)]
        plates.append(plate)

    return plates


class LicensePlateScanner:
    def __init__(self):
        self.model = yolov5.load('keremberke/yolov5m-license-plate')
        self.model.conf = 0.25  # NMS confidence threshold
        self.model.iou = 0.45  # NMS IoU threshold
        self.model.agnostic = False  # NMS class-agnostic
        self.model.multi_label = False  # NMS multiple labels per box
        self.model.max_det = 5  # maximum number of detections per imageself)

        self.pipeline = keras_ocr.pipeline.Pipeline()

    def get_license_plate_text(self, img_path: str):
        results = self.inference(img_path)
        predictions = results.pred[0]
        boxes = predictions[:, :4]
        plates = get_plate_images_from_image(img_path, boxes)

        # for plate in plates:
        #     cv.imshow("Plate", plate)
        #     cv.waitKey(0)
        #     cv.destroyAllWindows()

        return self.ocr_with_keras(plates), results.show()

    def inference(self, img_path: str):
        results = self.model(img_path, augment=True)  # x1, y1, x2, y2
        return results

    def ocr_with_keras(self, plates: list):
        predictions_from_keras = self.pipeline.recognize(plates)

        license_plates_text = []
        for i, plate in enumerate(plates):
            # print("Plate ", i)
            # damit die predicitions von links nach rechts sortiert sind
            sorted_predictions = sorted(predictions_from_keras[i], key=lambda x: x[1][0][0])
            license_plates_text.append("")
            for prediction in sorted_predictions:
                print(prediction[0])
                license_plates_text[i] += (prediction[0])

        return license_plates_text


scanner = LicensePlateScanner()
print(scanner.get_license_plate_text(
    r"C:\Users\bensc\Projects\additions_to_conv_lane_detect\data\license_plate_images\eu-1.jpg"))
