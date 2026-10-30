from dataset_doctor.loaders import yolo_hbb, yolo_obb, yolo_seg

FORMATS = {
    "yolo-hbb": yolo_hbb,
    "yolo-obb": yolo_obb,
    "yolo-seg": yolo_seg,
}
