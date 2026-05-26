# /home/inf431/Discord_Bot/trackguard/class_config.py

# ============================================================
# TrackGuard Model / Class Config
# ============================================================
# 只要改這一行，就可以切換 TrackGuard 使用的模型與 class ids
#
# 可選：
#   "coco_yolo11n"
#   "custom_best_new"
# ============================================================

TRACKGUARD_MODEL_PROFILE = "coco_yolo11n"
# TRACKGUARD_MODEL_PROFILE = "custom_best_new"


MODEL_PROFILES = {
    "coco_yolo11n": {
        "model_path": "/home/inf431/Discord_Bot/yolo_data/yolo11n.pt",
        "target_classes": ["car", "motorcycle", "bus", "truck"],
        "class_ids": [2, 3, 5, 7],
        "vehicle_classes": {
            "car",
            "motorcycle",
            "bus",
            "truck",
        },
        "person_classes": {
            "person",
        },
    },

    "custom_best_new": {
        "model_path": "/home/inf431/Discord_Bot/yolo_data/best_new.pt",
        "target_classes": ["bike", "car", "moto", "oloo"],
        "class_ids": [0, 1, 2, 3],
        "vehicle_classes": {
            "bike",
            "car",
            "moto",
            "oloo",
        },
        "person_classes": {
            "people",
        },
    },
}


def get_model_profile() -> dict:
    if TRACKGUARD_MODEL_PROFILE not in MODEL_PROFILES:
        raise ValueError(f"Unknown TRACKGUARD_MODEL_PROFILE: {TRACKGUARD_MODEL_PROFILE}")

    return MODEL_PROFILES[TRACKGUARD_MODEL_PROFILE]


def get_trackguard_model_path() -> str:
    return get_model_profile()["model_path"]


def get_target_classes() -> list:
    return list(get_model_profile()["target_classes"])


def get_class_ids() -> list:
    return list(get_model_profile()["class_ids"])


def get_vehicle_classes() -> set:
    return set(get_model_profile()["vehicle_classes"])


def get_person_classes() -> set:
    return set(get_model_profile()["person_classes"])


def normalize_class_name(name) -> str:
    return str(name or "unknown").strip().lower()


def is_person_class(name) -> bool:
    return normalize_class_name(name) in get_person_classes()


def is_vehicle_class(name) -> bool:
    name = normalize_class_name(name)

    if is_person_class(name):
        return False

    return name in get_vehicle_classes()


def is_wrong_way_vehicle_class(name) -> bool:
    return is_vehicle_class(name)


def is_collision_vehicle_class(name) -> bool:
    return is_vehicle_class(name)


# 給其他地方直接 import 用
TRACKGUARD_MODEL = get_trackguard_model_path()
TARGET_CLASSES = get_target_classes()
CLASS_IDS = get_class_ids()

VEHICLE_CLASSES = get_vehicle_classes()
PERSON_CLASSES = get_person_classes()

WRONG_WAY_VEHICLE_CLASSES = VEHICLE_CLASSES
COLLISION_VEHICLE_CLASSES = VEHICLE_CLASSES