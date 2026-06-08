TRACKGUARD_MODEL_PROFILE = "coco_yolo11n"

MODEL_PROFILES = {
    "coco_yolo11n": {
        "model_path": "/home/inf431/Discord_Bot/yolo_data/yolo11n.pt",

        # YOLO 實際要偵測的 class_ids
        # 0 person, 2 car, 3 motorcycle, 5 bus, 7 truck
        "target_classes": ["person", "car", "motorcycle", "bus", "truck"],
        "class_ids": [0, 2, 3, 5, 7],

        # wrong_way 方向場只允許車輛，不允許 person
        "wrong_way_vehicle_classes": {
            "car",
            "motorcycle",
            "bus",
            "truck",
        },

        # collision 可以允許 person
        "collision_participant_classes": {
            "person",
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

        # 0 bike, 1 car, 2 moto, 3 oloo, 4 people
        "target_classes": ["bike", "car", "moto", "oloo", "people"],
        "class_ids": [0, 1, 2, 3, 4],

        # wrong_way 不允許 people
        "wrong_way_vehicle_classes": {
            "bike",
            "car",
            "moto",
            "oloo",
        },

        # collision 可以允許 people
        "collision_participant_classes": {
            "people",
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


def normalize_class_name(name) -> str:
    return str(name or "unknown").strip().lower()


def get_trackguard_model_path() -> str:
    return get_model_profile()["model_path"]


def get_target_classes() -> list:
    return list(get_model_profile()["target_classes"])


def get_class_ids() -> list:
    return list(get_model_profile()["class_ids"])


def get_wrong_way_vehicle_classes() -> set:
    return set(get_model_profile()["wrong_way_vehicle_classes"])


def get_collision_participant_classes() -> set:
    return set(get_model_profile()["collision_participant_classes"])


def get_person_classes() -> set:
    return set(get_model_profile()["person_classes"])


def is_person_class(name) -> bool:
    return normalize_class_name(name) in get_person_classes()


def is_wrong_way_vehicle_class(name) -> bool:
    name = normalize_class_name(name)

    if is_person_class(name):
        return False

    return name in get_wrong_way_vehicle_classes()


def is_collision_vehicle_class(name) -> bool:
    return normalize_class_name(name) in get_collision_participant_classes()


def is_vehicle_class(name) -> bool:
    # 給舊程式相容用，預設用 wrong_way 的車輛定義
    return is_wrong_way_vehicle_class(name)


TRACKGUARD_MODEL = get_trackguard_model_path()
TARGET_CLASSES = get_target_classes()
CLASS_IDS = get_class_ids()

WRONG_WAY_VEHICLE_CLASSES = get_wrong_way_vehicle_classes()
COLLISION_VEHICLE_CLASSES = get_collision_participant_classes()
PERSON_CLASSES = get_person_classes()