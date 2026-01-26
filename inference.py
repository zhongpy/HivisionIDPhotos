import os
import cv2
import argparse
import json
import numpy as np
from hivision.error import FaceError, ComplianceError
from hivision.utils import hex_to_rgb, resize_image_to_kb, add_background, save_image_dpi_to_bytes
from hivision import IDCreator
from hivision.creator.layout_calculator import (
    generate_layout_array,
    generate_layout_image,
)
from hivision.creator.choose_handler import choose_handler
from hivision.utils import hex_to_rgb, resize_image_to_kb


INFERENCE_TYPE = [
    "idphoto",
    "human_matting",
    "add_background",
    "generate_layout_photos",
    "idphoto_crop",
]
MATTING_MODEL = [
    "hivision_modnet",
    "modnet_photographic_portrait_matting",
    "mnn_hivision_modnet",
    "rmbg-1.4",
    "birefnet-v1-lite",
]
FACE_DETECT_MODEL = [
    "mtcnn",
    "face_plusplus",
    "retinaface-resnet50",
]
RENDER = [0, 1, 2]

parser = argparse.ArgumentParser(description="HivisionIDPhotos 璇佷欢鐓у埗浣滄帹鐞嗙▼搴忋€?)
parser.add_argument(
    "-t",
    "--type",
    help="璇锋眰 API 鐨勭绫?,
    choices=INFERENCE_TYPE,
    default="idphoto",
)
parser.add_argument("-i", "--input_image_dir", help="杈撳叆鍥惧儚璺緞", required=True)
parser.add_argument("-o", "--output_image_dir", help="淇濆瓨鍥惧儚璺緞", required=True)
parser.add_argument("--height", help="璇佷欢鐓у昂瀵?楂?, default=413)
parser.add_argument("--width", help="璇佷欢鐓у昂瀵?瀹?, default=295)
parser.add_argument("-c", "--color", help="璇佷欢鐓ц儗鏅壊", default="638cce")
parser.add_argument("--hd", type=bool, help="鏄惁杈撳嚭楂樻竻鐓?, default=True)
parser.add_argument(
    "-k", "--kb", help="杈撳嚭鐓х墖鐨?KB 鍊硷紝浠呭鎹㈠簳鍜屽埗浣滄帓鐗堢収鐢熸晥", default=None
)
parser.add_argument(
    "-r",
    "--render",
    type=int,
    help="搴曡壊鍚堟垚鐨勬ā寮忥紝鏈?0:绾壊銆?:涓婁笅娓愬彉銆?:涓績娓愬彉 鍙€?,
    choices=RENDER,
    default=0,
)
parser.add_argument(
    "--dpi",
    type=int,
    help="杈撳嚭鐓х墖鐨?DPI 鍊?,
    default=300,
)
parser.add_argument(
    "--face_align",
    type=bool,
    help="鏄惁杩涜浜鸿劯鏃嬭浆鐭",
    default=False,
)
parser.add_argument(
    "--matting_model",
    help="鎶犲浘妯″瀷鏉冮噸",
    default="modnet_photographic_portrait_matting",
    choices=MATTING_MODEL,
)
parser.add_argument(
    "--face_detect_model",
    help="浜鸿劯妫€娴嬫ā鍨?,
    default="mtcnn",
    choices=FACE_DETECT_MODEL,
)

args = parser.parse_args()

# ------------------- 閫夋嫨鎶犲浘涓庝汉鑴告娴嬫ā鍨?-------------------
creator = IDCreator()
choose_handler(creator, args.matting_model, args.face_detect_model)

root_dir = os.path.dirname(os.path.abspath(__file__))
input_image = cv2.imread(args.input_image_dir, cv2.IMREAD_UNCHANGED)

# 濡傛灉妯″紡鏄敓鎴愯瘉浠剁収
if args.type == "idphoto":
    # 灏嗗瓧绗︿覆杞负鍏冪粍
    size = (int(args.height), int(args.width))
    try:
        result = creator(input_image, size=size, face_alignment=args.face_align)
    except FaceError:
        print("浜鸿劯鏁伴噺涓嶇瓑浜?1锛岃涓婁紶鍗曞紶浜鸿劯鐨勫浘鍍忋€?)
    except ComplianceError as exc:
        print(json.dumps(exc.report, ensure_ascii=False))
    else:
        print(json.dumps(result.compliance, ensure_ascii=False))
        # 淇濆瓨鏍囧噯鐓?
        save_image_dpi_to_bytes(cv2.cvtColor(result.standard, cv2.COLOR_RGBA2BGRA), args.output_image_dir, dpi=args.dpi)

        # 淇濆瓨楂樻竻鐓?
        file_name, file_extension = os.path.splitext(args.output_image_dir)
        new_file_name = file_name + "_hd" + file_extension
        save_image_dpi_to_bytes(cv2.cvtColor(result.hd, cv2.COLOR_RGBA2BGRA), new_file_name, dpi=args.dpi)

# 濡傛灉妯″紡鏄汉鍍忔姞鍥?
elif args.type == "human_matting":
    result = creator(input_image, change_bg_only=True)
    cv2.imwrite(args.output_image_dir, result.hd)

# 濡傛灉妯″紡鏄坊鍔犺儗鏅?
elif args.type == "add_background":

    render_choice = ["pure_color", "updown_gradient", "center_gradient"]

    # 灏嗗瓧绗︿覆杞负鍏冪粍
    color = hex_to_rgb(args.color)
    # 灏嗗厓绁栫殑 0 鍜?2 鍙锋暟瀛椾氦鎹?
    color = (color[2], color[1], color[0])

    result_image = add_background(
        input_image, bgr=color, mode=render_choice[args.render]
    )
    result_image = result_image.astype(np.uint8)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGBA2BGRA)
    
    if args.kb:
        resize_image_to_kb(result_image, args.output_image_dir, int(args.kb), dpi=args.dpi)
    else:
        save_image_dpi_to_bytes(cv2.cvtColor(result_image, cv2.COLOR_RGBA2BGRA), args.output_image_dir, dpi=args.dpi)

# 濡傛灉妯″紡鏄敓鎴愭帓鐗堢収
elif args.type == "generate_layout_photos":

    size = (int(args.height), int(args.width))

    typography_arr, typography_rotate = generate_layout_array(
        input_height=size[0], input_width=size[1]
    )

    result_layout_image = generate_layout_image(
        input_image,
        typography_arr,
        typography_rotate,
        height=size[0],
        width=size[1],
    )

    if args.kb:
        result_layout_image = cv2.cvtColor(result_layout_image, cv2.COLOR_RGB2BGR)
        result_layout_image = resize_image_to_kb(
            result_layout_image, args.output_image_dir, int(args.kb), dpi=args.dpi
        )
    else:
        save_image_dpi_to_bytes(cv2.cvtColor(result_layout_image, cv2.COLOR_RGBA2BGRA), args.output_image_dir, dpi=args.dpi)

# 濡傛灉妯″紡鏄瘉浠剁収瑁佸垏
elif args.type == "idphoto_crop":
    # 灏嗗瓧绗︿覆杞负鍏冪粍
    size = (int(args.height), int(args.width))
    try:
        result = creator(input_image, size=size, crop_only=True)
    except FaceError:
        print("浜鸿劯鏁伴噺涓嶇瓑浜?1锛岃涓婁紶鍗曞紶浜鸿劯鐨勫浘鍍忋€?)
    except ComplianceError as exc:
        print(json.dumps(exc.report, ensure_ascii=False))
    else:
        print(json.dumps(result.compliance, ensure_ascii=False))
        # 淇濆瓨鏍囧噯鐓?
        save_image_dpi_to_bytes(cv2.cvtColor(result.standard, cv2.COLOR_RGBA2BGRA), args.output_image_dir, dpi=args.dpi)

        # 淇濆瓨楂樻竻鐓?
        file_name, file_extension = os.path.splitext(args.output_image_dir)
        new_file_name = file_name + "_hd" + file_extension
        save_image_dpi_to_bytes(cv2.cvtColor(result.hd, cv2.COLOR_RGBA2BGRA), new_file_name, dpi=args.dpi)

