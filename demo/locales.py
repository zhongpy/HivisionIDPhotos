# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from demo.utils import csv_to_size_list
from demo.config import load_configuration
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
size_list_dict_CN = csv_to_size_list(os.path.join(base_dir, "assets/size_list_CN.csv"))
size_list_dict_EN = csv_to_size_list(os.path.join(base_dir, "assets/size_list_EN.csv"))
(
    size_list_config_CN,
    size_list_config_EN,
    color_list_dict_CN,
    color_list_dict_EN,
) = load_configuration(base_dir)


LOCALES = {
    "face_model": {
        "en": {
            "label": "Face detection model",
        },
        "zh": {
            "label": "Face detection model",
        },
        "ja": {
            "label": "Face detection model",
        },
        "ko": {
            "label": "Face detection model",
        },
    },
    "matting_model": {
        "en": {
            "label": "Matting model",
        },
        "zh": {
            "label": "Matting model",
        },
        "ja": {
            "label": "Matting model",
        },
        "ko": {
            "label": "Matting model",
        },
    },
    "key_param": {
        "en": {
            "label": "Key Parameters",
        },
        "zh": {
            "label": "Key Parameters",
        },
        "ja": {
            "label": "Key Parameters",
        },
        "ko": {
            "label": "Key Parameters",
        },
    },
    "advance_param": {
        "en": {
            "label": "Advance Parameters",
        },
        "zh": {
            "label": "Advance Parameters",
        },
        "ja": {
            "label": "Advance Parameters",
        },
        "ko": {
            "label": "Advance Parameters",
        },
    },
    "size_mode": {
        "en": {
            "label": "ID photo size options",
            "choices": [
                "Size List",
                "Only Change BG",
                "Custom(px)",
                "Custom(mm)",
            ],
            "custom_size_eror": "The width should not be greater than the length; the length and width should not be less than 100, and no more than 1800.",
        },
        "zh": {
            "label": "ID照片尺寸选项",
            "choices": ["尺寸列表", "仅更换背景", "自定义(px)", "自定义(mm)"],
            "custom_size_eror": "宽度不应大于长度；长度和宽度不应小于100，也不应大于1800。",
        },
        "ja": {
            "label": "ID写真サイズオプション",
            "choices": [
                "サイズリスト",
                "背景のみ変更",
                "カスタム(px)",
                "カスタム(mm)",
            ],
            "custom_size_eror": "幅は長さを超えてはいけません。長さと幅は100未満でも1800を超えてもいけません。",
        },
        "ko": {
            "label": "ID 사진 크기 옵션",
            "choices": [
                "크기 목록",
                "배경만 변경",
                "사용자 정의(px)",
                "사용자 정의(mm)",
            ],
            "custom_size_eror": "너비는 길이를 초과할 수 없습니다. 길이와 너비는 100 미만이거나 1800을 초과할 수 없습니다.",
        },
    },
    "custom_size_px": {
        "en": {
            "height": "Height(px)",
            "width": "Width(px)",
        },
        "zh": {
            "height": "高度(px)",
            "width": "宽度(px)",
        },
        "ja": {
            "height": "高さ(px)",
            "width": "幅(px)",
        },
        "ko": {
            "height": "높이(px)",
            "width": "너비(px)",
        },
    },
    "custom_size_mm": {
        "en": {
            "height": "Height(mm)",
            "width": "Width(mm)",
        },
        "zh": {
            "height": "高度(mm)",
            "width": "宽度(mm)",
        },
        "ja": {
            "height": "高さ(mm)",
            "width": "幅(mm)",
        },
        "ko": {
            "height": "높이(mm)",
            "width": "너비(mm)",
        },
    },
    "size_list": {
        "en": {
            "label": "Size list",
            "choices": list(size_list_dict_EN.keys()),
            "develop": size_list_config_EN,
        },
        "zh": {
            "label": "尺寸列表",
            "choices": list(size_list_dict_CN.keys()),
            "develop": size_list_config_CN,
        },
        "ja": {
            "label": "サイズリスト",
            "choices": list(size_list_dict_EN.keys()),
            "develop": size_list_config_EN,
        },
        "ko": {
            "label": "크기 목록",
            "choices": list(size_list_dict_EN.keys()),
            "develop": size_list_config_EN,
        },
    },
    "bg_color": {
        "en": {
            "label": "Background color",
            "choices": list(color_list_dict_EN.keys()) + ["American Style"] + ["Custom(RGB)", "Custom(HEX)"],
            "develop": color_list_dict_EN,
        },
        "zh": {
            "label": "背景颜色",
            "choices": list(color_list_dict_CN.keys()) + ["美式风格"] + ["自定义(RGB)", "自定义(HEX)"],
            "develop": color_list_dict_CN,
        },
        "ja": {
            "label": "背景色",
            "choices": list(color_list_dict_EN.keys()) + ["American Style"] + ["カスタム(RGB)", "カスタム(HEX)"],
            "develop": color_list_dict_EN,
        },
        "ko": {
            "label": "배경 색상",
            "choices": list(color_list_dict_EN.keys()) + ["American Style"] + ["사용자 정의(RGB)", "사용자 정의(HEX)"],
            "develop": color_list_dict_EN,
        },
    },
    "button": {
        "en": {
            "label": "Start",
        },
        "zh": {
            "label": "开始",
        },
        "ja": {
            "label": "開始",
        },
        "ko": {
            "label": "시작",
        },
    },
    "head_measure_ratio": {
        "en": {
            "label": "Head ratio",
        },
        "zh": {
            "label": "头部比例",
        },
        "ja": {
            "label": "頭部比率",
        },
        "ko": {
            "label": "머리 비율",
        },
    },
    "top_distance": {
        "en": {
            "label": "Top distance",
        },
        "zh": {
            "label": "顶部距离",
        },
        "ja": {
            "label": "上部距離",
        },
        "ko": {
            "label": "상단 거리",
        },
    },
    "image_kb": {
        "en": {
            "label": "Set KB size",
            "choices": ["Not Set", "Custom"],
        },
        "zh": {
            "label": "设置 KB 大小",
            "choices": ["不设置", "自定义"],
        },
        "ja": {
            "label": "KBサイズを設定",
            "choices": ["設定しない", "カスタム"],
        },
        "ko": {
            "label": "KB 크기 설정",
            "choices": ["설정하지 않음", "사용자 정의"],
        },
    },
    "image_kb_size": {
        "en": {
            "label": "KB size",
        },
        "zh": {
            "label": "KB 大小",
        },
        "ja": {
            "label": "KBサイズ",
        },
        "ko": {
            "label": "KB 크기",
        },
    },
    "image_dpi": {
        "en": {
            "label": "Set DPI",
            "choices": ["Not Set", "Custom"],
        },
        "zh": {
            "label": "设置 DPI 大小",
            "choices": ["不设置", "自定义"],
        },
        "ja": {
            "label": "DPIサイズを設定",
            "choices": ["設定しない", "カスタム"],
        },
        "ko": {
            "label": "DPI 크기 설정",
            "choices": ["설정하지 않음", "사용자 정의"],
        },
    },
    "image_dpi_size": {
        "en": {
            "label": "DPI size",
        },
        "zh": {
            "label": "DPI 大小",
        },
        "ja": {
            "label": "DPIサイズ",
        },
        "ko": {
            "label": "DPI 크기",
        },
    },
    "render_mode": {
        "en": {
            "label": "Render mode",
            "choices": [
                "Solid Color",
                "Up-Down Gradient (White)",
                "Center Gradient (White)",
            ],
        },
        "zh": {
            "label": "渲染模式",
            "choices": ["纯色", "上下渐变（白色）", "中心渐变（白色）"],
        },
        "ja": {
            "label": "レンダリングモード",
            "choices": [
                "単色",
                "上下グラデーション（白）",
                "中央グラデーション（白）",
            ],
        },
        "ko": {
            "label": "렌더링 모드",
            "choices": [
                "단색",
                "상하 그라데이션 (흰색)",
                "중앙 그라데이션 (흰색)",
            ],
        },
    },
    # Tab3 - 姘村嵃宸ヤ綔鍙?
    "watermark_tab": {
        "en": {
            "label": "Watermark",
        },
        "zh": {
            "label": "水印",
        },
        "ja": {
            "label": "ウォーターマーク",
        },
        "ko": {
            "label": "워터마크",
        },
    },
    "watermark_text": {
        "en": {
            "label": "Text",
            "value": "Hello",
            "placeholder": "up to 20 characters",
        },
        "zh": {
            "label": "水印文字",
            "value": "Hello",
            "placeholder": "最多 20 个字符",
        },
        "ja": {
            "label": "ウォーターマーク文字",
            "value": "Hello",
            "placeholder": "最大 20 文字",
        },
        "ko": {
            "label": "워터마크 문자",
            "value": "Hello",
            "placeholder": "최대 20자",
        },
    },
    "watermark_color": {
        "en": {
            "label": "Color",
        },
        "zh": {
            "label": "水印颜色",
        },
        "ja": {
            "label": "ウォーターマークの色",
        },
        "ko": {
            "label": "워터마크 색상",
        },
    },
    "watermark_size": {
        "en": {
            "label": "Size",
        },
        "zh": {
            "label": "水印大小",
        },
        "ja": {
            "label": "ウォーターマークのサイズ",
        },
        "ko": {
            "label": "워터마크 크기",
        },
    },
    "watermark_opacity": {
        "en": {
            "label": "Opacity",
        },
        "zh": {
            "label": "水印透明度",
        },
        "ja": {
            "label": "ウォーターマークの透明度",
        },
        "ko": {
            "label": "워터마크 투명도",
        },
    },
    "watermark_angle": {
        "en": {
            "label": "Angle",
        },
        "zh": {
            "label": "水印角度",
        },
        "ja": {
            "label": "ウォーターマークの角度",
        },
        "ko": {
            "label": "워터마크 각도",
        },
    },
    "watermark_space": {
        "en": {
            "label": "Space",
        },
        "zh": {
            "label": "水印间距",
        },
        "ja": {
            "label": "ウォーターマークの間隔",
        },
        "ko": {
            "label": "워터마크 간격",
        },
    },
    "watermark_switch": {
        "en": {
            "label": "Watermark",
            "value": "Not Add",
            "choices": ["Not Add", "Add"],
        },
        "zh": {
            "label": "水印",
            "value": "不添加",
            "choices": ["不添加", "添加"],
        },
        "ja": {
            "label": "ウォーターマーク",
            "value": "追加しない",
            "choices": ["追加しない", "追加"],
        },
        "ko": {
            "label": "워터마크",
            "value": "추가하지 않음",
            "choices": ["추가하지 않음", "추가"],
        },
    },
    # 杈撳嚭缁撴灉
    "notification": {
        "en": {
            "label": "notification",
            "face_error": "The number of faces is not equal to 1, please upload an image with a single face. If the actual number of faces is 1, it may be an issue with the accuracy of the detection model. Please switch to a different face detection model on the left or raise a Github Issue to notify the author.",
            "compliance_error": "Compliance check failed. Please review the report and adjust the photo.",
        },
        "zh": {
            "label": "通知",
            "face_error": "检测到的人脸数量不等于1，请上传一张单人脸的图片。如果实际人脸数量为1，可能是检测模型的准确性问题。请在左侧切换到其他人脸检测模型，或在Github上提交Issue通知作者。",
            "compliance_error": "合规性检查失败。请查看报告并调整照片。",
        },
        "ja": {
            "label": "通知",
            "face_error": "検出された顔の数が1ではありません。単一の顔の画像をアップロードしてください。実際の顔の数が1の場合、検出モデルの精度に問題がある可能性があります。左側で別の顔検出モデルに切り替えるか、GithubでIssueを提出して作者に通知してください。",
            "compliance_error": "コンプライアンスチェックに失敗しました。レポートを確認し、写真を調整してください。",
        },
        "ko": {
            "label": "알림",
            "face_error": "감지된 얼굴 수가 1이 아닙니다. 단일 얼굴 이미지 업로드를 해주세요. 실제 얼굴 수가 1인 경우, 감지 모델의 정확도 문제일 수 있습니다. 왼쪽에서 다른 얼굴 감지 모델로 전환하거나 Github에서 Issue를 제출하여 작성자에게 알리세요.",
            "compliance_error": "준수 검사에 실패했습니다. 보고서를 검토하고 사진을 조정하세요.",
        },
    },
    "standard_photo": {
        "en": {
            "label": "Standard photo",
        },
        "zh": {
            "label": "标准照片",
        },
        "ja": {
            "label": "標準写真",
        },
        "ko": {
            "label": "표준 사진",
        },
    },
    "hd_photo": {
        "en": {
            "label": "HD photo",
        },
        "zh": {
            "label": "高清照片",
        },
        "ja": {
            "label": "HD写真",
        },
        "ko": {
            "label": "HD 사진",
        },
    },
    "standard_photo_png": {
        "en": {
            "label": "Matting Standard photo",
        },
        "zh": {
            "label": "抠图标准照片",
        },
        "ja": {
            "label": "マッティング標準写真",
        },
        "ko": {
            "label": "매팅 표준 사진",
        },
    },
    "hd_photo_png": {
        "en": {
            "label": "Matting HD photo",
        },
        "zh": {
            "label": "抠图高清照片",
        },
        "ja": {
            "label": "マッティングHD写真",
        },
        "ko": {
            "label": "매팅 HD 사진",
        },
    },
    "layout_photo": {
        "en": {
            "label": "Layout photo",
        },
        "zh": {
            "label": "布局照片",
        },
        "ja": {
            "label": "レイアウト写真",
        },
        "ko": {
            "label": "레이아웃 사진",
        },
    },
    "download": {
        "en": {
            "label": "Download the photo after adjusting the DPI or KB size",
        },
        "zh": {
            "label": "下载调整 DPI 或 KB 大小后的照片",
        },
        "ja": {
            "label": "DPIやKBサイズを調整した後の写真をダウンロード",
        },
        "ko": {
            "label": "DPI 또는 KB 크기를 조정한 후 사진 다운로드",
        },
    },
    "matting_image": {
        "en": {
            "label": "Matting image",
        },
        "zh": {
            "label": "抠图图片",
        },
        "ja": {
            "label": "マッティング画像",
        },
        "ko": {
            "label": "매팅 이미지",
        },
    },
    "beauty_tab": {
        "en": {
            "label": "Beauty",
        },
        "zh": {
            "label": "美颜",
        },
        "ja": {
            "label": "美顔",
        },
        "ko": {
            "label": "미용",
        },
    },
    "whitening_strength": {
        "en": {
            "label": "whitening strength",
        },
        "zh": {
            "label": "美白强度",
        },
        "ja": {
            "label": "美白強度",
        },
        "ko": {
            "label": "미백 강도",
        },
    },
    "brightness_strength": {
        "en": {
            "label": "brightness strength",
        },
        "zh": {
            "label": "亮度强度",
        },
        "ja": {
            "label": "明るさ強度",
        },
        "ko": {
            "label": "밝기 강도",
        },
    },
    "contrast_strength": {
        "en": {
            "label": "contrast strength",
        },
        "zh": {
            "label": "对比度强度",
        },
        "ja": {
            "label": "コントラスト強度",
        },
        "ko": {
            "label": "대비 강도",
        },
    },
    "sharpen_strength": {
        "en": {
            "label": "sharpen strength",
        },
        "zh": {
            "label": "锐化强度",
        },
        "ja": {
            "label": "シャープネス強度",
        },
        "ko": {
            "label": "선명도 강도",
        },
    },
    "saturation_strength": {
        "en": {
            "label": "saturation strength",
        },
        "zh": {
            "label": "饱和度强度",
        },
        "ja": {
            "label": "彩度強度",
        },
        "ko": {
            "label": "채도 강도",
        },
    },
    "plugin": {
        "en": {
            "label": "Plugin",
            "choices": ["Face Alignment", "Horizontal Flip", "Layout Photo Crop Line", "JPEG Format", "Five Inch Paper"],
            "value": ["Layout Photo Crop Line"]
        },
        "zh": {
            "label": "插件",
            "choices": ["人脸对齐", "水平翻转", "布局照片裁剪线", "JPEG格式", "五寸纸"],
            "value": ["布局照片裁剪线"]
        },
        "ja": {
            "label": "プラグイン",
            "choices": ["顔の整列", "水平反転", "レイアウト写真のトリミングライン", "JPEG形式", "五インチ用紙"],
            "value": ["レイアウト写真のトリミングライン"]
        },
        "ko": {
            "label": "플러그인",
            "choices": ["얼굴 정렬", "수평 뒤집기", "레이아웃 사진 자르기 라인", "JPEG 형식", "5인치 용지"],
            "value": ["레이아웃 사진 자르기 라인"]
        },
    },
    "template_photo": {
        "en": {
            "label": "Social Media Template Photo",
        },
        "zh": {
            "label": "社交媒体模板照片",
        },
        "ja": {
            "label": "SNS テンプレート写真",
        },
        "ko": {
            "label": "SNS 템플릿 사진",
        },
    },
    "print_tab": {
        "en": {
            "label": "Print Layout",
        },
        "zh": {
            "label": "打印布局",
        },
        "ja": {
            "label": "印刷レイアウト",
        },
        "ko": {
            "label": "인쇄 레이아웃",
        },
    },
    "print_switch": {
        "shape": [[1205, 1795], [1051, 1500], [2479, 3508], [1051, 1500], [1205, 1795]],
        "en": {
            "label": "Paper size",
            "choices": ["6 inch", "5 inch", "A4", "3R", "4R"],
        },
        "zh": {
            "label": "纸张大小",
            "choices": ["6寸", "5寸", "A4", "3R", "4R"],
        },
        "ja": {
            "label": "用紙サイズ",
            "choices": ["6インチ", "5インチ", "A4", "3R", "4R"],
        },
        "ko": {
            "label": "용지 크기",
            "choices": ["6인치", "5인치", "A4", "3R", "4R"],
        },
    },
    "compliance_report" : {
        "en": {
            "label": "Compliance Report",
        },
        "zh": {
            "label": "合规报告",
        },
        "ja": {
            "label": "コンプライアンスレポート",
        },
        "ko": {
            "label": "준수 보고서",
        },
    },
}