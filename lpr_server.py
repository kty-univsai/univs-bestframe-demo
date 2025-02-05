import ultimateAlprSdk
import argparse
import json
import os.path
from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)
JSON_CONFIG = {
    "debug_level": "info",
    "debug_write_input_image_enabled": False,
    "debug_internal_data_path": ".",
    
    "num_threads": -1,
    "gpgpu_enabled": True,
    "max_latency": -1,

    "klass_vcr_gamma": 1.5,
    
    "detect_roi": [0, 0, 0, 0],
    "detect_minscore": 0.1,

    "car_noplate_detect_min_score": 0.8,
    
    "pyramidal_search_enabled": True,
    "pyramidal_search_sensitivity": 0.28,
    "pyramidal_search_minscore": 0.3,
    "pyramidal_search_min_image_size_inpixels": 800,
    
    "recogn_rectify_enabled": True,
    "recogn_minscore": 0.3,
    "recogn_score_type": "min",

    "assets_folder": "/home/univs/samples/ultimateALPR-SDK/assets",
    "charset": "korean", 
    "openvino_enabled": False
}

IMAGE_TYPES_MAPPING = { 
        'RGB': ultimateAlprSdk.ULTALPR_SDK_IMAGE_TYPE_RGB24,
        'RGBA': ultimateAlprSdk.ULTALPR_SDK_IMAGE_TYPE_RGBA32,
        'L': ultimateAlprSdk.ULTALPR_SDK_IMAGE_TYPE_Y
}


def load_pil_image(file):
    from PIL import Image, ExifTags, ImageOps
    import traceback
    pil_image = Image.open(file)
    img_exif = pil_image.getexif()
    ret = {}
    orientation  = 1
    try:
        if img_exif:
            for tag, value in img_exif.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                ret[decoded] = value
            orientation  = ret["Orientation"]
    except Exception as e:
        print("An exception occurred: {}".format(e))
        traceback.print_exc()

    if orientation > 1:
        pil_image = ImageOps.exif_transpose(pil_image)

    if pil_image.mode in IMAGE_TYPES_MAPPING:
        imageType = IMAGE_TYPES_MAPPING[pil_image.mode]
    else:
        raise ValueError("Invalid mode: %s" % pil_image.mode)

    return pil_image, imageType


@app.route('/lpr_proc', methods=['POST'])
def upload_image():
    # 클라이언트가 업로드한 파일 가져오기
    if 'image' not in request.files:
        return {"error": "No image part"}, 400
    
    file = request.files['image']
    
    if file.filename == '':
        return {"error": "No selected file"}, 400

    try:
        # 이미지를 Pillow 객체로 변환
        image, imageType = load_pil_image(file)
        width, height = image.size
        result = ultimateAlprSdk.UltAlprSdkEngine_process(
                    imageType,
                    image.tobytes(), # type(x) == bytes
                    width,
                    height,
                    0, # stride
                    1 # exifOrientation (already rotated in load_image -> use default value: 1)
                )
        if not result.isOK():
            return jsonify({            
                "code": "failure"            
            })
        else:
            return jsonify({            
                "code": "success",
                "data": result.json()            
            })
    

    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == '__main__':
    ultimateAlprSdk.UltAlprSdkEngine_init(json.dumps(JSON_CONFIG))
    app.run(host='0.0.0.0', port=8020, debug=True)
