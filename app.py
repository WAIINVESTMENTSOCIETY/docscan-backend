from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import re

app = Flask(__name__)
CORS(app)

def b64_to_img(b64):
    b64 = re.sub(r'^data:image/\w+;base64,', '', b64)
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def img_to_b64(img, quality=92):
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()

def enhance(img, mode='auto'):
    # Redresse et améliore l'image
    h, w = img.shape[:2]

    if mode == 'document':
        # Convertit en gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Seuillage adaptatif = fond blanc / texte noir parfait
        result = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10
        )
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    elif mode == 'nb':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # CLAHE = contraste adaptatif local
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        result = clahe.apply(gray)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    else:  # auto ou couleur
        # CLAHE sur luminosité
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Sharpening sur tous les modes
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    result = cv2.filter2D(result, -1, kernel)

    # Débruitage léger
    result = cv2.fastNlMeansDenoisingColored(result, None, 5, 5, 7, 21)

    return result

def remove_bg(img):
    h, w = img.shape[:2]
    # Masque GrabCut pour isoler le document du fond
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1,65), np.float64)
    fgd = np.zeros((1,65), np.float64)
    rect = (int(w*0.05), int(h*0.05), int(w*0.9), int(h*0.9))
    cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    result = img * mask2[:,:,np.newaxis]
    # Remplace le fond par du blanc
    white = np.ones_like(img) * 255
    result = np.where(mask2[:,:,np.newaxis]==0, white, result)
    return result

@app.route('/enhance', methods=['POST'])
def enhance_route():
    try:
        data = request.json
        img = b64_to_img(data['image'])
        mode = data.get('mode', 'auto')
        bg_remove = data.get('bg_remove', False)

        if bg_remove:
            img = remove_bg(img)

        result = enhance(img, mode)
        return jsonify({'image': img_to_b64(result), 'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
