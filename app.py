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

def order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(img, pts):
    rect = order_points(pts)
    tl, tr, br, bl = rect
    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)
    maxW = max(int(wA), int(wB))
    maxH = max(int(hA), int(hB))
    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype='float32')
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (maxW, maxH))

def detect_document(img):
    h, w = img.shape[:2]
    # Resize pour la détection
    scale = 800 / max(h, w)
    small = cv2.resize(img, (int(w*scale), int(h*scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 30, 100)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    doc_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > (small.shape[0]*small.shape[1]*0.1):
            doc_cnt = approx
            break
    if doc_cnt is not None:
        doc_cnt = (doc_cnt.reshape(4,2) / scale).astype('float32')
        return four_point_transform(img, doc_cnt), True
    return img, False

def enhance(img, mode='auto'):
    if mode == 'document':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10
        )
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    elif mode == 'nb':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        result = clahe.apply(gray)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    result = cv2.filter2D(result, -1, kernel)
    result = cv2.fastNlMeansDenoisingColored(result, None, 5, 5, 7, 21)
    return result

@app.route('/enhance', methods=['POST'])
def enhance_route():
    try:
        data = request.json
        img = b64_to_img(data['image'])
        mode = data.get('mode', 'auto')
        auto_crop = data.get('auto_crop', True)
        cropped = False
        if auto_crop:
            img, cropped = detect_document(img)
        result = enhance(img, mode)
        return jsonify({'image': img_to_b64(result), 'ok': True, 'cropped': cropped})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
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
