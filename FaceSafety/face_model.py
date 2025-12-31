import cv2
import numpy as np
from insightface.app import FaceAnalysis
from config import DEVICE, DETSCOREBAR  # 确保导入配置

class FaceRecognitionModel:
    def __init__(self, model_name="buffalo_l", device=DEVICE):
        """初始化ArcFace模型，强制使用指定设备"""
        
        # 根据设备配置providers
        if "cuda" in device.lower():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            ctx_id = 0  # GPU device id
            print(f"🚀 尝试使用GPU: {device}")
        else:
            providers = ['CPUExecutionProvider']
            ctx_id = -1
            print(f"💻 使用CPU")
        
        # 初始化模型并强制指定providers
        self.app = FaceAnalysis(name=model_name, providers=providers)
        
        # 准备模型时明确指定设备
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        
        # 验证实际使用的设备
        session = self.app.models['detection'].session
        actual_providers = session.get_providers()
        print(f"✅ 模型实际使用: {actual_providers[0]}")
        
        if 'CUDAExecutionProvider' not in actual_providers:
            print("⚠️ 警告：未使用GPU，请检查CUDA和onnxruntime-gpu安装")


    def _preprocess_image(self, img):
        """图像预处理：去噪、锐化、颜色校正"""
        
        # 1. 非局部均值去噪（保留细节，适合自然图像）
        #    h=10: 颜色强度去噪强度, templateWindowSize=7, searchWindowSize=21
        denoised = cv2.fastNlMeansDenoisingColored(img, None, h=10, 
                                                    templateWindowSize=7, 
                                                    searchWindowSize=21)
        
        # 2. 锐化（补偿去噪导致的轻微模糊）
        kernel = np.array([[-0.1,-0.1,-0.1],
                        [-0.1, 2.0,-0.1],
                        [-0.1,-0.1,-0.1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # 3. 对比度受限自适应直方图均衡化（CLAHE）
        #    改善光照不均，提升低光照区域细节
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        img_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 4. 亮度/对比度微调（可选，防止过曝/过暗）
        #    alpha: 对比度 (1.0-3.0), beta: 亮度 (0-100)
        alpha, beta = 1.2, 10
        adjusted = cv2.convertScaleAbs(img_enhanced, alpha=alpha, beta=beta)
        
        return adjusted
    

    def extract_feature(self, image_path, use_enhancement=True):
        # ...（保持不变）...
        img = cv2.imread(image_path)
        if img is None:
            return False, None, None
        
        # 添加超时保护，防止卡死
        try:
            faces = self.app.get(img)
        except Exception as e:
            print(f"❌ 处理 {image_path} 时出错: {e}")
            return False, None, None
        
        if len(faces) == 0 or faces[0].det_score < DETSCOREBAR:
            if use_enhancement:
                img = self._preprocess_image(img)
                faces = self.app.get(img)
                if len(faces) != 0:
                    face = faces[0]
                    embedding = face.embedding
                    bbox = face.bbox
                    return True, embedding, bbox
            return False, None, None
        
        face = faces[0]
        # print(face)
        embedding = face.embedding
        bbox = face.bbox
        
        return True, embedding, bbox
    
    # ...（其余方法保持不变）...

    def detect_face(self, image_path):
        """检测图片中是否有人脸"""
        img = cv2.imread(image_path)
        if img is None:
            return False

        faces = self.app.get(img)
        return len(faces) > 0
