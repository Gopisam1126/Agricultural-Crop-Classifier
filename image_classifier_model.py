import os
import cv2
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import torch
import torchvision.transforms as transforms
from torchvision import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = models.VGG16_Weights.DEFAULT
cnn = models.vgg16(pretrained=True).to(device)
cnn.eval()
# Preprocessing pipeline for PyTorch VGG16
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(img_path, lbp_params=(8, 1), gabor_params=None,
                     orb_params=None, bovw_model=None):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = []
    # LBP
    lbp = local_binary_pattern(gray, P=lbp_params[0], R=lbp_params[1], method='uniform')
    (hist_lbp, _) = np.histogram(lbp.ravel(), bins=np.arange(0, lbp_params[0] + 3), range=(0, lbp_params[0] + 2))
    hist_lbp = hist_lbp.astype("float")
    hist_lbp /= hist_lbp.sum()
    features.append(hist_lbp)

    # GLCM
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))
    features.append([contrast, entropy])

    # Gabor filters
    if gabor_params is None:
        gabor_params = {'ksize': (21, 21), 'sigma': 4.0,
                        'theta_list': np.arange(0, np.pi, np.pi/4),
                        'lambd': 10.0, 'gamma': 0.5, 'psi': 0}
    gabor_feats = []
    for theta in gabor_params['theta_list']:
        kern = cv2.getGaborKernel(
            gabor_params['ksize'], gabor_params['sigma'], theta,
            gabor_params['lambd'], gabor_params['gamma'], gabor_params['psi'], ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kern)
        gabor_feats.append(filtered.mean())
    features.append(gabor_feats)

    # ORB + BoVW
    orb = cv2.ORB_create(**(orb_params or {}))
    kps, des = orb.detectAndCompute(gray, None)
    if des is None:
        des = np.zeros((1, orb.descriptorSize()))
    if bovw_model is not None:
        hist_bovw = np.zeros(bovw_model.n_clusters)
        words = bovw_model.predict(des)
        for w in words:
            hist_bovw[w] += 1
        hist_bovw = hist_bovw.astype('float')
        hist_bovw /= (hist_bovw.sum() + 1e-10)
        features.append(hist_bovw)
    else:
        features.append(des.flatten())

    # Deep CNN embeddings via PyTorch VGG16
    img_pil = Image.open(img_path).convert('RGB')
    x = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = cnn.features(x)
        emb = feats.view(feats.size(0), -1)
    features.append(emb.cpu().numpy().flatten())

    feat_vec = np.hstack([f if isinstance(f, np.ndarray) else np.array(f).ravel() for f in features])
    return feat_vec

descriptor_list = []
for root, _, files in os.walk('Agricultural-crops'):
    for fname in files:
        path = os.path.join(root, fname)
        gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        _, des = orb.detectAndCompute(gray, None)
        if des is not None:
            descriptor_list.extend(des)
bovw = MiniBatchKMeans(n_clusters=50, random_state=42)
bovw.fit(descriptor_list)

X, y = [], []
for cls in os.listdir('Agricultural-crops'):
    cls_dir = os.path.join('Agricultural-crops', cls)
    for fname in os.listdir(cls_dir):
        path = os.path.join(cls_dir, fname)
        vec = extract_features(path, bovw_model=bovw)
        X.append(vec)
        y.append(cls)
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
pca = PCA(n_components=100, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
selector = RFE(estimator=SVC(kernel='linear'), n_features_to_select=200)
X_train_sel = selector.fit_transform(X_train_pca, y_train)

X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)
X_test_sel = selector.transform(X_test_pca)

svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train_sel, y_train)

y_pred = svm.predict(X_test_sel)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump({'scaler': scaler, 'pca': pca, 'selector': selector, 'svm': svm, 'bovw': bovw}, 'crop_classifier_pipeline_pytorch.pkl')

def predict_image(img_path, pipeline_path='crop_classifier_pipeline_pytorch.pkl'):
    data = joblib.load(pipeline_path)
    feat = extract_features(img_path, bovw_model=data['bovw'])
    scaled = data['scaler'].transform([feat])
    pca_f = data['pca'].transform(scaled)
    sel = data['selector'].transform(pca_f)
    return data['svm'].predict(sel)[0]

if __name__ == '__main__':
    print("Predicted class:", predict_image('cherry.jpeg'))
