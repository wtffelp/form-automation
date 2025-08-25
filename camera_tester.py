import cv2

# Forçar backend Media Foundation
web_cam = 0
ip = "http://192.168.1.6:4747/video"
cap = cv2.VideoCapture(web_cam, cv2.CAP_MSMF)

if not cap.isOpened():
    print("❌ Não conseguiu abrir a câmera com Media Foundation")
    exit()

# Pega resolução da câmera
width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps    = cap.get(cv2.CAP_PROP_FPS)
print(f"📸 Resolução: {int(width)}x{int(height)} @ {fps:.1f} FPS")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Falha ao capturar frame.")
        break

    cv2.imshow("Câmera (Media Foundation)", frame)

    # ESC para sair
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
