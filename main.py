import cv2
import pytesseract
from PIL import Image
import re
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import os
import platform

class EnhancedCameraDetector:
    def __init__(self):
        self.available_cameras = []

    def find_available_cameras(self):
        """Detecta câmeras disponíveis"""
        print("🔍 Detectando câmeras disponíveis...")

        self.available_cameras = []

        # Backends para testar baseado no OS
        system = platform.system().lower()
        if "windows" in system:
            backends_to_test = [
                (cv2.CAP_DSHOW, "DirectShow"),
                (cv2.CAP_MSMF, "Media Foundation"),
                (cv2.CAP_ANY, "Padrão")
            ]
        else:
            backends_to_test = [
                (cv2.CAP_V4L2, "V4L2"),
                (cv2.CAP_ANY, "Padrão")
            ]

        for backend_id, backend_name in backends_to_test:
            for cam_index in range(4):
                try:
                    cap = cv2.VideoCapture(cam_index, backend_id)

                    if cap.isOpened():
                        ret, frame = cap.read()

                        if ret and frame is not None:
                            height, width = frame.shape[:2]

                            camera_info = {
                                'index': cam_index,
                                'backend': backend_id,
                                'backend_name': backend_name,
                                'width': width,
                                'height': height,
                                'working': True
                            }

                            if not any(cam['index'] == cam_index and cam['backend'] == backend_id
                                     for cam in self.available_cameras):
                                self.available_cameras.append(camera_info)
                                print(f"   ✅ Câmera {cam_index} ({backend_name}): {width}x{height}")

                    cap.release()

                except Exception:
                    continue

        if self.available_cameras:
            print(f"✅ {len(self.available_cameras)} câmera(s) detectada(s)")
            return True
        else:
            print("❌ Nenhuma câmera detectada")
            return False

    def get_best_camera(self):
        if not self.available_cameras:
            return None

        # Prioriza câmeras com maior resolução
        best_camera = max(self.available_cameras,
                         key=lambda x: x['width'] * x['height'])

        return best_camera

class BalancedDocumentProcessor:
    def __init__(self, tesseract_path=None):
        if tesseract_path and os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            print("✅ Tesseract configurado")
        else:
            print("⚠️  Usando Tesseract do PATH do sistema")

    def enhance_image_basic(self, frame):
        """
        Melhorias BÁSICAS e eficazes para OCR
        """
        # 1. Converte para escala de cinza
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # 2. Melhora contraste com CLAHE (menos agressivo)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # 3. Filtro bilateral suave
        bilateral = cv2.bilateralFilter(enhanced, 5, 50, 50)

        # 4. Threshold adaptativo simples
        thresh = cv2.adaptiveThreshold(bilateral, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

        return thresh

    def extract_text_balanced(self, frame):
        """
        OCR EQUILIBRADO - funciona melhor que o anterior
        """
        try:
            # Tenta PRIMEIRO com imagem original
            print("🔍 Tentando OCR básico primeiro...")

            # Converte para PIL
            if len(frame.shape) == 3:
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.fromarray(frame)

            # OCR simples primeiro
            text_simple = pytesseract.image_to_string(pil_img, lang='por')
            # lang = "por" (para tradução e identificação em portugues com acentos, deve conter o por.traineddata)

            if text_simple and len(text_simple.strip()) > 3:
                print(f"✅ OCR básico funcionou: {text_simple[:50]}...")
                return text_simple.strip()

            # Se não funcionou, tenta com melhorias
            print("🔧 Aplicando melhorias na imagem...")
            enhanced_frame = self.enhance_image_basic(frame)

            # Upscale moderado (só x1.5)
            height, width = enhanced_frame.shape[:2]
            upscaled = cv2.resize(enhanced_frame,
                                (int(width * 1.5), int(height * 1.5)),
                                interpolation=cv2.INTER_CUBIC)

            pil_enhanced = Image.fromarray(upscaled)

            # Configurações do Tesseract mais simples
            configs = [
                r'--oem 3 --psm 6 -l por',
                r'--oem 3 --psm 7 -l por',
                r'--oem 3 --psm 8 -l por',
                r'--oem 3 --psm 6',
                r'--oem 3 --psm 13'
            ]

            best_text = ""

            for config in configs:
                try:
                    text = pytesseract.image_to_string(pil_enhanced, config=config)
                    if text and len(text.strip()) > len(best_text):
                        best_text = text.strip()
                except:
                    continue

            if best_text:
                print(f"✅ OCR melhorado funcionou: {best_text[:50]}...")
                return best_text

            print("❌ Nenhum texto detectado com OCR")
            return ""

        except Exception as e:
            print(f"❌ Erro no OCR: {e}")
            return ""

    def parse_flexible_data(self, text):
        """
        Parser com FILTRO INTELIGENTE para nomes
        """
        if not text or len(text.strip()) < 2:
            return {}

        data = {}
        text_clean = text.strip()

        print(f"🔤 Analisando: '{text_clean}'")

        # PALAVRAS PROIBIDAS para nomes (lista extensa)
        palavras_proibidas = {
            # Rótulos de campos
            'nome', 'name', 'telefone', 'tel', 'celular', 'fone', 'whats', 'whatsapp',
            'cliente', 'client', 'person', 'pessoa', 'contato', 'contact','fone', 'lefone',
            'elefone', 'tel', 'tell', 'elefone', 'one'
            # Palavras técnicas
            'cpf', 'rg', 'cep', 'endereco', 'email', 'data', 'nascimento',
            'profissao', 'cargo', 'empresa', 'trabalho', 'app', '.app'
            # Palavras comuns em documentos
            'documento', 'registro', 'numero', 'codigo', 'protocolo',
            'servico', 'produto', 'valor', 'preco', 'total', 'arquivo', 'rquivo', 'quivo', 'editar'
            'formatar', 'exibir', 'h1', 'h2', 'h3', 'título', 'subtítulo'
            # Conectivos e preposições
            'para', 'com', 'sem', 'por', 'em', 'da', 'do', 'dos', 'das', 'me'
            # Outras
            'favor', 'obrigado', 'atenciosamente', 'cordialmente'
        }

        # NOME - Padrões com validação inteligente
        nome_patterns = [
            # Nome após "nome:", "name:", etc.
            r'(?:nome|name|client|cliente)\s*:?\s*([a-záàâãéêíóôõúç\s]{3,50})',
            # Nome no início de linha (2+ palavras)
            r'^([A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]+\s+[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]+.*?)(?:\s*\d|\s*$)',
            # Qualquer sequência de 2+ palavras com inicial maiúscula
            r'([A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]{2,}\s+[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]{2,})',
            # Palavras em maiúsculas separadas por espaços
            r'([A-ZÁÀÂÃÉÊÍÓÔÕÚÇ]{3,}\s+[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ]{3,})'
        ]

        # TELEFONE - Padrões flexíveis
        telefone_patterns = [
            # Com rótulos
            r'(?:tel|telefone|celular|fone|whats)\s*:?\s*([\(\)\d\s\-\.]{8,15})',
            # Formato (XX) XXXXX-XXXX
            r'(\(\d{2}\)\s*\d{4,5}[\s\-]*\d{4})',
            # Formato XX XXXXX-XXXX
            r'(\d{2}\s+\d{4,5}[\s\-]+\d{4})',
            # 11 dígitos seguidos
            r'(\d{11})',
            # 10 dígitos seguidos
            r'(\d{10})',
            # Números com separadores
            r'(\d{2}[\s\-\.]+\d{4,5}[\s\-\.]+\d{4})',
        ]

        def is_valid_nome(nome_candidato):
            """
            Validação INTELIGENTE para nomes
            """
            nome_limpo = ' '.join(nome_candidato.split()).strip()
            palavras = nome_limpo.lower().split()

            # Verifica tamanho mínimo
            if len(nome_limpo) < 5 or len(palavras) < 2:
                return False

            # Verifica se não tem muitos dígitos
            if re.search(r'\d{3,}', nome_limpo):
                return False

            # FILTRO PRINCIPAL: verifica palavras proibidas
            for palavra in palavras:
                palavra_clean = re.sub(r'[^\w]', '', palavra.lower())
                if palavra_clean in palavras_proibidas:
                    print(f"❌ Nome rejeitado (palavra proibida '{palavra_clean}'): {nome_limpo}")
                    return False

            # Verifica se todas as palavras têm tamanho mínimo
            if any(len(p) < 2 for p in palavras):
                return False

            # Verifica se não é só números ou caracteres especiais
            if not re.search(r'[a-záàâãéêíóôõúç]', nome_limpo.lower()):
                return False

            # Verifica padrões suspeitos (muitas maiúsculas seguidas sem espaços)
            if re.search(r'[A-Z]{8,}', nome_limpo):
                print(f"❌ Nome rejeitado (muitas maiúsculas): {nome_limpo}")
                return False

            return True

        # Busca NOME com validação inteligente
        for i, pattern in enumerate(nome_patterns):
            matches = re.finditer(pattern, text_clean, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                nome_candidato = match.group(1).strip()

                if is_valid_nome(nome_candidato):
                    nome_final = ' '.join(nome_candidato.split()).title()
                    data['nome'] = nome_final
                    print(f"✅ Nome VÁLIDO encontrado: {data['nome']}")
                    break

            if 'nome' in data:
                break

        # Busca TELEFONE (sem mudanças, está funcionando bem)
        for i, pattern in enumerate(telefone_patterns):
            matches = re.finditer(pattern, text_clean, re.IGNORECASE)
            for match in matches:
                tel_candidato = match.group(1).strip()
                tel_numeros = re.sub(r'[^\d]', '', tel_candidato)

                # Validação básica para Brasil
                if len(tel_numeros) >= 10 and len(tel_numeros) <= 11:
                    data['telefone'] = tel_numeros
                    print(f"✅ Telefone encontrado: {tel_numeros}")
                    break

            if 'telefone' in data:
                break

        return data

class EnhancedFormFiller:
    def __init__(self, headless=False):
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')

        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        self.wait = WebDriverWait(self.driver, 10)
        print("✅ Navegador configurado")

    def open_form(self, url):
        self.driver.get(url)
        time.sleep(2)
        print(f"✅ Formulário aberto: {os.path.basename(url)}")

    def fill_field(self, field_id, value):
        try:
            element = self.wait.until(EC.presence_of_element_located((By.ID, field_id)))
            self.driver.execute_script("arguments[0].scrollIntoView();", element)
            time.sleep(0.5)
            element.clear()
            element.send_keys(str(value))

            # Destaca o campo preenchido
            self.driver.execute_script("arguments[0].style.borderColor = '#28a745'; arguments[0].style.backgroundColor = '#f8fff8';", element)

            return True
        except Exception as e:
            print(f"❌ Erro ao preencher {field_id}: {e}")
            return False

    def fill_form(self, data):
        successful_fills = 0

        if 'nome' in data and data['nome']:
            if self.fill_field('nome_completo', data['nome']):
                successful_fills += 1
                print(f"   ✅ Nome preenchido: {data['nome']}")

        if 'telefone' in data and data['telefone']:
            # Formata o telefone para exibição
            tel_clean = data['telefone']
            if len(tel_clean) == 11:
                tel_formatted = f"({tel_clean[:2]}) {tel_clean[2:7]}-{tel_clean[7:]}"
            else:
                tel_formatted = f"({tel_clean[:2]}) {tel_clean[2:6]}-{tel_clean[6:]}"

            if self.fill_field('telefone_field', tel_formatted):
                successful_fills += 1
                print(f"   ✅ Telefone preenchido: {tel_formatted}")

        return successful_fills

    def close(self):
        try:
            self.driver.quit()
        except:
            pass

class BalancedLiveAutomation:
    def __init__(self, tesseract_path=None):
        self.processor = BalancedDocumentProcessor(tesseract_path)
        self.camera_detector = EnhancedCameraDetector()
        self.form_filler = None
        self.camera = None
        self.is_running = False
        self.last_successful_data = {}
        self.last_process_time = 0
        self.process_cooldown = 3  # Aumentado para evitar spam
        self.frame_skip = 0

    def setup_camera(self):
        print("🎥 Configurando câmera...")

        if not self.camera_detector.find_available_cameras():
            return False

        best_camera = self.camera_detector.get_best_camera()

        if not best_camera:
            return False

        try:
            self.camera = cv2.VideoCapture(best_camera['index'], best_camera['backend'])

            # Configurações básicas
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Resolução menor para melhor performance
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ret, frame = self.camera.read()

            if ret and frame is not None:
                print(f"✅ Câmera configurada: {frame.shape[1]}x{frame.shape[0]}")
                return True
            else:
                print("❌ Câmera não consegue ler frames")
                return False

        except Exception as e:
            print(f"❌ Erro ao configurar câmera: {e}")
            return False

    def setup_form_automation(self, form_url, headless=False):
        try:
            self.form_url = form_url
            self.form_filler = EnhancedFormFiller(headless=headless)
            self.form_filler.open_form(form_url)
            return True
        except Exception as e:
            print(f"❌ Erro ao configurar formulário: {e}")
            return False

    def process_frame(self, frame):
        current_time = time.time()

        # Verifica cooldown
        if current_time - self.last_process_time < self.process_cooldown:
            return None

        # Pula frames para não sobrecarregar
        self.frame_skip += 1
        if self.frame_skip < 15:  # Processa a cada 15 frames
            return None
        self.frame_skip = 0

        try:
            print("🔍 Processando frame...")
            time.sleep(2)
            # Usa OCR balanceado
            text = self.processor.extract_text_balanced(frame)

            if not text:
                print("❌ Nenhum texto detectado")
                return None

            # Usa parser flexível
            data = self.processor.parse_flexible_data(text)

            print(f"📊 Dados extraídos: {data}")

            # Aceita se tem pelo menos 1 campo
            if data and len(data) >= 1:
                if self.data_changed(data, self.last_successful_data):
                    self.last_process_time = current_time
                    return data

            return None

        except Exception as e:
            print(f"❌ Erro ao processar frame: {e}")
            return None

    def data_changed(self, new_data, old_data):
        if not old_data:
            return True

        for field in ['nome', 'telefone']:
            if field in new_data and field in old_data:
                if new_data[field] != old_data[field]:
                    return True
            elif field in new_data or field in old_data:
                return True

        return False

    def fill_form_with_confirmation(self, data):
        try:
            print("\n" + "🎯" + "="*50 + "🎯")
            print("        DADOS DETECTADOS!")
            print("🎯" + "="*50 + "🎯")

            print("📋 Dados encontrados:")
            for key, value in data.items():
                if key == 'telefone' and len(value) >= 10:
                    # Formata telefone
                    if len(value) == 11:
                        formatted = f"({value[:2]}) {value[2:7]}-{value[7:]}"
                    else:
                        formatted = f"({value[:2]}) {value[2:6]}-{value[6:]}"
                    print(f"   📞 {key.upper()}: {formatted}")
                else:
                    print(f"   📝 {key.upper()}: {value}")

            print(f"\n📝 Preenchendo formulário...")
            successful_fills = self.form_filler.fill_form(data) # type: ignore

            if successful_fills > 0:
                print(f"\n🎉 {successful_fills} campo(s) preenchido(s)!")

                print(f"\n🔍 VERIFIQUE OS DADOS!")
                print(f"   ✅ Pressione ENTER para continuar")
                print(f"   ❌ Ou 'q' para sair")

                user_input = input("\n➡️  ENTER para próximo (ou 'q' para sair): ")

                if user_input.lower() == 'q':
                    self.is_running = False
                    return False

                self.last_successful_data = data.copy()
                print("✅ Continuando...")
                return True
            else:
                print("❌ Falha ao preencher")
                return False

        except Exception as e:
            print(f"❌ Erro: {e}")
            return False

    def run_live_processing(self):
        print("\n🚀 INICIANDO OCR EQUILIBRADO")
        print("="*50)
        print("📹 Aponte para NOME e/ou TELEFONE")
        print("💡 Dicas:")
        print("   • 💡 Boa iluminação")
        print("   • 📏 Distância: 15-20cm")
        print("   • ⏸️  Mantenha parado 3-4 segundos")
        print("   • 🔍 Texto grande e legível")
        print("   • ❌ 'q' ou ESC para sair")
        print("="*50)

        self.is_running = True

        while self.is_running:
            try:
                ret, frame = self.camera.read() # type: ignore

                if not ret:
                    print("❌ Erro ao capturar frame")
                    break

                # Interface simples
                display_frame = frame.copy()
                h, w = display_frame.shape[:2]

                # Título
                cv2.putText(display_frame, "OCR - Nome + Telefone",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Status
                current_time = time.time()
                time_left = max(0, int(self.process_cooldown - (current_time - self.last_process_time)))

                if time_left > 0:
                    cv2.putText(display_frame, f"Aguarde: {time_left}s",
                               (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                else:
                    cv2.putText(display_frame, "DETECTANDO...",
                               (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Área de foco
                margin = 50
                cv2.rectangle(display_frame, (margin, margin),
                             (w - margin, h - margin), (255, 255, 0), 2)

                cv2.imshow('OCR Equilibrado', display_frame)

                # Processa frame
                data = self.process_frame(frame)

                if data:
                    cv2.destroyAllWindows()
                    if not self.fill_form_with_confirmation(data):
                        break

                    if self.is_running:
                        print("📹 Câmera reativada...")

                # Verifica teclas
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

            except KeyboardInterrupt:
                print("\n⏹️  Interrompido")
                break
            except Exception as e:
                print(f"❌ Erro: {e}")
                break

        self.is_running = False

    def cleanup(self):
        self.is_running = False

        if self.camera:
            self.camera.release()

        cv2.destroyAllWindows()

        if self.form_filler:
            self.form_filler.close()

        print("✅ Recursos liberados")

def main():
    print("🤖 AUTOMAÇÃO OCR EQUILIBRADA")
    print("=" * 50)

    # Configurações padrão do tesseract
    tesseract_path = None

    # Para Windows (descomente para ativar o tesseract padrão):
    tesseract_path = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

    # Arquivo HTML ("Mudar de acordo com o Path certo para o html")
    html_file = "C:/Users/user/Documents/Paulao_piton/Marketing/Prototipo/formulario_simples.html"

    if not os.path.exists(html_file):
        print(f"❌ Arquivo {html_file} não encontrado!")
        return

    form_url = "file://" + os.path.abspath(html_file)

    # Cria automação equilibrada
    automation = BalancedLiveAutomation(tesseract_path)

    try:
        print("🎥 Configurando câmera...")
        if not automation.setup_camera():
            print("❌ Falha na câmera!")
            return

        print("🌐 Configurando formulário...")
        if not automation.setup_form_automation(form_url, headless=False):
            print("❌ Erro no formulário")
            return

        print("✅ Sistema configurado!")
        print(f"\n📋 Configuração:")
        print(f"   📹 Câmera: 640x480")
        print(f"   🌐 Formulário: {os.path.basename(html_file)}")
        print(f"   📝 Campos: Nome + Telefone")
        print(f"   🧠 OCR: Equilibrado e flexível")

        input(f"\n▶️  ENTER para iniciar...")

        # Inicia processamento
        automation.run_live_processing()
    except KeyboardInterrupt:
        print("\n⏹️  Interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
    finally:
        automation.cleanup()
        print(f"👋 Finalizado!")
if __name__ == "__main__":
    main()
