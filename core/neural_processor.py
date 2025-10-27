import re
import hashlib
import torch
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class NeuralNewsProcessor:
    def __init__(self):
        logger.info("Загрузка нейросетевых моделей...")
        
        try:
            # Модель для эмбеддингов (русский язык) на GPU
            self.embedding_model = SentenceTransformer(
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                device=self._setup_gpu()
            )
            
            logger.info("✅ Нейросетевые модели загружены успешно")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки моделей: {e}")
            raise

    def _setup_gpu(self):
        """Настройка использования GPU для нейросетей"""
        try:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"🎮 Используется GPU: {gpu_name}")
            else:
                device = torch.device("cpu")
                logger.info("❌ GPU не доступен, используется CPU")
            return device
        except Exception as e:
            logger.error(f"Ошибка настройки GPU: {e}")
            return torch.device("cpu")

    def create_fingerprint(self, text):
        """Создание цифрового отпечатка текста"""
        if not text or len(text.strip()) < 10:
            return "0" * 64
            
        try:
            # Создаем эмбеддинг на GPU
            embedding = self.embedding_model.encode(text, convert_to_tensor=True)
            
            # Конвертируем в хэш
            embedding_np = embedding.cpu().numpy()
            embedding_bytes = embedding_np.tobytes()
            return hashlib.sha256(embedding_bytes).hexdigest()
        except Exception as e:
            logger.error(f"Error creating fingerprint: {e}")
            return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def calculate_interest_score(self, text):
        """Оценка интересности текста"""
        if not text or len(text.strip()) < 20:
            return 0.0
        
        try:
            scores = []
            
            # 1. Оценка длины текста
            length_score = min(len(text) / 500, 1.0) * 0.3
            
            # 2. Оценка информативности через разнообразие слов
            words = text.split()
            if len(words) > 0:
                unique_words = set(words)
                diversity_score = len(unique_words) / len(words)
                scores.append(diversity_score * 0.3)
            
            # 3. Оценка структуры текста
            structure_score = self._calculate_structure_score(text)
            scores.append(structure_score * 0.4)
            
            total_score = length_score + sum(scores)
            return min(total_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating interest score: {e}")
            return 0.5

    def _calculate_structure_score(self, text):
        """Оценка структурного качества текста"""
        score = 0.0
        
        # Наличие чисел
        if any(char.isdigit() for char in text):
            score += 0.2
        
        # Наличие заглавных букв
        if any(char.isupper() for char in text):
            score += 0.2
        
        # Длина предложений
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 0:
            avg_sentence_length = sum(len(sent.split()) for sent in sentences) / len(sentences)
            if 5 <= avg_sentence_length <= 20:
                score += 0.3
        
        # Наличие ключевых слов новостей
        news_keywords = ['новость', 'событие', 'сообщение', 'заявление', 'интервью', 
                        'анализ', 'данные', 'исследование', 'эксперт', 'официально']
        if any(keyword in text.lower() for keyword in news_keywords):
            score += 0.3
            
        return min(score, 1.0)

    def are_posts_similar(self, fingerprint1, fingerprint2):
        """Проверка схожести двух постов"""
        return fingerprint1 == fingerprint2