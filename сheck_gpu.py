#!/usr/bin/env python3
"""
Проверка доступности GPU для нейросетей
"""

import torch
import sys

def check_gpu():
    print("🔍 ПРОВЕРКА GPU ДЛЯ НЕЙРОСЕТЕЙ")
    print("=" * 50)
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ Доступно GPU: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({memory:.1f} GB)")
            
        print(f"🎮 CUDA версия: {torch.version.cuda}")
        print("🚀 Нейросети будут использовать GPU!")
        
    else:
        print("❌ GPU не доступен")
        print("💡 Установите CUDA и torch с GPU поддержкой")
        print("💡 Или нейросети будут работать на CPU")
    
    print("=" * 50)

if __name__ == '__main__':
    check_gpu()