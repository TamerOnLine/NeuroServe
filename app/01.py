def model_size(in_features: int, hidden: int, out_features: int, dtype_bytes: int = 4):
    """
    يحسب عدد الأوزان وحجم الذاكرة المطلوبة لنموذج MLP بسيط.
    - in_features: حجم المدخل
    - hidden: حجم الطبقة المخفية
    - out_features: حجم المخرجات
    - dtype_bytes: حجم كل وزن (افتراضياً float32 = 4 بايت)
    """
    weights = in_features * hidden + hidden * out_features
    memory_bytes = weights * dtype_bytes
    return {
        "in_features": in_features,
        "hidden": hidden,
        "out_features": out_features,
        "total_weights": weights,
        "memory_MB": round(memory_bytes / (1024**2), 2),
        "memory_GB": round(memory_bytes / (1024**3), 2),
    }

# 🟢 أمثلة جاهزة:
tiny = model_size(512, 1024, 10)
big = model_size(1000, 100000, 1000)

print("TinyNet:", tiny)
print("BigNet:", big)
