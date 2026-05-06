import streamlit as st
import joblib
import os
import sys

# إضافة مجلد src إلى مسار النظام لاستيراد دوال استخراج الخصائص
sys.path.append(os.path.abspath('src'))
from feature_extraction import extract_features

# إعدادات الصفحة
st.set_page_config(
    page_title="Cat vs Dog Audio Classifier",
    page_icon="🐱🐶",
    layout="centered"
)

# عنوان التطبيق
st.title("🐱🐶 تصنيف أصوات الحيوانات: قطة أم كلب؟")
st.write("قم برفع ملف صوتي بصيغة (WAV) وسيقوم نموذج الذكاء الاصطناعي بتحديد ما إذا كان الصوت يخص قطة أم كلباً.")

# رفع الملف
uploaded_file = st.file_uploader("اختر ملفاً صوتياً...", type=["wav"])

if uploaded_file is not None:
    # عرض مشغل الصوت
    st.audio(uploaded_file, format='audio/wav')
    
    # عند الضغط على زر التنبؤ
    if st.button("توقع الصوت 🚀"):
        with st.spinner("⏳ جاري تحليل الصوت..."):
            # حفظ الملف المرفوع بشكل مؤقت للتمكن من معالجته
            temp_path = "temp_audio.wav"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            try:
                model_path = os.path.join("model", "trained_model.pkl")
                
                # التحقق من وجود النموذج
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    
                    # استخراج الخصائص
                    features = extract_features(temp_path)
                    
                    # التنبؤ
                    prediction = model.predict(features.reshape(1, -1))
                    
                    # تعيين النتيجة
                    classes = {0: 'النتيجة: قطة 🐱', 1: 'النتيجة: كلب 🐶'}
                    result = classes[prediction[0]]
                    
                    # عرض النتيجة
                    st.success(f"### {result}")
                else:
                    st.error("⚠️ لم يتم العثور على النموذج المدرب! يرجى إضافة بيانات في المجلدات وتدريب النموذج أولاً باستخدام `train.py`.")
                    
            except Exception as e:
                st.error(f"حدث خطأ أثناء معالجة الملف: {e}")
                
            finally:
                # مسح الملف المؤقت
                if os.path.exists(temp_path):
                    os.remove(temp_path)

st.markdown("---")
st.markdown("💡 **ملاحظة:** تأكد من أن الملف الصوتي واضح وأن مدته مناسبة للحصول على دقة أفضل.")
